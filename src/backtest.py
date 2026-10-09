"""
Backtesting del modelo de pools de liquidación — Fase 2b

Validación walk-forward del modelo (pool_model.py) contra liquidaciones reales.
Los resultados se guardan en results/last_backtest.md (el CI los commitea al repo
para poder leerlos sin depender de la API de Actions).

  1. En cada paso t (cada STEP_H horas):
     - Entrena el modelo con la ventana de WINDOW_DAYS anterior a t
     - Toma los TOP_N pools más grandes (nivel, lado, USD)
  2. Mira hacia delante LOOKAHEAD_H horas y comprueba:
     - HIT: ¿el precio tocó el nivel del pool?
     - CAPTURA: qué % de las liquidaciones reales ocurrió a < NEAR_PCT del nivel
  3. Compara contra BASELINE aleatorio: mismos N niveles posicionados al azar
     en el rango ±10% del precio. Si el modelo no supera al azar, no hay señal.

Limitaciones honestas de esta v1:
  - Liquidaciones de Binance capadas (hasta 1 BTC/evento): magnitudes sesgadas
    a la baja, pero las UBICACIONES son reales.
  - El hit-rate tiene sesgo mecánico; el baseline mitiga pero no elimina.
  - Muestra pequeña (free tier). Ampliable a meses con API key gratuita.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pool_model import (
    BUCKET_USD,
    build_pools,
    build_snapshots,
    fetch_dataset,
    hourly_targets,
)

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
DAYS = int(os.environ.get("LIQ_DAYS", "7"))

WINDOW_DAYS = 1      # historia para construir el mapa en cada paso
LOOKAHEAD_H = 12     # horizonte de predicción a validar
STEP_H = 6           # paso entre evaluaciones
TOP_N = 15           # pools más grandes a evaluar
NEAR_PCT = 0.005     # 0.5% de distancia para considerar liquidación "en el pool"
BASELINE_DRAWS = 20  # muestras aleatorias del baseline
RNG = np.random.default_rng(42)


def prep_liquidations(liq_raw: pd.DataFrame) -> pd.DataFrame:
    """Liquidaciones reales: timestamp, precio de ejecución, USD, lado."""
    df = liq_raw.copy()
    df["exec_px"] = df["average_price"].astype(float)
    df["usd"] = df["quantity"].astype(float) * df["exec_px"]
    df["is_long_liq"] = df["side"] == "SELL"  # SELL = long liquidado
    df["ts"] = pd.to_datetime(df["event_time"].astype("int64"), unit="ms")
    return df[["ts", "exec_px", "usd", "is_long_liq"]]


def pools_from_cohorts(cohorts: list) -> pd.DataFrame:
    df = pd.DataFrame(cohorts)
    df["bucket"] = df["level"] // BUCKET_USD * BUCKET_USD
    return df.groupby(["bucket", "side"]).usd.sum().reset_index()


def evaluate(snap: pd.DataFrame, liq: pd.DataFrame, start: pd.Timestamp) -> dict | None:
    """Una evaluación walk-forward: mapa en t, validación en [t, t+LOOKAHEAD)."""
    train = snap[(snap.ts >= start - pd.Timedelta(days=WINDOW_DAYS)) & (snap.ts < start)]
    test = snap[(snap.ts >= start) & (snap.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(train) < 100 or len(test) < 10:
        return None

    cohorts = build_pools(train)
    pools = pools_from_cohorts(cohorts).sort_values("usd", ascending=False).head(TOP_N)
    if pools.empty:
        return None

    px0 = float(train["px"].iloc[-1])
    lo, hi = float(test["px"].min()), float(test["px"].max())

    model_hits = []
    for _, p in pools.iterrows():
        touched = (p.side == "long" and lo <= p.bucket) or (p.side == "short" and hi >= p.bucket)
        model_hits.append(bool(touched))
    model_hit = float(np.mean(model_hits))

    base_hits = []
    for _ in range(BASELINE_DRAWS):
        lv = RNG.uniform(px0 * 0.90, px0 * 1.10, size=len(pools))
        sd = RNG.choice(["long", "short"], size=len(pools))
        th = np.mean([
            (s == "long" and lo <= l) or (s == "short" and hi >= l)
            for l, s in zip(lv, sd)
        ])
        base_hits.append(float(th))
    base_hit = float(np.mean(base_hits))

    look = liq[(liq.ts >= start) & (liq.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(look) > 0 and look.usd.sum() > 0:
        near_usd = 0.0
        for _, p in pools.iterrows():
            side_mask = look.is_long_liq if p.side == "long" else ~look.is_long_liq
            dist = (look.exec_px - p.bucket).abs() / p.bucket
            near_usd += float(look[side_mask & (dist <= NEAR_PCT)].usd.sum())
        capture = near_usd / float(look.usd.sum())
    else:
        capture = float("nan")

    return {
        "start": start,
        "n_pools": len(pools),
        "model_hit": model_hit,
        "base_hit": base_hit,
        "capture": capture,
    }


def main() -> None:
    hours = hourly_targets()
    print(f"Backtest {SYMBOL}: {DAYS}d de datos | ventana {WINDOW_DAYS}d | "
          f"lookahead {LOOKAHEAD_H}h | paso {STEP_H}h | top {TOP_N} pools")

    snap = build_snapshots(fetch_dataset(hours, "open_interest"))
    liq = prep_liquidations(fetch_dataset(hours, "liquidations"))
    print(f"Snapshots de OI: {len(snap):,} | Liquidaciones reales: {len(liq):,}\n")

    t = snap.ts.min() + pd.Timedelta(days=WINDOW_DAYS)
    t_end = snap.ts.max() - pd.Timedelta(hours=LOOKAHEAD_H)
    rows = []
    while t <= t_end:
        r = evaluate(snap, liq, t)
        if r:
            rows.append(r)
        t += pd.Timedelta(hours=STEP_H)

    assert len(rows) >= 5, f"Muy pocas evaluaciones ({len(rows)}) — ampliar LIQ_DAYS"
    df = pd.DataFrame(rows)
    df["start"] = df["start"].dt.strftime("%m-%d %H:%M")

    print("========== RESULTADOS POR EVALUACIÓN ==========")
    print(f"{'Paso':>12} | {'Hit modelo':>10} | {'Hit azar':>9} | {'Captura':>8}")
    print("-" * 50)
    for _, r in df.iterrows():
        cap = f"{r.capture*100:.1f}%" if r.capture == r.capture else "n/a"
        print(f"{r['start']:>12} | {r.model_hit*100:>9.0f}% | {r.base_hit*100:>8.0f}% | {cap:>8}")

    print("\n========== RESUMEN ==========")
    print(f"Evaluaciones:            {len(df)}")
    print(f"Hit-rate del modelo:     {df.model_hit.mean()*100:.1f}%")
    print(f"Hit-rate del azar:       {df.base_hit.mean()*100:.1f}%")
    edge = (df.model_hit.mean() - df.base_hit.mean()) * 100
    print(f"Ventaja sobre el azar:   {edge:+.1f} puntos")
    capture_mean = df.capture.mean() if df.capture.notna().any() else float("nan")
    if capture_mean == capture_mean:
        print(f"Captura media:           {capture_mean*100:.1f}% "
              f"(liquidaciones reales ocurridas en pools predichos)")

    if edge > 0:
        verdict = "El modelo SUPERA al baseline aleatorio en esta muestra."
    else:
        verdict = "El modelo NO supera al azar en esta muestra: ajustar parámetros."
    print(f"\n{verdict}")

    # ---- Guardar resultados en el repo (el CI los commitea) ----
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Resultado del último backtest — {SYMBOL}",
        "",
        f"- Ejecutado: {now}",
        f"- Datos: {DAYS} días · ventana {WINDOW_DAYS}d · lookahead {LOOKAHEAD_H}h · "
        f"paso {STEP_H}h · top {TOP_N} pools",
        f"- Evaluaciones: {len(df)}",
        f"- Hit-rate modelo: **{df.model_hit.mean()*100:.1f}%**",
        f"- Hit-rate azar: **{df.base_hit.mean()*100:.1f}%**",
        f"- Ventaja: **{edge:+.1f} puntos**",
    ]
    if capture_mean == capture_mean:
        lines.append(f"- Captura media: **{capture_mean*100:.1f}%**")
    lines += ["", "## Por evaluación", "",
              "| Paso | Hit modelo | Hit azar | Captura |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        cap = f"{r.capture*100:.1f}%" if r.capture == r.capture else "n/a"
        lines.append(f"| {r['start']} | {r.model_hit*100:.0f}% | {r.base_hit*100:.0f}% | {cap} |")
    lines += ["", f"**Veredicto:** {verdict}",
              "", "_Muestra limitada por el free tier; ampliable a meses con API key gratuita._"]
    os.makedirs("results", exist_ok=True)
    with open("results/last_backtest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n📄 Resultados guardados en results/last_backtest.md")

    assert df.model_hit.between(0, 1).all()
    assert df.base_hit.between(0, 1).all()
    print("✅ Test OK: backtest ejecutado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ Backtest falló: {exc}")
        sys.exit(1)
