"""
Backtesting del modelo de pools de liquidación — Fase 2b (v4)

Historia de versiones:
  v1: hit-rate vs azar uniforme — injusta (los pools del modelo están lejos por
      construcción; el azar ganaba por colocar niveles cerca).
  v2: captura vs nulls (uniforme y distancia-emparejada), lookahead 12h.
      Gana al emparejado (+7 pts) pero pierde al uniforme (-6.5 pts).
  v3: lookahead 48h, 14d de datos, 44 evaluaciones. La señal se diluye:
      -2.5 vs uniforme, +0.6 vs emparejado -> modelo v1 marginal.
  v4: valida el MODELO v4 (funding-split + FIFO + margen de mantenimiento).
      Mismas métricas y nulls que v3 para comparación directa apples-to-apples.

Resultados -> results/last_backtest.md (el CI los commitea al repo).

Limitaciones honestas:
  - Liquidaciones de Binance capadas (1 BTC/evento): magnitudes sesgadas a la
    baja, pero las UBICACIONES son reales (la validación se basa en ubicación).
  - Muestra de semanas (free tier). Ampliable a meses con API key gratuita.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pool_model import (
    BUCKET_USD,
    build_funding_series,
    build_pools,
    build_snapshots,
    fetch_dataset,
    hourly_targets,
)

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
DAYS = int(os.environ.get("LIQ_DAYS", "14"))
LOOKAHEAD_H = int(os.environ.get("LIQ_LOOKAHEAD_H", "48"))

WINDOW_DAYS = 1      # historia para construir el mapa en cada paso
STEP_H = 6           # paso entre evaluaciones
TOP_N = 15           # pools más grandes a evaluar
NEAR_PCT = 0.005     # 0.5% de distancia para considerar liquidación "en el pool"
BASELINE_DRAWS = 20  # muestras aleatorias de cada null
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


def capture_of(levels, exec_px: np.ndarray, is_long: np.ndarray, usd: np.ndarray) -> float:
    """% del USD liquidado que cae a < NEAR_PCT de algún nivel del lado correcto."""
    if len(usd) == 0 or usd.sum() <= 0:
        return float("nan")
    near = np.zeros(len(exec_px), dtype=bool)
    for bucket, side in levels:
        same = is_long if side == "long" else ~is_long
        d = np.abs(exec_px - bucket) / bucket
        near |= same & (d <= NEAR_PCT)
    return float(usd[near].sum() / usd.sum())


def evaluate(snap, funding, liq, start: pd.Timestamp) -> dict | None:
    """Una evaluación walk-forward: mapa v4 en t, validación en [t, t+LOOKAHEAD)."""
    train = snap[(snap.ts >= start - pd.Timedelta(days=WINDOW_DAYS)) & (snap.ts < start)]
    test = snap[(snap.ts >= start) & (snap.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(train) < 100 or len(test) < 10:
        return None

    cohorts = build_pools(train, funding)
    pools = pools_from_cohorts(cohorts).sort_values("usd", ascending=False).head(TOP_N)
    if pools.empty:
        return None

    px0 = float(train["px"].iloc[-1])
    lo, hi = float(test["px"].min()), float(test["px"].max())

    # Descriptivo: perfil de distancia y pools tocados (NO es prueba de habilidad)
    model_levels = [(float(b), s) for b, s in zip(pools.bucket, pools.side)]
    dists = np.abs(pools.bucket.values / px0 - 1)
    touched = [
        (s == "long" and lo <= b) or (s == "short" and hi >= b)
        for b, s in model_levels
    ]
    model_hit = float(np.mean(touched))
    mean_dist = float(np.mean(dists))

    # Nulls: mismas N pools, colocaciones aleatorias
    long_frac = float((pools.side == "long").mean())
    p_side = [long_frac, 1 - long_frac]

    look = liq[(liq.ts >= start) & (liq.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(look) == 0 or look.usd.sum() <= 0:
        return None
    exec_px = look.exec_px.values
    is_long = look.is_long_liq.values
    usd = look.usd.values

    capture_model = capture_of(model_levels, exec_px, is_long, usd)

    null_u, null_m = [], []
    for _ in range(BASELINE_DRAWS):
        du = RNG.uniform(0, 0.10, size=len(pools))
        su = RNG.choice(["long", "short"], size=len(pools), p=p_side)
        lv_u = [(px0 * (1 - d if s == "long" else 1 + d), s) for d, s in zip(du, su)]
        null_u.append(capture_of(lv_u, exec_px, is_long, usd))
        dm = RNG.permutation(dists)
        sm = RNG.choice(["long", "short"], size=len(pools), p=p_side)
        lv_m = [(px0 * (1 - d if s == "long" else 1 + d), s) for d, s in zip(dm, sm)]
        null_m.append(capture_of(lv_m, exec_px, is_long, usd))

    return {
        "start": start,
        "n_pools": len(pools),
        "liq_usd": float(usd.sum()),
        "model_hit": model_hit,
        "mean_dist": mean_dist,
        "capture": capture_model,
        "null_u": float(np.mean(null_u)),
        "null_m": float(np.mean(null_m)),
    }


def main() -> None:
    hours = hourly_targets()
    print(f"Backtest v4 {SYMBOL}: {DAYS}d | ventana {WINDOW_DAYS}d | "
          f"lookahead {LOOKAHEAD_H}h | paso {STEP_H}h | top {TOP_N} pools")
    print("Modelo v4: funding-split + FIFO + margen de mantenimiento\n")

    snap = build_snapshots(fetch_dataset(hours, "open_interest"))
    funding = build_funding_series(fetch_dataset(hours, "mark_price"))
    liq = prep_liquidations(fetch_dataset(hours, "liquidations"))
    print(f"Snapshots OI: {len(snap):,} | Puntos funding: {len(funding):,} | "
          f"Liquidaciones: {len(liq):,}\n")

    t = snap.ts.min() + pd.Timedelta(days=WINDOW_DAYS)
    t_end = snap.ts.max() - pd.Timedelta(hours=LOOKAHEAD_H)
    rows = []
    while t <= t_end:
        r = evaluate(snap, funding, liq, t)
        if r:
            rows.append(r)
        t += pd.Timedelta(hours=STEP_H)

    assert len(rows) >= 5, f"Muy pocas evaluaciones ({len(rows)}) — ampliar LIQ_DAYS"
    df = pd.DataFrame(rows)
    df["start"] = df["start"].dt.strftime("%m-%d %H:%M")

    cap_m = df.capture.mean()
    cap_u = df.null_u.mean()
    cap_mm = df.null_m.mean()

    print("========== RESULTADOS POR EVALUACIÓN ==========")
    print(f"{'Paso':>12} | {'Liq USD':>10} | {'Cap modelo':>10} | {'Cap azar±10%':>12} | {'Cap dist-emp':>12}")
    print("-" * 68)
    for _, r in df.iterrows():
        print(f"{r['start']:>12} | {r.liq_usd:>10,.0f} | {r.capture*100:>9.1f}% | "
              f"{r.null_u*100:>11.1f}% | {r.null_m*100:>11.1f}%")

    print("\n========== RESUMEN ==========")
    print(f"Evaluaciones:                     {len(df)}")
    print(f"Lookahead:                        {LOOKAHEAD_H}h")
    print(f"Distancia media de los pools:     {df.mean_dist.mean()*100:.1f}% del precio")
    print(f"Pools tocados (descriptivo):      {df.model_hit.mean()*100:.1f}%")
    print(f"Captura del modelo v4:            {cap_m*100:.1f}%")
    print(f"Captura null uniforme (±10%):     {cap_u*100:.1f}%")
    print(f"Captura null dist-emparejada:     {cap_mm*100:.1f}%")
    print(f"Ventaja vs null uniforme:         {(cap_m-cap_u)*100:+.1f} puntos")
    print(f"Ventaja vs dist-emparejada:       {(cap_m-cap_mm)*100:+.1f} puntos")

    edge_u = (cap_m - cap_u) * 100
    edge_m = (cap_m - cap_mm) * 100
    if edge_m > 0 and edge_u > 0:
        verdict = ("El modelo v4 captura más liquidaciones reales que ambos nulls: "
                   "señal de localización confirmada.")
    elif edge_m > 0:
        verdict = ("El modelo v4 supera el null estricto pero no al uniforme: "
                   "mejora, horizonte aún corto.")
    else:
        verdict = "El modelo v4 NO supera a los nulls: iterar de nuevo."

    print(f"\n{verdict}")
    print("(Comparar con v3: captura 39.6%, uniforme 42.1%, emparejada 39.0%)")

    # ---- Guardar resultados en el repo (el CI los commitea) ----
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Resultado del último backtest (v4) — {SYMBOL}",
        "",
        f"- Ejecutado: {now}",
        f"- Modelo: **v4** (funding-split + FIFO + margen de mantenimiento)",
        f"- Datos: {DAYS} días · ventana {WINDOW_DAYS}d · lookahead {LOOKAHEAD_H}h · "
        f"paso {STEP_H}h · top {TOP_N} pools",
        f"- Evaluaciones: {len(df)}",
        f"- Captura del modelo v4: **{cap_m*100:.1f}%** (v3: 39.6%)",
        f"- Captura null uniforme: **{cap_u*100:.1f}%** (v3: 42.1%)",
        f"- Captura null distancia-emparejada: **{cap_mm*100:.1f}%** (v3: 39.0%)",
        f"- Ventaja vs uniforme: **{edge_u:+.1f} pts** (v3: -2.5) · "
        f"vs emparejada: **{edge_m:+.1f} pts** (v3: +0.6)",
        "",
        "## Por evaluación",
        "",
        "| Paso | Liq USD | Cap modelo | Cap uniforme | Cap emparejada |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['start']} | {r.liq_usd:,.0f} | {r.capture*100:.1f}% | "
                     f"{r.null_u*100:.1f}% | {r.null_m*100:.1f}% |")
    lines += ["", f"**Veredicto:** {verdict}",
              "", "_Muestra de semanas (free tier); ampliable a meses con API key gratuita._"]
    os.makedirs("results", exist_ok=True)
    with open("results/last_backtest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n📄 Resultados guardados en results/last_backtest.md")

    assert df.capture.between(0, 1).all()
    assert df.null_u.between(0, 1).all()
    assert df.null_m.between(0, 1).all()
    print("✅ Test OK: backtest v4 ejecutado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ Backtest falló: {exc}")
        sys.exit(1)
