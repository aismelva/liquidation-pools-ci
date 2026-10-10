"""
Backtesting del modelo de pools de liquidación — Fase 2b (v4.2: ranking por peligrosidad)

Historia de versiones:
  v1: hit-rate vs azar uniforme — injusta (los pools del modelo están lejos por
      construcción; el azar ganaba por colocar niveles cerca).
  v2: captura vs nulls (uniforme y distancia-emparejada), lookahead 12h.
      Gana al emparejado (+7 pts) pero pierde al uniforme (-6.5 pts).
  v3: lookahead 48h, 14d, 44 eval. Señal diluida: -2.5 uniforme, +0.6 emparejado.
  v4.1: funding-split + FIFO sembrado + margen. 14d: +3.6/+2.4 (ambos nulls).
      30d/108 eval: la ventaja NO se mantiene (−1.4 uniforme, +1.8 emparejado).
      Diagnóstico: la UBICACIÓN es correcta pero el RANKING por USD puro falla —
      un pool de $500M a 8% capta menos que uno de $80M a 1%.
  v4.2: experimento de ranking — se seleccionan los top N pools por
      PELIGROSIDAD = usd / distancia_al_precio, en vez de USD puro.
      Hipótesis: ponderar por alcanzabilidad mejora la captura vs ambos nulls.
      Todo lo demás (modelo, métricas, nulls, ventanas) idéntico a v4.1 para
      comparación apples-to-apples.

Resultados -> results/last_backtest.md (el CI los commitea al repo).

Limitaciones honestas:
  - Ranking por peligrosidad es heurístico (1/distancia); la calibración con
    distribución real de Hyperliquid y el ratio cascada/absorción (v5, order
    books) son los siguientes pasos naturales.
  - Liquidaciones de Binance capadas (1 BTC/evento): magnitudes sesgadas a la
    baja, pero las UBICACIONES son reales.
"""

import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pool_model import (
    build_funding_series,
    build_pools,
    build_snapshots,
    fetch_dataset,
    hourly_targets,
    pools_from_cohorts,
)

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
DAYS = int(os.environ.get("LIQ_DAYS", "30"))
LOOKAHEAD_H = int(os.environ.get("LIQ_LOOKAHEAD_H", "48"))

WINDOW_DAYS = 1      # historia para construir el mapa en cada paso
STEP_H = 6           # paso entre evaluaciones
TOP_N = 15           # pools a evaluar
NEAR_PCT = 0.005     # 0.5% de distancia para considerar liquidación "en el pool"
BASELINE_DRAWS = 20  # muestras aleatorias de cada null
RNG = np.random.default_rng(42)

# v4.2: ranking por peligrosidad (usd / distancia) en vez de usd puro.
# RANK_MODE = "usd" reproduce v4.1 exactamente (para comparar en el mismo run).
RANK_MODE = os.environ.get("LIQ_RANK_MODE", "danger")


def prep_liquidations(liq_raw: pd.DataFrame) -> pd.DataFrame:
    """Liquidaciones reales: timestamp, precio de ejecución, USD, lado."""
    df = liq_raw.copy()
    df["exec_px"] = df["average_price"].astype(float)
    df["usd"] = df["quantity"].astype(float) * df["exec_px"]
    df["is_long_liq"] = df["side"] == "SELL"  # SELL = long liquidado
    df["ts"] = pd.to_datetime(df["event_time"].astype("int64"), unit="ms")
    return df[["ts", "exec_px", "usd", "is_long_liq"]]


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


def select_pools(pools: pd.DataFrame, px0: float, mode: str) -> pd.DataFrame:
    """Selecciona los TOP_N pools según el modo de ranking."""
    df = pools.copy()
    df["dist"] = (df["bucket"] / px0 - 1).abs()
    if mode == "danger":
        # peligrosidad = USD por punto porcentual de distancia (cap al 0.1%
        # para no div-bidir por cero y premiar pools pegados al precio)
        df["danger"] = df["usd"] / np.maximum(df["dist"], 0.001)
        df = df.sort_values("danger", ascending=False)
    else:  # "usd" — comportamiento v4.1
        df = df.sort_values("usd", ascending=False)
    return df.head(TOP_N).reset_index(drop=True)


def evaluate(snap, funding, liq, start: pd.Timestamp) -> dict | None:
    """Una evaluación walk-forward: mapa v4.2 en t, validación en [t, t+LOOKAHEAD)."""
    train = snap[(snap.ts >= start - pd.Timedelta(days=WINDOW_DAYS)) & (snap.ts < start)]
    test = snap[(snap.ts >= start) & (snap.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(train) < 100 or len(test) < 10:
        return None

    cohorts = build_pools(train, funding)
    all_pools = pools_from_cohorts(cohorts)
    if all_pools.empty:
        return None

    px0 = float(train["px"].iloc[-1])
    pools = select_pools(all_pools, px0, RANK_MODE)
    if pools.empty:
        return None

    lo, hi = float(test["px"].min()), float(test["px"].max())

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
    print(f"Backtest v4.2 {SYMBOL}: {DAYS}d | ventana {WINDOW_DAYS}d | "
          f"lookahead {LOOKAHEAD_H}h | paso {STEP_H}h | top {TOP_N} pools "
          f"| ranking: {RANK_MODE}")
    print("Modelo v4.2 = v4.1 + ranking por peligrosidad (usd / distancia)\n")

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
    print(f"Modo de ranking:                  {RANK_MODE}")
    print(f"Distancia media de los pools:     {df.mean_dist.mean()*100:.1f}% del precio")
    print(f"Pools tocados (descriptivo):      {df.model_hit.mean()*100:.1f}%")
    print(f"Captura del modelo v4.2:          {cap_m*100:.1f}%")
    print(f"Captura null uniforme (±10%):     {cap_u*100:.1f}%")
    print(f"Captura null dist-emparejada:     {cap_mm*100:.1f}%")
    print(f"Ventaja vs null uniforme:         {(cap_m-cap_u)*100:+.1f} puntos")
    print(f"Ventaja vs dist-emparejada:       {(cap_m-cap_mm)*100:+.1f} puntos")

    edge_u = (cap_m - cap_u) * 100
    edge_m = (cap_m - cap_mm) * 100
    if edge_m > 0 and edge_u > 0:
        verdict = ("El modelo v4.2 captura más liquidaciones reales que ambos nulls: "
                   "señal de localización confirmada.")
    elif edge_m > 0:
        verdict = ("El modelo v4.2 supera el null estricto pero no al uniforme: "
                   "mejora, horizonte aún corto.")
    else:
        verdict = "El modelo v4.2 NO supera a los nulls: iterar de nuevo."

    print(f"\n{verdict}")
    print("(Referencia v4.1 con 30d: captura 39.2%, uniforme 40.7% (-1.4), emparejada 37.4% (+1.8))")

    # ---- Guardar resultados en el repo (el CI los commitea) ----
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Resultado del último backtest (v4.2) — {SYMBOL}",
        "",
        f"- Ejecutado: {now}",
        f"- Modelo: **v4.2** (v4.1 + ranking por peligrosidad usd/distancia, modo `{RANK_MODE}`)",
        f"- Datos: {DAYS} días · ventana {WINDOW_DAYS}d · lookahead {LOOKAHEAD_H}h · "
        f"paso {STEP_H}h · top {TOP_N} pools",
        f"- Evaluaciones: {len(df)}",
        f"- Captura del modelo v4.2: **{cap_m*100:.1f}%** (v4.1/30d: 39.2%)",
        f"- Captura null uniforme: **{cap_u*100:.1f}%** (v4.1/30d: 40.7%)",
        f"- Captura null distancia-emparejada: **{cap_mm*100:.1f}%** (v4.1/30d: 37.4%)",
        f"- Ventaja vs uniforme: **{edge_u:+.1f} pts** (v4.1/30d: -1.4) · "
        f"vs emparejada: **{edge_m:+.1f} pts** (v4.1/30d: +1.8)",
        f"- Distancia media de pools seleccionados: {df.mean_dist.mean()*100:.1f}% "
        f"(v4.1 priorizaba pools lejanos; v4.2 debería bajar esto)",
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
              "", "_Con API key gratuita; ampliable a más meses._"]
    os.makedirs("results", exist_ok=True)
    with open("results/last_backtest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n📄 Resultados guardados en results/last_backtest.md")

    assert df.capture.between(0, 1).all()
    assert df.null_u.between(0, 1).all()
    assert df.null_m.between(0, 1).all()
    print("✅ Test OK: backtest v4.2 ejecutado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ Backtest falló: {exc}")
        traceback.print_exc()
        sys.exit(1)
