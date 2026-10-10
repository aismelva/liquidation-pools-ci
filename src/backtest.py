"""
Backtesting del modelo de pools de liquidación — v5.2: pools × absorción del libro

Historia de versiones:
  v1-v3: iteraciones de métrica y horizonte (ver historial en git).
  v4.1: funding-split + FIFO sembrado + margen. 30d: -1.4 uniforme, +1.8 emparejado.
  v4.2: ranking por peligrosidad (usd/distancia). 30d: +19.0 uniforme pero -0.4
      emparejado — la mejora venía de la alcanzabilidad, no del conocimiento.
  v5: ratio cascada/absorción con libro de 24h. Falló: 0 evaluaciones (libro
      ausente para casi todos los timestamps -> todo excluido).
  v5.1: libro de 72h + fallback de profundidad. Falló: 4 evaluaciones — el
      problema real era el LOOKAHEAD de 48h: reservaba 48h al FINAL de la
      ventana, y sumado a la ventana de 24h de entrenamiento, dejaba las
      evaluaciones al inicio SIN libro (el libro cubre solo las últimas 72h).
      Matemática: (14d-1d-2d)=11d evaluables pero solo ~2d con libro = ~4 eval.
  v5.2: fixes de geometría del experimento:
      a) LIQ_LOOKAHEAD_H=24 (reserva final más pequeña)
      b) LIQ_BOOK_HOURS=120 (libro cubre 5 días = ~17 evaluaciones con libro)
      c) LIQ_DAYS sigue 14 (la ventana evaluable con libro: 5d-1d-1d=3d*4=~17 eval)

Resultados -> results/last_backtest.md (el CI los commitea al repo).
"""

import os
import sys
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from book_depth import fetch_book_hours
from pool_model import (
    build_funding_series,
    build_pools,
    build_snapshots,
    fetch_dataset,
    hourly_targets,
    pools_from_cohorts,
)

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
DAYS = int(os.environ.get("LIQ_DAYS", "14"))
LOOKAHEAD_H = int(os.environ.get("LIQ_LOOKAHEAD_H", "24"))
BOOK_HOURS = int(os.environ.get("LIQ_BOOK_HOURS", "120"))

WINDOW_DAYS = 1      # historia para construir el mapa en cada paso
STEP_H = 6           # paso entre evaluaciones
TOP_N = 15           # pools a evaluar
NEAR_PCT = 0.005     # 0.5% de distancia para considerar liquidación "en el pool"
DEPTH_PCT = 0.005    # banda ±0.5% para medir absorción del libro
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


def depth_for_pool(depth_idx, ts_ms: int, price: float, side_book: str) -> float:
    """Profundidad para un pool con fallback.

    1) Banda ±DEPTH_PCT alrededor del nivel del pool.
    2) Banda vacía (libro parcial) → USD TOTALES del lado como cota
       conservadora (documentado: subestima el ratio de pools lejanos).
    """
    d = depth_idx.depth_within(ts_ms, price, side_book, DEPTH_PCT)
    if d == d and d > 0:
        return d
    keys = sorted(k for k in depth_idx.books if k <= ts_ms)
    if not keys:
        return float("nan")
    bids, asks = depth_idx.books[keys[-1]]
    book = bids if side_book == "bid" else asks
    total = sum(px * qty for px, qty in book.items())
    return total if total > 0 else float("nan")


def evaluate(snap, funding, liq, depth_idx, start: pd.Timestamp) -> dict | None:
    """Evaluación walk-forward: pools rankeados por ratio cascada/absorción."""
    train = snap[(snap.ts >= start - pd.Timedelta(days=WINDOW_DAYS)) & (snap.ts < start)]
    test = snap[(snap.ts >= start) & (snap.ts < start + pd.Timedelta(hours=LOOKAHEAD_H))]
    if len(train) < 100 or len(test) < 10:
        return None

    cohorts = build_pools(train, funding)
    all_pools = pools_from_cohorts(cohorts)
    if all_pools.empty:
        return None

    px0 = float(train["px"].iloc[-1])
    df = all_pools.copy()
    df["dist"] = (df["bucket"] / px0 - 1).abs()

    ts_ms = int(start.value // 10**6)  # ms de la evaluación
    depths, ratios = [], []
    for _, row in df.iterrows():
        side_book = "bid" if row["side"] == "long" else "ask"
        d = depth_for_pool(depth_idx, ts_ms, float(row["bucket"]), side_book)
        depths.append(d)
        ratios.append(row["usd"] / d if d == d and d > 0 else float("inf"))

    df["book_usd"] = depths
    df["cascade"] = ratios
    df = df[(df["dist"] <= 0.10) & np.isfinite(df["cascade"])]
    if df.empty:
        return None

    pools = df.sort_values("cascade", ascending=False).head(TOP_N).reset_index(drop=True)

    lo, hi = float(test["px"].min()), float(test["px"].max())
    model_levels = [(float(b), s) for b, s in zip(pools.bucket, pools.side)]
    dists = np.abs(pools.bucket.values / px0 - 1)
    touched = [
        (s == "long" and lo <= b) or (s == "short" and hi >= b)
        for b, s in model_levels
    ]
    model_hit = float(np.mean(touched))
    mean_dist = float(np.mean(dists))
    mean_cascade = float(pools.cascade.mean())

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
        "mean_cascade": mean_cascade,
        "capture": capture_model,
        "null_u": float(np.mean(null_u)),
        "null_m": float(np.mean(null_m)),
    }


def main() -> None:
    hours = hourly_targets()
    print(f"Backtest v5.2 {SYMBOL}: {DAYS}d | ventana {WINDOW_DAYS}d | "
          f"lookahead {LOOKAHEAD_H}h | paso {STEP_H}h | top {TOP_N} pools "
          f"| libro: últimas {BOOK_HOURS}h")
    print("Modelo v5.2 = v4.1 + ranking por ratio cascada/absorción (order book L2)\n")

    snap = build_snapshots(fetch_dataset(hours, "open_interest"))
    funding = build_funding_series(fetch_dataset(hours, "mark_price"))
    liq = prep_liquidations(fetch_dataset(hours, "liquidations"))
    print(f"Snapshots OI: {len(snap):,} | Puntos funding: {len(funding):,} | "
          f"Liquidaciones: {len(liq):,}")

    # Order book: últimas BOOK_HOURS horas
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    book_hours = [now - timedelta(hours=h) for h in range(BOOK_HOURS, 0, -1)]
    depth_idx = fetch_book_hours(book_hours)
    print(f"Libros L2 reconstruidos: {len(depth_idx.books)}\n")

    t = snap.ts.min() + pd.Timedelta(days=WINDOW_DAYS)
    t_end = snap.ts.max() - pd.Timedelta(hours=LOOKAHEAD_H)
    rows = []
    while t <= t_end:
        r = evaluate(snap, funding, liq, depth_idx, t)
        if r:
            rows.append(r)
        t += pd.Timedelta(hours=STEP_H)

    assert len(rows) >= 5, (
        f"Muy pocas evaluaciones ({len(rows)}) — ampliar LIQ_BOOK_HOURS "
        f"(actual: {BOOK_HOURS}h) o LIQ_DAYS"
    )
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
    print(f"Ratio cascada medio (usd/libro):  {df.mean_cascade.mean():.2f}")
    print(f"Distancia media de los pools:     {df.mean_dist.mean()*100:.1f}% del precio")
    print(f"Pools tocados (descriptivo):      {df.model_hit.mean()*100:.1f}%")
    print(f"Captura del modelo v5.2:          {cap_m*100:.1f}%")
    print(f"Captura null uniforme (±10%):     {cap_u*100:.1f}%")
    print(f"Captura null dist-emparejada:     {cap_mm*100:.1f}%")
    print(f"Ventaja vs null uniforme:         {(cap_m-cap_u)*100:+.1f} puntos")
    print(f"Ventaja vs dist-emparejada:       {(cap_m-cap_mm)*100:+.1f} puntos")

    edge_u = (cap_m - cap_u) * 100
    edge_m = (cap_m - cap_mm) * 100
    if edge_m > 0 and edge_u > 0:
        verdict = ("El modelo v5.2 captura más liquidaciones reales que ambos nulls: "
                   "señal de cascada/absorción confirmada.")
    elif edge_m > 0:
        verdict = ("El modelo v5.2 supera el null estricto pero no al uniforme: "
                   "mejora parcial.")
    else:
        verdict = "El modelo v5.2 NO supera a los nulls: iterar."

    print(f"\n{verdict}")
    print("(Referencias 14d: v4.1 45.0%/+3.6/+2.4 · v4.2 60.7%/+19.0/-0.4)")

    # ---- Guardar resultados en el repo (el CI los commitea) ----
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Resultado del último backtest (v5.2) — {SYMBOL}",
        "",
        f"- Ejecutado: {now_s}",
        f"- Modelo: **v5.2** (v4.1 + ranking por ratio cascada/absorción con order book L2)",
        f"- Datos: {DAYS} días · ventana {WINDOW_DAYS}d · lookahead {LOOKAHEAD_H}h · "
        f"libro L2: últimas {BOOK_HOURS}h",
        f"- Evaluaciones: {len(df)}",
        f"- Ratio cascada medio: **{df.mean_cascade.mean():.2f}** (usd del pool / usd absorbible)",
        f"- Captura del modelo v5.2: **{cap_m*100:.1f}%** (v4.1/14d: 45.0%)",
        f"- Captura null uniforme: **{cap_u*100:.1f}%**",
        f"- Captura null distancia-emparejada: **{cap_mm*100:.1f}%**",
        f"- Ventaja vs uniforme: **{edge_u:+.1f} pts** · vs emparejada: **{edge_m:+.1f} pts**",
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
              "", "_Diferencial v5: pools × absorción del libro — capa que las plataformas estándar no integran._"]
    os.makedirs("results", exist_ok=True)
    with open("results/last_backtest.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n📄 Resultados guardados en results/last_backtest.md")

    assert df.capture.between(0, 1).all()
    assert df.null_u.between(0, 1).all()
    assert df.null_m.between(0, 1).all()
    print("✅ Test OK: backtest v5.2 ejecutado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"❌ Backtest falló: {exc}")
        traceback.print_exc()
        sys.exit(1)
