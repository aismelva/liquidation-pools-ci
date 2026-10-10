"""
Modelo de pools de liquidación — Fase 2 (v4: funding + FIFO + margen de mantenimiento)

v4 incorpora las tres mejoras con base teórica identificadas tras el backtest v3:

  1. SPLIT LONG/SHORT BASADO EN FUNDING (antes: momentum 60/40)
     El funding rate es la medida real de sesgo de posicionamiento: funding
     positivo = los longs pagan a los shorts = exceso de longs. Se mapea a
     long_share en [0.35, 0.65] con la escala FUNDING_SENS.
     Fuente: dataset mark_price de CryptoHFTData (columna funding_rate, ~1s).

  2. DESCUENTO FIFO POR CAÍDAS DE OI (antes: solo barrido por precio)
     Cuando el OI baja, las cohortes más antiguas se dan por cerradas y salen
     del mapa (las posiciones cerradas voluntariamente suelen ser antiguas).

  3. MARGEN DE MANTENIMIENTO (antes: liquidación a 1/L exacto)
     Nivel de liquidación aproximado para Binance USDT-M (margen ~0.5%):
       long:  P * (1 - 1/L + MM)
       short: P * (1 + 1/L - MM)
     Con MM=0.005: un 10x long se liquida a -9.5% en vez de -10%.

Datasets (naming verificado):
  binance_futures/{fecha}/{hora}/{SYMBOL}_open_interest.parquet
    -> sum_open_interest (BTC), sum_open_interest_value (USD), timestamp (ms)
  binance_futures/{fecha}/{hora}/{SYMBOL}_mark_price.parquet
    -> mark_price, funding_rate, event_time (ms)
  binance_futures/{fecha}/{hora}/{SYMBOL}_liquidations.parquet
    -> esquema 13 columnas (ver liquidation_analysis.py)

Simplificaciones honestas restantes:
  - El FIFO asume que las posiciones cerradas son las más antiguas (aprox razonable)
  - MM fijo del 0.5% (Binance usa escalones; aproximación de primer orden)
  - Distribución de apalancamientos sigue asumida (10/25/50/100x)
"""

import io
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CRYPTOHFTDATA_DOWNLOAD = "https://api.cryptohftdata.com/download"

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
EXCHANGE = "binance_futures"
DAYS = int(os.environ.get("LIQ_DAYS", "3"))  # días de historia a modelar

# (apalancamiento, peso en la distribución asumida)
LEVERAGES = [(10, 0.40), (25, 0.30), (50, 0.20), (100, 0.10)]
MM = 0.005                      # margen de mantenimiento aproximado (0.5%)
FUNDING_SENS = 3000.0           # escala: funding diario de 0.1% -> long_share 0.65
LONG_SHARE_MIN, LONG_SHARE_MAX = 0.35, 0.65
BUCKET_USD = 250                # tamaño del bucket de precio para agrupar pools


def hourly_targets() -> list[datetime]:
    """Horas UTC a descargar: las últimas DAYS*24, sin la hora en curso."""
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [now - timedelta(hours=h) for h in range(DAYS * 24, 0, -1)]


def fetch_dataset(hours: list[datetime], dataset: str) -> pd.DataFrame:
    """Descarga un dataset horario respetando el rate limit del free tier (~60 req/min)."""
    frames = []
    ok = 0
    for i, dt in enumerate(hours, 1):
        path = f"{EXCHANGE}/{dt:%Y-%m-%d}/{dt:%H}/{SYMBOL}_{dataset}.parquet"
        url = f"{CRYPTOHFTDATA_DOWNLOAD}?file={path}"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                frames.append(pd.read_parquet(io.BytesIO(resp.content)))
                ok += 1
        except requests.RequestException:
            pass  # hora perdida: continuamos
        if i % 24 == 0:
            print(f"  [{dataset}] {i}/{len(hours)} horas procesadas ({ok} con datos)")
        time.sleep(1.05)  # rate limit free tier
    if not frames:
        raise RuntimeError(f"No se pudo descargar ningún archivo de {dataset}")
    return pd.concat(frames, ignore_index=True)


def build_snapshots(oi_raw: pd.DataFrame) -> pd.DataFrame:
    """Convierte los snapshots de OI en una serie temporal (ts, precio, OI BTC, OI USD)."""
    df = oi_raw.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["oi_qty"] = df["sum_open_interest"].astype(float)
    df["oi_usd"] = df["sum_open_interest_value"].astype(float)
    df["px"] = df["oi_usd"] / df["oi_qty"]  # precio implícito en cada snapshot
    df = df.drop_duplicates("timestamp").sort_values("ts").reset_index(drop=True)
    return df


def build_funding_series(mp_raw: pd.DataFrame) -> pd.Series:
    """Serie temporal de funding rate: índice ts (ms), valor float."""
    df = mp_raw.copy()
    df["ts"] = df["event_time"].astype("int64")
    df["fr"] = df["funding_rate"].astype(float)
    return df.drop_duplicates("ts").set_index("ts")["fr"].sort_index()


def long_share_from_funding(funding_8h: float) -> float:
    """Mapea el funding al % de longs de la cohorte: positivo = exceso de longs."""
    daily = funding_8h * 3.0
    share = 0.5 + daily * FUNDING_SENS
    return max(LONG_SHARE_MIN, min(LONG_SHARE_MAX, share))


def build_pools(snap: pd.DataFrame, funding: pd.Series | None = None) -> list[dict]:
    """Núcleo v4: cohortes con funding-split, barrido, descuento FIFO y margen."""
    cohorts: deque = deque()   # FIFO: los más antiguos al frente
    swept_usd = closed_usd = 0.0
    prev_qty = prev_px = None
    fr = None

    for ts, px, qty in zip(snap["timestamp"], snap["px"], snap["oi_qty"]):
        # funding vigente en este snapshot (búsqueda hacia atrás, o 0 si no hay)
        if funding is not None and len(funding) > 0:
            idx = funding.index.searchsorted(ts, side="right") - 1
            fr = float(funding.iloc[idx]) if idx >= 0 else 0.0

        # 1) Barrido: niveles tocados por el precio salen del mapa
        survivors = deque()
        for c in cohorts:
            if (c["side"] == "long" and px <= c["level"]) or (
                c["side"] == "short" and px >= c["level"]
            ):
                swept_usd += c["usd"]
            else:
                survivors.append(c)
        cohorts = survivors

        # 2) FIFO: caída de OI cierra las cohortes más antiguas
        if prev_qty is not None and qty < prev_qty:
            to_close = prev_qty - qty
            while to_close > 0 and cohorts:
                c = cohorts[0]  # más antigua
                qty_c = c["usd"] / c["entry_px"]
                if qty_c <= to_close:
                    closed_usd += c["usd"]
                    to_close -= qty_c
                    cohorts.popleft()
                else:
                    c["usd"] -= to_close * c["entry_px"]
                    to_close = 0

        # 3) Creación: subida de OI = nuevas posiciones a este precio
        if prev_qty is not None:
            d_oi = qty - prev_qty
            if d_oi > 0:
                usd = d_oi * px
                long_share = long_share_from_funding(fr) if fr is not None else 0.5
                for lev, weight in LEVERAGES:
                    cohorts.append({
                        "side": "long",
                        "entry_px": px,
                        "level": px * (1 - 1 / lev + MM),
                        "usd": usd * long_share * weight,
                    })
                    cohorts.append({
                        "side": "short",
                        "entry_px": px,
                        "level": px * (1 + 1 / lev - MM),
                        "usd": usd * (1 - long_share) * weight,
                    })
        prev_qty, prev_px = qty, px

    print(f"  Cohortes -> supervivientes: {len(cohorts)} | barridas: ${swept_usd:,.0f} "
          f"| cerradas FIFO: ${closed_usd:,.0f}")
    return list(cohorts)


def summarize(cohorts: list[dict], snap: pd.DataFrame) -> pd.DataFrame:
    px_now = snap["px"].iloc[-1]
    df = pd.DataFrame(cohorts)
    df["bucket"] = (df["level"] // BUCKET_USD * BUCKET_USD).astype(int)
    pools = df.groupby(["bucket", "side"]).usd.sum().reset_index()

    print("\n========== MAPA DE POOLS DE LIQUIDACIÓN (estimación v4) ==========")
    print(f"Símbolo: {SYMBOL} | Ventana: últimos {DAYS} días | Precio actual: ${px_now:,.0f}")

    longs = pools[pools.side == "long"].sort_values("usd", ascending=False).head(10)
    shorts = pools[pools.side == "short"].sort_values("usd", ascending=False).head(10)

    print("\n--- POOLS DE LONGS (debajo del precio: ventas forzadas si se tocan) ---")
    print(f"{'Nivel (USD)':>12} | {'USD en riesgo':>15} | {'Distancia':>10}")
    for _, r in longs.iterrows():
        dist = (r.bucket / px_now - 1) * 100
        print(f"{r.bucket:>12,} | {r.usd:>15,.0f} | {dist:>9.1f}%")

    print("\n--- POOLS DE SHORTS (encima del precio: compras forzadas si se tocan) ---")
    for _, r in shorts.iterrows():
        dist = (r.bucket / px_now - 1) * 100
        print(f"{r.bucket:>12,} | {r.usd:>15,.0f} | {dist:>9.1f}%")

    return pools


def main() -> None:
    hours = hourly_targets()
    print(f"Descargando {len(hours)} horas de {SYMBOL} (OI + mark price + liquidaciones)...")

    oi_raw = fetch_dataset(hours, "open_interest")
    snap = build_snapshots(oi_raw)
    print(f"Snapshots de OI: {len(snap):,} (cada ~5 min)")

    mp_raw = fetch_dataset(hours, "mark_price")
    funding = build_funding_series(mp_raw)
    print(f"Puntos de funding/mark price: {len(funding):,}")

    cohorts = build_pools(snap, funding)
    pools = summarize(cohorts, snap)

    # Contexto: liquidaciones reales del mismo periodo
    liq = fetch_dataset(hours, "liquidations")
    liq_usd = liq.quantity.astype(float) * liq.average_price.astype(float)
    print("\n========== LIQUIDACIONES REALES EN LA VENTANA ==========")
    print(f"Eventos: {len(liq):,} | Total: ${liq_usd.sum():,.0f}")
    print(f"  Longs liquidados:  ${liq_usd[liq.side == 'SELL'].sum():,.0f}")
    print(f"  Shorts liquidados: ${liq_usd[liq.side == 'BUY'].sum():,.0f}")

    # Sanity checks del test de CI
    assert len(snap) > 50, "Muy pocos snapshots de OI"
    assert len(funding) > 50, "Muy pocos puntos de funding"
    assert len(cohorts) > 0, "El modelo no generó niveles"
    assert pools["usd"].sum() > 0
    print("\n✅ Test OK: modelo v4 generado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"❌ Error de red: {exc}")
        sys.exit(1)
