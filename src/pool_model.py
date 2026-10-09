"""
Modelo de pools de liquidación — Fase 2 (cohortes de OI)

Construye un mapa de niveles de liquidación estimados ("pools") a partir de:
  - Open Interest: snapshots de ~5 min de Binance Futures (CryptoHFTData, gratuito, sin API key)
  - Liquidaciones reales: para contexto y validación (mismo proveedor)

Datasets (naming verificado en la documentación):
  binance_futures/{fecha}/{hora}/BTCUSDT_open_interest.parquet
    -> received_time, symbol, sum_open_interest (BTC), sum_open_interest_value (USD), timestamp (ms)
  binance_futures/{fecha}/{hora}/BTCUSDT_liquidations.parquet
    -> esquema 13 columnas (ver liquidation_analysis.py)

Algoritmo v1:
  1. Cada subida de OI a un precio P = nuevas posiciones abiertas ahí (cohorte)
  2. Split long/short heurístico según momentum del precio en el snapshot
  3. Distribución de apalancamientos asumida: 10x/25x/50x/100x (pesos 40/30/20/10 %)
  4. Nivel de liquidación estimado: long ≈ P*(1-1/L), short ≈ P*(1+1/L)
  5. Al avanzar el precio, los niveles tocados se consideran barridos y salen del mapa
  6. Los niveles supervivientes se agrupan en buckets de precio -> "pools"

Simplificaciones honestas de v1 (a iterar en backtesting):
  - Split long/short es heurístico (sin datos reales de posicionamiento por venue)
  - Caídas de OI aún no descuentan cohortes (v2: descuento FIFO)
  - Fórmula de liquidación sin margen de mantenimiento
"""

import io
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

CRYPTOHFTDATA_DOWNLOAD = "https://api.cryptohftdata.com/download"

SYMBOL = os.environ.get("LIQ_SYMBOL", "BTCUSDT")
EXCHANGE = "binance_futures"
DAYS = int(os.environ.get("LIQ_DAYS", "3"))  # días de historia a modelar

# (apalancamiento, peso en la distribución asumida)
LEVERAGES = [(10, 0.40), (25, 0.30), (50, 0.20), (100, 0.10)]
LONG_SHARE_UP = 0.60      # si el precio sube al abrirse posiciones: % asumido de longs
LONG_SHARE_DOWN = 0.40    # si baja: % asumido de longs (más probables shorts)
BUCKET_USD = 250          # tamaño del bucket de precio para agrupar pools


def hourly_targets() -> list[datetime]:
    """Horas UTC a descargar: las últimas DAYS*24, sin la hora en curso (aún no publicada)."""
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


def build_pools(snap: pd.DataFrame) -> list[dict]:
    """Núcleo del modelo: cohortes de OI -> niveles de liquidación -> pools supervivientes."""
    cohorts: list[dict] = []
    swept_usd = 0.0
    prev_qty = prev_px = None

    for px, qty in zip(snap["px"], snap["oi_qty"]):
        # 1) Barrido: niveles tocados por el precio salen del mapa
        survivors = []
        for c in cohorts:
            if (c["side"] == "long" and px <= c["level"]) or (
                c["side"] == "short" and px >= c["level"]
            ):
                swept_usd += c["usd"]
            else:
                survivors.append(c)
        cohorts = survivors

        # 2) Creación: subida de OI = nuevas posiciones a este precio
        if prev_qty is not None:
            d_oi = qty - prev_qty
            if d_oi > 0:
                usd = d_oi * px
                momentum = px - prev_px
                long_share = (
                    LONG_SHARE_UP if momentum > 0 else LONG_SHARE_DOWN if momentum < 0 else 0.5
                )
                for lev, weight in LEVERAGES:
                    cohorts.append({
                        "side": "long",
                        "level": px * (1 - 1 / lev),
                        "usd": usd * long_share * weight,
                    })
                    cohorts.append({
                        "side": "short",
                        "level": px * (1 + 1 / lev),
                        "usd": usd * (1 - long_share) * weight,
                    })
        prev_qty, prev_px = qty, px

    print(f"  Cohortes creadas -> supervivientes: {len(cohorts)} | barridas: ${swept_usd:,.0f}")
    return cohorts


def summarize(cohorts: list[dict], snap: pd.DataFrame) -> pd.DataFrame:
    px_now = snap["px"].iloc[-1]
    df = pd.DataFrame(cohorts)
    df["bucket"] = (df["level"] // BUCKET_USD * BUCKET_USD).astype(int)
    pools = df.groupby(["bucket", "side"]).usd.sum().reset_index()

    print("\n========== MAPA DE POOLS DE LIQUIDACIÓN (estimación v1) ==========")
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
    print(f"Descargando {len(hours)} horas de {SYMBOL} (OI + liquidaciones)...")

    oi_raw = fetch_dataset(hours, "open_interest")
    snap = build_snapshots(oi_raw)
    print(f"Snapshots de OI: {len(snap):,} (cada ~5 min)")

    cohorts = build_pools(snap)
    pools = summarize(cohorts, snap)

    # Contexto: liquidaciones reales del mismo periodo
    liq = fetch_dataset(hours, "liquidations")
    liq_usd = liq.quantity.astype(float) * liq.average_price.astype(float)
    print("\n========== LIQUIDACIONES REALES EN LA VENTANA (validación) ==========")
    print(f"Eventos: {len(liq):,} | Total: ${liq_usd.sum():,.0f}")
    print(f"  Longs liquidados:  ${liq_usd[liq.side == 'SELL'].sum():,.0f}")
    print(f"  Shorts liquidados: ${liq_usd[liq.side == 'BUY'].sum():,.0f}")

    # Sanity checks del test de CI
    assert len(snap) > 50, "Muy pocos snapshots de OI"
    assert len(cohorts) > 0, "El modelo no generó niveles"
    assert pools["usd"].sum() > 0
    print("\n✅ Test OK: modelo de pools generado con datos reales.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"❌ Error de red: {exc}")
        sys.exit(1)
