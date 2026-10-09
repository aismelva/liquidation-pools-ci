"""
Análisis de pools de liquidación — Fase 1 (prototipo)

Descarga liquidaciones reales de CryptoHFTData (nivel gratuito, sin API key)
y calcula:
  1. Total liquidado (USD), separado longs vs shorts
  2. Clústeres de liquidación por nivel de precio (el embrión del heatmap)
  3. Slippage medio de las liquidaciones (precio ejecución vs precio de quiebra)

Esquema Parquet (verificado con la muestra real de Binance Futures):
  received_time, event_time, symbol, side, order_type, time_in_force,
  quantity, price, average_price, order_status,
  last_filled_quantity, filled_quantity, trade_time

  side == "SELL" -> liquidación de LONGS (venta forzada)
  side == "BUY"  -> liquidación de SHORTS (compra forzada)
  price          -> precio de quiebra (bankruptcy)
  average_price  -> precio medio de ejecución
"""

import io
import sys

import pandas as pd
import requests

CRYPTOHFTDATA_DOWNLOAD = "https://api.cryptohftdata.com/download"

# Muestra gratuita verificada: BTCUSDT en Binance Futures, 2026-09-02 12:00 UTC
SAMPLE_FILES = [
    "binance_futures/2026-09-02/12/BTCUSDT_liquidations.parquet",
    "binance_futures/2026-09-02/13/BTCUSDT_liquidations.parquet",
]

PRICE_BUCKET_USD = 100  # tamaño del clúster de precio para agrupar niveles


def fetch_liquidations(file_path: str) -> pd.DataFrame:
    """Descarga un archivo Parquet de liquidaciones (nivel gratuito, sin API key)."""
    url = f"{CRYPTOHFTDATA_DOWNLOAD}?file={file_path}"
    print(f"Descargando: {file_path}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return pd.read_parquet(io.BytesIO(resp.content))


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el dataframe para análisis de pools."""
    out = df.copy()
    out["qty"] = out["quantity"].astype(float)
    out["bankrupt_px"] = out["price"].astype(float)
    out["exec_px"] = out["average_price"].astype(float)
    out["usd"] = out["qty"] * out["exec_px"]
    out["is_long_liq"] = out["side"] == "SELL"
    return out


def main() -> None:
    frames = [fetch_liquidations(f) for f in SAMPLE_FILES]
    raw = pd.concat(frames, ignore_index=True)
    df = analyze(raw)

    total_usd = df["usd"].sum()
    longs = df[df["is_long_liq"]]
    shorts = df[~df["is_long_liq"]]

    print("\n========== RESUMEN DE LIQUIDACIONES ==========")
    print(f"Eventos:                {len(df):,}")
    print(f"Total liquidado:        ${total_usd:,.0f}")
    print(f"  Longs (venta forz.):  ${longs['usd'].sum():,.0f}  ({len(longs):,} eventos)")
    print(f"  Shorts (compra forz.): ${shorts['usd'].sum():,.0f}  ({len(shorts):,} eventos)")

    # ---- Clústeres por nivel de precio (prototipo de pool) ----
    df["bucket"] = (df["exec_px"] // PRICE_BUCKET_USD * PRICE_BUCKET_USD).astype(int)
    clusters = (
        df.groupby("bucket")
        .agg(usd=("usd", "sum"), eventos=("usd", "size"))
        .sort_values("usd", ascending=False)
        .head(10)
    )
    print("\n========== TOP 10 CLÚSTERES DE LIQUIDACIÓN (pool barrido) ==========")
    print(f"{'Nivel de precio (USD)':>22} | {'USD liquidado':>15} | {'Eventos':>8}")
    print("-" * 55)
    for bucket, row in clusters.iterrows():
        print(f"{bucket:>20,} | {row['usd']:>15,.0f} | {int(row['eventos']):>8}")

    # ---- Slippage de liquidaciones (ejecución vs quiebra) ----
    liq_slippage = (df["exec_px"] / df["bankrupt_px"] - 1).abs()
    print("\n========== SLIPPAGE ==========")
    print(f"Medio:   {liq_slippage.mean() * 100:.3f}%")
    print(f"Máximo:  {liq_slippage.max() * 100:.3f}%")

    # Sanity checks para el test de CI
    assert len(df) > 0, "No se descargaron eventos"
    assert total_usd > 0, "El volumen liquidado debe ser positivo"
    assert df["symbol"].nunique() >= 1
    print("\n✅ Test OK: datos reales descargados y analizados correctamente.")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        print(f"❌ Error de red descargando datos: {exc}")
        sys.exit(1)
