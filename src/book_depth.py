"""
Profundidad del order book (L2) — Fase 5: pools × absorción del libro

Reconstruye la profundidad agregada del libro a partir del dataset orderbook
de CryptoHFTData (L2 snapshots + deltas, ~5.4M filas/hora en BTCUSDT).

Estrategia de peso controlado (los archivos son enormes):
  - No descargamos deltas completos: para cada hora tomamos SOLO la primera
    ventana de eventos y acumulamos niveles hasta tener profundidad suficiente
    (±DEPTH_PCT del precio). Es una aproximación del "libro visible" al inicio
    de la hora, suficiente para rankear absorción.
  - Alternativa simple usada aquí: por cada hora, parsear solo las filas con
    event_type="snapshot" (si existen) o las primeras N filas, acumulando
    (side, price, quantity) en un dict. Con quantity>0 añade, quantity=0 quita.

API:
  build_depth_index(hours) -> DepthIndex
  DepthIndex.usd_within(precio, lado, pct) -> USD absorbible a ±pct del precio

El ratio cascada/absorción de un pool = pool.usd / depth.usd_within(nivel).
Ratio > 1 → el libro no puede absorber la liquidación → candidata a cascada.
"""

import io
import time
from datetime import datetime

import pandas as pd
import requests

from pool_model import CRYPTOHFTDATA_DOWNLOAD, API_KEY, EXCHANGE, SYMBOL


class DepthIndex:
    """Índice de profundidad por hora: dict[ts_ms] -> {bid: {px: qty}, ask: {px: qty}}."""

    def __init__(self):
        self.books = {}  # ts_ms -> (bids: dict[float,float], asks: dict[float,float])

    def add_hour(self, ts_ms: int, bids: dict, asks: dict):
        self.books[ts_ms] = (bids, asks)

    def depth_within(self, ts_ms: int, price: float, side: str, pct: float) -> float:
        """USD absorbible a ±pct de `price` en el lado `side` (bid=debajo, ask=encima).

        Usa el snapshot de libro más reciente <= ts_ms.
        """
        keys = sorted(k for k in self.books if k <= ts_ms)
        if not keys:
            return float("nan")
        bids, asks = self.books[keys[-1]]
        book = bids if side == "bid" else asks
        usd = 0.0
        lo, hi = price * (1 - pct), price * (1 + pct)
        for px, qty in book.items():
            if lo <= px <= hi:
                usd += px * qty
        return usd


def fetch_book_hours(hours: list[datetime], max_rows_per_hour: int = 400_000) -> DepthIndex:
    """Descarga orderbook por hora y acumula el libro visible de cada hora.

    max_rows_per_hour limita el parseo por archivo (los archivos tienen ~5M
    filas; con 400k cubrimos snapshot inicial + primeros deltas, suficiente
    para profundidad cercana al mid).
    """
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    idx = DepthIndex()
    ok = 0
    for i, dt in enumerate(hours, 1):
        path = f"{EXCHANGE}/{dt:%Y-%m-%d}/{dt:%H}/{SYMBOL}_orderbook.parquet"
        url = f"{CRYPTOHFTDATA_DOWNLOAD}?file={path}"
        try:
            resp = requests.get(url, headers=headers, timeout=120)
            if resp.status_code == 200:
                df = pd.read_parquet(io.BytesIO(resp.content))
                if len(df) > max_rows_per_hour:
                    df = df.iloc[:max_rows_per_hour]
                # pyarrow filtra columnas automáticamente al leer: seleccionar las presentes
                cols = {c for c in ("event_time", "side", "price", "quantity") if c in df.columns}
                df = df[list(cols)]
                # timestamp de la hora (ms) — usamos el event_time máximo parseado
                ts_ms = int(df["event_time"].astype("int64").max())
                bids, asks = {}, {}
                for side, px, qty in zip(df["side"], df["price"].astype(float),
                                         df["quantity"].astype(float)):
                    book = bids if side == "bid" else asks
                    if qty > 0:
                        book[px] = book.get(px, 0.0) + qty
                    else:
                        book.pop(px, None)
                idx.add_hour(ts_ms, bids, asks)
                ok += 1
        except (requests.RequestException, Exception):
            pass  # hora perdida: continuamos
        if i % 12 == 0:
            print(f"  [orderbook] {i}/{len(hours)} horas procesadas ({ok} con libro)")
        time.sleep(0.2 if API_KEY else 1.05)
    if ok == 0:
        raise RuntimeError("No se pudo descargar ningún archivo de orderbook")
    print(f"  [orderbook] libro reconstruido para {ok}/{len(hours)} horas")
    return idx
