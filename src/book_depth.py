"""
Profundidad del order book (L2) — Fase 5: pools × absorción del libro

Reconstruye la profundidad agregada del libro a partir del dataset orderbook
de CryptoHFTData (L2 snapshots + deltas, ~5.4M filas/hora en BTCUSDT).

Estrategia de peso controlado (los archivos son enormes):
  - Por cada hora, parseamos solo las primeras N filas del archivo
    (max_rows_per_hour, default 1.2M): cubre snapshot inicial + deltas,
    suficiente para profundidad cercana al mid de esa hora.
  - Acumulamos (side, price, quantity) en dicts: quantity>0 añade, 0 quita.
  - Un libro por hora; depth_within usa el snapshot <= timestamp pedido.

Fallback de profundidad (fix v5.1): si la banda ±pct alrededor del precio no
tiene USD medidos (libro parcial lejos del precio pedido), usamos el MEJOR
precio disponible del lado correcto más cercano como aproximación del mid y
devolvemos la profundidad total acumulada de ese lado — mejor que inf/nan,
que excluía evaluaciones enteras. Ver caveats en backtest.py.

API:
  fetch_book_hours(hours) -> DepthIndex
  DepthIndex.depth_within(ts_ms, price, side, pct) -> USD absorbible (>=0)
"""

import io
import time
from datetime import datetime

import pandas as pd
import requests

from pool_model import API_KEY, CRYPTOHFTDATA_DOWNLOAD, EXCHANGE, SYMBOL


class DepthIndex:
    """Índice de profundidad por hora: dict[ts_ms] -> (bids, asks) de px->qty."""

    def __init__(self):
        self.books = {}  # ts_ms -> (bids: dict[float,float], asks: dict[float,float])

    def add_hour(self, ts_ms: int, bids: dict, asks: dict):
        self.books[ts_ms] = (bids, asks)

    def best_price(self, ts_ms: int, side: str) -> float:
        """Mejor precio del lado (bid: máximo, ask: mínimo) del libro <= ts_ms."""
        keys = sorted(k for k in self.books if k <= ts_ms)
        if not keys:
            return float("nan")
        bids, asks = self.books[keys[-1]]
        book = bids if side == "bid" else asks
        if not book:
            return float("nan")
        return max(book) if side == "bid" else min(book)

    def depth_within(self, ts_ms: int, price: float, side: str, pct: float) -> float:
        """USD absorbible a ±pct de `price` en el lado `side` (bid/ask).

        Usa el snapshot de libro más reciente <= ts_ms. Si no hay niveles en la
        banda (libro parcial), devuelve 0.0 — el llamador decide el fallback.
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


def fetch_book_hours(hours: list[datetime], max_rows_per_hour: int = 1_200_000) -> DepthIndex:
    """Descarga orderbook por hora y acumula el libro visible de cada hora."""
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    idx = DepthIndex()
    ok = 0
    for i, dt in enumerate(hours, 1):
        path = f"{EXCHANGE}/{dt:%Y-%m-%d}/{dt:%H}/{SYMBOL}_orderbook.parquet"
        url = f"{CRYPTOHFTDATA_DOWNLOAD}?file={path}"
        try:
            resp = requests.get(url, headers=headers, timeout=180)
            if resp.status_code == 200:
                df = pd.read_parquet(io.BytesIO(resp.content))
                if len(df) > max_rows_per_hour:
                    df = df.iloc[:max_rows_per_hour]
                cols = [c for c in ("event_time", "side", "price", "quantity")
                        if c in df.columns]
                df = df[cols]
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
        except Exception:
            pass  # hora perdida: continuamos
        if i % 12 == 0:
            print(f"  [orderbook] {i}/{len(hours)} horas procesadas ({ok} con libro)")
        time.sleep(0.2 if API_KEY else 1.05)
    if ok == 0:
        raise RuntimeError("No se pudo descargar ningún archivo de orderbook")
    print(f"  [orderbook] libro reconstruido para {ok}/{len(hours)} horas")
    return idx
