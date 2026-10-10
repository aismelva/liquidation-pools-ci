"""
Dashboard v6 — Pools de Liquidación (diseño moderno)

Tres capas documentadas (ver manual-proyecto-pools.md):
  1. Mapa de pools (heatmap interactivo) — validado (Glassnode + backtests)
  2. Severidad esperada por pool (usd vs profundidad) — avalado, etiqueta no ranking
  3. Dinámica del mapa (evolución de pools) — proxy de convicción (diferencial)

Estilo: dark theme, tipografía Inter, tarjetas glassmorphism, acentos
  neón (ámbar para longs / cian para shorts), Plotly interactivo.

Ejecutar: streamlit run src/dashboard.py
(env opcional: CRYPTOHFTDATA_API_KEY, LIQ_SYMBOL, LIQ_DAYS)
"""

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from book_depth import fetch_book_hours
from pool_model import (
    BUCKET_USD,
    DAYS,
    SYMBOL,
    build_funding_series,
    build_pools,
    build_snapshots,
    fetch_dataset,
    hourly_targets,
    pools_from_cohorts,
)

# ------------------------- Configuración de página -------------------------

st.set_page_config(
    page_title=f"Pools · {SYMBOL}",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT_LONG = "#f5a623"   # ámbar — pools de longs (debajo del precio)
ACCENT_SHORT = "#00d4ff"  # cian — pools de shorts (encima del precio)
BG = "#0b0e14"
CARD = "#151a23"
TEXT = "#e8ecf3"
MUTED = "#8b93a7"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp {{ background: {BG}; color: {TEXT}; font-family: 'Inter', sans-serif; }}
    [data-testid="stSidebar"] {{ background: {CARD}; border-right: 1px solid #232a37; }}
    [data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
    h1, h2, h3 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
    .glass {{
        background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px 24px;
        backdrop-filter: blur(8px);
    }}
    .kpi-label {{ font-size: 12px; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 600; }}
    .kpi-value {{ font-size: 28px; font-weight: 800; margin-top: 2px; }}
    .kpi-sub {{ font-size: 12px; color: {MUTED}; margin-top: 4px; }}
    .badge {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .badge-long {{ background: rgba(245,166,35,0.15); color: {ACCENT_LONG}; border: 1px solid rgba(245,166,35,0.4); }}
    .badge-short {{ background: rgba(0,212,255,0.12); color: {ACCENT_SHORT}; border: 1px solid rgba(0,212,255,0.4); }}
    .badge-sev-high {{ background: rgba(255,77,109,0.15); color: #ff4d6d; border: 1px solid rgba(255,77,109,0.45); }}
    .badge-sev-mid {{ background: rgba(245,166,35,0.12); color: {ACCENT_LONG}; border: 1px solid rgba(245,166,35,0.35); }}
    .badge-sev-low {{ background: rgba(139,147,167,0.12); color: {MUTED}; border: 1px solid rgba(139,147,167,0.3); }}
    .header-grad {{
        background: linear-gradient(90deg, {TEXT} 0%, {MUTED} 120%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-size: 30px; font-weight: 800; letter-spacing: -0.03em;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid #232a37; border-radius: 14px; }}
    .footer {{ color: {MUTED}; font-size: 11.5px; line-height: 1.5; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------ Carga de datos ------------------------------


@st.cache_resource(show_spinner="Descargando datos de CryptoHFTData…")
def load_data(days: int):
    hours = hourly_targets()
    snap = build_snapshots(fetch_dataset(hours, "open_interest"))
    funding = build_funding_series(fetch_dataset(hours, "mark_price"))
    liq_raw = fetch_dataset(hours, "liquidations")
    liq = liq_raw.copy()
    liq["exec_px"] = liq["average_price"].astype(float)
    liq["usd"] = liq["quantity"].astype(float) * liq["exec_px"]
    liq["is_long"] = liq["side"] == "SELL"
    liq["ts"] = pd.to_datetime(liq["event_time"].astype("int64"), unit="ms")
    return snap, funding, liq


@st.cache_resource(show_spinner="Reconstruyendo profundidad del libro L2…")
def load_depth(book_hours: int):
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hours = [now - timedelta(hours=h) for h in range(book_hours, 0, -1)]
    return fetch_book_hours(hours)


# ------------------------------- Interfaz ----------------------------------

with st.sidebar:
    st.markdown("### 💧 Liquidation Pools")
    st.caption("v6 · diseño documentado")
    st.divider()
    days = st.slider("Ventana del mapa (días)", 1, 14, 3)
    top_n = st.slider("Pools destacados", 5, 25, 12)
    show_depth = st.toggle("Capa 2 · Severidad (libro L2)", value=False,
                           help="Descarga order book L2 (pesado). Etiqueta cada pool con su ratio usd/profundidad — avalado como modificador de severidad, no como selector.")
    st.divider()
    st.caption("Fuente: CryptoHFTData (gratis) · Modelo v4.1 (cohortes OI + funding + margen)")

snap, funding, liq = load_data(days)
cohorts = build_pools(snap, funding)
pools = pools_from_cohorts(cohorts)
px_now = float(snap["px"].iloc[-1])
ts_ms_now = int(snap["ts"].iloc[-1].value // 10**6)

pools["dist"] = (pools["bucket"] / px_now - 1).abs()
pools["dist_pct"] = (pools["dist"] * 100).round(2)
pools["usd_m"] = pools["usd"] / 1e6
pools = pools.sort_values("usd", ascending=False)

# Capa 3 — dinámica: recomputar el mapa con ventana reducida y comparar
half = snap[snap["ts"] >= snap["ts"].max() - pd.Timedelta(days=max(days - 1, 1))]
cohorts_half = build_pools(half, funding)
pools_half = pools_from_cohorts(cohorts_half)
prev = pools_half.set_index(["bucket", "side"])["usd"]
pools["usd_prev"] = [prev.get((b, s), 0.0) for b, s in zip(pools.bucket, pools.side)]
pools["delta_pct"] = ((pools["usd"] - pools["usd_prev"]) / pools["usd_prev"].clip(lower=1)).mul(100).round(0)

# Capa 2 — severidad
depth_idx = None
if show_depth:
    depth_idx = load_depth(24)


def depth_for(bucket: float, side: str) -> float:
    if depth_idx is None:
        return float("nan")
    side_book = "bid" if side == "long" else "ask"
    d = depth_idx.depth_within(ts_ms_now, bucket, side_book, 0.005)
    if d == d and d > 0:
        return d
    keys = sorted(k for k in depth_idx.books if k <= ts_ms_now)
    if not keys:
        return float("nan")
    bids, asks = depth_idx.books[keys[-1]]
    book = bids if side_book == "bid" else asks
n    return sum(px * qty for px, qty in book.items()) or float("nan")


if show_depth:
    pools["book_usd"] = [depth_for(b, s) for b, s in zip(pools.bucket, pools.side)]
    pools["severity"] = pools["usd"] / pools["book_usd"]
else:
    pools["severity"] = np.nan

# ------------------------------ Encabezado ---------------------------------

col1, col2, col3, col4 = st.columns(4)
long_usd = pools.loc[pools.side == "long", "usd_m"].sum()
short_usd = pools.loc[pools.side == "short", "usd_m"].sum()
liq_24h = liq.loc[liq.ts >= snap["ts"].max() - pd.Timedelta(hours=24), "usd"].sum() / 1e6


def kpi(col, label, value, sub, color=TEXT):
    col.markdown(
        f'<div class="glass"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


kpi(col1, "Precio actual", f"${px_now:,.0f}", f"{SYMBOL} · Binance Futures")
kpi(col2, "Pools de longs", f"${long_usd:,.0f}M", "debajo del precio", ACCENT_LONG)
kpi(col3, "Pools de shorts", f"${short_usd:,.0f}M", "encima del precio", ACCENT_SHORT)
kpi(col4, "Liquidado 24h", f"${liq_24h:,.1f}M", "real (stream)")

st.markdown(
    f'<div style="margin: 10px 0 2px;">'
    f'<span class="header-grad">Mapa de Pools</span> '
    f'<span style="color:{MUTED}; font-size:13px;">capa 1 · ubicación validada · ventana {days}d</span></div>',
    unsafe_allow_html=True,
)

# ------------------------- Heatmap interactivo ------------------------------

fig = go.Figure()

for side, color, name in (("long", ACCENT_LONG, "Longs (debajo)"),
                          ("short", ACCENT_SHORT, "Shorts (encima)")):
    sub = pools[pools.side == side].sort_values("usd", ascending=False).head(60)
    sub = sub[sub.dist <= 0.12]
    fig.add_trace(go.Bar(
        x=sub["bucket"], y=sub["usd_m"],
        name=name, marker_color=color, opacity=0.85,
        marker_line_width=0, width=BUCKET_USD * 0.92,
        customdata=np.stack((sub["dist_pct"], sub["delta_pct"], sub["severity"]), axis=-1),
        hovertemplate=(
            "<b>%{x:,.0f} USD</b><br>"
            "USD en riesgo: %{y:.1f}M<br>"
            "Distancia: %{customdata[0]:.2f}%<br>"
            "Δ vs ventana previa: %{customdata[1]:+.0f}%<br>"
            "Severidad (usd/libro): %{customdata[2]:.1f}x"
            "<extra></extra>"
        ),
    ))

# precio como línea
px_series = snap.set_index("ts")["px"].resample("30min").last().dropna()
fig.add_trace(go.Scatter(
    x=px_series.index, y=px_series.values, name="Precio",
    mode="lines", line=dict(color="#ffffff", width=1.6, shape="spline"),
    hovertemplate="%{x|%H:%M} · %{y:,.0f}<extra>Precio</extra>",
))

fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT),
    height=460, margin=dict(l=10, r=10, t=30, b=10),
    xaxis=dict(title="Nivel de precio (USD)", gridcolor="#232a37",
               zeroline=False, spikecolor="#3a4354"),
    yaxis=dict(title="USD en riesgo (M)", gridcolor="#232a37"),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor=CARD, bordercolor="#232a37", font_size=12),
)
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------- Tabla de pools (3 capas) -------------------------

top = pools[pools.dist <= 0.15].sort_values("usd", ascending=False).head(top_n).copy()


def sev_badge(r):
    if r.severity != r.severity:
        return '<span class="badge badge-sev-low">libro n/d</span>'
    if r.severity >= 5:
        return f'<span class="badge badge-sev-high">{r.severity:.1f}x · violento</span>'
    if r.severity >= 1.5:
        return f'<span class="badge badge-sev-mid">{r.severity:.1f}x · moderado</span>'
    return f'<span class="badge badge-sev-low">{r.severity:.1f}x · absorbible</span>'


def side_badge(s):
    cls = "badge-long" if s == "long" else "badge-short"
    txt = "LONG · venta forz." if s == "long" else "SHORT · compra forz."
    return f'<span class="badge {cls}">{txt}</span>'


def dyn_arrow(d):
    if d > 25:
        return "▲ apilándose"
    if d < -25:
        return "▼ disolviéndose"
    return "· estable"


rows = []
for _, r in top.iterrows():
    rows.append({
        "Nivel": f"${r.bucket:,.0f}",
        "Distancia": f"{r.dist_pct:+.2f}%",
        "USD en riesgo": f"${r.usd_m:,.1f}M",
        "Lado": side_badge(r.side),
        "Dinámica": dyn_arrow(r.delta_pct),
        "Severidad": sev_badge(r),
    })

df_show = pd.DataFrame(rows)
st.markdown("##### Pools destacados")
st.markdown(
    df_show.to_html(escape=False, index=False, classes=""),
    unsafe_allow_html=True,
)

# ---------------------------- Notas honestas -------------------------------

st.divider()
st.markdown(
    f"""
    <div class="footer">
    <b>Cómo leerlo:</b> capa 1 (mapa) validada — los clústeres anticipan zonas de liquidación
    (Glassnode, posiciones reales Hyperliquid; nuestros backtests +2 a +7 pts vs null estricto).
    Capa 2 (severidad): ratio usd/profundidad — si el nivel se toca, cuánto overshoot esperar
    (modificador de severidad, NO selector de destino — 63% de absorción es off-book, arXiv:2608.03616).
    Capa 3 (dinámica): pools que crecen = apilamiento (riesgo de squeeze); que se disuelven =
    deleveraging. Uso recomendado: contexto de swing (días), no ejecución intradía.
    <br><br>Datos: CryptoHFTData · Liquidations: Binance (cap 1 BTC/evento) · Modelo: cohortes de OI
    + funding-split + margen 0.5% · <b>No es asesoramiento financiero.</b>
    </div>
    """,
    unsafe_allow_html=True,
)
