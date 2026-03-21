# -*- coding: utf-8 -*-
"""
Go Live Page - Bloomberg Terminal Style
========================================
Real-time ML predictions + peer comparison inspired by demo-stockpeers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pysimfin import PySimFin
from etl import FEATURE_COLUMNS, transform

st.set_page_config(page_title="Go Live", page_icon="🚀", layout="wide")

# ---------------------------------------------------------------------------
# BLOOMBERG CSS (matches Home.py and Backtesting.py)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .stApp { background-color: #0a0a0a !important; font-family: 'IBM Plex Sans', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #8a8a9a !important; }
    header[data-testid="stHeader"] { background-color: #0a0a0a !important; }

    /* Nav links */
    [data-testid="stSidebarNavLink"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        border-radius: 2px !important;
        margin: 1px 0 !important;
    }
    [data-testid="stSidebarNavLink"]:hover {
        background-color: #1a1a2e !important;
    }
    [data-testid="stSidebarNavLink"][aria-current="page"] {
        background-color: #1a1a2e !important;
        border-left: 2px solid #ff6600 !important;
    }
    [data-testid="stSidebarNavLink"] span {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
    }

    .bb-topbar {
        background: #1a1a2e; border-bottom: 2px solid #ff6600;
        padding: 0.6rem 1.5rem; display: flex;
        justify-content: space-between; align-items: center;
        margin: -1rem -1rem 1.5rem -1rem;
    }
    .bb-topbar-left { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.85rem; color: #ff6600; letter-spacing: 0.08em; }
    .bb-topbar-right { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #5a5a6a; }

    .bb-panel { background: #0f0f14; border: 1px solid #1a1a2e; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
    .bb-panel-header {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; font-weight: 600;
        color: #ff6600; text-transform: uppercase; letter-spacing: 0.12em;
        margin-bottom: 0.8rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1a1a2e;
    }

    /* Signal card */
    .bb-signal-card {
        background: #0f0f14; border: 1px solid #1a1a2e;
        padding: 1.5rem; text-align: center;
    }
    .bb-signal-card.buy  { border: 2px solid #00d4aa; }
    .bb-signal-card.sell { border: 2px solid #ff4444; }
    .bb-signal-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #5a5a6a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
    .bb-signal-value { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 2.2rem; }
    .bb-signal-value.buy  { color: #00d4aa; }
    .bb-signal-value.sell { color: #ff4444; }
    .bb-signal-sub { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #5a5a6a; margin-top: 0.3rem; }

    /* Metric tiles */
    .bb-metric { background: #0f0f14; border: 1px solid #1a1a2e; padding: 1rem; text-align: center; }
    .bb-metric-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: #5a5a6a; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
    .bb-metric-value { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1.25rem; color: #c0c0d0; }
    .bb-metric-value.pos { color: #00d4aa; }
    .bb-metric-value.neg { color: #ff4444; }

    /* Peer comparison cards */
    .bb-peer-card { background: #0f0f14; border: 1px solid #1a1a2e; padding: 1rem 1.2rem; margin-bottom: 0.4rem; display: flex; justify-content: space-between; align-items: center; }
    .bb-peer-ticker { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 1rem; color: #ffffff; }
    .bb-peer-name { font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #5a5a6a; }
    .bb-peer-ret { font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 1rem; }
    .bb-peer-ret.pos { color: #00d4aa; }
    .bb-peer-ret.neg { color: #ff4444; }

    /* Pill buttons (time horizon) */
    div[data-testid="stHorizontalBlock"] .stRadio > div { flex-direction: row; gap: 0.4rem; }
    .stRadio label { background: #1a1a2e !important; border: 1px solid #252540 !important;
        color: #5a5a6a !important; padding: 0.3rem 1rem !important;
        font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
        cursor: pointer; }
    .stRadio label:has(input:checked) { background: #ff6600 !important; color: #ffffff !important; border-color: #ff6600 !important; }

    .bb-divider { height: 1px; background: #1a1a2e; margin: 1.5rem 0; }

    .bb-status { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: #3a3a4a; text-align: center; padding: 0.8rem 0; border-top: 1px solid #1a1a2e; }
    .bb-status .active { color: #00d4aa; }

    .stMarkdown, .stText, p, span, li { color: #c0c0d0 !important; }
    h1, h2, h3 { color: #ffffff !important; }
    div[data-testid="stMetric"] { background: #0f0f14; border: 1px solid #1a1a2e; padding: 0.8rem; border-radius: 0; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
    .stSelectbox > div > div { background-color: #0f0f14 !important; border: 1px solid #1a1a2e !important; color: #c0c0d0 !important; }
    .stExpander { border: 1px solid #1a1a2e !important; }
    div[data-testid="stExpander"] details { background: #0f0f14 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar title — injected as real HTML, more reliable than ::before CSS
st.sidebar.markdown("""
<div style="
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    color: #ff6600;
    letter-spacing: 0.18em;
    padding: 1.2rem 1rem 0.8rem 1rem;
    border-bottom: 1px solid #1a1a2e;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
">AUTOMATED TRADING SYSTEM</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
TICKER_NAMES = {
    "AAPL": "APPLE INC",
    "MSFT": "MICROSOFT CORP",
    "GOOG": "ALPHABET INC",
    "AMZN": "AMAZON.COM INC",
    "TSLA": "TESLA INC",
}
TICKER_COLORS = {
    "AAPL": "#1f77b4",
    "MSFT": "#ff7f0e",
    "GOOG": "#2ca02c",
    "AMZN": "#d62728",
    "TSLA": "#9467bd",
}
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

TIME_HORIZON_DAYS = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
@st.cache_resource
def get_simfin_client():
    api_key = st.secrets.get("SIMFIN_API_KEY", "")
    if not api_key:
        st.error("SimFin API key not found. Set SIMFIN_API_KEY in .streamlit/secrets.toml")
        st.stop()
    return PySimFin(api_key=api_key)


@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    client = get_simfin_client()
    try:
        return client.get_share_prices(ticker, start, end)
    except Exception as e:
        st.error(f"SimFin API error for {ticker}: {e}")
        return pd.DataFrame()


def rename_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename SimFin API columns to match bulk-download schema."""
    return df.rename(columns={
        "Last Closing Price":    "Close",
        "Highest Price":         "High",
        "Lowest Price":          "Low",
        "Opening Price":         "Open",
        "Trading Volume":        "Volume",
        "Adjusted Closing Price":"Adj. Close",
    })


def apply_etl_transformations(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Rename API columns then run through shared transform() function."""
    df = rename_api_columns(df.copy())
    df["Ticker"] = ticker
    return transform(df, ticker, include_target=False)


def load_model(ticker: str):
    mp = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    sp = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None
    return joblib.load(mp), joblib.load(sp)


def predict(model, scaler, row: pd.Series) -> dict:
    X = row[FEATURE_COLUMNS].values.reshape(1, -1)
    X_s = scaler.transform(X)
    prob = model.predict_proba(X_s)[0]
    pred = 1 if prob[1] >= 0.5 else 0
    return {
        "prediction": int(pred),
        "label":      "BUY" if pred == 1 else "SELL",
        "confidence": float(max(prob)),
        "prob_up":    float(prob[1]),
        "prob_down":  float(prob[0]),
    }


def load_peer_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Get closing prices for a ticker using the already-cached fetch_prices()."""
    df = fetch_prices(ticker, start, end)
    if df.empty:
        return pd.DataFrame()
    df = rename_api_columns(df.copy())
    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"])
    return df[["Date", "Close"]].dropna()


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="bb-topbar">
    <div class="bb-topbar-left">GO LIVE — REAL-TIME SIGNAL GENERATOR</div>
    <div class="bb-topbar-right">LIVE &nbsp;|&nbsp; {now}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONTROLS: ticker + time horizon
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 600;
     color: #ff6600; text-transform: uppercase; letter-spacing: 0.14em; margin-bottom: 0.6rem;">
    Live Market Intelligence &nbsp;·&nbsp; ML-Powered Trading Signals
</div>
""", unsafe_allow_html=True)

ctrl_col1, ctrl_col2 = st.columns([1, 2])

with ctrl_col1:
    selected_ticker = st.selectbox(
        "TICKER",
        TICKERS,
        format_func=lambda t: f"{t}  ·  {TICKER_NAMES.get(t, t)}"
    )

with ctrl_col2:
    horizon_label = st.radio(
        "TIME HORIZON",
        list(TIME_HORIZON_DAYS.keys()),
        index=1,          # default: 3M
        horizontal=True,
    )

horizon_days = TIME_HORIZON_DAYS[horizon_label]
end_date   = datetime.today()
start_date = end_date - timedelta(days=horizon_days)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FETCH LIVE DATA
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching {selected_ticker} from SimFin..."):
    raw_df = fetch_prices(
        selected_ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

if raw_df.empty:
    st.error(f"No data returned for {selected_ticker}. Check your API key or try again.")
    st.stop()

# Rename columns for price display
display_df = rename_api_columns(raw_df.copy())
if "Date" in display_df.columns:
    display_df["Date"] = pd.to_datetime(display_df["Date"])
    display_df = display_df.sort_values("Date").reset_index(drop=True)

# Apply ETL for ML features
transformed_df = apply_etl_transformations(raw_df, selected_ticker)
latest_complete = transformed_df.dropna(subset=FEATURE_COLUMNS)

# ---------------------------------------------------------------------------
# SECTION 1: ML PREDICTION SIGNAL
# ---------------------------------------------------------------------------
st.markdown('<div class="bb-panel-header">ML SIGNAL — NEXT TRADING DAY</div>', unsafe_allow_html=True)

model, scaler = load_model(selected_ticker)

if model is None:
    st.warning(f"No trained model for {selected_ticker}. Run `python src/model.py --ticker {selected_ticker}`")
elif latest_complete.empty:
    st.warning("Not enough data to generate a signal (need at least 20 trading days).")
else:
    latest_row = latest_complete.iloc[-1]
    result = predict(model, scaler, latest_row)

    sig_class = "buy" if result["prediction"] == 1 else "sell"
    sig_icon  = "▲" if result["prediction"] == 1 else "▼"
    as_of     = str(latest_row.get("Date", ""))[:10]

    s1, s2, s3, s4, s5 = st.columns(5)

    with s1:
        st.markdown(f"""
        <div class="bb-signal-card {sig_class}">
            <div class="bb-signal-label">SIGNAL</div>
            <div class="bb-signal-value {sig_class}">{sig_icon} {result['label']}</div>
            <div class="bb-signal-sub">{selected_ticker} · {as_of}</div>
        </div>""", unsafe_allow_html=True)

    with s2:
        st.markdown(f"""
        <div class="bb-metric">
            <div class="bb-metric-label">CONFIDENCE</div>
            <div class="bb-metric-value">{result['confidence']*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with s3:
        rsi_val = latest_row.get("RSI_14", 0)
        rsi_cls = "neg" if rsi_val > 70 else ("pos" if rsi_val < 30 else "")
        st.markdown(f"""
        <div class="bb-metric">
            <div class="bb-metric-label">RSI (14)</div>
            <div class="bb-metric-value {rsi_cls}">{rsi_val:.1f}</div>
        </div>""", unsafe_allow_html=True)

    with s4:
        sma_diff = latest_row.get("SMA_5", 0) - latest_row.get("SMA_20", 0)
        sma_cls  = "pos" if sma_diff > 0 else "neg"
        st.markdown(f"""
        <div class="bb-metric">
            <div class="bb-metric-label">SMA5 vs SMA20</div>
            <div class="bb-metric-value {sma_cls}">{sma_diff:+.2f}</div>
        </div>""", unsafe_allow_html=True)

    with s5:
        vol = latest_row.get("Volatility_20", 0)
        st.markdown(f"""
        <div class="bb-metric">
            <div class="bb-metric-label">VOLATILITY (20d)</div>
            <div class="bb-metric-value">{vol:.4f}</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)
    prob_bar_up   = int(result["prob_up"]   * 100)
    prob_bar_down = int(result["prob_down"] * 100)
    st.markdown(f"""
    <div style="display:flex; gap:0; height:6px; margin-bottom:0.3rem;">
        <div style="width:{prob_bar_up}%; background:#00d4aa;"></div>
        <div style="width:{prob_bar_down}%; background:#ff4444;"></div>
    </div>
    <div style="display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#5a5a6a;">
        <span>▲ UP {result['prob_up']*100:.1f}%</span>
        <span>▼ DOWN {result['prob_down']*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SECTION 2: PRICE CHART + MOVING AVERAGES
# ---------------------------------------------------------------------------
st.markdown(f'<div class="bb-panel-header">{selected_ticker} — PRICE & MOVING AVERAGES ({horizon_label})</div>', unsafe_allow_html=True)

if not display_df.empty and "Date" in display_df.columns:
    ohlc = display_df[["Date", "Open", "High", "Low", "Close"]].dropna()
    mas  = transformed_df[["Date", "SMA_5", "SMA_20", "EMA_12"]].dropna() if not transformed_df.empty else pd.DataFrame()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc["Date"],
        open=ohlc["Open"], high=ohlc["High"],
        low=ohlc["Low"],   close=ohlc["Close"],
        name="Price",
        increasing_line_color="#00d4aa", increasing_fillcolor="#00d4aa",
        decreasing_line_color="#ff4444", decreasing_fillcolor="#ff4444",
    ))

    # Moving average overlays
    if not mas.empty:
        fig.add_trace(go.Scatter(x=mas["Date"], y=mas["EMA_12"], name="EMA_12",
            line=dict(color="#ff6600", width=1.2)))
        fig.add_trace(go.Scatter(x=mas["Date"], y=mas["SMA_20"], name="SMA_20",
            line=dict(color="#ffffff", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=mas["Date"], y=mas["SMA_5"], name="SMA_5",
            line=dict(color="#5a5a9a", width=1)))

    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0f0f14",
        font=dict(family="IBM Plex Mono", color="#8a8a9a", size=11),
        xaxis=dict(gridcolor="#1a1a2e", showgrid=True, rangeslider_visible=False,
                   tickfont=dict(color="#5a5a6a")),
        yaxis=dict(gridcolor="#1a1a2e", showgrid=True, tickprefix="$",
                   tickfont=dict(color="#5a5a6a")),
        legend=dict(bgcolor="#0f0f14", bordercolor="#1a1a2e", borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=0, r=0, t=10, b=0),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

with st.expander("VIEW RAW PRICE DATA"):
    show_cols = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in display_df.columns]
    st.dataframe(display_df[show_cols].tail(20).round(2), use_container_width=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SECTION 3: PEER COMPARISON — normalized price (stockpeers style)
# ---------------------------------------------------------------------------
st.markdown('<div class="bb-panel-header">PEER COMPARISON — NORMALIZED PRICE (PROCESSED DATA)</div>', unsafe_allow_html=True)
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; color:#5a5a6a; margin-bottom:0.8rem;">Prices normalized to 1.0 at start of period. Easily compare performance across all tickers.</div>', unsafe_allow_html=True)

# Fetch closing prices for all tickers via the SimFin API (already cached per ticker)
peer_frames = {}
for t in TICKERS:
    df_t = load_peer_prices(t, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
    if not df_t.empty and len(df_t) > 5:
        peer_frames[t] = df_t.set_index("Date")["Close"]

if peer_frames:
    norm_df = pd.DataFrame(peer_frames)
    norm_df = norm_df.dropna(how="all")
    # Normalize each series to 1.0 at its first valid value
    norm_df = norm_df.div(norm_df.bfill().iloc[0])

    # Best and worst over the period
    returns = (norm_df.iloc[-1] - 1) * 100
    best_t  = returns.idxmax()
    worst_t = returns.idxmin()

    b1, b2 = st.columns(2)
    with b1:
        bret = returns[best_t]
        st.markdown(f"""
        <div class="bb-metric" style="border-left:3px solid #00d4aa;">
            <div class="bb-metric-label">BEST PERFORMER ({horizon_label})</div>
            <div class="bb-metric-value" style="font-size:2rem; color:#00d4aa;">{best_t}</div>
            <div class="bb-metric-value pos" style="font-size:1rem;">{bret:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

    with b2:
        wret = returns[worst_t]
        st.markdown(f"""
        <div class="bb-metric" style="border-left:3px solid #ff4444;">
            <div class="bb-metric-label">WORST PERFORMER ({horizon_label})</div>
            <div class="bb-metric-value" style="font-size:2rem; color:#ff4444;">{worst_t}</div>
            <div class="bb-metric-value neg" style="font-size:1rem;">{wret:+.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
    st.line_chart(norm_df)

    # Return leaderboard
    st.markdown('<div style="height:0.5rem;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="bb-panel-header">RETURN LEADERBOARD</div>', unsafe_allow_html=True)
    sorted_tickers = returns.sort_values(ascending=False).index.tolist()
    for t in sorted_tickers:
        ret  = returns[t]
        cls  = "pos" if ret >= 0 else "neg"
        sign = "▲" if ret >= 0 else "▼"
        st.markdown(f"""
        <div class="bb-peer-card">
            <div>
                <div class="bb-peer-ticker">{t}</div>
                <div class="bb-peer-name">{TICKER_NAMES.get(t, t)}</div>
            </div>
            <div class="bb-peer-ret {cls}">{sign} {abs(ret):.2f}%</div>
        </div>""", unsafe_allow_html=True)
else:
    st.warning("Could not fetch peer data from SimFin. Check your API key or try again.")

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SECTION 4: INDIVIDUAL STOCK vs PEER AVERAGE (stockpeers style)
# ---------------------------------------------------------------------------
st.markdown('<div class="bb-panel-header">INDIVIDUAL vs PEER AVERAGE</div>', unsafe_allow_html=True)
st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.72rem; color:#5a5a6a; margin-bottom:1rem;">For each stock, the peer average excludes that stock itself. Delta = stock minus peer average.</div>', unsafe_allow_html=True)

if peer_frames and len(peer_frames) >= 2:
    cols = st.columns(len(TICKERS))
    for i, t in enumerate(TICKERS):
        if t not in peer_frames:
            continue
        with cols[i]:
            st.markdown(f'<div style="font-family:\'IBM Plex Mono\',monospace; font-size:0.75rem; font-weight:700; color:#ff6600; margin-bottom:0.4rem;">{t}</div>', unsafe_allow_html=True)

            stock_norm = norm_df[t]
            peers_norm = norm_df[[x for x in norm_df.columns if x != t]]
            peer_avg   = peers_norm.mean(axis=1)

            comparison = pd.DataFrame({
                t:            stock_norm,
                "Peer Avg":   peer_avg,
            }).dropna()

            st.line_chart(comparison, height=160, color=["#00d4aa", "#5a5a6a"])

            delta = (stock_norm - peer_avg).dropna()
            delta_df = pd.DataFrame({"Delta": delta})
            st.area_chart(delta_df, height=100, color=["#ff6600"])

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SECTION 5: TECHNICAL INDICATORS TABLE
# ---------------------------------------------------------------------------
if not transformed_df.empty and not latest_complete.empty:
    latest_row = latest_complete.iloc[-1]

    st.markdown('<div class="bb-panel-header">TECHNICAL INDICATORS — LATEST VALUES</div>', unsafe_allow_html=True)

    t1, t2, t3, t4, t5, t6, t7, t8 = st.columns(8)
    indicator_data = [
        ("RETURNS",   f"{latest_row.get('Returns', 0)*100:.3f}%", latest_row.get('Returns', 0) > 0),
        ("SMA 5",     f"{latest_row.get('SMA_5', 0):.2f}",        None),
        ("SMA 20",    f"{latest_row.get('SMA_20', 0):.2f}",       None),
        ("EMA 12",    f"{latest_row.get('EMA_12', 0):.2f}",       None),
        ("RSI 14",    f"{latest_row.get('RSI_14', 0):.1f}",       latest_row.get('RSI_14', 50) < 50),
        ("VOL 20d",   f"{latest_row.get('Volatility_20', 0):.4f}",None),
        ("VOL CHG",   f"{latest_row.get('Volume_Change', 0)*100:.1f}%", latest_row.get('Volume_Change', 0) > 0),
        ("H-L RANGE", f"{latest_row.get('High_Low_Range', 0):.2f}",None),
    ]

    for col, (label, value, is_positive) in zip([t1,t2,t3,t4,t5,t6,t7,t8], indicator_data):
        with col:
            if is_positive is None:
                cls = ""
            elif is_positive:
                cls = "pos"
            else:
                cls = "neg"
            st.markdown(f"""
            <div class="bb-metric">
                <div class="bb-metric-label">{label}</div>
                <div class="bb-metric-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("VIEW FULL INDICATOR HISTORY"):
        display_cols = [c for c in ["Date"] + FEATURE_COLUMNS if c in transformed_df.columns]
        st.dataframe(transformed_df[display_cols].tail(30).round(4), use_container_width=True)

# ---------------------------------------------------------------------------
# STATUS BAR
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="bb-status">
    SIGNAL <span class="active">●</span> LIVE &nbsp;&nbsp;|&nbsp;&nbsp;
    {selected_ticker} &nbsp;&nbsp;|&nbsp;&nbsp;
    HORIZON {horizon_label} &nbsp;&nbsp;|&nbsp;&nbsp;
    {now} &nbsp;&nbsp;|&nbsp;&nbsp;
    ⚠ EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
