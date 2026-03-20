"""
Backtesting Page - Bloomberg Terminal Style
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from strategy import buy_and_sell_strategy, buy_and_hold_baseline, calculate_performance_metrics

st.set_page_config(page_title="Backtesting", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# BLOOMBERG CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .stApp { background-color: #0a0a0a !important; font-family: 'IBM Plex Sans', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #0f0f0f !important; border-right: 1px solid #1a1a2e; }
    section[data-testid="stSidebar"] * { color: #8a8a9a !important; }
    header[data-testid="stHeader"] { background-color: #0a0a0a !important; }

    .bb-topbar {
        background: #1a1a2e;
        border-bottom: 2px solid #ff6600;
        padding: 0.6rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
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

    .bb-result-card {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1.2rem;
        text-align: center;
    }
    .bb-result-card.winner { border: 1px solid #00d4aa; }
    .bb-result-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem; color: #5a5a6a;
        text-transform: uppercase; letter-spacing: 0.1em;
        margin-bottom: 0.3rem;
    }
    .bb-result-value {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700; font-size: 1.5rem; color: #c0c0d0;
    }
    .bb-result-delta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem; font-weight: 600; margin-top: 0.2rem;
    }
    .bb-result-delta.pos { color: #00d4aa; }
    .bb-result-delta.neg { color: #ff4444; }

    .bb-strategy-rule {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        border-left: 3px solid #ff6600;
        padding: 0.6rem 1.2rem;
        margin-bottom: 0.3rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #c0c0d0;
    }

    .bb-stat-table {
        width: 100%;
        font-family: 'IBM Plex Mono', monospace;
    }
    .bb-stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1a1a2e;
    }
    .bb-stat-row:last-child { border-bottom: none; }
    .bb-stat-key { font-size: 0.8rem; color: #5a5a6a; }
    .bb-stat-val { font-size: 0.8rem; font-weight: 600; color: #00d4aa; }

    .bb-divider { height: 1px; background: #1a1a2e; margin: 1.5rem 0; }

    .bb-status {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        color: #3a3a4a; text-align: center; padding: 0.8rem 0; border-top: 1px solid #1a1a2e;
    }
    .bb-status .active { color: #00d4aa; }

    .stMarkdown, .stText, p, span, li { color: #c0c0d0 !important; }
    h1, h2, h3 { color: #ffffff !important; }
    div[data-testid="stMetric"] { background: #0f0f14; border: 1px solid #1a1a2e; padding: 0.8rem; border-radius: 0; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
    .stSelectbox > div > div { background-color: #0f0f14 !important; border: 1px solid #1a1a2e !important; }
    .stNumberInput > div > div > input { background-color: #0f0f14 !important; border: 1px solid #1a1a2e !important; color: #c0c0d0 !important; }
    .stExpander { border: 1px solid #1a1a2e !important; }
    div[data-testid="stExpander"] details { background: #0f0f14 !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
FEATURE_COLUMNS = ["Returns", "SMA_5", "SMA_20", "EMA_12", "RSI_14", "Volatility_20", "Volume_Change", "High_Low_Range"]
TICKER_NAMES = {"AAPL": "APPLE INC", "MSFT": "MICROSOFT CORP", "GOOG": "ALPHABET INC", "AMZN": "AMAZON.COM INC", "TSLA": "TESLA INC"}

def load_processed_data(ticker):
    fp = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(fp): return pd.DataFrame()
    df = pd.read_csv(fp)
    if "Date" in df.columns: df["Date"] = pd.to_datetime(df["Date"])
    return df

def load_model_and_scaler(ticker):
    mp = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    sp = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp): return None, None
    return joblib.load(mp), joblib.load(sp)

# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="bb-topbar">
    <div class="bb-topbar-left">BACKTESTING — STRATEGY SIMULATOR</div>
    <div class="bb-topbar-right">HISTORICAL &nbsp;|&nbsp; {now}</div>
</div>
""", unsafe_allow_html=True)

# Strategy rules
st.markdown("""
<div class="bb-strategy-rule">▲ MODEL PREDICTS UP  →  BUY 1 SHARE AT CLOSE</div>
<div class="bb-strategy-rule">▼ MODEL PREDICTS DOWN  →  SELL 1 SHARE AT CLOSE</div>
<div class="bb-strategy-rule">— CANNOT EXECUTE  →  HOLD POSITION</div>
""", unsafe_allow_html=True)

st.write("")

col1, col2 = st.columns(2)
with col1:
    selected_ticker = st.selectbox("TICKER", TICKERS, format_func=lambda t: f"{t}  ·  {TICKER_NAMES.get(t, t)}")
with col2:
    initial_cash = st.number_input("INITIAL CAPITAL ($)", min_value=1000, max_value=100000, value=10000, step=1000)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# Load & run
df = load_processed_data(selected_ticker)
model, scaler = load_model_and_scaler(selected_ticker)

if df.empty:
    st.error(f"NO DATA FOR {selected_ticker}. RUN ETL FIRST.")
    st.stop()
if model is None:
    st.error(f"NO MODEL FOR {selected_ticker}. TRAIN FIRST.")
    st.stop()

with st.spinner("RUNNING BACKTEST..."):
    X = df[FEATURE_COLUMNS].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)
    predictions = pd.Series(model.predict(X_scaled))
    close_prices = df["Close"].reset_index(drop=True)

    strat = buy_and_sell_strategy(predictions, close_prices, initial_cash)
    metrics = calculate_performance_metrics(strat, initial_cash)
    baseline = buy_and_hold_baseline(close_prices, initial_cash)
    bl_final = baseline["Portfolio_Value"].iloc[-1]
    bl_return = ((bl_final - initial_cash) / initial_cash) * 100

ml_return = metrics["total_return_pct"]
ml_wins = ml_return > bl_return

# --- Results ---
st.markdown(f'<div class="bb-panel-header">{selected_ticker} — BACKTEST RESULTS</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    w = "winner" if ml_wins else ""
    d = "pos" if ml_return >= 0 else "neg"
    st.markdown(f"""
    <div class="bb-result-card {w}">
        <div class="bb-result-label">ML STRATEGY</div>
        <div class="bb-result-value">${metrics['final_value']:,.2f}</div>
        <div class="bb-result-delta {d}">{ml_return:+.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col2:
    w = "winner" if not ml_wins else ""
    d = "pos" if bl_return >= 0 else "neg"
    st.markdown(f"""
    <div class="bb-result-card {w}">
        <div class="bb-result-label">BUY & HOLD</div>
        <div class="bb-result-value">${bl_final:,.2f}</div>
        <div class="bb-result-delta {d}">{bl_return:+.2f}%</div>
    </div>""", unsafe_allow_html=True)

with col3:
    winner = "ML STRATEGY" if ml_wins else "BUY & HOLD"
    diff = ml_return - bl_return
    st.markdown(f"""
    <div class="bb-result-card">
        <div class="bb-result-label">OUTPERFORMER</div>
        <div class="bb-result-value" style="font-size:1.1rem;color:#ff6600;">{winner}</div>
        <div class="bb-result-delta {'pos' if diff>=0 else 'neg'}">{diff:+.2f}% SPREAD</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="bb-result-card">
        <div class="bb-result-label">INITIAL CAPITAL</div>
        <div class="bb-result-value">${initial_cash:,.2f}</div>
        <div class="bb-result-delta" style="color:#5a5a6a;">STARTING</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# --- Chart ---
st.markdown('<div class="bb-panel-header">PORTFOLIO VALUE — TIME SERIES</div>', unsafe_allow_html=True)

chart_data = pd.DataFrame({
    "ML Strategy": strat["Portfolio_Value"].values,
    "Buy & Hold": baseline["Portfolio_Value"].values,
})
if "Date" in df.columns:
    chart_data.index = df["Date"].reset_index(drop=True)

st.line_chart(chart_data)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# --- Stats ---
st.markdown('<div class="bb-panel-header">DETAILED STATISTICS</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    <div class="bb-panel">
        <div class="bb-panel-header">ML STRATEGY</div>
        <div class="bb-stat-row"><span class="bb-stat-key">BUY ORDERS</span><span class="bb-stat-val">{metrics['total_buys']}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">SELL ORDERS</span><span class="bb-stat-val">{metrics['total_sells']}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">HOLD DAYS</span><span class="bb-stat-val">{metrics['total_holds']}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">PEAK VALUE</span><span class="bb-stat-val">${metrics['max_portfolio_value']:,.2f}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">TROUGH VALUE</span><span class="bb-stat-val">${metrics['min_portfolio_value']:,.2f}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">AVG DAILY RTN</span><span class="bb-stat-val">{metrics['avg_daily_return_pct']:.4f}%</span></div>
    </div>""", unsafe_allow_html=True)

with col2:
    bh_max = baseline["Portfolio_Value"].max()
    bh_min = baseline["Portfolio_Value"].min()
    shares = int(initial_cash / close_prices.iloc[0])
    st.markdown(f"""
    <div class="bb-panel">
        <div class="bb-panel-header">BUY & HOLD</div>
        <div class="bb-stat-row"><span class="bb-stat-key">STRATEGY</span><span class="bb-stat-val">FULL ALLOC DAY 1</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">SHARES</span><span class="bb-stat-val">{shares}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">ENTRY PRICE</span><span class="bb-stat-val">${close_prices.iloc[0]:,.2f}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">PEAK VALUE</span><span class="bb-stat-val">${bh_max:,.2f}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">TROUGH VALUE</span><span class="bb-stat-val">${bh_min:,.2f}</span></div>
        <div class="bb-stat-row"><span class="bb-stat-key">FINAL VALUE</span><span class="bb-stat-val">${bl_final:,.2f}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# Trade log
st.markdown('<div class="bb-panel-header">TRADE LOG</div>', unsafe_allow_html=True)
with st.expander("VIEW FULL LOG"):
    display_df = strat.copy()
    if "Date" in df.columns:
        display_df.insert(0, "Date", df["Date"].reset_index(drop=True))
    st.dataframe(display_df, use_container_width=True)

# Status
st.markdown(f"""
<div class="bb-status">
    BACKTEST <span class="active">●</span> COMPLETE &nbsp;&nbsp;|&nbsp;&nbsp;
    {selected_ticker} &nbsp;&nbsp;|&nbsp;&nbsp;
    {len(df)} TRADING DAYS &nbsp;&nbsp;|&nbsp;&nbsp;
    {now} &nbsp;&nbsp;|&nbsp;&nbsp;
    ⚠ SIMULATED — NOT ACTUAL RETURNS
</div>
""", unsafe_allow_html=True)
