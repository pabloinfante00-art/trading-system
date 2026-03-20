"""
Home Page - Automated Daily Trading System
Bloomberg Terminal-Inspired Design
"""

import streamlit as st

st.set_page_config(page_title="Trading System", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# BLOOMBERG TERMINAL CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    /* Force dark background everywhere */
    .stApp {
        background-color: #0a0a0a !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f0f0f !important;
        border-right: 1px solid #1a1a2e;
    }
    section[data-testid="stSidebar"] * {
        color: #8a8a9a !important;
    }
    header[data-testid="stHeader"] {
        background-color: #0a0a0a !important;
    }

    /* Top bar — Bloomberg-style orange/black bar */
    .bb-topbar {
        background: #1a1a2e;
        border-bottom: 2px solid #ff6600;
        padding: 0.6rem 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: -1rem -1rem 1.5rem -1rem;
        border-radius: 0;
    }
    .bb-topbar-left {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 0.85rem;
        color: #ff6600;
        letter-spacing: 0.08em;
    }
    .bb-topbar-right {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #5a5a6a;
    }

    /* Main title block */
    .bb-title-block {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        border-left: 3px solid #ff6600;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
    }
    .bb-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
        margin-bottom: 0.3rem;
    }
    .bb-subtitle {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: #5a5a6a;
        letter-spacing: 0.02em;
    }

    /* Data panels */
    .bb-panel {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
    }
    .bb-panel-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        color: #ff6600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1a1a2e;
    }
    .bb-panel-body {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.88rem;
        color: #c0c0d0;
        line-height: 1.7;
    }

    /* Ticker display — Bloomberg style */
    .bb-ticker-row {
        display: flex;
        gap: 0;
        margin-bottom: 1rem;
    }
    .bb-ticker-cell {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1rem;
        flex: 1;
        text-align: center;
        transition: all 0.15s ease;
    }
    .bb-ticker-cell:hover {
        border-color: #ff6600;
        background: #12121a;
    }
    .bb-ticker-symbol {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 1.4rem;
        color: #00d4aa;
        letter-spacing: 0.05em;
    }
    .bb-ticker-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #5a5a6a;
        margin-top: 0.2rem;
        letter-spacing: 0.02em;
    }

    /* Pipeline steps */
    .bb-step {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        border-top: 2px solid #ff6600;
        padding: 1.2rem 1.5rem;
        height: 100%;
    }
    .bb-step-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #ff6600;
        margin-bottom: 0.3rem;
    }
    .bb-step-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
    }
    .bb-step-desc {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: #5a5a6a;
        line-height: 1.6;
    }

    /* Tech tags */
    .bb-tag {
        display: inline-block;
        background: #1a1a2e;
        color: #00d4aa;
        padding: 0.3rem 0.8rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        margin: 0.15rem;
        letter-spacing: 0.05em;
        border: 1px solid #252540;
    }

    /* Section divider */
    .bb-divider {
        height: 1px;
        background: #1a1a2e;
        margin: 1.5rem 0;
    }

    /* Status line */
    .bb-status {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3a3a4a;
        text-align: center;
        padding: 0.8rem 0;
        border-top: 1px solid #1a1a2e;
        letter-spacing: 0.03em;
    }
    .bb-status .active {
        color: #00d4aa;
    }

    /* Override Streamlit defaults */
    .stMarkdown, .stText, p, span, li { color: #c0c0d0 !important; }
    h1, h2, h3 { color: #ffffff !important; }
    .stAlert > div { background-color: #1a1a2e !important; border: 1px solid #252540 !important; }
    div[data-testid="stMetric"] { background: #0f0f14; border: 1px solid #1a1a2e; padding: 1rem; border-radius: 0; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; color: #00d4aa !important; }
    div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; color: #5a5a6a !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

st.markdown(f"""
<div class="bb-topbar">
    <div class="bb-topbar-left">AUTOMATED TRADING SYSTEM</div>
    <div class="bb-topbar-right">SESSION: ACTIVE &nbsp;&nbsp;|&nbsp;&nbsp; {now} &nbsp;&nbsp;|&nbsp;&nbsp; v1.0</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TITLE
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bb-title-block">
    <div class="bb-title">DAILY TRADING SYSTEM</div>
    <div class="bb-subtitle">ML-POWERED EQUITY SIGNAL GENERATOR  ·  5 US LARGE-CAP EQUITIES  ·  LOGISTIC REGRESSION MODEL</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SYSTEM OVERVIEW
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bb-panel">
    <div class="bb-panel-header">System Overview</div>
    <div class="bb-panel-body">
        Machine learning system that analyzes daily equity price data and generates next-day
        directional signals (BUY/SELL). The pipeline extracts real-time data from SimFin,
        computes technical indicators (SMA, EMA, RSI, volatility), and feeds them into a
        trained Logistic Regression classifier. Includes historical backtesting with
        Buy-and-Sell strategy simulation.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3, gap="small")

with col1:
    st.markdown("""
    <div class="bb-step">
        <div class="bb-step-num">01</div>
        <div class="bb-step-title">Data Ingestion</div>
        <div class="bb-step-desc">
            Real-time price data via SimFin API.<br>
            OHLCV bars · 90-day lookback window.<br>
            Rate-limited: 2 req/sec (free tier).
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="bb-step">
        <div class="bb-step-num">02</div>
        <div class="bb-step-title">Feature Engine</div>
        <div class="bb-step-desc">
            8 technical indicators computed:<br>
            SMA(5), SMA(20), EMA(12), RSI(14),<br>
            Returns, Vol(20), VolChg, H-L Range.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="bb-step">
        <div class="bb-step-num">03</div>
        <div class="bb-step-title">Signal Output</div>
        <div class="bb-step-desc">
            Logistic Regression classifier.<br>
            Binary output: UP(1) / DOWN(0).<br>
            Confidence score via predict_proba.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UNIVERSE
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bb-panel-header" style="margin-bottom:0.8rem;">Coverage Universe</div>
<div class="bb-ticker-row">
    <div class="bb-ticker-cell">
        <div class="bb-ticker-symbol">AAPL</div>
        <div class="bb-ticker-name">APPLE INC</div>
    </div>
    <div class="bb-ticker-cell">
        <div class="bb-ticker-symbol">MSFT</div>
        <div class="bb-ticker-name">MICROSOFT CORP</div>
    </div>
    <div class="bb-ticker-cell">
        <div class="bb-ticker-symbol">GOOG</div>
        <div class="bb-ticker-name">ALPHABET INC</div>
    </div>
    <div class="bb-ticker-cell">
        <div class="bb-ticker-symbol">AMZN</div>
        <div class="bb-ticker-name">AMAZON.COM INC</div>
    </div>
    <div class="bb-ticker-cell">
        <div class="bb-ticker-symbol">TSLA</div>
        <div class="bb-ticker-name">TESLA INC</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TECH STACK
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bb-panel-header" style="margin-bottom:0.8rem;">Technology Stack</div>
<div>
    <span class="bb-tag">PYTHON 3.11</span>
    <span class="bb-tag">PANDAS</span>
    <span class="bb-tag">SCIKIT-LEARN</span>
    <span class="bb-tag">STREAMLIT</span>
    <span class="bb-tag">SIMFIN API</span>
    <span class="bb-tag">NUMPY</span>
    <span class="bb-tag">JOBLIB</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# TEAM
# ---------------------------------------------------------------------------
st.markdown("""
<div class="bb-panel">
    <div class="bb-panel-header">Development Team</div>
    <div class="bb-panel-body">
        University Group Project — Python for Data Analysis<br>
        Team Members: [Add your names here]
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# STATUS BAR
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="bb-status">
    SYS <span class="active">●</span> OPERATIONAL &nbsp;&nbsp;|&nbsp;&nbsp;
    API <span class="active">●</span> CONNECTED &nbsp;&nbsp;|&nbsp;&nbsp;
    MODEL <span class="active">●</span> LOADED &nbsp;&nbsp;|&nbsp;&nbsp;
    {now} &nbsp;&nbsp;|&nbsp;&nbsp;
    ⚠ EDUCATIONAL USE ONLY — NOT FINANCIAL ADVICE
</div>
""", unsafe_allow_html=True)
