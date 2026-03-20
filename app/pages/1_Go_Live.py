"""
Go Live Page - Bloomberg Terminal Style
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from pysimfin import PySimFin

st.set_page_config(page_title="Go Live", page_icon="🚀", layout="wide")

# ---------------------------------------------------------------------------
# BLOOMBERG CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0a0a0a !important;
        font-family: 'IBM Plex Sans', sans-serif;
    }
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

    .bb-panel {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
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

    /* Signal display */
    .bb-signal-box {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1.5rem;
        text-align: center;
    }
    .bb-signal-box.buy { border-top: 3px solid #00d4aa; }
    .bb-signal-box.sell { border-top: 3px solid #ff4444; }
    .bb-signal-box.neutral { border-top: 3px solid #ff6600; }

    .bb-signal-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #5a5a6a;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
    }
    .bb-signal-value {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 700;
        font-size: 1.6rem;
    }
    .bb-signal-value.up { color: #00d4aa; }
    .bb-signal-value.down { color: #ff4444; }
    .bb-signal-value.neutral { color: #ff6600; }

    .bb-signal-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #5a5a6a;
        margin-top: 0.3rem;
    }

    /* Data cells */
    .bb-data-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0;
    }
    .bb-data-cell {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .bb-data-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: #5a5a6a;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .bb-data-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.15rem;
        font-weight: 600;
        color: #00d4aa;
        margin-top: 0.2rem;
    }

    /* Probability bar */
    .bb-prob-container {
        background: #0f0f14;
        border: 1px solid #1a1a2e;
        padding: 1rem 1.5rem;
    }
    .bb-prob-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 1rem;
    }
    .bb-prob-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #5a5a6a;
        width: 60px;
        text-align: right;
    }
    .bb-prob-bar-bg {
        flex: 1;
        height: 20px;
        background: #1a1a2e;
        position: relative;
    }
    .bb-prob-bar {
        height: 100%;
        position: absolute;
        left: 0;
        top: 0;
    }
    .bb-prob-bar.up { background: #00d4aa; }
    .bb-prob-bar.down { background: #ff4444; }
    .bb-prob-pct {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        font-weight: 600;
        color: #c0c0d0;
        width: 55px;
    }

    .bb-divider { height: 1px; background: #1a1a2e; margin: 1.5rem 0; }

    .bb-status {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: #3a3a4a;
        text-align: center;
        padding: 0.8rem 0;
        border-top: 1px solid #1a1a2e;
    }
    .bb-status .active { color: #00d4aa; }

    /* Override Streamlit */
    .stMarkdown, .stText, p, span, li { color: #c0c0d0 !important; }
    h1, h2, h3 { color: #ffffff !important; }
    div[data-testid="stMetric"] { background: #0f0f14; border: 1px solid #1a1a2e; padding: 0.8rem; border-radius: 0; }
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }
    div[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; color: #5a5a6a !important; }
    .stSelectbox > div > div { background-color: #0f0f14 !important; border: 1px solid #1a1a2e !important; color: #c0c0d0 !important; font-family: 'IBM Plex Mono', monospace !important; }
    .stExpander { border: 1px solid #1a1a2e !important; background: #0f0f14 !important; }
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
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
FEATURE_COLUMNS = [
    "Returns", "SMA_5", "SMA_20", "EMA_12",
    "RSI_14", "Volatility_20", "Volume_Change", "High_Low_Range",
]
TICKER_NAMES = {
    "AAPL": "APPLE INC", "MSFT": "MICROSOFT CORP", "GOOG": "ALPHABET INC",
    "AMZN": "AMAZON.COM INC", "TSLA": "TESLA INC",
}

# ---------------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------------
@st.cache_resource
def get_simfin_client() -> PySimFin:
    api_key = st.secrets.get("SIMFIN_API_KEY", "")
    if not api_key:
        st.error("⚠️ SimFin API key not found.")
        st.stop()
    return PySimFin(api_key=api_key)

@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    client = get_simfin_client()
    try:
        return client.get_share_prices(ticker, start, end)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def apply_etl_transformations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {
        "Last Closing Price": "Close", "Highest Price": "High",
        "Lowest Price": "Low", "Opening Price": "Open",
        "Trading Volume": "Volume", "Adjusted Closing Price": "Adj. Close",
    }
    df = df.rename(columns=rename_map)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    if "Volume" in df.columns: df["Volume"] = df["Volume"].fillna(0)
    if "High" in df.columns: df["High"] = df["High"].fillna(df["Close"])
    if "Low" in df.columns: df["Low"] = df["Low"].fillna(df["Close"])

    df["Returns"] = df["Close"].pct_change()
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_Change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["Volume_Change"] = 0.0
    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Range"] = df["High"] - df["Low"]
    else:
        df["High_Low_Range"] = 0.0
    return df

def load_model_and_scaler(ticker: str):
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return joblib.load(model_path), joblib.load(scaler_path)

def make_prediction(model, scaler, features_row):
    X = features_row[FEATURE_COLUMNS].values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    return {
        "prediction": int(prediction),
        "label": "UP" if prediction == 1 else "DOWN",
        "confidence": float(max(probability)),
        "prob_up": float(probability[1]),
        "prob_down": float(probability[0]),
    }

# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------
now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="bb-topbar">
    <div class="bb-topbar-left">GO LIVE — SIGNAL GENERATOR</div>
    <div class="bb-topbar-right">REAL-TIME &nbsp;|&nbsp; {now}</div>
</div>
""", unsafe_allow_html=True)

# --- Ticker Selection ---
selected_ticker = st.selectbox(
    "TICKER", TICKERS,
    format_func=lambda t: f"{t}  ·  {TICKER_NAMES.get(t, t)}"
)

end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# --- Fetch ---
with st.spinner(f"FETCHING {selected_ticker} DATA FROM SIMFIN..."):
    raw_df = fetch_prices(selected_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if raw_df.empty:
    st.error("NO DATA RETURNED. CHECK TICKER OR TRY AGAIN.")
    st.stop()

transformed_df = apply_etl_transformations(raw_df)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# --- Price Chart ---
st.markdown(f'<div class="bb-panel-header">{selected_ticker} — PRICE (LAST 90 DAYS)</div>', unsafe_allow_html=True)

if "Date" in transformed_df.columns and "Close" in transformed_df.columns:
    chart_df = transformed_df.set_index("Date")[["Close"]].copy()
    st.line_chart(chart_df)

    # Show latest price info
    latest_close = transformed_df["Close"].iloc[-1]
    prev_close = transformed_df["Close"].iloc[-2] if len(transformed_df) > 1 else latest_close
    day_change = latest_close - prev_close
    day_change_pct = (day_change / prev_close) * 100 if prev_close != 0 else 0
    latest_high = transformed_df["High"].iloc[-1] if "High" in transformed_df.columns else 0
    latest_low = transformed_df["Low"].iloc[-1] if "Low" in transformed_df.columns else 0
    latest_vol = transformed_df["Volume"].iloc[-1] if "Volume" in transformed_df.columns else 0

    change_color = "#00d4aa" if day_change >= 0 else "#ff4444"
    sign = "+" if day_change >= 0 else ""

    st.markdown(f"""
    <div class="bb-data-grid">
        <div class="bb-data-cell">
            <div class="bb-data-label">LAST CLOSE</div>
            <div class="bb-data-value">${latest_close:,.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">DAY CHANGE</div>
            <div class="bb-data-value" style="color:{change_color}">{sign}{day_change:,.2f} ({sign}{day_change_pct:.2f}%)</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">DAY RANGE</div>
            <div class="bb-data-value" style="font-size:1rem;">${latest_low:,.2f} — ${latest_high:,.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">VOLUME</div>
            <div class="bb-data-value">{latest_vol:,.0f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("RAW PRICE DATA"):
    st.dataframe(raw_df.tail(10), use_container_width=True)

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# --- Technical Indicators ---
st.markdown('<div class="bb-panel-header">TECHNICAL INDICATORS</div>', unsafe_allow_html=True)

latest = transformed_df.dropna(subset=FEATURE_COLUMNS)
if not latest.empty:
    lr = latest.iloc[-1]
    rsi_color = "#ff4444" if lr["RSI_14"] > 70 else ("#00d4aa" if lr["RSI_14"] < 30 else "#ff6600")

    st.markdown(f"""
    <div class="bb-data-grid">
        <div class="bb-data-cell">
            <div class="bb-data-label">RSI (14)</div>
            <div class="bb-data-value" style="color:{rsi_color}">{lr['RSI_14']:.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">SMA (5)</div>
            <div class="bb-data-value">{lr['SMA_5']:.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">SMA (20)</div>
            <div class="bb-data-value">{lr['SMA_20']:.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">VOLATILITY (20D)</div>
            <div class="bb-data-value">{lr['Volatility_20']:.4f}</div>
        </div>
    </div>
    <div class="bb-data-grid" style="margin-top:0;">
        <div class="bb-data-cell">
            <div class="bb-data-label">EMA (12)</div>
            <div class="bb-data-value">{lr['EMA_12']:.2f}</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">DAILY RETURN</div>
            <div class="bb-data-value" style="color:{'#00d4aa' if lr['Returns']>=0 else '#ff4444'}">{lr['Returns']*100:.3f}%</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">VOL CHANGE</div>
            <div class="bb-data-value">{lr['Volume_Change']*100:.2f}%</div>
        </div>
        <div class="bb-data-cell">
            <div class="bb-data-label">H-L RANGE</div>
            <div class="bb-data-value">${lr['High_Low_Range']:.2f}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("INSUFFICIENT DATA FOR INDICATOR COMPUTATION.")
    st.stop()

st.markdown('<div class="bb-divider"></div>', unsafe_allow_html=True)

# --- ML Signal ---
st.markdown('<div class="bb-panel-header">MODEL SIGNAL</div>', unsafe_allow_html=True)

model, scaler = load_model_and_scaler(selected_ticker)

if model is None:
    st.error(f"NO MODEL FOUND FOR {selected_ticker}. TRAIN FIRST.")
else:
    latest_complete = transformed_df.dropna(subset=FEATURE_COLUMNS)
    if not latest_complete.empty:
        latest_row = latest_complete.iloc[-1]
        result = make_prediction(model, scaler, latest_row)

        is_up = result["prediction"] == 1
        direction = "buy" if is_up else "sell"
        val_class = "up" if is_up else "down"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="bb-signal-box {direction}">
                <div class="bb-signal-label">NEXT DAY FORECAST</div>
                <div class="bb-signal-value {val_class}">{"▲ UP" if is_up else "▼ DOWN"}</div>
                <div class="bb-signal-sub">Logistic Regression · Next Session</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="bb-signal-box neutral">
                <div class="bb-signal-label">CONFIDENCE</div>
                <div class="bb-signal-value neutral">{result['confidence']*100:.1f}%</div>
                <div class="bb-signal-sub">predict_proba max</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            signal = "BUY" if is_up else "SELL"
            st.markdown(f"""
            <div class="bb-signal-box {direction}">
                <div class="bb-signal-label">TRADE SIGNAL</div>
                <div class="bb-signal-value {val_class}">{signal}</div>
                <div class="bb-signal-sub">Based on directional forecast</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability bars
        st.write("")
        st.markdown(f"""
        <div class="bb-prob-container">
            <div class="bb-prob-row">
                <div class="bb-prob-label">UP</div>
                <div class="bb-prob-bar-bg"><div class="bb-prob-bar up" style="width:{result['prob_up']*100}%;"></div></div>
                <div class="bb-prob-pct">{result['prob_up']*100:.1f}%</div>
            </div>
            <div class="bb-prob-row">
                <div class="bb-prob-label">DOWN</div>
                <div class="bb-prob-bar-bg"><div class="bb-prob-bar down" style="width:{result['prob_down']*100}%;"></div></div>
                <div class="bb-prob-pct">{result['prob_down']*100:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        data_date = latest_row.get("Date", "N/A")
        st.markdown(f"""
        <div class="bb-signal-sub" style="text-align:center; margin-top:0.8rem;">
            DATA AS OF: {data_date} &nbsp;&nbsp;|&nbsp;&nbsp; MODEL: LOGISTIC REGRESSION &nbsp;&nbsp;|&nbsp;&nbsp; SCALER: STANDARD
        </div>
        """, unsafe_allow_html=True)

# Status bar
st.markdown(f"""
<div class="bb-status">
    FEED <span class="active">●</span> SIMFIN API &nbsp;&nbsp;|&nbsp;&nbsp;
    TICKER: {selected_ticker} &nbsp;&nbsp;|&nbsp;&nbsp;
    {now} &nbsp;&nbsp;|&nbsp;&nbsp;
    ⚠ EDUCATIONAL USE ONLY
</div>
""", unsafe_allow_html=True)
