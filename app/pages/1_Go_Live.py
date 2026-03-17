"""
Go Live Page - Real-Time Stock Predictions
============================================
This page allows users to:
- Select a stock ticker
- View real-time and historical price data (fetched from SimFin API)
- See the ML model's prediction for the next trading day
- View technical indicators used by the model
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the src directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pysimfin import PySimFin

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Go Live", page_icon="🚀", layout="wide")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")

FEATURE_COLUMNS = [
    "Returns",
    "SMA_5",
    "SMA_20",
    "EMA_12",
    "RSI_14",
    "Volatility_20",
    "Volume_Change",
    "High_Low_Range",
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
@st.cache_resource
def get_simfin_client() -> PySimFin:
    """Create a SimFin API client using the API key from Streamlit secrets."""
    api_key = st.secrets.get("SIMFIN_API_KEY", "")
    if not api_key:
        st.error(
            "⚠️ SimFin API key not found. Please set it in `.streamlit/secrets.toml` "
            "or in Streamlit Cloud secrets."
        )
        st.stop()
    return PySimFin(api_key=api_key)


@st.cache_data(ttl=3600)
def fetch_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch share prices from SimFin API. Cached for 1 hour."""
    client = get_simfin_client()
    try:
        df = client.get_share_prices(ticker, start, end)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def apply_etl_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the SAME transformations used in src/etl.py to the live data.
    This ensures the model receives data in the same format it was trained on.
    """
    df = df.copy()
# Rename API columns to match bulk download names
    rename_map = {
        "Last Closing Price": "Close",
        "Highest Price": "High",
        "Lowest Price": "Low",
        "Opening Price": "Open",
        "Trading Volume": "Volume",
        "Adjusted Closing Price": "Adj. Close",
    }
    df = df.rename(columns=rename_map)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # Fill missing Volume with 0 (same as ETL)
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)
    if "High" in df.columns:
        df["High"] = df["High"].fillna(df["Close"])
    if "Low" in df.columns:
        df["Low"] = df["Low"].fillna(df["Close"])

    # Daily Returns
    df["Returns"] = df["Close"].pct_change()

    # Simple Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Volatility
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()

    # Volume Change
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_Change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["Volume_Change"] = 0.0

    # High-Low Range
    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Range"] = df["High"] - df["Low"]
    else:
        df["High_Low_Range"] = 0.0

    return df


def load_model_and_scaler(ticker: str):
    """Load the trained ML model and scaler for a specific ticker."""
    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def make_prediction(model, scaler, features_row: pd.DataFrame) -> dict:
    """Use the trained model to predict UP or DOWN for the next day."""
    X = features_row[FEATURE_COLUMNS].values.reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]

    return {
        "prediction": int(prediction),
        "label": "UP 📈" if prediction == 1 else "DOWN 📉",
        "confidence": float(max(probability)),
        "prob_up": float(probability[1]),
        "prob_down": float(probability[0]),
    }


# ---------------------------------------------------------------------------
# PAGE CONTENT
# ---------------------------------------------------------------------------
st.title("🚀 Go Live — Real-Time Predictions")
st.markdown("---")

# Ticker Selection
col_select, col_info = st.columns([1, 3])

with col_select:
    selected_ticker = st.selectbox("Select a stock ticker:", TICKERS)

with col_info:
    st.info(
        f"Showing data and prediction for **{selected_ticker}**. "
        "The model predicts whether tomorrow's closing price will be higher or lower."
    )

# Date Range
from datetime import datetime, timedelta

end_date = datetime.today()
start_date = end_date - timedelta(days=90)

st.markdown("---")

# Fetch Data
with st.spinner(f"Fetching data for {selected_ticker} from SimFin..."):
    raw_df = fetch_prices(
        selected_ticker,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

if raw_df.empty:
    st.warning("No data returned from SimFin. Please check the ticker or try again later.")
    st.stop()

# Display Price Chart
st.header(f"📊 {selected_ticker} — Stock Price Data")

if "Date" in raw_df.columns and "Close" in raw_df.columns:
    chart_df = raw_df.set_index("Date")[["Close"]].copy()
    st.line_chart(chart_df)

with st.expander("View raw price data (last 10 rows)"):
    st.dataframe(raw_df.tail(10), use_container_width=True)

st.markdown("---")

# Apply ETL Transformations
st.header("🔧 Technical Indicators")

transformed_df = apply_etl_transformations(raw_df)

if not transformed_df.empty:
    latest = transformed_df.dropna(subset=FEATURE_COLUMNS)

    if not latest.empty:
        latest_row = latest.iloc[-1]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RSI (14)", f"{latest_row['RSI_14']:.2f}")
        with col2:
            st.metric("SMA 5", f"{latest_row['SMA_5']:.2f}")
        with col3:
            st.metric("SMA 20", f"{latest_row['SMA_20']:.2f}")
        with col4:
            st.metric("Volatility", f"{latest_row['Volatility_20']:.4f}")

        with st.expander("View all technical indicators (last 10 rows)"):
            display_cols = ["Date"] + FEATURE_COLUMNS
            available_cols = [c for c in display_cols if c in transformed_df.columns]
            st.dataframe(
                transformed_df[available_cols].tail(10).round(4),
                use_container_width=True,
            )
    else:
        st.warning("Not enough data to compute technical indicators (need at least 20 trading days).")
        st.stop()

st.markdown("---")

# ML Prediction
st.header("🤖 Model Prediction")

model, scaler = load_model_and_scaler(selected_ticker)

if model is None:
    st.error(
        f"No trained model found for {selected_ticker}. "
        f"Train the model first: `python src/model.py --ticker {selected_ticker}`"
    )
else:
    latest_complete = transformed_df.dropna(subset=FEATURE_COLUMNS)

    if not latest_complete.empty:
        latest_row = latest_complete.iloc[-1]
        result = make_prediction(model, scaler, latest_row)

        pred_col1, pred_col2, pred_col3 = st.columns(3)

        with pred_col1:
            st.metric(label="Prediction for Next Day", value=result["label"])

        with pred_col2:
            st.metric(label="Confidence", value=f"{result['confidence']*100:.1f}%")

        with pred_col3:
            signal = "🟢 BUY" if result["prediction"] == 1 else "🔴 SELL"
            st.metric(label="Trading Signal", value=signal)

        st.write("**Probability breakdown:**")
        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.progress(result["prob_up"], text=f"Probability UP: {result['prob_up']*100:.1f}%")
        with prob_col2:
            st.progress(result["prob_down"], text=f"Probability DOWN: {result['prob_down']*100:.1f}%")

        st.caption(
            f"Based on data up to: {latest_row.get('Date', 'N/A')} | "
            f"Model: Logistic Regression"
        )
    else:
        st.warning("Not enough transformed data to make a prediction.")

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This is an educational project. "
    "Predictions should NOT be used for real trading decisions."
)
