"""
Home Page - Automated Daily Trading System
============================================
Main entry point for the Streamlit web application.

To run locally:
    streamlit run app/Home.py
"""

import streamlit as st

st.set_page_config(page_title="Trading System", page_icon="📈", layout="wide")

st.title("📈 Automated Daily Trading System")
st.markdown("---")

st.header("Welcome")
st.write(
    """
    This is an automated daily trading system that uses **machine learning** to predict 
    stock market movements. The system analyzes historical price data, identifies patterns, 
    and generates trading signals to help you make informed decisions.
    """
)

st.header("How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1️⃣ Data Collection")
    st.write(
        """
        We collect historical and real-time stock price data from **SimFin**, 
        a financial data platform. This includes daily open, high, low, close 
        prices and trading volume.
        """
    )

with col2:
    st.subheader("2️⃣ ML Prediction")
    st.write(
        """
        Our machine learning model analyzes technical indicators like 
        moving averages, RSI, and volatility to predict whether the stock 
        price will go **UP** or **DOWN** the next trading day.
        """
    )

with col3:
    st.subheader("3️⃣ Trading Signal")
    st.write(
        """
        Based on the model's prediction, the system generates a clear 
        trading signal: **BUY** (price expected to rise) or **SELL** 
        (price expected to fall).
        """
    )

st.markdown("---")

st.header("Available Companies")
st.write("The system currently supports the following US companies:")

companies = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOG": "Alphabet (Google)",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
}

cols = st.columns(5)
for i, (ticker, name) in enumerate(companies.items()):
    with cols[i]:
        st.metric(label=ticker, value=name)

st.markdown("---")

st.header("Development Team")
st.write(
    """
    This project was developed as part of a university group assignment 
    focused on building an end-to-end trading system using Python.
    """
)
st.info("👤 **Team Members:** [Add your names here]")

st.markdown("---")

st.header("Technology Stack")
st.write(
    """
    - **Python** — Core programming language
    - **Pandas** — Data processing and analysis
    - **Scikit-learn** — Machine learning model (Logistic Regression)
    - **Streamlit** — Web application framework
    - **SimFin API** — Financial data source
    """
)

st.markdown("---")
st.caption("⚠️ Disclaimer: This is an educational project. Predictions are not financial advice.")
