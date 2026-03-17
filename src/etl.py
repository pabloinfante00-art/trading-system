"""
ETL Pipeline for Stock Market Data
===================================
This script handles:
- Extract: Reading raw CSV files downloaded from SimFin bulk download
- Transform: Cleaning data, creating features for the ML model
- Load: Saving the processed data as clean CSVs ready for modeling

Usage:
    python src/etl.py --ticker AAPL
    python src/etl.py --ticker MSFT
    python src/etl.py --all
"""

import pandas as pd
import numpy as np
import argparse
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# These are the 5 US companies we will work with in this project.
# Note: Google uses "GOOG" (not "GOOGL") in SimFin.
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

# Paths (relative to project root)
RAW_PRICES_PATH = "data/us-shareprices-daily.csv"
RAW_COMPANIES_PATH = "data/us-companies.csv"
PROCESSED_DATA_DIR = "data/processed"


# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------
def extract_share_prices(filepath: str) -> pd.DataFrame:
    """
    Read the raw share prices CSV downloaded from SimFin bulk download.
    SimFin CSVs use semicolon (;) as delimiter.
    """
    df = pd.read_csv(filepath, sep=";")
    print(f"[EXTRACT] Loaded {len(df)} rows from {filepath}")
    return df


def extract_companies(filepath: str) -> pd.DataFrame:
    """
    Read the raw companies CSV downloaded from SimFin bulk download.
    """
    df = pd.read_csv(filepath, sep=";")
    print(f"[EXTRACT] Loaded {len(df)} companies from {filepath}")
    return df


# ---------------------------------------------------------------------------
# TRANSFORM
# ---------------------------------------------------------------------------
def filter_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Filter the share prices dataframe to keep only rows for a specific ticker.
    """
    filtered = df[df["Ticker"] == ticker].copy()
    print(f"[TRANSFORM] Filtered {len(filtered)} rows for ticker '{ticker}'")
    return filtered


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw share prices data:
    - Parse dates, sort by date
    - Drop rows with missing Close price
    - Fill missing Volume/High/Low values
    - Remove duplicates
    """
    df = df.copy()

    # Convert Date column to datetime type
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically (oldest first)
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop rows where the Close price is missing
    rows_before = len(df)
    df = df.dropna(subset=["Close"])
    rows_after = len(df)
    if rows_before != rows_after:
        print(f"[TRANSFORM] Dropped {rows_before - rows_after} rows with missing Close price")

    # Fill missing Volume with 0 (SimFin free tier often has missing Volume)
    if "Volume" in df.columns:
        missing_vol = df["Volume"].isna().sum()
        if missing_vol > 0:
            print(f"[TRANSFORM] Filling {missing_vol} missing Volume values with 0")
            df["Volume"] = df["Volume"].fillna(0)

    # Fill missing High/Low with Close price
    if "High" in df.columns:
        df["High"] = df["High"].fillna(df["Close"])
    if "Low" in df.columns:
        df["Low"] = df["Low"].fillna(df["Close"])

    # Remove duplicate dates (keep the last one)
    df = df.drop_duplicates(subset=["Date"], keep="last")

    df = df.reset_index(drop=True)
    print(f"[TRANSFORM] Cleaned data: {len(df)} rows remaining")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features (input variables) for the ML model.
    Technical indicators: Returns, SMA, EMA, RSI, Volatility, Volume Change, Range.
    """
    df = df.copy()

    # Daily Returns (percentage change in Close price)
    df["Returns"] = df["Close"].pct_change()

    # Simple Moving Averages
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()

    # Exponential Moving Average
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Volatility (rolling standard deviation of returns)
    df["Volatility_20"] = df["Returns"].rolling(window=20).std()

    # Volume Change — handle safely (Volume may be 0 in free dataset)
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["Volume_Change"] = df["Volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    else:
        df["Volume_Change"] = 0.0

    # High-Low Range
    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Range"] = df["High"] - df["Low"]
    else:
        df["High_Low_Range"] = 0.0

    print(f"[TRANSFORM] Created 8 features")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target variable:
    Target = 1 if tomorrow's Close > today's Close (price goes UP)
    Target = 0 if tomorrow's Close <= today's Close (price goes DOWN)
    """
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    print(f"[TRANSFORM] Target distribution: {df['Target'].value_counts().to_dict()}")
    return df


def drop_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN in ESSENTIAL columns only (not all columns).
    This is important because columns like Dividend may have NaN and that's fine.
    """
    essential_columns = [
        "Returns", "SMA_5", "SMA_20", "EMA_12", "RSI_14",
        "Volatility_20", "Volume_Change", "High_Low_Range", "Target"
    ]
    available = [c for c in essential_columns if c in df.columns]

    rows_before = len(df)
    df = df.dropna(subset=available).reset_index(drop=True)
    rows_after = len(df)
    print(f"[TRANSFORM] Dropped {rows_before - rows_after} rows with NaN. {rows_after} rows remaining.")
    return df


# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
def save_processed_data(df: pd.DataFrame, ticker: str, output_dir: str) -> str:
    """Save the processed dataframe to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{ticker}_processed.csv")
    df.to_csv(output_path, index=False)
    print(f"[LOAD] Saved processed data to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# MAIN ETL PIPELINE
# ---------------------------------------------------------------------------
def run_etl(ticker: str) -> pd.DataFrame:
    """Run the full ETL pipeline for a single ticker."""
    print(f"\n{'='*60}")
    print(f"  Running ETL for: {ticker}")
    print(f"{'='*60}")

    # Step 1: Extract
    prices_df = extract_share_prices(RAW_PRICES_PATH)
    companies_df = extract_companies(RAW_COMPANIES_PATH)

    # Step 2: Filter for this ticker
    df = filter_ticker(prices_df, ticker)
    if len(df) == 0:
        print(f"[ERROR] No data found for ticker '{ticker}'.")
        return pd.DataFrame()

    # Company info
    company_info = companies_df[companies_df["Ticker"] == ticker]
    if len(company_info) > 0:
        company_name = company_info.iloc[0].get("Company Name", ticker)
        print(f"[INFO] Company: {company_name}")

    # Step 3: Clean
    df = clean_data(df)

    # Step 4: Create features
    df = create_features(df)

    # Step 5: Create target
    df = create_target(df)

    # Step 6: Drop missing rows
    df = drop_missing_rows(df)

    # Step 7: Save
    if len(df) > 0:
        save_processed_data(df, ticker, PROCESSED_DATA_DIR)
        print(f"[DONE] ETL complete for {ticker}. Shape: {df.shape}\n")
    else:
        print(f"[ERROR] No data remaining after processing for {ticker}.\n")

    return df


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL pipeline for stock data")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--all", action="store_true", help="Run ETL for all configured tickers")
    args = parser.parse_args()

    if args.all:
        for t in TICKERS:
            run_etl(t)
    elif args.ticker:
        run_etl(args.ticker.upper())
    else:
        print("Usage: python src/etl.py --ticker AAPL")
        print("       python src/etl.py --all")
