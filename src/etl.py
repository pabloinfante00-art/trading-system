"""
etl.py — Automated Trading System ETL Pipeline
================================================
Loads raw SimFin bulk CSV files, applies all transformations, and saves
one processed CSV per ticker to data/processed/.

Usage
-----
Run for all tickers (default):
    python src/etl.py

Run for a specific ticker:
    python src/etl.py --ticker AAPL

Assumptions
-----------
- Raw data files live in data/ relative to the project root.
- SimFin bulk CSV files use semicolons (;) as delimiters.
- Required files: us-shareprices-daily.csv, us-companies.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']

FEATURE_COLUMNS = [
    'Returns',
    'SMA_5',
    'SMA_20',
    'EMA_12',
    'RSI_14',
    'Volatility_20',
    'Volume_Change',
    'High_Low_Range',
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_raw_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the raw SimFin bulk CSV files.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the SimFin bulk downloads.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (prices_df, companies_df)
    """
    prices_path    = os.path.join(data_dir, 'us-shareprices-daily.csv')
    companies_path = os.path.join(data_dir, 'us-companies.csv')

    if not os.path.exists(prices_path):
        raise FileNotFoundError(f"Share prices file not found: {prices_path}")
    if not os.path.exists(companies_path):
        raise FileNotFoundError(f"Companies file not found: {companies_path}")

    prices_df    = pd.read_csv(prices_path, sep=';')
    companies_df = pd.read_csv(companies_path, sep=';')

    print(f"Loaded {len(prices_df):,} price rows and {len(companies_df):,} companies.")
    return prices_df, companies_df


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

def transform(df_raw: pd.DataFrame, ticker: str, include_target: bool = True) -> pd.DataFrame:
    """
    Apply all ETL transformations to raw SimFin share price data for one ticker.

    This is the single source of truth for feature engineering. It is used
    both here (offline bulk processing) and in the Streamlit web app (live data),
    so the model always receives identically-structured input.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw share price DataFrame from SimFin bulk CSV or API.
        Expected columns: Ticker, Date, Open, High, Low, Close, Volume.
    ticker : str
        Stock ticker symbol to process (e.g. 'AAPL').
    include_target : bool
        If True, adds a Target column (next-day direction: 1=UP, 0=DOWN).
        Set to False when transforming live data where tomorrow is unknown.

    Returns
    -------
    pd.DataFrame
        Cleaned and feature-engineered DataFrame ready for ML model input.

    Raises
    ------
    ValueError
        If no rows are found for the given ticker.
    """
    df = df_raw[df_raw['Ticker'] == ticker].copy()

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'")

    # --- CLEANING ---

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Rows with no Close price cannot be used at all
    df = df.dropna(subset=['Close'])

    # SimFin free tier omits Volume for some dates; fill with 0
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)

    # Use Close as a fallback when High/Low are missing
    if 'High' in df.columns:
        df['High'] = df['High'].fillna(df['Close'])
    if 'Low' in df.columns:
        df['Low'] = df['Low'].fillna(df['Close'])

    # Keep only the last entry when the same date appears more than once
    df = df.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)

    # --- FEATURE ENGINEERING ---

    # Daily percentage return
    df['Returns'] = df['Close'].pct_change()

    # Short and medium-term trend signals
    df['SMA_5']  = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # EMA weights recent prices more heavily than SMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()

    # RSI — momentum indicator (overbought > 70, oversold < 30)
    delta    = df['Close'].diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # Rolling volatility of daily returns over 20 days
    df['Volatility_20'] = df['Returns'].rolling(window=20).std()

    # Volume change detects unusual trading activity
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['Volume_Change'] = (
            df['Volume']
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )
    else:
        df['Volume_Change'] = 0.0

    # Intraday price range
    df['High_Low_Range'] = df['High'] - df['Low']

    # --- TARGET VARIABLE ---
    if include_target:
        # 1 = next day closes higher (UP), 0 = next day closes lower or flat (DOWN)
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        essential = FEATURE_COLUMNS + ['Target']
    else:
        essential = FEATURE_COLUMNS

    # --- FINAL CLEANUP ---

    # Remove rows where any essential column is still NaN
    df = df.dropna(subset=essential).reset_index(drop=True)

    # Replace any remaining infinities (can appear in pct_change on zero values)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# ETL Runner
# ---------------------------------------------------------------------------

def run_etl(
    ticker: str,
    data_dir: str,
    output_dir: str,
    prices_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Run the full ETL pipeline for a single ticker and save the output CSV.

    Parameters
    ----------
    ticker : str
        Stock ticker to process.
    data_dir : str
        Directory containing raw SimFin bulk CSVs (used if prices_df is None).
    output_dir : str
        Directory where the processed CSV will be saved.
    prices_df : pd.DataFrame, optional
        Pre-loaded prices DataFrame. Pass this when processing multiple tickers
        to avoid reloading the file each time.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame for this ticker.
    """
    if prices_df is None:
        prices_df, _ = load_raw_data(data_dir)

    df_processed = transform(prices_df, ticker, include_target=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{ticker}_processed.csv')
    df_processed.to_csv(output_path, index=False)

    print(f"  {ticker}: {len(df_processed):,} rows -> {output_path}")
    return df_processed


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run the SimFin ETL pipeline for one or all tickers.'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help=f'Ticker to process. If omitted, runs for all: {TICKERS}',
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing raw SimFin bulk CSVs (default: data/)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Directory to save processed CSVs (default: data/processed/)',
    )
    args = parser.parse_args()

    tickers_to_run = [args.ticker] if args.ticker else TICKERS

    print(f"Loading raw data from '{args.data_dir}'...")
    try:
        prices_df, _ = load_raw_data(args.data_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\nRunning ETL for: {tickers_to_run}\n")
    results = {}
    for ticker in tickers_to_run:
        try:
            df = run_etl(ticker, args.data_dir, args.output_dir, prices_df=prices_df)
            results[ticker] = f"OK ({len(df):,} rows)"
        except Exception as e:
            results[ticker] = f"FAILED: {e}"
            print(f"  {ticker}: FAILED — {e}")

    print("\n--- ETL Summary ---")
    for ticker, status in results.items():
        print(f"  {ticker}: {status}")


if __name__ == '__main__':
    main()
