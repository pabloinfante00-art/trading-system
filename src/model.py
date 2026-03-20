"""
model.py — Automated Trading System ML Training Script
=======================================================
Loads processed CSVs from the ETL pipeline, trains one Logistic Regression
model per ticker, evaluates it, and exports model + scaler as .pkl files.

Usage
-----
Train all tickers (default):
    python src/model.py

Train a specific ticker:
    python src/model.py --ticker AAPL

Custom directories:
    python src/model.py --processed-dir data/processed --models-dir models
"""

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# src/ is already in path when running from project root, but be explicit
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from etl import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(df: pd.DataFrame) -> tuple:
    """
    Train a Logistic Regression model on a processed ticker DataFrame.

    Splits into train (80%) / test (20%) without shuffling — we must train
    on past data and test on future data, never the reverse.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame from the ETL pipeline. Must contain all
        FEATURE_COLUMNS and a 'Target' column.

    Returns
    -------
    tuple[LogisticRegression, StandardScaler, dict]
        Trained model, fitted scaler, and evaluation metrics dict.
    """
    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df['Target']

    # Drop any rows where a feature is still NaN after ETL
    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    # shuffle=False: time-series rule — train on past, test on future
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Fit scaler only on training data to avoid leakage into the test set
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # compensates for imbalance between UP/DOWN days
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    metrics = {
        'accuracy':          accuracy_score(y_test, y_pred),
        'confusion_matrix':  confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(
            y_test, y_pred, target_names=['DOWN', 'UP']
        ),
        'n_train': len(X_train),
        'n_test':  len(X_test),
    }

    return model, scaler, metrics


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def train_ticker(ticker: str, processed_dir: str, models_dir: str) -> dict:
    """
    Load the processed CSV for one ticker, train, evaluate, and export.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL').
    processed_dir : str
        Directory containing processed CSVs from the ETL pipeline.
    models_dir : str
        Directory where .pkl files will be saved.

    Returns
    -------
    dict
        Evaluation metrics for this ticker.
    """
    csv_path = os.path.join(processed_dir, f'{ticker}_processed.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Processed file not found: {csv_path}\n"
            f"Run 'python src/etl.py' first."
        )

    df = pd.read_csv(csv_path)
    model, scaler, metrics = train(df)

    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model,  os.path.join(models_dir, f'{ticker}_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, f'{ticker}_scaler.pkl'))

    return metrics


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train and export ML models for the trading system.'
    )
    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help=f'Ticker to train. If omitted, trains all: {TICKERS}',
    )
    parser.add_argument(
        '--processed-dir',
        type=str,
        default='data/processed',
        help='Directory with processed CSVs from ETL (default: data/processed/)',
    )
    parser.add_argument(
        '--models-dir',
        type=str,
        default='models',
        help='Directory to save .pkl files (default: models/)',
    )
    args = parser.parse_args()

    tickers_to_run = [args.ticker] if args.ticker else TICKERS

    print(f"Training models for: {tickers_to_run}\n")

    results = {}
    for ticker in tickers_to_run:
        print(f"  {ticker}...", end=' ')
        try:
            metrics = train_ticker(ticker, args.processed_dir, args.models_dir)
            results[ticker] = metrics
            print(f"accuracy = {metrics['accuracy']:.1%}  "
                  f"(train: {metrics['n_train']:,}, test: {metrics['n_test']:,})")
        except Exception as e:
            results[ticker] = None
            print(f"FAILED — {e}")

    # Print detailed report for each successful ticker
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    for ticker, metrics in results.items():
        if metrics is None:
            print(f"\n{ticker}: FAILED")
            continue
        print(f"\n{ticker}  (accuracy: {metrics['accuracy']:.1%})")
        print(metrics['classification_report'])
        cm = metrics['confusion_matrix']
        print(f"  Confusion matrix:")
        print(f"                Pred DOWN  Pred UP")
        print(f"  Actual DOWN   {cm[0][0]:>9}  {cm[0][1]:>7}")
        print(f"  Actual UP     {cm[1][0]:>9}  {cm[1][1]:>7}")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ticker, metrics in results.items():
        status = f"{metrics['accuracy']:.1%}" if metrics else "FAILED"
        model_path = os.path.join(args.models_dir, f'{ticker}_model.pkl')
        saved = "saved" if metrics and os.path.exists(model_path) else "not saved"
        print(f"  {ticker}: {status} — {saved} to {model_path}")


if __name__ == '__main__':
    main()
