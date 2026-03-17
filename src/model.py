"""
Machine Learning Model for Stock Price Prediction
===================================================
This script handles:
- Loading processed data (output of the ETL pipeline)
- Training a Logistic Regression model to predict if price goes UP or DOWN
- Evaluating the model's performance
- Exporting the trained model as a .pkl file for use in the web app

Usage:
    python src/model.py --ticker AAPL
    python src/model.py --all
"""

import pandas as pd
import numpy as np
import joblib
import os
import argparse
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

# Feature columns must match the features created in the ETL pipeline.
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

TARGET_COLUMN = "Target"  # 1 = price goes up, 0 = price goes down


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_processed_data(ticker: str) -> pd.DataFrame:
    """Load the processed CSV file created by the ETL pipeline."""
    filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Processed data not found at {filepath}. Run the ETL first: python src/etl.py --ticker {ticker}"
        )
    df = pd.read_csv(filepath)
    print(f"[DATA] Loaded {len(df)} rows for {ticker}")
    return df


# ---------------------------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------------------------
def prepare_data(df: pd.DataFrame):
    """
    Split into features (X) and target (y), then into train/test sets.
    80% training, 20% testing. Features are scaled with StandardScaler.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Replace infinities with NaN, then drop
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        raise ValueError(f"Not enough data to train: only {len(X)} rows available (need at least 10)")

    # Split: train on past, test on future (shuffle=False for time series)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Scale features (standardize to mean=0, std=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[DATA] Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    print(f"[DATA] Target balance (train): {dict(y_train.value_counts())}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train, y_train):
    """Train a Logistic Regression model for binary classification (UP/DOWN)."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    print("[MODEL] Logistic Regression trained successfully")
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model: accuracy, classification report, confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n[EVALUATION] Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n[EVALUATION] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["DOWN (0)", "UP (1)"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"[EVALUATION] Confusion Matrix:")
    print(f"                 Predicted DOWN  Predicted UP")
    print(f"  Actual DOWN    {cm[0][0]:>13}  {cm[0][1]:>12}")
    print(f"  Actual UP      {cm[1][0]:>13}  {cm[1][1]:>12}")

    return accuracy


# ---------------------------------------------------------------------------
# MODEL EXPORT
# ---------------------------------------------------------------------------
def export_model(model, scaler, ticker: str):
    """Export the trained model and scaler as .pkl files."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, f"{ticker}_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, f"{ticker}_scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[EXPORT] Model saved to {model_path}")
    print(f"[EXPORT] Scaler saved to {scaler_path}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_model_pipeline(ticker: str):
    """Full ML pipeline: load data → prepare → train → evaluate → export."""
    print(f"\n{'='*60}")
    print(f"  Training ML Model for: {ticker}")
    print(f"{'='*60}")

    df = load_processed_data(ticker)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    export_model(model, scaler, ticker)

    print(f"\n[DONE] Model pipeline complete for {ticker}\n")


# ---------------------------------------------------------------------------
# SCRIPT ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model for stock prediction")
    parser.add_argument("--ticker", type=str, help="Ticker symbol (e.g., AAPL)")
    parser.add_argument("--all", action="store_true", help="Train models for all tickers")
    args = parser.parse_args()

    if args.all:
        for t in TICKERS:
            try:
                run_model_pipeline(t)
            except (FileNotFoundError, ValueError) as e:
                print(f"[SKIP] {e}")
    elif args.ticker:
        run_model_pipeline(args.ticker.upper())
    else:
        print("Usage: python src/model.py --ticker AAPL")
        print("       python src/model.py --all")
