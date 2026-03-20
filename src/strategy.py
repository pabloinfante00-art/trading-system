"""
Trading Strategy Module
========================
This module implements simple trading strategies based on ML model predictions.

Strategies:
- Buy-and-Sell: Buy when model predicts UP, sell when model predicts DOWN
- Buy-and-Hold: Buy when model predicts UP, hold when model predicts DOWN, sell at profit target

Usage:
    from strategy import backtest_strategy
    results = backtest_strategy(df, predictions)
"""

import pandas as pd
import numpy as np


def buy_and_sell_strategy(predictions: pd.Series, close_prices: pd.Series, 
                          initial_cash: float = 10000.0) -> pd.DataFrame:
    """
    Buy-and-Sell Strategy:
    - If prediction = 1 (UP): Buy 1 share at today's close price
    - If prediction = 0 (DOWN): Sell 1 share at today's close price
    - Track portfolio value (cash + shares * current price)

    Parameters
    ----------
    predictions : pd.Series
        Model predictions (1 = UP, 0 = DOWN).
    close_prices : pd.Series
        Closing prices aligned with predictions.
    initial_cash : float
        Starting cash amount (default $10,000).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Action, Shares_Held, Cash, Portfolio_Value
    """
    cash = initial_cash
    shares = 0
    records = []

    for i in range(len(predictions)):
        price = close_prices.iloc[i]
        pred = predictions.iloc[i]

        if pred == 1 and cash >= price:
            # BUY: invest 50% of available cash
            invest_amount = cash * 0.5
            shares_to_buy = int(invest_amount / price)
            if shares_to_buy < 1:
                shares_to_buy = 1
            if cash >= shares_to_buy * price:
                shares += shares_to_buy
                cash -= shares_to_buy * price
                action = f"BUY x{shares_to_buy}"
            else:
                action = "HOLD"
        elif pred == 0 and shares > 0:
            # SELL: sell 50% of held shares (minimum 1)
            shares_to_sell = max(1, shares // 2)
            shares -= shares_to_sell
            cash += shares_to_sell * price
            action = f"SELL x{shares_to_sell}"
        else:
            # HOLD: can't buy (no cash) or can't sell (no shares)
            action = "HOLD"

        portfolio_value = cash + (shares * price)
        records.append({
            "Action": action,
            "Price": price,
            "Shares_Held": shares,
            "Cash": round(cash, 2),
            "Portfolio_Value": round(portfolio_value, 2),
        })

    return pd.DataFrame(records)


def buy_and_hold_baseline(close_prices: pd.Series, 
                          initial_cash: float = 10000.0) -> pd.DataFrame:
    """
    Buy-and-Hold Baseline Strategy:
    - Buy as many shares as possible on day 1
    - Hold until the end
    - This is the "do nothing" benchmark to compare against

    Parameters
    ----------
    close_prices : pd.Series
        Closing prices.
    initial_cash : float
        Starting cash amount.

    Returns
    -------
    pd.DataFrame
        DataFrame with Portfolio_Value column.
    """
    first_price = close_prices.iloc[0]
    shares = int(initial_cash / first_price)
    remaining_cash = initial_cash - (shares * first_price)

    records = []
    for i in range(len(close_prices)):
        price = close_prices.iloc[i]
        portfolio_value = remaining_cash + (shares * price)
        records.append({
            "Portfolio_Value": round(portfolio_value, 2),
        })

    return pd.DataFrame(records)


def calculate_performance_metrics(strategy_results: pd.DataFrame, 
                                  initial_cash: float = 10000.0) -> dict:
    """
    Calculate performance metrics for a trading strategy.

    Parameters
    ----------
    strategy_results : pd.DataFrame
        Output from buy_and_sell_strategy().
    initial_cash : float
        Starting cash amount.

    Returns
    -------
    dict
        Dictionary with performance metrics.
    """
    final_value = strategy_results["Portfolio_Value"].iloc[-1]
    total_return = ((final_value - initial_cash) / initial_cash) * 100
    
    # Count trades
    total_buys = strategy_results["Action"].str.startswith("BUY").sum()
    total_sells = strategy_results["Action"].str.startswith("SELL").sum()
    total_holds = (strategy_results["Action"] == "HOLD").sum()

    # Max portfolio value and max drawdown
    max_value = strategy_results["Portfolio_Value"].max()
    min_value = strategy_results["Portfolio_Value"].min()

    # Calculate daily returns of portfolio
    portfolio_returns = strategy_results["Portfolio_Value"].pct_change().dropna()
    
    return {
        "initial_cash": initial_cash,
        "final_value": round(final_value, 2),
        "total_return_pct": round(total_return, 2),
        "total_buys": int(total_buys),
        "total_sells": int(total_sells),
        "total_holds": int(total_holds),
        "max_portfolio_value": round(max_value, 2),
        "min_portfolio_value": round(min_value, 2),
        "avg_daily_return_pct": round(portfolio_returns.mean() * 100, 4) if len(portfolio_returns) > 0 else 0,
    }


def backtest_strategy(df: pd.DataFrame, model, scaler, 
                      feature_columns: list, initial_cash: float = 10000.0) -> dict:
    """
    Run a full backtest: generate predictions on historical data and simulate trading.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataframe with features (output of ETL).
    model : LogisticRegression
        Trained ML model.
    scaler : StandardScaler
        Fitted scaler.
    feature_columns : list
        List of feature column names.
    initial_cash : float
        Starting cash.

    Returns
    -------
    dict
        Dictionary containing strategy results, baseline results, and metrics.
    """
    # Get features and scale them
    X = df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X)

    # Generate predictions for all rows
    predictions = pd.Series(model.predict(X_scaled), index=df.index)
    close_prices = df["Close"].reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)

    # Run strategy
    strategy_results = buy_and_sell_strategy(predictions, close_prices, initial_cash)
    
    # Run baseline (buy-and-hold)
    baseline_results = buy_and_hold_baseline(close_prices, initial_cash)

    # Calculate metrics
    strategy_metrics = calculate_performance_metrics(strategy_results, initial_cash)
    baseline_final = baseline_results["Portfolio_Value"].iloc[-1]
    baseline_return = ((baseline_final - initial_cash) / initial_cash) * 100

    return {
        "strategy_results": strategy_results,
        "baseline_results": baseline_results,
        "strategy_metrics": strategy_metrics,
        "baseline_return_pct": round(baseline_return, 2),
        "baseline_final_value": round(baseline_final, 2),
        "predictions": predictions,
        "dates": df["Date"].reset_index(drop=True) if "Date" in df.columns else None,
    }
