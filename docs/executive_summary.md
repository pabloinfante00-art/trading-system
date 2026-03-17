# Executive Summary — Automated Daily Trading System

## Introduction

This project implements an automated daily trading system that predicts stock market movements using machine learning. The system covers five major US companies (Apple, Microsoft, Google, Amazon, and Tesla) and is accessible through a web application deployed on Streamlit Cloud.

## Data Sources

All financial data is sourced from SimFin (simfin.com), a platform providing free access to historical and real-time stock market data. We used two main datasets:

- **Share Prices (Bulk Download):** Five years of daily prices including open, high, low, close, and volume for all US-listed companies. This was used to train the machine learning model.
- **SimFin Data API:** Used in the web application to fetch fresh price data in real time for generating live predictions.

## ETL Process

The ETL (Extract, Transform, Load) pipeline processes raw share price data into a format suitable for machine learning. The key transformations include cleaning missing data, computing technical indicators (moving averages, RSI, volatility), and creating the target variable (whether the next day's price goes up or down). The pipeline is designed to be reusable across any ticker symbol.

## Machine Learning Model

We chose a Logistic Regression model for its simplicity and interpretability. The model takes eight technical indicators as input and outputs a binary prediction: UP (price increases) or DOWN (price decreases). The data is split chronologically (80% training, 20% testing) to avoid data leakage. Features are standardized using a StandardScaler. While the model's accuracy is modest (as expected for stock prediction), the focus of this project is on the engineering and methodology rather than prediction quality.

## Web Application

The Streamlit web application has two main pages:

1. **Home Page:** Provides an overview of the system, explains how it works, and lists the supported companies.
2. **Go Live Page:** Allows users to select a stock ticker, view real-time price data fetched from SimFin, see computed technical indicators, and receive the model's prediction for the next trading day.

The application uses a Python API wrapper class (PySimFin) to interact with SimFin's REST API, handling authentication and rate limiting automatically.

## Challenges

- **Rate Limiting:** SimFin's free tier allows only 2 requests per second, requiring careful caching and request management in the web app.
- **Data Quality:** Some tickers had missing data points that needed handling in the ETL pipeline.
- **Model Limitations:** Stock markets are inherently difficult to predict; the model serves as a proof of concept rather than a reliable trading tool.

## Conclusions

This project demonstrates an end-to-end data engineering workflow: from raw data extraction through machine learning to a deployed web application. The system is modular, well-documented, and easily extensible to include more companies or more sophisticated models.
