# 📈 Automated Daily Trading System

An automated daily trading system that uses machine learning to predict stock market movements for 5 US companies. Built with Python, Scikit-learn, and Streamlit.

## Team Members

- Selim El Khoury, Jan Wilhelm Pietsch, Nuria Etemadi, Tenaw Belete, Pablo Infante

## 🌐 Live Application

[![Live App](https://img.shields.io/badge/Live_App-Trading_System-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://trading-system-gnvcjp4j8kqekggfhwpfl2.streamlit.app)

## Overview

This system consists of two main parts:

1. **Data Analytics Module** — An ETL pipeline processes historical stock data from SimFin, then a Logistic Regression model is trained to predict whether the next day's closing price will go UP or DOWN.
2. **Web Application** — A Streamlit app fetches real-time data from the SimFin API, applies the same transformations, and uses the trained model to generate live trading signals.

## Project Structure

```
trading-system/
├── app/                    # Streamlit web application
│   ├── Home.py             # Home page (main entry point)
│   ├── pages/
│   │   └── 1_Go_Live.py   # Real-time predictions page
│   └── .streamlit/
│       └── secrets.toml.example  # API key template
├── src/                    # Core Python modules
│   ├── etl.py              # ETL pipeline
│   ├── model.py            # ML model training & export
│   └── pysimfin.py         # SimFin API wrapper class
├── data/                   # Raw and processed data (not in Git)
│   └── processed/          # Output of ETL pipeline
├── models/                 # Trained model files (.pkl)
├── notebooks/              # Jupyter notebooks for exploration
├── docs/                   # Executive summary
├── requirements.txt        # Python dependencies
├── AI_USAGE_LOG.md         # AI tool usage documentation
└── README.md               # This file
```

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/trading-system.git
cd trading-system
```

### 2. Create a Python environment

```bash
conda create -n trading-app python=3.11
conda activate trading-app
pip install -r requirements.txt
```

### 3. Set up SimFin

1. Register at [simfin.com](https://www.simfin.com/) and get your API key.
2. Download bulk data files (Share Prices + Companies) and place them in the `data/` folder:
   - `data/us-shareprices-daily.csv`
   - `data/us-companies.csv`
3. Create the secrets file for the Streamlit app:
   ```bash
   cp app/.streamlit/secrets.toml.example app/.streamlit/secrets.toml
   ```
   Edit `app/.streamlit/secrets.toml` and paste your API key.

### 4. Run the ETL pipeline

```bash
python src/etl.py --all
```

### 5. Train the ML models

```bash
python src/model.py --all
```

### 6. Run the Streamlit app

```bash
streamlit run app/Home.py
```

The app will open in your browser at `http://localhost:8501`.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core language |
| Pandas | Data manipulation |
| Scikit-learn | Machine learning (Logistic Regression) |
| Streamlit | Web application |
| SimFin API | Financial data source |
| Joblib | Model serialization |
