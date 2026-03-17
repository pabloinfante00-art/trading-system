# AI Usage Log

## Tools Used

- **Claude (Anthropic)** — Used throughout the project for code generation, debugging, and learning.

## Usage Details

| Task | AI Tool | What I Did | What Worked / What I Learned |
|---|---|---|---|
| ETL pipeline structure | Claude | Asked Claude to help design the ETL pipeline with clear Extract/Transform/Load steps | Learned how to structure a data pipeline with separate functions for each step. Understood why we need rolling windows for technical indicators. |
| Feature engineering | Claude | Used Claude to generate technical indicators (SMA, RSI, EMA, Volatility) | Learned what each indicator means and why they're useful for stock prediction. Had to adjust window sizes to match the data we had. |
| ML model training | Claude | Asked Claude to build a Logistic Regression model with proper train/test split | Learned why we use shuffle=False for time series data (can't train on future data), and why StandardScaler is needed. |
| API wrapper | Claude | Generated the PySimFin class with Claude's help | Learned about OOP in Python (classes, methods, __init__), HTTP requests, rate limiting, and error handling. |
| Streamlit app | Claude | Used Claude to scaffold the multi-page Streamlit app | Learned how Streamlit's page system works, how to use st.cache for performance, and how to display metrics and charts. |
| Debugging | Claude | Used Claude to fix various errors during development | Helped understand error messages and tracebacks. Most common issues were path-related and missing data columns. |

## Reflection

Using Claude significantly accelerated development, especially for boilerplate code and understanding new libraries. The most valuable learning came from asking Claude to *explain* the code it generated rather than just accepting it. Understanding concepts like feature scaling, time-series splitting, and API rate limiting will be useful beyond this project.

The main risk of using AI is accepting code without understanding it. To mitigate this, I made sure to read through every function and add my own comments explaining what each part does.
