# Data Directory

Place your SimFin bulk download files here:

- `us-shareprices-daily.csv` — Download from SimFin > Bulk Download > Share Prices (Daily)
- `us-companies.csv` — Download from SimFin > Bulk Download > Companies

These raw files are **not committed to Git** (too large). Download them from [simfin.com](https://www.simfin.com/) and place them in this folder before running the ETL pipeline.

The `processed/` subdirectory contains feature-engineered CSVs that are committed to the repository so that the Streamlit Cloud deployment can serve predictions without re-running the ETL.
