"""
PySimFin - Python API Wrapper for SimFin
==========================================
This module provides an object-oriented interface to the SimFin Data API.
It simplifies making HTTP requests to SimFin and returns data as Pandas DataFrames.

Usage:
    from pysimfin import PySimFin

    client = PySimFin(api_key="your-api-key-here")
    prices = client.get_share_prices("AAPL", start="2024-01-01", end="2024-12-31")
    print(prices.head())
"""

import requests
import pandas as pd
import time
from typing import Optional


class PySimFin:
    """
    Python wrapper for the SimFin Data API.

    This class handles authentication, rate limiting, and data retrieval
    from SimFin's REST API endpoints.

    Attributes
    ----------
    api_key : str
        Your SimFin API key for authentication.
    base_url : str
        The base URL for all SimFin API requests.
    headers : dict
        HTTP headers sent with every request (includes the API key).
    last_request_time : float
        Timestamp of the last API request (used for rate limiting).

    Example
    -------
    >>> client = PySimFin(api_key="your-key")
    >>> df = client.get_share_prices("AAPL", "2024-01-01", "2024-06-01")
    >>> print(df.head())
    """

    # SimFin free tier limit: max 2 requests per second
    RATE_LIMIT_DELAY = 0.55  # seconds between requests (slightly over 0.5 to be safe)

    def __init__(self, api_key: str):
        """
        Initialize the PySimFin client.

        Parameters
        ----------
        api_key : str
            Your SimFin API key. Get it from https://www.simfin.com/ after registering.

        Raises
        ------
        ValueError
            If the API key is empty or None.
        """
        if not api_key:
            raise ValueError("API key cannot be empty. Please provide a valid SimFin API key.")

        self.api_key = api_key
        self.base_url = "https://backend.simfin.com/api/v3"
        self.headers = {
            "Authorization": f"api-key {self.api_key}",
            "Accept": "application/json",
        }
        self.last_request_time = 0.0

    def _rate_limit(self):
        """
        Enforce rate limiting to avoid exceeding SimFin's free tier limits.
        Waits if the time since the last request is less than RATE_LIMIT_DELAY.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            wait_time = self.RATE_LIMIT_DELAY - elapsed
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """
        Make an HTTP GET request to the SimFin API.

        This is a private method used internally by the public methods.
        It handles rate limiting, error checking, and JSON parsing.

        Parameters
        ----------
        endpoint : str
            API endpoint path (e.g., "/companies/prices/compact").
        params : dict, optional
            Query parameters to include in the request.

        Returns
        -------
        dict
            Parsed JSON response from the API.

        Raises
        ------
        ConnectionError
            If the API request fails (non-200 status code).
        Exception
            If there is a network error.
        """
        # Respect rate limits
        self._rate_limit()

        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)

            # Check for HTTP errors
            if response.status_code == 401:
                raise ConnectionError("Authentication failed. Check your API key.")
            elif response.status_code == 403:
                raise ConnectionError("Access forbidden. You may have exceeded rate limits.")
            elif response.status_code == 404:
                raise ConnectionError(f"Endpoint not found: {url}")
            elif response.status_code != 200:
                raise ConnectionError(
                    f"API request failed with status {response.status_code}: {response.text}"
                )

            return response.json()

        except requests.exceptions.Timeout:
            raise ConnectionError("Request timed out. SimFin server may be slow. Try again.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Could not connect to SimFin. Check your internet connection.")

    def _json_to_dataframe(self, data: list) -> pd.DataFrame:
        """
        Convert SimFin's compact JSON response into a Pandas DataFrame.

        SimFin returns data in a compact format:
        [
            {"columns": ["col1", "col2", ...], "data": [[val1, val2, ...], ...]}
        ]

        Parameters
        ----------
        data : list
            Raw JSON response from SimFin.

        Returns
        -------
        pd.DataFrame
            Data in a tabular format.
        """
        if not data or len(data) == 0:
            return pd.DataFrame()

        # SimFin returns a list where first element has columns and data
        entry = data[0] if isinstance(data, list) else data

        if "columns" in entry and "data" in entry:
            df = pd.DataFrame(entry["data"], columns=entry["columns"])
        else:
            # Fallback: try to build DataFrame directly
            df = pd.DataFrame(data)

        return df

    def get_share_prices(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Retrieve daily share prices for a specific company and time period.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol (e.g., "AAPL", "MSFT").
        start : str
            Start date in "YYYY-MM-DD" format (e.g., "2024-01-01").
        end : str
            End date in "YYYY-MM-DD" format (e.g., "2024-12-31").

        Returns
        -------
        pd.DataFrame
            DataFrame with columns like Date, Open, High, Low, Close, Volume, etc.

        Example
        -------
        >>> client = PySimFin(api_key="your-key")
        >>> prices = client.get_share_prices("AAPL", "2024-01-01", "2024-06-01")
        >>> print(prices.head())
        """
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
        }

        data = self._make_request("/companies/prices/compact", params=params)
        df = self._json_to_dataframe(data)

        # Parse date column if it exists
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

        return df

    def get_financial_statement(
        self, ticker: str, start: str, end: str, statement: str = "pl"
    ) -> pd.DataFrame:
        """
        Retrieve financial statements for a specific company.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        start : str
            Start date in "YYYY-MM-DD" format.
        end : str
            End date in "YYYY-MM-DD" format.
        statement : str, optional
            Type of statement: "pl" (Profit & Loss / Income Statement),
            "bs" (Balance Sheet), "cf" (Cash Flow). Default is "pl".

        Returns
        -------
        pd.DataFrame
            DataFrame with financial statement data.

        Example
        -------
        >>> client = PySimFin(api_key="your-key")
        >>> income = client.get_financial_statement("AAPL", "2023-01-01", "2024-01-01", "pl")
        """
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
        }

        endpoint = f"/companies/statements/compact?statement={statement}"
        data = self._make_request(endpoint, params=params)
        df = self._json_to_dataframe(data)

        return df

    def get_company_info(self, ticker: str) -> pd.DataFrame:
        """
        Retrieve general information about a company.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

        Returns
        -------
        pd.DataFrame
            DataFrame with company information (name, industry, etc.).
        """
        params = {"ticker": ticker}
        data = self._make_request("/companies/general/compact", params=params)
        df = self._json_to_dataframe(data)
        return df


# ---------------------------------------------------------------------------
# Quick test (only runs if you execute this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("PySimFin API Wrapper")
    print("=" * 40)
    print("To test, create an instance:")
    print('  client = PySimFin(api_key="your-key")')
    print('  df = client.get_share_prices("AAPL", "2024-01-01", "2024-06-01")')
    print("  print(df.head())")
