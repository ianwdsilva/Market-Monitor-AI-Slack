"""Alpaca client for fetching market data."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for fetching stock data from Alpaca."""

    def __init__(self, api_key: str, secret_key: str):
        """Initialize the Alpaca client.

        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
        """
        self._client = StockHistoricalDataClient(api_key, secret_key)
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = timedelta(seconds=30)  # Cache bars for 30 seconds

    def get_bars(
        self,
        ticker: str,
        timeframe: str = "5Min",
        limit: int = 250,
    ) -> pd.DataFrame:
        """Fetch recent OHLCV bars for a ticker.

        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            limit: Number of bars to fetch (max ~10000)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, VWAP, Trades
        """
        ticker = ticker.upper().strip()
        cache_key = f"{ticker}_{timeframe}_{limit}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached bars for {ticker}")
                return cached_df.copy()

        # Map timeframe string to Alpaca TimeFrame
        tf = self._parse_timeframe(timeframe)

        # Calculate start time based on limit and timeframe
        # For 5Min bars, 250 bars = ~21 hours of market time
        end = datetime.utcnow()
        if timeframe == "5Min":
            start = end - timedelta(days=5)  # Fetch extra to account for market hours
        elif timeframe == "1Min":
            start = end - timedelta(days=2)
        elif timeframe == "15Min":
            start = end - timedelta(days=10)
        elif timeframe == "1Hour":
            start = end - timedelta(days=30)
        else:
            start = end - timedelta(days=365)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            end=end,
            limit=limit,
        )

        try:
            bars = self._client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                logger.warning(f"No bars returned for {ticker}")
                return pd.DataFrame()

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Handle multi-index if multiple symbols
            if "symbol" in df.columns:
                df = df[df["symbol"] == ticker].drop(columns=["symbol"])

            # Rename columns to standard format
            df = df.rename(
                columns={
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "vwap": "VWAP",
                    "trade_count": "Trades",
                }
            )

            # Ensure timezone-naive datetime
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                df = df.set_index("Date").sort_index()

            # Add AdjClose (same as Close for recent data)
            df["AdjClose"] = df["Close"]

            # Add return columns
            df["Return"] = df["Close"].pct_change()

            # Cache the result
            self._cache[cache_key] = (datetime.utcnow(), df.copy())

            logger.info(f"Fetched {len(df)} bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching bars for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_bar(self, ticker: str) -> Optional[dict]:
        """Get the most recent bar for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dict with bar data or None if unavailable
        """
        df = self.get_bars(ticker, timeframe="5Min", limit=1)
        if df.empty:
            return None

        row = df.iloc[-1]
        return {
            "timestamp": df.index[-1],
            "open": row["Open"],
            "high": row["High"],
            "low": row["Low"],
            "close": row["Close"],
            "volume": row["Volume"],
            "vwap": row.get("VWAP"),
        }

    def get_historical_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        timeframe: str = "5Min",
    ) -> pd.DataFrame:
        """Fetch historical bars for backtesting.

        Args:
            ticker: Stock symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe

        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker.upper().strip()
        tf = self._parse_timeframe(timeframe)

        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=tf,
            start=start,
            end=end,
        )

        try:
            bars = self._client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                logger.warning(f"No historical bars for {ticker} from {start} to {end}")
                return pd.DataFrame()

            df = df.reset_index()
            if "symbol" in df.columns:
                df = df[df["symbol"] == ticker].drop(columns=["symbol"])

            df = df.rename(
                columns={
                    "timestamp": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                    "vwap": "VWAP",
                    "trade_count": "Trades",
                }
            )

            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                df = df.set_index("Date").sort_index()

            df["AdjClose"] = df["Close"]
            df["Return"] = df["Close"].pct_change()

            logger.info(f"Fetched {len(df)} historical bars for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching historical bars for {ticker}: {e}")
            return pd.DataFrame()

    def get_bars_batch(
        self,
        tickers: List[str],
        timeframe: str = "5Min",
        limit: int = 250,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple tickers efficiently.

        Args:
            tickers: List of stock symbols
            timeframe: Bar timeframe
            limit: Number of bars per ticker

        Returns:
            Dict mapping ticker -> DataFrame
        """
        result = {}
        for ticker in tickers:
            result[ticker.upper()] = self.get_bars(ticker, timeframe, limit)
        return result

    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Convert timeframe string to Alpaca TimeFrame.

        Args:
            timeframe: String like '5Min', '1Hour', '1Day'

        Returns:
            Alpaca TimeFrame object
        """
        tf_map = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
        }
        return tf_map.get(timeframe, TimeFrame(5, TimeFrameUnit.Minute))

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        logger.debug("Cleared Alpaca data cache")


# Singleton instance (initialized lazily)
_client_instance: Optional[AlpacaClient] = None


def get_alpaca_client() -> AlpacaClient:
    """Get or create the Alpaca client singleton.

    Returns:
        AlpacaClient instance
    """
    global _client_instance
    if _client_instance is None:
        from app.config import settings

        _client_instance = AlpacaClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
    return _client_instance
