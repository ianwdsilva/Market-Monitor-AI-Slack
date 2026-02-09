"""Yahoo Finance client for fetching market data.

No API key required. Data has ~15-minute delay on free tier.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YFinanceClient:
    """Client for fetching stock data from Yahoo Finance."""

    def __init__(self):
        """Initialize the yfinance client."""
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = timedelta(seconds=60)  # Cache bars for 60 seconds

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
            limit: Number of bars to fetch (approximate)

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        ticker = ticker.upper().strip()
        cache_key = f"{ticker}_{timeframe}_{limit}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached bars for {ticker}")
                return cached_df.copy()

        # Map timeframe to yfinance interval and period
        interval, period = self._parse_timeframe(timeframe)

        try:
            # Fetch data from yfinance
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No bars returned for {ticker}")
                return pd.DataFrame()

            # Rename columns to standard format
            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })

            # Ensure timezone-naive datetime index
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Add AdjClose
            df["AdjClose"] = df["Close"]

            # Add return column
            df["Return"] = df["Close"].pct_change()

            # Limit rows if needed
            if len(df) > limit:
                df = df.iloc[-limit:]

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
        interval, _ = self._parse_timeframe(timeframe)

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start, end=end, interval=interval)

            if df.empty:
                logger.warning(f"No historical bars for {ticker} from {start} to {end}")
                return pd.DataFrame()

            df = df.rename(columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Volume": "Volume",
            })

            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

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
        """Fetch bars for multiple tickers.

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

    def _parse_timeframe(self, timeframe: str) -> tuple[str, str]:
        """Convert timeframe string to yfinance interval and period.

        Args:
            timeframe: String like '5Min', '1Hour', '1Day'

        Returns:
            Tuple of (interval, period)
        """
        tf_map = {
            "1Min": ("1m", "7d"),      # 1-min data available for 7 days
            "5Min": ("5m", "60d"),     # 5-min data available for 60 days
            "15Min": ("15m", "60d"),
            "30Min": ("30m", "60d"),
            "1Hour": ("1h", "730d"),   # Hourly data for 2 years
            "1Day": ("1d", "max"),
        }
        return tf_map.get(timeframe, ("5m", "60d"))

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        logger.debug("Cleared yfinance data cache")


# Singleton instance
_client_instance: Optional[YFinanceClient] = None


def get_yfinance_client() -> YFinanceClient:
    """Get or create the yfinance client singleton.

    Returns:
        YFinanceClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = YFinanceClient()
    return _client_instance
