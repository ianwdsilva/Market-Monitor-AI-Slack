"""Polygon.io client for fetching market data."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from polygon import RESTClient
from polygon.rest.models import Agg

logger = logging.getLogger(__name__)


class PolygonClient:
    """Client for fetching stock data from Polygon.io."""

    def __init__(self, api_key: str):
        """Initialize the Polygon client.

        Args:
            api_key: Polygon.io API key
        """
        self._client = RESTClient(api_key)
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._cache_ttl = timedelta(seconds=30)  # Cache bars for 30 seconds
        self._last_request_time: float = 0
        self._min_request_interval: float = 12.5  # 5 calls/min = 1 call per 12 seconds

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
            limit: Number of bars to fetch

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, VWAP
        """
        ticker = ticker.upper().strip()
        cache_key = f"{ticker}_{timeframe}_{limit}"

        # Check cache
        if cache_key in self._cache:
            cached_time, cached_df = self._cache[cache_key]
            if datetime.utcnow() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached bars for {ticker}")
                return cached_df.copy()

        # Rate limiting - wait if needed
        self._wait_for_rate_limit()

        # Parse timeframe
        multiplier, timespan = self._parse_timeframe(timeframe)

        # Calculate date range
        end = datetime.utcnow()
        if timeframe == "5Min":
            start = end - timedelta(days=5)
        elif timeframe == "1Min":
            start = end - timedelta(days=2)
        elif timeframe == "15Min":
            start = end - timedelta(days=10)
        elif timeframe == "1Hour":
            start = end - timedelta(days=30)
        else:
            start = end - timedelta(days=365)

        try:
            # Fetch aggregates from Polygon
            aggs = list(self._client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                limit=limit,
                sort="asc",
            ))

            if not aggs:
                logger.warning(f"No bars returned for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    "Date": datetime.fromtimestamp(agg.timestamp / 1000),
                    "Open": agg.open,
                    "High": agg.high,
                    "Low": agg.low,
                    "Close": agg.close,
                    "Volume": agg.volume,
                    "VWAP": agg.vwap if hasattr(agg, 'vwap') and agg.vwap else agg.close,
                    "Trades": agg.transactions if hasattr(agg, 'transactions') else 0,
                })

            df = pd.DataFrame(data)
            df = df.set_index("Date").sort_index()

            # Add AdjClose (same as Close for recent data)
            df["AdjClose"] = df["Close"]

            # Add return column
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
        multiplier, timespan = self._parse_timeframe(timeframe)

        # Rate limiting
        self._wait_for_rate_limit()

        try:
            aggs = list(self._client.list_aggs(
                ticker=ticker,
                multiplier=multiplier,
                timespan=timespan,
                from_=start.strftime("%Y-%m-%d"),
                to=end.strftime("%Y-%m-%d"),
                limit=50000,  # Max for historical
                sort="asc",
            ))

            if not aggs:
                logger.warning(f"No historical bars for {ticker} from {start} to {end}")
                return pd.DataFrame()

            data = []
            for agg in aggs:
                data.append({
                    "Date": datetime.fromtimestamp(agg.timestamp / 1000),
                    "Open": agg.open,
                    "High": agg.high,
                    "Low": agg.low,
                    "Close": agg.close,
                    "Volume": agg.volume,
                    "VWAP": agg.vwap if hasattr(agg, 'vwap') and agg.vwap else agg.close,
                })

            df = pd.DataFrame(data)
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

    def _wait_for_rate_limit(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            sleep_time = self._min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def _parse_timeframe(self, timeframe: str) -> tuple[int, str]:
        """Convert timeframe string to Polygon multiplier and timespan.

        Args:
            timeframe: String like '5Min', '1Hour', '1Day'

        Returns:
            Tuple of (multiplier, timespan)
        """
        tf_map = {
            "1Min": (1, "minute"),
            "5Min": (5, "minute"),
            "15Min": (15, "minute"),
            "30Min": (30, "minute"),
            "1Hour": (1, "hour"),
            "1Day": (1, "day"),
        }
        return tf_map.get(timeframe, (5, "minute"))

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        logger.debug("Cleared Polygon data cache")


# Singleton instance (initialized lazily)
_client_instance: Optional[PolygonClient] = None


def get_polygon_client() -> PolygonClient:
    """Get or create the Polygon client singleton.

    Returns:
        PolygonClient instance
    """
    global _client_instance
    if _client_instance is None:
        from app.config import settings

        _client_instance = PolygonClient(api_key=settings.polygon_api_key)
    return _client_instance
