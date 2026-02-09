"""Market Monitor - Main entry point and monitoring loop.

Runs during NYSE market hours, analyzing tickers and sending
Slack alerts when trading signals are detected.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Optional

from app.config import settings
from app.services.alerts import AlertManager, get_alert_manager
from app.services.alpaca_client import AlpacaClient, get_alpaca_client
from app.services.backtest import load_latest_backtest, run_backtest, save_backtest_results
from app.services.indicators import build_indicator_frame
from app.services.market_hours import (
    async_wait_for_market_open,
    get_market_status,
    is_market_open,
)
from app.services.patterns import detect_all_patterns
from app.services.strategies import Strategy, get_all_strategies, get_strategy

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Graceful shutdown flag
_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global _shutdown_requested
    logger.info("Shutdown signal received, stopping...")
    _shutdown_requested = True


class MarketMonitor:
    """Main market monitoring class."""

    def __init__(self):
        self.client: Optional[AlpacaClient] = None
        self.alert_manager: Optional[AlertManager] = None
        self.best_strategies: Dict[str, str] = {}  # ticker -> strategy name
        self.strategies: Dict[str, Strategy] = {}  # strategy name -> instance

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Market Monitor starting up...")
        logger.info("=" * 60)

        # Initialize Alpaca client
        logger.info("Initializing Alpaca client...")
        self.client = get_alpaca_client()

        # Initialize alert manager
        logger.info("Initializing Slack alert manager...")
        self.alert_manager = get_alert_manager()

        # Load or create strategy instances
        for strategy in get_all_strategies():
            self.strategies[strategy.name] = strategy

        # Load existing backtest or run new one
        await self._load_or_run_backtest()

        logger.info(f"Monitoring {len(settings.watchlist_tickers)} tickers: {settings.watchlist_tickers}")
        logger.info(f"Poll interval: {settings.poll_interval_seconds}s")
        logger.info(f"Alert cooldown: {settings.alert_cooldown_minutes} min")
        logger.info(f"Confidence threshold: {settings.signal_confidence_threshold:.0%}")
        logger.info("=" * 60)

    async def _load_or_run_backtest(self):
        """Load existing backtest results or run new one."""
        logger.info("Loading backtest results...")

        # Try to load existing results
        result = load_latest_backtest()

        if result is None:
            logger.info("No existing backtest found, running initial backtest...")
            result = run_backtest(
                self.client,
                settings.watchlist_tickers,
                days=settings.backtest_days,
            )
            save_backtest_results(result)

        # Store best strategy per ticker
        self.best_strategies = result.best_strategy_per_ticker

        logger.info("Best strategies per ticker:")
        for ticker, strategy in self.best_strategies.items():
            logger.info(f"  {ticker}: {strategy}")

    async def analyze_ticker(self, ticker: str):
        """Analyze a single ticker and send alert if signal detected.

        Args:
            ticker: Stock symbol to analyze
        """
        try:
            # Fetch recent bars
            df = self.client.get_bars(ticker, timeframe="5Min", limit=250)
            if df.empty:
                logger.warning(f"No data for {ticker}")
                return

            # Build indicators
            indicator_df = build_indicator_frame(df)
            if indicator_df.empty:
                logger.warning(f"Insufficient data for {ticker} after indicators")
                return

            # Detect patterns
            patterns = detect_all_patterns(indicator_df)

            # Get best strategy for this ticker
            strategy_name = self.best_strategies.get(ticker, "ensemble")
            strategy = self.strategies.get(strategy_name)

            if strategy is None:
                strategy = self.strategies["ensemble"]

            # Check for signal
            signal = strategy.check_signal(indicator_df, ticker, patterns)

            if signal is None:
                return

            # Check confidence threshold
            if signal.confidence < settings.signal_confidence_threshold:
                logger.debug(
                    f"{ticker}: Signal below threshold "
                    f"({signal.confidence:.0%} < {settings.signal_confidence_threshold:.0%})"
                )
                return

            # Send alert (respects cooldown)
            self.alert_manager.send_alert(signal)

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)

    async def run_monitoring_cycle(self):
        """Run one complete monitoring cycle for all tickers."""
        logger.info(f"Running monitoring cycle at {datetime.utcnow().strftime('%H:%M:%S')} UTC")

        for ticker in settings.watchlist_tickers:
            await self.analyze_ticker(ticker)
            # Small delay between tickers to avoid rate limits
            await asyncio.sleep(0.5)

    async def run(self):
        """Main monitoring loop."""
        await self.initialize()

        while not _shutdown_requested:
            # Check if market is open
            if not is_market_open():
                status = get_market_status()
                logger.info(f"Market closed. {status}")
                await async_wait_for_market_open()

                if _shutdown_requested:
                    break

                # Run fresh backtest at market open
                logger.info("Market opened, refreshing backtest...")
                result = run_backtest(
                    self.client,
                    settings.watchlist_tickers,
                    days=settings.backtest_days,
                )
                save_backtest_results(result)
                self.best_strategies = result.best_strategy_per_ticker

            # Run monitoring cycle
            await self.run_monitoring_cycle()

            # Wait for next cycle
            if not _shutdown_requested:
                await asyncio.sleep(settings.poll_interval_seconds)

        logger.info("Market Monitor stopped.")


async def main():
    """Entry point."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    monitor = MarketMonitor()
    await monitor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)
