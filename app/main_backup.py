"""Market Monitor - Main entry point and monitoring loop.

Runs during NYSE market hours, analyzing tickers and sending
Slack alerts when trading signals are detected. Includes ML-enhanced
confidence scoring and paper trading simulation.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional

from app.config import settings
from app.models.schemas import BacktestResult, PatternSignal, Signal
from app.services.alerts import AlertManager, get_alert_manager
from app.services.yfinance_client import YFinanceClient, get_yfinance_client
from app.services.backtest import load_latest_backtest, run_backtest, save_backtest_results
from app.services.indicators import build_indicator_frame
from app.services.market_hours import (
    async_wait_for_market_open,
    get_market_status,
    is_market_open,
)
from app.services.ml_scorer import (
    MLScorer,
    compute_enhanced_confidence,
    compute_indicator_extremity,
    compute_pattern_strength,
    get_ml_scorer,
)
from app.services.paper_trader import PaperTrader, get_paper_trader
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
        self.client: Optional[YFinanceClient] = None
        self.alert_manager: Optional[AlertManager] = None
        self.best_strategies: Dict[str, str] = {}  # ticker -> strategy name
        self.strategies: Dict[str, Strategy] = {}  # strategy name -> instance
        self.ml_scorer: Optional[MLScorer] = None
        self.paper_trader: Optional[PaperTrader] = None
        self.current_prices: Dict[str, float] = {}  # ticker -> latest price
        self._backtest_result: Optional[BacktestResult] = None
        self._market_was_open: bool = False

    async def initialize(self):
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("Market Monitor starting up...")
        logger.info("=" * 60)

        # Initialize yfinance client
        logger.info("Initializing Yahoo Finance client...")
        self.client = get_yfinance_client()

        # Initialize alert manager
        logger.info("Initializing Slack alert manager...")
        self.alert_manager = get_alert_manager()

        # Initialize ML scorer
        logger.info("Initializing ML signal scorer...")
        self.ml_scorer = get_ml_scorer()

        # Initialize paper trader
        logger.info("Initializing paper trader...")
        self.paper_trader = get_paper_trader()

        # Load or create strategy instances
        for strategy in get_all_strategies():
            self.strategies[strategy.name] = strategy

        # Load existing backtest or run new one
        await self._load_or_run_backtest()

        logger.info(f"Monitoring {len(settings.watchlist_tickers)} tickers: {settings.watchlist_tickers}")
        logger.info(f"Poll interval: {settings.poll_interval_seconds}s")
        logger.info(f"Alert cooldown: {settings.alert_cooldown_minutes} min")
        logger.info(f"Confidence threshold: {settings.signal_confidence_threshold:.0%}")
        logger.info(f"Paper trading: $100,000 starting capital")
        logger.info(f"ML scoring: {'enabled' if self.ml_scorer.is_trained else 'training on first backtest'}")
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

        # Store backtest result for ML training
        self._backtest_result = result

        # Store best strategy per ticker
        self.best_strategies = result.best_strategy_per_ticker

        # Train/update ML model with backtest data
        if self.ml_scorer:
            if not self.ml_scorer.is_trained:
                logger.info("Training ML model on backtest data...")
                self.ml_scorer.train(result)
            else:
                # Update historical accuracy from new backtest
                self.ml_scorer.update_from_backtest(result)

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

            # Get current price for paper trading
            price_col = "AdjClose" if "AdjClose" in indicator_df.columns else "Close"
            current_price = float(indicator_df[price_col].iloc[-1])
            self.current_prices[ticker] = current_price

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

            # Log that we found a signal (before enhancement)
            logger.info(
                f"📊 {ticker}: {signal.direction.value} signal detected by {strategy_name} "
                f"(base confidence: {signal.confidence:.0%}, {signal.confirmations} confirmations)"
            )

            # Enhance confidence with ML, pattern strength, and historical accuracy
            signal = self._enhance_signal_confidence(
                signal, indicator_df, patterns, strategy_name, ticker
            )

            # Check confidence threshold
            if signal.confidence < settings.signal_confidence_threshold:
                logger.info(
                    f"   ↳ {ticker}: Below threshold after enhancement "
                    f"({signal.confidence:.0%} < {settings.signal_confidence_threshold:.0%}) - skipping"
                )
                return

            # Send alert (respects cooldown)
            alert_sent = self.alert_manager.send_alert(signal)

            # Execute paper trade independently from alert/cooldown status
            if self.paper_trader:
                action = self.paper_trader.process_signal(signal, self.current_prices)
                if action and not alert_sent:
                    logger.info(
                        f"   ↳ {ticker}: Executed paper trade without Slack alert "
                        "(alert suppressed or failed)"
                    )

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}", exc_info=True)

    def _enhance_signal_confidence(
        self,
        signal: Signal,
        indicator_df,
        patterns: List[PatternSignal],
        strategy_name: str,
        ticker: str,
    ) -> Signal:
        """Enhance signal confidence using ML and additional factors."""
        if not self.ml_scorer:
            return signal

        # Extract indicator values for ML scoring
        row = indicator_df.iloc[-1]
        indicators = {
            "rsi": row.get("RSI14"),
            "macd": row.get("MACD"),
            "macd_signal": row.get("MACDSignal"),
            "macd_hist": row.get("MACDHist"),
            "stoch_k": row.get("STOCHK"),
            "stoch_d": row.get("STOCHD"),
            "adx": row.get("ADX"),
            "plus_di": row.get("PLUS_DI"),
            "minus_di": row.get("MINUS_DI"),
            "mfi": row.get("MFI14"),
            "willr": row.get("WILLR14"),
            "vol_ratio": row.get("VolRatio"),
            "bb_upper": row.get("BB_Up"),
            "bb_lower": row.get("BB_Low"),
        }

        # Get ML score
        ml_score = self.ml_scorer.score_signal(signal, indicators, patterns)

        # Get historical accuracy for this strategy-ticker combo
        historical_accuracy = self.ml_scorer.get_strategy_historical_accuracy(
            strategy_name, ticker
        )

        # Compute pattern strength
        pattern_strength = compute_pattern_strength(patterns)

        # Compute indicator extremity
        indicator_extremity = compute_indicator_extremity(indicators)

        # Compute enhanced confidence
        enhanced_confidence = compute_enhanced_confidence(
            base_confidence=signal.confidence,
            ml_score=ml_score,
            historical_accuracy=historical_accuracy,
            pattern_strength=pattern_strength,
            indicator_extremity=indicator_extremity,
        )

        # Log confidence breakdown
        logger.info(
            f"   ↳ {ticker} confidence: base={signal.confidence:.0%}, ml={ml_score:.0%}, "
            f"hist={historical_accuracy:.0%}, pattern={pattern_strength:.2f}, "
            f"extremity={indicator_extremity:.2f} → enhanced={enhanced_confidence:.0%}"
        )

        # Create new signal with enhanced confidence
        return Signal(
            ticker=signal.ticker,
            timestamp=signal.timestamp,
            strategy_name=signal.strategy_name,
            direction=signal.direction,
            confidence=enhanced_confidence,
            current_price=signal.current_price,
            indicators=signal.indicators,
            patterns=signal.patterns,
            reason=signal.reason,
            confirmations=signal.confirmations,
        )

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
            market_open = is_market_open()

            # Handle market state transitions
            if market_open and not self._market_was_open:
                # Market just opened
                logger.info("Market opened, starting trading day...")
                if self.paper_trader:
                    self.paper_trader.start_trading_day(self.current_prices)

                # Run fresh backtest at market open
                logger.info("Refreshing backtest...")
                result = run_backtest(
                    self.client,
                    settings.watchlist_tickers,
                    days=settings.backtest_days,
                )
                save_backtest_results(result)
                self._backtest_result = result
                self.best_strategies = result.best_strategy_per_ticker

                # Update ML model
                if self.ml_scorer:
                    self.ml_scorer.update_from_backtest(result)

            elif not market_open and self._market_was_open:
                # Market just closed
                logger.info("Market closed, ending trading day...")
                if self.paper_trader and self.current_prices:
                    self.paper_trader.end_trading_day(self.current_prices)

            self._market_was_open = market_open

            # Check if market is open
            if not market_open:
                status = get_market_status()
                logger.info(f"Market closed. {status}")
                await async_wait_for_market_open()

                if _shutdown_requested:
                    break
                continue

            # Check for auto-close on paper positions
            if self.paper_trader and self.current_prices:
                self.paper_trader.check_auto_close(self.current_prices)

            # Run monitoring cycle
            await self.run_monitoring_cycle()

            # Wait for next cycle
            if not _shutdown_requested:
                await asyncio.sleep(settings.poll_interval_seconds)

        # Send final daily summary if we had an active trading day
        if self._market_was_open and self.paper_trader and self.current_prices:
            logger.info("Shutdown during market hours, sending final summary...")
            self.paper_trader.end_trading_day(self.current_prices)

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
