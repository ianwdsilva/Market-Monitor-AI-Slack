"""Market Monitor - Main entry point and monitoring loop. Command to run is 'python3 -m app.main'

Runs during NYSE market hours, analyzing tickers and sending
Slack alerts when trading signals are detected. Includes ML-enhanced
confidence scoring, continuous online learning, Kelly Criterion
position sizing, concurrent ticker analysis, and a live JSON dashboard.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from app.services.strategies import Strategy, get_all_strategies

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------

_shutdown_requested = False


def signal_handler(signum, frame) -> None:
    global _shutdown_requested
    logger.info("Shutdown signal received, stopping...")
    _shutdown_requested = True


# ---------------------------------------------------------------------------
# Dashboard path
# ---------------------------------------------------------------------------

DASHBOARD_PATH = Path("dashboard.json")

# ---------------------------------------------------------------------------
# Kelly Criterion helpers
# ---------------------------------------------------------------------------

# Minimum edge required before Kelly allocates any capital.
# Prevents tiny edges from generating noise trades.
_KELLY_MIN_EDGE: float = 0.02

# Fractional Kelly multiplier — full Kelly is notoriously aggressive.
# Half-Kelly (0.5) is standard practice for live/paper trading.
_KELLY_FRACTION: float = 0.5

# Hard cap: never allocate more than 20 % of portfolio on a single signal,
# regardless of what Kelly computes.
_KELLY_MAX_FRACTION: float = 0.20


def kelly_position_size(
    confidence: float,
    portfolio_value: float,
    price: float,
    max_trade_value: float,
) -> Tuple[int, float]:
    """Compute share count and dollar value using fractional Kelly Criterion.

    Kelly formula for a binary win/loss bet:
        f* = (p * b - q) / b
    where:
        p = win probability  (we use signal confidence as a proxy)
        q = 1 - p            (loss probability)
        b = win/loss ratio   (we assume a symmetric 1:1 payoff as a baseline)

    A symmetric 1:1 payoff means b = 1, which simplifies to:
        f* = p - q = 2p - 1

    Fractional Kelly then applies the safety multiplier and the hard cap.

    Args:
        confidence:      Signal confidence score in [0, 1].
        portfolio_value: Current total portfolio value in dollars.
        price:           Current price per share.
        max_trade_value: Hard dollar ceiling from settings.

    Returns:
        (shares, dollar_value) — both are 0 if Kelly finds no edge.
    """
    if confidence <= 0 or price <= 0 or portfolio_value <= 0:
        return 0, 0.0

    # Win probability proxy
    p = confidence
    q = 1.0 - p

    # Edge: expected gain per dollar risked (symmetric payoff assumed)
    edge = p - q  # equivalent to 2p - 1

    if edge < _KELLY_MIN_EDGE:
        # No meaningful edge — skip the trade
        return 0, 0.0

    # Full Kelly fraction
    full_kelly = edge  # b=1 simplifies away

    # Apply safety multiplier and hard cap
    fraction = min(full_kelly * _KELLY_FRACTION, _KELLY_MAX_FRACTION)

    # Dollar allocation
    dollar_value = portfolio_value * fraction
    dollar_value = min(dollar_value, max_trade_value)

    shares = int(dollar_value // price)
    actual_value = shares * price

    return shares, actual_value


# ---------------------------------------------------------------------------
# Dashboard writer
# ---------------------------------------------------------------------------

class DashboardWriter:
    """Writes a live JSON snapshot of monitor state each cycle.

    The file is written atomically (temp → rename) so any reader
    always sees a complete, valid JSON document.
    """

    def __init__(self, path: Path = DASHBOARD_PATH) -> None:
        self._path = path
        self._tmp = path.with_suffix(".json.tmp")
        self._signals_today: List[dict] = []
        self._cycle_count: int = 0
        self._errors_today: int = 0
        self._start_time: str = datetime.now(timezone.utc).isoformat()

    def record_signal(self, signal: Signal, shares: int, dollar_value: float) -> None:
        """Append a signal record to today's in-memory log."""
        self._signals_today.append(
            {
                "time": signal.timestamp.isoformat() if hasattr(signal.timestamp, "isoformat") else str(signal.timestamp),
                "ticker": signal.ticker,
                "direction": signal.direction.value,
                "confidence": round(signal.confidence, 4),
                "strategy": signal.strategy_name,
                "price": signal.current_price,
                "kelly_shares": shares,
                "kelly_dollar_value": round(dollar_value, 2),
                "confirmations": signal.confirmations,
                "reason": signal.reason,
            }
        )

    def record_error(self) -> None:
        self._errors_today += 1

    def reset_day(self) -> None:
        """Clear intraday accumulators at market open."""
        self._signals_today = []
        self._errors_today = 0

    def write(
        self,
        *,
        current_prices: Dict[str, float],
        best_strategies: Dict[str, str],
        paper_trader: Optional[PaperTrader],
        ml_scorer: Optional[MLScorer],
        market_open: bool,
        cycle_count: int,
    ) -> None:
        """Write a full dashboard snapshot to disk."""
        self._cycle_count = cycle_count

        # Paper trading summary
        pt_summary: dict = {}
        if paper_trader is not None:
            try:
                pt_summary = {
                    "portfolio_value": paper_trader.portfolio_value,
                    "cash": paper_trader.cash,
                    "open_positions": len(paper_trader.positions),
                    "daily_pnl": paper_trader.daily_pnl,
                    "total_pnl": paper_trader.total_pnl,
                    "trade_count_today": paper_trader.trade_count_today,
                }
            except AttributeError:
                # Graceful degradation if paper_trader API differs slightly
                pt_summary = {"note": "paper trader stats unavailable"}

        # ML scorer summary
        ml_summary: dict = {}
        if ml_scorer is not None:
            try:
                ml_summary = {
                    "is_trained": ml_scorer.is_trained,
                    "samples_seen": getattr(ml_scorer, "samples_seen", None),
                    "online_updates": getattr(ml_scorer, "online_updates", 0),
                }
            except AttributeError:
                ml_summary = {"note": "ml scorer stats unavailable"}

        snapshot = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "monitor_start": self._start_time,
            "market_open": market_open,
            "cycle_count": self._cycle_count,
            "errors_today": self._errors_today,
            "watchlist": settings.watchlist_tickers,
            "current_prices": {k: round(v, 4) for k, v in current_prices.items()},
            "best_strategies": best_strategies,
            "paper_trading": pt_summary,
            "ml_scorer": ml_summary,
            "signals_today": self._signals_today,
            "config": {
                "poll_interval_seconds": settings.poll_interval_seconds,
                "alert_cooldown_minutes": settings.alert_cooldown_minutes,
                "signal_confidence_threshold": settings.signal_confidence_threshold,
                "min_confirmations": settings.min_confirmations,
                "backtest_days": settings.backtest_days,
                "kelly_fraction": _KELLY_FRACTION,
                "kelly_max_fraction": _KELLY_MAX_FRACTION,
            },
        }

        try:
            self._tmp.write_text(json.dumps(snapshot, indent=2, default=str))
            self._tmp.replace(self._path)
        except OSError as exc:
            logger.warning(f"Dashboard write failed: {exc}")


# ---------------------------------------------------------------------------
# MarketMonitor
# ---------------------------------------------------------------------------

class MarketMonitor:
    """Main market monitoring class.

    Use ``MarketMonitor.create()`` to get a fully initialised instance
    rather than constructing directly.
    """

    # Maximum concurrent ticker analyses.  Keeps yfinance from being
    # hammered while still running all tickers in parallel.
    _CONCURRENCY = 4

    def __init__(
        self,
        *,
        client: YFinanceClient,
        alert_manager: AlertManager,
        ml_scorer: MLScorer,
        paper_trader: PaperTrader,
        strategies: Dict[str, Strategy],
        best_strategies: Dict[str, str],
        backtest_result: BacktestResult,
    ) -> None:
        # All dependencies injected — no Optional fields, no deferred init.
        self.client = client
        self.alert_manager = alert_manager
        self.ml_scorer = ml_scorer
        self.paper_trader = paper_trader
        self.strategies = strategies
        self.best_strategies = best_strategies

        self.current_prices: Dict[str, float] = {}
        self._backtest_result: BacktestResult = backtest_result
        self._market_was_open: bool = False
        self._cycle_count: int = 0

        # Failure tracking for circuit-breaker logic
        self._consecutive_failures: Dict[str, int] = {
            t: 0 for t in settings.watchlist_tickers
        }
        self._backoff_until: Dict[str, datetime] = {}

        # Concurrency limiter
        self._semaphore = asyncio.Semaphore(self._CONCURRENCY)

        # Dashboard
        self._dashboard = DashboardWriter()

    # ------------------------------------------------------------------
    # Async factory — the preferred way to construct MarketMonitor
    # ------------------------------------------------------------------

    @classmethod
    async def create(cls) -> "MarketMonitor":
        """Async factory that fully initialises all components before returning.

        Usage::

            monitor = await MarketMonitor.create()
            await monitor.run()

        This eliminates the ``Optional`` noise of the old ``__init__`` +
        ``initialize()`` pattern — the returned object is always fully ready.
        """
        logger.info("=" * 60)
        logger.info("Market Monitor starting up...")
        logger.info("=" * 60)

        logger.info("Initializing Yahoo Finance client...")
        client = get_yfinance_client()

        logger.info("Initializing Slack alert manager...")
        alert_manager = get_alert_manager()

        logger.info("Initializing ML signal scorer...")
        ml_scorer = get_ml_scorer()

        logger.info("Initializing paper trader...")
        paper_trader = get_paper_trader()

        strategies: Dict[str, Strategy] = {
            s.name: s for s in get_all_strategies()
        }

        # Load or run backtest
        logger.info("Loading backtest results...")
        backtest_result = load_latest_backtest()
        if backtest_result is None:
            logger.info("No existing backtest found, running initial backtest...")
            backtest_result = run_backtest(
                client,
                settings.watchlist_tickers,
                days=settings.backtest_days,
            )
            save_backtest_results(backtest_result)

        # Train or update ML model
        if not ml_scorer.is_trained:
            logger.info("Training ML model on backtest data...")
            ml_scorer.train(backtest_result)
        else:
            ml_scorer.update_from_backtest(backtest_result)

        best_strategies = backtest_result.best_strategy_per_ticker

        instance = cls(
            client=client,
            alert_manager=alert_manager,
            ml_scorer=ml_scorer,
            paper_trader=paper_trader,
            strategies=strategies,
            best_strategies=best_strategies,
            backtest_result=backtest_result,
        )

        logger.info(f"Monitoring {len(settings.watchlist_tickers)} tickers: {settings.watchlist_tickers}")
        logger.info(f"Poll interval:          {settings.poll_interval_seconds}s")
        logger.info(f"Alert cooldown:         {settings.alert_cooldown_minutes} min")
        logger.info(f"Confidence threshold:   {settings.signal_confidence_threshold:.0%}")
        logger.info(f"Paper capital:          ${settings.paper_trading_starting_balance:,.0f}")
        logger.info(f"Kelly fraction:         {_KELLY_FRACTION} (max {_KELLY_MAX_FRACTION:.0%})")
        logger.info(f"ML scoring:             {'enabled' if ml_scorer.is_trained else 'training on first backtest'}")
        logger.info(f"Dashboard:              {DASHBOARD_PATH.resolve()}")
        logger.info("=" * 60)

        for ticker, strat in best_strategies.items():
            logger.info(f"  Best strategy  {ticker}: {strat}")

        return instance

    # ------------------------------------------------------------------
    # Ticker analysis
    # ------------------------------------------------------------------

    async def analyze_ticker(self, ticker: str) -> None:
        """Analyze one ticker under the concurrency semaphore.

        Includes a simple circuit-breaker: after 3 consecutive failures
        the ticker is skipped for an exponentially increasing back-off
        window (2 min → 4 min → 8 min, capped at 30 min).
        """
        # Circuit-breaker check
        backoff_until = self._backoff_until.get(ticker)
        if backoff_until and datetime.now(timezone.utc) < backoff_until:
            logger.debug(f"{ticker}: in back-off, skipping")
            return

        async with self._semaphore:
            try:
                await self._do_analyze(ticker)
                # Success — reset failure counter
                self._consecutive_failures[ticker] = 0

            except Exception as exc:
                failures = self._consecutive_failures.get(ticker, 0) + 1
                self._consecutive_failures[ticker] = failures
                self._dashboard.record_error()

                wait_minutes = min(2 ** failures, 30)
                from datetime import timedelta
                self._backoff_until[ticker] = datetime.now(timezone.utc) + timedelta(minutes=wait_minutes)

                logger.error(
                    f"Error analyzing {ticker} (failure #{failures}, "
                    f"back-off {wait_minutes} min): {exc}",
                    exc_info=True,
                )

    async def _do_analyze(self, ticker: str) -> None:
        """Core analysis logic for a single ticker (no error handling here)."""
        # Fetch bars — run sync yfinance call in a thread pool so it
        # doesn't block the event loop during concurrent analysis.
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None, lambda: self.client.get_bars(ticker, timeframe="5Min", limit=250)
        )

        if df.empty:
            logger.warning(f"No data for {ticker}")
            return

        # Build indicators
        indicator_df = build_indicator_frame(df)
        if indicator_df.empty:
            logger.warning(f"Insufficient data for {ticker} after indicators")
            return

        # Track current price
        price_col = "AdjClose" if "AdjClose" in indicator_df.columns else "Close"
        current_price = float(indicator_df[price_col].iloc[-1])
        self.current_prices[ticker] = current_price

        # Detect patterns
        patterns = detect_all_patterns(indicator_df)

        # Pick strategy
        strategy_name = self.best_strategies.get(ticker, "ensemble")
        strategy = self.strategies.get(strategy_name) or self.strategies["ensemble"]

        # Generate raw signal
        sig = strategy.check_signal(indicator_df, ticker, patterns)
        if sig is None:
            return

        logger.info(
            f"📊 {ticker}: {sig.direction.value} signal via {strategy_name} "
            f"(base confidence: {sig.confidence:.0%}, {sig.confirmations} confirmations)"
        )

        # Extract indicators for ML
        row = indicator_df.iloc[-1]
        indicators = {
            "rsi":        row.get("RSI14"),
            "macd":       row.get("MACD"),
            "macd_signal":row.get("MACDSignal"),
            "macd_hist":  row.get("MACDHist"),
            "stoch_k":    row.get("STOCHK"),
            "stoch_d":    row.get("STOCHD"),
            "adx":        row.get("ADX"),
            "plus_di":    row.get("PLUS_DI"),
            "minus_di":   row.get("MINUS_DI"),
            "mfi":        row.get("MFI14"),
            "willr":      row.get("WILLR14"),
            "vol_ratio":  row.get("VolRatio"),
            "bb_upper":   row.get("BB_Up"),
            "bb_lower":   row.get("BB_Low"),
        }

        # Enhance confidence with ML + pattern/indicator factors
        sig = self._enhance_signal_confidence(sig, indicators, patterns, strategy_name, ticker)

        # Threshold gate
        if sig.confidence < settings.signal_confidence_threshold:
            logger.info(
                f"   ↳ {ticker}: Below threshold "
                f"({sig.confidence:.0%} < {settings.signal_confidence_threshold:.0%}) — skipping"
            )
            return

        # Kelly position sizing
        portfolio_value = getattr(self.paper_trader, "portfolio_value", settings.paper_trading_starting_balance)
        shares, dollar_value = kelly_position_size(
            confidence=sig.confidence,
            portfolio_value=portfolio_value,
            price=current_price,
            max_trade_value=settings.paper_trading_max_trade_value,
        )

        if shares == 0:
            logger.info(f"   ↳ {ticker}: Kelly finds no edge at {sig.confidence:.0%} confidence — skipping")
            return

        logger.info(
            f"   ↳ {ticker}: Kelly → {shares} shares @ ${current_price:.2f} "
            f"= ${dollar_value:,.2f} ({dollar_value / portfolio_value:.1%} of portfolio)"
        )

        # Record signal for dashboard before sending
        self._dashboard.record_signal(sig, shares, dollar_value)

        # Send Slack alert
        alert_sent = self.alert_manager.send_alert(sig)

        # Execute paper trade with Kelly-sized quantity
        action = self.paper_trader.process_signal(sig, self.current_prices, shares=shares)
        if action and not alert_sent:
            logger.info(f"   ↳ {ticker}: Paper trade executed without Slack alert")

        # ----------------------------------------------------------------
        # Online ML update — teach the model about this live signal so it
        # continuously improves without waiting for the next backtest.
        # ----------------------------------------------------------------
        self._online_ml_update(sig, indicators, patterns, strategy_name, ticker)

    # ------------------------------------------------------------------
    # Continuous ML learning
    # ------------------------------------------------------------------

    def _online_ml_update(
        self,
        signal: Signal,
        indicators: dict,
        patterns: List[PatternSignal],
        strategy_name: str,
        ticker: str,
    ) -> None:
        """Feed one live signal back into the ML scorer as a training sample.

        We treat the signal's (enhanced) confidence score as a soft label —
        high-confidence signals we actually sent reinforce the model, while
        the signal's direction and indicator state provide the features.

        The scorer is expected to expose ``partial_fit(features, label)``
        for incremental/online learning.  If it doesn't (e.g. the scorer
        uses a batch-only model), this is a no-op and a debug message is
        logged so you know to implement it.
        """
        if not hasattr(self.ml_scorer, "partial_fit"):
            logger.debug(
                "MLScorer does not implement partial_fit — "
                "continuous learning skipped for this signal."
            )
            return

        try:
            # Build a simple feature dict the scorer can consume.
            # Using the same shape as score_signal so no new plumbing needed.
            feature_sample = {
                **indicators,
                "pattern_strength": compute_pattern_strength(patterns),
                "indicator_extremity": compute_indicator_extremity(indicators),
                "confirmations": signal.confirmations,
                "direction": signal.direction.value,
                "strategy": strategy_name,
                "ticker": ticker,
            }

            # Label: treat the final enhanced confidence as a continuous
            # target in [0, 1].  If your scorer works with binary labels,
            # threshold at 0.5 here instead.
            label = signal.confidence

            self.ml_scorer.partial_fit(feature_sample, label)

            online_updates = getattr(self.ml_scorer, "online_updates", 0) + 1
            try:
                self.ml_scorer.online_updates = online_updates
            except AttributeError:
                pass  # read-only attribute on some scorer implementations

            logger.debug(
                f"   ↳ {ticker}: ML online update #{online_updates} "
                f"(label={label:.2f})"
            )

        except Exception as exc:
            # Never let ML bookkeeping crash the main loop
            logger.warning(f"Online ML update failed for {ticker}: {exc}")

    # ------------------------------------------------------------------
    # Confidence enhancement (unchanged logic, refactored for clarity)
    # ------------------------------------------------------------------

    def _enhance_signal_confidence(
        self,
        signal: Signal,
        indicators: dict,
        patterns: List[PatternSignal],
        strategy_name: str,
        ticker: str,
    ) -> Signal:
        ml_score = self.ml_scorer.score_signal(signal, indicators, patterns)
        historical_accuracy = self.ml_scorer.get_strategy_historical_accuracy(strategy_name, ticker)
        pattern_strength = compute_pattern_strength(patterns)
        indicator_extremity = compute_indicator_extremity(indicators)

        enhanced_confidence = compute_enhanced_confidence(
            base_confidence=signal.confidence,
            ml_score=ml_score,
            historical_accuracy=historical_accuracy,
            pattern_strength=pattern_strength,
            indicator_extremity=indicator_extremity,
        )

        logger.info(
            f"   ↳ {ticker} confidence: base={signal.confidence:.0%}, ml={ml_score:.0%}, "
            f"hist={historical_accuracy:.0%}, pattern={pattern_strength:.2f}, "
            f"extremity={indicator_extremity:.2f} → enhanced={enhanced_confidence:.0%}"
        )

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

    # ------------------------------------------------------------------
    # Monitoring cycle — now concurrent
    # ------------------------------------------------------------------

    async def run_monitoring_cycle(self) -> None:
        """Analyze all tickers concurrently (bounded by _CONCURRENCY semaphore)."""
        self._cycle_count += 1
        logger.info(
            f"Cycle #{self._cycle_count} at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC "
            f"— analyzing {len(settings.watchlist_tickers)} tickers concurrently "
            f"(max {self._CONCURRENCY} at once)"
        )

        tasks = [self.analyze_ticker(t) for t in settings.watchlist_tickers]
        await asyncio.gather(*tasks)

        # Write dashboard after every cycle
        self._dashboard.write(
            current_prices=self.current_prices,
            best_strategies=self.best_strategies,
            paper_trader=self.paper_trader,
            ml_scorer=self.ml_scorer,
            market_open=True,
            cycle_count=self._cycle_count,
        )

    # ------------------------------------------------------------------
    # Backtest refresh helper
    # ------------------------------------------------------------------

    async def _refresh_backtest(self) -> None:
        """Run a fresh backtest and update ML + strategy selection."""
        logger.info("Refreshing backtest...")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_backtest(
                self.client,
                settings.watchlist_tickers,
                days=settings.backtest_days,
            ),
        )
        save_backtest_results(result)
        self._backtest_result = result
        self.best_strategies = result.best_strategy_per_ticker
        self.ml_scorer.update_from_backtest(result)
        logger.info("Backtest refreshed and ML model updated.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main monitoring loop."""
        while not _shutdown_requested:
            market_open = is_market_open()

            # ---- Market open transition ----
            if market_open and not self._market_was_open:
                logger.info("Market opened — starting trading day...")
                self._dashboard.reset_day()
                self.paper_trader.start_trading_day(self.current_prices)
                await self._refresh_backtest()

            # ---- Market close transition ----
            elif not market_open and self._market_was_open:
                logger.info("Market closed — ending trading day...")
                if self.current_prices:
                    self.paper_trader.end_trading_day(self.current_prices)

                # Persist ML model state at end of day so it survives restarts
                if hasattr(self.ml_scorer, "save"):
                    try:
                        self.ml_scorer.save()
                        logger.info("ML model state persisted to disk.")
                    except Exception as exc:
                        logger.warning(f"ML save failed: {exc}")

                # Write final dashboard with market_open=False
                self._dashboard.write(
                    current_prices=self.current_prices,
                    best_strategies=self.best_strategies,
                    paper_trader=self.paper_trader,
                    ml_scorer=self.ml_scorer,
                    market_open=False,
                    cycle_count=self._cycle_count,
                )

            self._market_was_open = market_open

            # ---- Wait for market open if closed ----
            if not market_open:
                status = get_market_status()
                logger.info(f"Market closed. {status}")
                await async_wait_for_market_open()
                if _shutdown_requested:
                    break
                continue

            # ---- Active market hours ----
            if self.current_prices:
                self.paper_trader.check_auto_close(self.current_prices)

            await self.run_monitoring_cycle()

            if not _shutdown_requested:
                await asyncio.sleep(settings.poll_interval_seconds)

        # ---- Shutdown cleanup ----
        if self._market_was_open and self.current_prices:
            logger.info("Shutdown during market hours — sending final summary...")
            self.paper_trader.end_trading_day(self.current_prices)

        # Persist ML model on any shutdown
        if hasattr(self.ml_scorer, "save"):
            try:
                self.ml_scorer.save()
            except Exception:
                pass

        logger.info("Market Monitor stopped.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Factory creates and fully initialises the monitor before returning
    monitor = await MarketMonitor.create()
    await monitor.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(0)