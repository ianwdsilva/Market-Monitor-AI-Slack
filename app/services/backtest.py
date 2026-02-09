"""Backtesting engine for strategy evaluation.

Evaluates strategy performance on historical data and selects
the best strategy per ticker.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import (
    BacktestResult,
    SignalDirection,
    StrategyPerformance,
    TradeOutcome,
)
from app.services.indicators import build_indicator_frame
from app.services.patterns import detect_all_patterns
from app.services.strategies import Strategy, get_all_strategies

logger = logging.getLogger(__name__)


def backtest_strategy(
    strategy: Strategy,
    ticker: str,
    df: pd.DataFrame,
    eval_windows: List[int] = [3, 12, 48],  # 15min, 1hr, 4hr in 5-min bars
) -> StrategyPerformance:
    """Run backtest for a single strategy on a single ticker.

    Args:
        strategy: Strategy instance to test
        ticker: Stock symbol
        df: DataFrame with indicators (from build_indicator_frame)
        eval_windows: Number of bars for each evaluation window

    Returns:
        StrategyPerformance with metrics
    """
    if df.empty or len(df) < max(eval_windows) + 50:
        return StrategyPerformance(
            strategy_name=strategy.name,
            ticker=ticker,
            total_signals=0,
        )

    outcomes: List[TradeOutcome] = []
    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"

    # Walk through data, checking for signals at each point
    # Start after warmup period, end before eval window
    start_idx = 50
    end_idx = len(df) - max(eval_windows)

    for i in range(start_idx, end_idx):
        # Get data up to current point
        current_df = df.iloc[: i + 1]

        # Check for signal
        patterns = detect_all_patterns(current_df)
        signal = strategy.check_signal(current_df, ticker, patterns)

        if signal is None:
            continue

        entry_time = current_df.index[-1]
        entry_price = current_df[price_col].iloc[-1]

        # Calculate outcomes at different time horizons
        outcome = TradeOutcome(
            entry_time=entry_time,
            entry_price=entry_price,
            direction=signal.direction,
        )

        # 15-min outcome (3 bars)
        if i + eval_windows[0] < len(df):
            exit_idx = i + eval_windows[0]
            exit_price = df[price_col].iloc[exit_idx]
            outcome.exit_time_15m = df.index[exit_idx]
            outcome.exit_price_15m = exit_price
            outcome.return_15m = (exit_price - entry_price) / entry_price

        # 1-hour outcome (12 bars)
        if i + eval_windows[1] < len(df):
            exit_idx = i + eval_windows[1]
            exit_price = df[price_col].iloc[exit_idx]
            outcome.exit_time_1h = df.index[exit_idx]
            outcome.exit_price_1h = exit_price
            outcome.return_1h = (exit_price - entry_price) / entry_price

        # 4-hour outcome (48 bars)
        if i + eval_windows[2] < len(df):
            exit_idx = i + eval_windows[2]
            exit_price = df[price_col].iloc[exit_idx]
            outcome.exit_time_4h = df.index[exit_idx]
            outcome.exit_price_4h = exit_price
            outcome.return_4h = (exit_price - entry_price) / entry_price

        outcomes.append(outcome)

    # Calculate performance metrics
    return _calculate_performance(strategy.name, ticker, outcomes)


def _calculate_performance(
    strategy_name: str, ticker: str, outcomes: List[TradeOutcome]
) -> StrategyPerformance:
    """Calculate performance metrics from trade outcomes."""
    if not outcomes:
        return StrategyPerformance(
            strategy_name=strategy_name,
            ticker=ticker,
            total_signals=0,
        )

    # Win rates (use == True instead of 'is True' to handle numpy bools)
    winners_15m = [o for o in outcomes if o.is_winner_15m == True]
    winners_1h = [o for o in outcomes if o.is_winner_1h == True]

    valid_15m = [o for o in outcomes if o.return_15m is not None]
    valid_1h = [o for o in outcomes if o.return_1h is not None]
    valid_4h = [o for o in outcomes if o.return_4h is not None]

    win_rate_15m = len(winners_15m) / max(1, len(valid_15m))
    win_rate_1h = len(winners_1h) / max(1, len(valid_1h))

    # Average returns (adjusted for direction)
    returns_15m = []
    returns_1h = []
    returns_4h = []

    for o in outcomes:
        direction_mult = 1 if o.direction == SignalDirection.BUY else -1
        if o.return_15m is not None:
            returns_15m.append(o.return_15m * direction_mult)
        if o.return_1h is not None:
            returns_1h.append(o.return_1h * direction_mult)
        if o.return_4h is not None:
            returns_4h.append(o.return_4h * direction_mult)

    avg_return_15m = np.mean(returns_15m) if returns_15m else 0.0
    avg_return_1h = np.mean(returns_1h) if returns_1h else 0.0
    avg_return_4h = np.mean(returns_4h) if returns_4h else 0.0

    # Profit factor (gross profit / gross loss)
    profits = [r for r in returns_1h if r > 0]
    losses = [abs(r) for r in returns_1h if r < 0]
    gross_profit = sum(profits) if profits else 0
    gross_loss = sum(losses) if losses else 0.001  # Avoid division by zero
    profit_factor = gross_profit / gross_loss

    # Max consecutive losses
    max_consec_losses = 0
    current_streak = 0
    for o in outcomes:
        if o.is_winner_1h == False:
            current_streak += 1
            max_consec_losses = max(max_consec_losses, current_streak)
        else:
            current_streak = 0

    # Sharpe ratio (annualized, assuming 78 5-min bars per day)
    if returns_1h and len(returns_1h) > 1:
        mean_return = np.mean(returns_1h)
        std_return = np.std(returns_1h)
        if std_return > 0:
            # Annualize: ~6.5 hours/day * 252 days / 12 bars per hour
            annualization = np.sqrt(252 * 6.5)
            sharpe_ratio = (mean_return / std_return) * annualization
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    return StrategyPerformance(
        strategy_name=strategy_name,
        ticker=ticker,
        total_signals=len(outcomes),
        win_rate_15m=win_rate_15m,
        win_rate_1h=win_rate_1h,
        win_rate_4h=len([o for o in outcomes if o.return_4h is not None and (
            (o.direction == SignalDirection.BUY and o.return_4h > 0) or
            (o.direction == SignalDirection.SELL and o.return_4h < 0)
        )]) / max(1, len(valid_4h)),
        avg_return_15m=avg_return_15m,
        avg_return_1h=avg_return_1h,
        avg_return_4h=avg_return_4h,
        profit_factor=profit_factor,
        max_consecutive_losses=max_consec_losses,
        sharpe_ratio=sharpe_ratio,
        outcomes=outcomes,
    )


def run_backtest(
    client,
    tickers: List[str],
    strategies: Optional[List[Strategy]] = None,
    days: int = 30,
) -> BacktestResult:
    """Run backtest for all strategies on all tickers.

    Args:
        client: AlpacaClient instance
        tickers: List of stock symbols
        strategies: List of strategies to test (default: all)
        days: Number of days of historical data

    Returns:
        BacktestResult with performance metrics and best strategy per ticker
    """
    if strategies is None:
        strategies = get_all_strategies()

    result = BacktestResult(
        run_timestamp=datetime.utcnow(),
        backtest_days=days,
        strategies={},
        best_strategy_per_ticker={},
    )

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    for ticker in tickers:
        logger.info(f"Backtesting {ticker}...")

        # Fetch historical data
        df = client.get_historical_bars(ticker, start, end, timeframe="5Min")
        if df.empty:
            logger.warning(f"No data for {ticker}, skipping")
            continue

        # Build indicators
        try:
            indicator_df = build_indicator_frame(df)
        except Exception as e:
            logger.error(f"Error building indicators for {ticker}: {e}")
            continue

        if indicator_df.empty or len(indicator_df) < 100:
            logger.warning(f"Insufficient data for {ticker} after indicators")
            continue

        # Test each strategy
        best_performance: Optional[StrategyPerformance] = None
        best_score = -float("inf")

        for strategy in strategies:
            performance = backtest_strategy(strategy, ticker, indicator_df)

            # Store performance
            if strategy.name not in result.strategies:
                result.strategies[strategy.name] = {}
            result.strategies[strategy.name][ticker] = performance

            # Score strategy (weighted combination of metrics)
            # Higher is better
            if performance.total_signals >= 5:  # Minimum signals for validity
                score = (
                    performance.win_rate_1h * 0.3
                    + performance.profit_factor * 0.3
                    + performance.sharpe_ratio * 0.2
                    + (1 - performance.max_consecutive_losses / 10) * 0.2
                )

                if score > best_score:
                    best_score = score
                    best_performance = performance

            logger.info(
                f"  {strategy.name}: {performance.total_signals} signals, "
                f"win_rate={performance.win_rate_1h:.1%}, "
                f"profit_factor={performance.profit_factor:.2f}"
            )

        # Record best strategy for this ticker
        if best_performance:
            result.best_strategy_per_ticker[ticker] = best_performance.strategy_name
            logger.info(f"  Best for {ticker}: {best_performance.strategy_name}")
        else:
            # Default to ensemble if no clear winner
            result.best_strategy_per_ticker[ticker] = "ensemble"

    return result


def save_backtest_results(result: BacktestResult, path: str = "data/backtest_results"):
    """Save backtest results to JSON file."""
    os.makedirs(path, exist_ok=True)
    filename = f"backtest_{result.run_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(path, filename)

    # Convert to serializable format
    data = result.model_dump(mode="json")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    logger.info(f"Saved backtest results to {filepath}")
    return filepath


def load_latest_backtest(path: str = "data/backtest_results") -> Optional[BacktestResult]:
    """Load the most recent backtest results."""
    if not os.path.exists(path):
        return None

    files = sorted(
        [f for f in os.listdir(path) if f.startswith("backtest_") and f.endswith(".json")],
        reverse=True,
    )

    if not files:
        return None

    filepath = os.path.join(path, files[0])
    with open(filepath, "r") as f:
        data = json.load(f)

    return BacktestResult(**data)
