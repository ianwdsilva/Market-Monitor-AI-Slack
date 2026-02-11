"""Paper trading simulator with virtual portfolio management.

Executes virtual trades based on signals, tracks P&L, and sends
notifications about trade activity.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from app.models.portfolio import DailySummary, Portfolio, Position, Trade
from app.models.schemas import Signal, SignalDirection

logger = logging.getLogger(__name__)


class PaperTrader:
    """Paper trading simulator."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size_pct: float = 0.10,  # 10% of portfolio per trade
        max_positions: int = 10,
        auto_close_hours: float = 4.0,
        data_dir: str = "data",
        webhook_url: Optional[str] = None,
    ):
        """Initialize paper trader.

        Args:
            initial_capital: Starting cash amount
            position_size_pct: Percentage of portfolio per trade
            max_positions: Maximum concurrent positions
            auto_close_hours: Auto-close positions after this many hours
            data_dir: Directory to persist portfolio state
            webhook_url: Slack webhook for trade notifications
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.auto_close_hours = auto_close_hours
        self.data_dir = data_dir
        self.webhook_url = webhook_url
        self.portfolio_file = os.path.join(data_dir, "paper_portfolio.json")

        # Track daily stats
        self._day_start_equity: Optional[float] = None
        self._trades_opened_today: int = 0
        self._trades_closed_today: int = 0
        self._winning_today: int = 0
        self._losing_today: int = 0

        # Load or create portfolio
        self.portfolio = self._load_portfolio()

    def _load_portfolio(self) -> Portfolio:
        """Load portfolio from disk or create new one."""
        os.makedirs(self.data_dir, exist_ok=True)

        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, "r") as f:
                    data = json.load(f)
                portfolio = Portfolio(**data)
                logger.info(
                    f"Loaded portfolio: ${portfolio.cash:,.2f} cash, "
                    f"{len(portfolio.positions)} positions, "
                    f"{portfolio.total_trades} closed trades"
                )
                return portfolio
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")

        # Create new portfolio
        portfolio = Portfolio(
            cash=self.initial_capital,
            initial_capital=self.initial_capital,
            created_at=datetime.utcnow(),
        )
        logger.info(f"Created new portfolio with ${self.initial_capital:,.2f}")
        return portfolio

    def _save_portfolio(self):
        """Save portfolio state to disk."""
        self.portfolio.last_updated = datetime.utcnow()
        data = self.portfolio.model_dump(mode="json")

        with open(self.portfolio_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def process_signal(
        self,
        signal: Signal,
        current_prices: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Process a trading signal and execute paper trade if appropriate.

        Args:
            signal: Trading signal to process
            current_prices: Current prices for all tickers (for equity calc)

        Returns:
            Action taken (opened/closed/skipped) or None
        """
        ticker = signal.ticker.upper()

        if current_prices is None:
            current_prices = {ticker: signal.current_price}
        else:
            current_prices[ticker] = signal.current_price

        # Check for existing position
        has_position = ticker in self.portfolio.positions

        if signal.direction == SignalDirection.BUY:
            if has_position:
                # Already have a position - check if it's a SELL position we should close
                position = self.portfolio.positions[ticker]
                if position.direction == SignalDirection.SELL:
                    return self._close_position(ticker, signal.current_price, "Opposite signal")
                return None  # Already have a BUY position

            return self._open_position(signal, current_prices)

        else:  # SELL signal
            if has_position:
                position = self.portfolio.positions[ticker]
                if position.direction == SignalDirection.BUY:
                    return self._close_position(ticker, signal.current_price, "Opposite signal")
                return None  # Already have a SELL position

            return self._open_position(signal, current_prices)

    def _open_position(
        self,
        signal: Signal,
        current_prices: Dict[str, float],
    ) -> Optional[str]:
        """Open a new position."""
        ticker = signal.ticker.upper()

        # Check max positions
        if len(self.portfolio.positions) >= self.max_positions:
            logger.debug(f"Max positions ({self.max_positions}) reached, skipping {ticker}")
            return None

        # Calculate position size
        equity = self.portfolio.total_equity(current_prices)
        position_value = equity * self.position_size_pct

        # Check if we have enough cash
        if position_value > self.portfolio.cash:
            position_value = self.portfolio.cash * 0.95  # Use 95% of available cash

        if position_value < 100:  # Minimum position size
            logger.debug(f"Insufficient cash for {ticker}")
            return None

        # Calculate shares
        shares = position_value / signal.current_price

        # Create position
        position = Position(
            ticker=ticker,
            direction=signal.direction,
            entry_price=signal.current_price,
            shares=shares,
            entry_time=datetime.utcnow(),
            signal_confidence=signal.confidence,
            strategy_name=signal.strategy_name,
        )

        # Update portfolio
        self.portfolio.positions[ticker] = position
        self.portfolio.cash -= position.cost_basis
        self._trades_opened_today += 1

        self._save_portfolio()

        # Send notification
        action = "BUY" if signal.direction == SignalDirection.BUY else "SHORT"
        message = (
            f"📈 Opened {action} position: {ticker}\n"
            f"• Shares: {shares:.2f}\n"
            f"• Price: ${signal.current_price:.2f}\n"
            f"• Value: ${position.cost_basis:,.2f}\n"
            f"• Strategy: {signal.strategy_name}\n"
            f"• Confidence: {signal.confidence:.0%}"
        )
        self._send_notification(message, color="#3b82f6")  # Blue

        logger.info(f"Opened {action} position: {ticker} {shares:.2f} shares @ ${signal.current_price:.2f}")
        return f"opened_{action.lower()}"

    def _close_position(
        self,
        ticker: str,
        exit_price: float,
        reason: str = "",
    ) -> str:
        """Close an existing position."""
        ticker = ticker.upper()

        if ticker not in self.portfolio.positions:
            return "no_position"

        position = self.portfolio.positions[ticker]

        # Create trade record
        trade = Trade(
            ticker=ticker,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            shares=position.shares,
            entry_time=position.entry_time,
            exit_time=datetime.utcnow(),
            signal_confidence=position.signal_confidence,
            strategy_name=position.strategy_name,
        )

        # Update portfolio
        exit_value = exit_price * position.shares
        self.portfolio.cash += exit_value
        del self.portfolio.positions[ticker]
        self.portfolio.closed_trades.append(trade)

        # Track daily stats
        self._trades_closed_today += 1
        if trade.is_winner:
            self._winning_today += 1
        else:
            self._losing_today += 1

        self._save_portfolio()

        # Send notification
        pnl = trade.pnl
        pnl_pct = trade.pnl_pct * 100
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        pnl_sign = "+" if pnl >= 0 else ""

        action = "LONG" if position.direction == SignalDirection.BUY else "SHORT"
        message = (
            f"{pnl_emoji} Closed {action} position: {ticker}\n"
            f"• Entry: ${position.entry_price:.2f} → Exit: ${exit_price:.2f}\n"
            f"• P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\n"
            f"• Hold time: {trade.hold_duration_minutes:.0f} min\n"
            f"• Reason: {reason}"
        )

        color = "#22c55e" if pnl >= 0 else "#ef4444"  # Green or red
        self._send_notification(message, color=color)

        logger.info(
            f"Closed {action} position: {ticker} @ ${exit_price:.2f}, "
            f"P&L: {pnl_sign}${pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)"
        )
        return "closed"

    def check_auto_close(self, current_prices: Dict[str, float]):
        """Check for positions that should be auto-closed due to time limit."""
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=self.auto_close_hours)

        positions_to_close = []
        for ticker, position in self.portfolio.positions.items():
            if position.entry_time < cutoff:
                positions_to_close.append(ticker)

        for ticker in positions_to_close:
            price = current_prices.get(ticker)
            if price:
                self._close_position(ticker, price, f"Auto-close after {self.auto_close_hours}h")

    def start_trading_day(self, current_prices: Dict[str, float]):
        """Initialize daily tracking at market open."""
        self._day_start_equity = self.portfolio.total_equity(current_prices)
        self._trades_opened_today = 0
        self._trades_closed_today = 0
        self._winning_today = 0
        self._losing_today = 0
        logger.info(f"Started trading day with equity: ${self._day_start_equity:,.2f}")

    def end_trading_day(self, current_prices: Dict[str, float]):
        """Send daily summary at market close."""
        ending_equity = self.portfolio.total_equity(current_prices)
        starting_equity = self._day_start_equity or self.portfolio.initial_capital

        daily_pnl = ending_equity - starting_equity
        daily_return_pct = daily_pnl / starting_equity if starting_equity > 0 else 0

        summary = DailySummary(
            date=datetime.utcnow(),
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            daily_pnl=daily_pnl,
            daily_return_pct=daily_return_pct,
            trades_opened=self._trades_opened_today,
            trades_closed=self._trades_closed_today,
            winning_trades=self._winning_today,
            losing_trades=self._losing_today,
            total_positions=len(self.portfolio.positions),
            cash_available=self.portfolio.cash,
            total_return_pct=self.portfolio.total_return_pct(current_prices),
            cumulative_trades=self.portfolio.total_trades,
            cumulative_win_rate=self.portfolio.win_rate,
        )

        self._send_daily_summary(summary)
        return summary

    def _send_daily_summary(self, summary: DailySummary):
        """Send daily summary to Slack."""
        daily_sign = "+" if summary.daily_pnl >= 0 else ""
        total_sign = "+" if summary.total_return_pct >= 0 else ""
        daily_emoji = "📈" if summary.daily_pnl >= 0 else "📉"

        message = {
            "attachments": [
                {
                    "color": "#6366f1",  # Indigo
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{daily_emoji} Paper Trading Daily Summary",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Today's P&L:*\n{daily_sign}${summary.daily_pnl:,.2f} ({daily_sign}{summary.daily_return_pct:.2%})",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Portfolio Value:*\n${summary.ending_equity:,.2f}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Total Return:*\n{total_sign}{summary.total_return_pct:.2%}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Cash Available:*\n${summary.cash_available:,.2f}",
                                },
                            ],
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Today's Trades:*\n{summary.trades_opened} opened, {summary.trades_closed} closed",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Today's W/L:*\n{summary.winning_trades}W / {summary.losing_trades}L",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Open Positions:*\n{summary.total_positions}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Cumulative:*\n{summary.cumulative_trades} trades, {summary.cumulative_win_rate:.1%} win rate",
                                },
                            ],
                        },
                    ],
                }
            ]
        }

        self._send_slack_message(message)

    def _send_notification(self, text: str, color: str = "#6366f1"):
        """Send a simple notification to Slack."""
        message = {
            "attachments": [
                {
                    "color": color,
                    "text": text,
                }
            ]
        }
        self._send_slack_message(message)

    def _send_slack_message(self, payload: dict) -> bool:
        """Send message to Slack webhook."""
        if not self.webhook_url:
            return False

        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False

    def get_status(self, current_prices: Dict[str, float]) -> dict:
        """Get current portfolio status."""
        return {
            "cash": self.portfolio.cash,
            "equity": self.portfolio.total_equity(current_prices),
            "unrealized_pnl": self.portfolio.unrealized_pnl(current_prices),
            "realized_pnl": self.portfolio.total_realized_pnl,
            "total_return_pct": self.portfolio.total_return_pct(current_prices),
            "positions": len(self.portfolio.positions),
            "total_trades": self.portfolio.total_trades,
            "win_rate": self.portfolio.win_rate,
            "profit_factor": self.portfolio.profit_factor,
        }


# Singleton instance
_paper_trader: Optional[PaperTrader] = None


def get_paper_trader() -> PaperTrader:
    """Get or create the paper trader singleton."""
    global _paper_trader
    if _paper_trader is None:
        from app.config import settings

        _paper_trader = PaperTrader(
            initial_capital=100_000.0,
            position_size_pct=0.10,
            max_positions=10,
            auto_close_hours=4.0,
            webhook_url=settings.slack_webhook_url,
        )
    return _paper_trader
