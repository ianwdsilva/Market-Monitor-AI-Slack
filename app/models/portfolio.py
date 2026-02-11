"""Portfolio models for paper trading simulation."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from app.models.schemas import SignalDirection


class Position(BaseModel):
    """An open position in the paper portfolio."""

    ticker: str
    direction: SignalDirection
    entry_price: float
    shares: float
    entry_time: datetime
    signal_confidence: float = 0.0
    strategy_name: str = ""

    @property
    def cost_basis(self) -> float:
        """Total cost of the position."""
        return self.entry_price * self.shares

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L at current price."""
        if self.direction == SignalDirection.BUY:
            return (current_price - self.entry_price) * self.shares
        else:
            # Short position
            return (self.entry_price - current_price) * self.shares

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.BUY:
            return (current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - current_price) / self.entry_price


class Trade(BaseModel):
    """A completed (closed) trade."""

    ticker: str
    direction: SignalDirection
    entry_price: float
    exit_price: float
    shares: float
    entry_time: datetime
    exit_time: datetime
    signal_confidence: float = 0.0
    strategy_name: str = ""

    @property
    def pnl(self) -> float:
        """Realized P&L of the trade."""
        if self.direction == SignalDirection.BUY:
            return (self.exit_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - self.exit_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        """Realized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        if self.direction == SignalDirection.BUY:
            return (self.exit_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.exit_price) / self.entry_price

    @property
    def is_winner(self) -> bool:
        """Whether the trade was profitable."""
        return self.pnl > 0

    @property
    def hold_duration_minutes(self) -> float:
        """Duration of the trade in minutes."""
        return (self.exit_time - self.entry_time).total_seconds() / 60


class Portfolio(BaseModel):
    """Paper trading portfolio state."""

    cash: float = Field(default=100_000.0, description="Available cash")
    initial_capital: float = Field(default=100_000.0, description="Starting capital")
    positions: Dict[str, Position] = Field(default_factory=dict)
    closed_trades: List[Trade] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_realized_pnl(self) -> float:
        """Total realized P&L from closed trades."""
        return sum(t.pnl for t in self.closed_trades)

    @property
    def total_trades(self) -> int:
        """Total number of closed trades."""
        return len(self.closed_trades)

    @property
    def winning_trades(self) -> int:
        """Number of winning trades."""
        return sum(1 for t in self.closed_trades if t.is_winner)

    @property
    def win_rate(self) -> float:
        """Win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_win(self) -> float:
        """Average winning trade P&L."""
        winners = [t.pnl for t in self.closed_trades if t.is_winner]
        return sum(winners) / len(winners) if winners else 0.0

    @property
    def avg_loss(self) -> float:
        """Average losing trade P&L (as positive number)."""
        losers = [abs(t.pnl) for t in self.closed_trades if not t.is_winner]
        return sum(losers) / len(losers) if losers else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss."""
        gross_profit = sum(t.pnl for t in self.closed_trades if t.is_winner)
        gross_loss = sum(abs(t.pnl) for t in self.closed_trades if not t.is_winner)
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized P&L given current prices."""
        total = 0.0
        for ticker, position in self.positions.items():
            if ticker in prices:
                total += position.unrealized_pnl(prices[ticker])
        return total

    def total_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total portfolio value (cash + positions)."""
        return self.cash + sum(
            position.shares * prices.get(ticker, position.entry_price)
            for ticker, position in self.positions.items()
        )

    def total_return_pct(self, prices: Dict[str, float]) -> float:
        """Calculate total return percentage."""
        equity = self.total_equity(prices)
        return (equity - self.initial_capital) / self.initial_capital


class DailySummary(BaseModel):
    """Daily performance summary for Slack notification."""

    date: datetime
    starting_equity: float
    ending_equity: float
    daily_pnl: float
    daily_return_pct: float
    trades_opened: int
    trades_closed: int
    winning_trades: int
    losing_trades: int
    total_positions: int
    cash_available: float
    total_return_pct: float
    cumulative_trades: int
    cumulative_win_rate: float
