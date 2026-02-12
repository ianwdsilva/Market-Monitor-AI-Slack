"""Configuration management using Pydantic Settings."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Slack
    slack_webhook_url: str = Field(..., description="Slack webhook URL for alerts")

    # Watchlist
    watchlist: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,DIS",
        description="Comma-separated list of tickers to monitor",
    )

    # Monitoring
    poll_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Seconds between each monitoring cycle",
    )
    alert_cooldown_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Minutes to wait before alerting on same ticker again",
    )

    # Strategy
    signal_confidence_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to trigger alert",
    )
    min_confirmations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Minimum indicator confirmations required",
    )
    backtest_days: int = Field(
        default=30,
        ge=7,
        le=365,
        description="Days of historical data for backtesting",
    )
    paper_trading_starting_balance: float = Field(
        default=100_000.0,
        ge=100.0,
        description="Initial capital for paper trading",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")

    @field_validator("watchlist")
    @classmethod
    def validate_watchlist(cls, v: str) -> str:
        """Ensure watchlist is non-empty."""
        tickers = [t.strip().upper() for t in v.split(",") if t.strip()]
        if not tickers:
            raise ValueError("Watchlist must contain at least one ticker")
        return ",".join(tickers)

    @property
    def watchlist_tickers(self) -> List[str]:
        """Return watchlist as a list of tickers."""
        return [t.strip().upper() for t in self.watchlist.split(",") if t.strip()]



@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience access
settings = get_settings()
