"""Configuration management with pydantic and stdlib fallback."""

from __future__ import annotations

import os
from importlib import import_module
from importlib.util import find_spec
from functools import lru_cache
from typing import List

_HAS_PYDANTIC = find_spec("pydantic") is not None and find_spec("pydantic_settings") is not None


if _HAS_PYDANTIC:
    pydantic = import_module("pydantic")
    pydantic_settings = import_module("pydantic_settings")
    Field = pydantic.Field
    field_validator = pydantic.field_validator
    BaseSettings = pydantic_settings.BaseSettings
    SettingsConfigDict = pydantic_settings.SettingsConfigDict

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
        paper_trading_max_trade_value: float = Field(
            default=5_000.0,
            ge=100.0,
            description="Maximum dollar value for any single paper trade",
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

else:

    def _load_env_file(path: str = ".env") -> None:
        """Load .env values into process env without overriding existing values."""
        if not os.path.exists(path):
            return

        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


    def _env_value(name: str, default: str | None = None, required: bool = False) -> str:
        """Read an environment value with optional default/required semantics."""
        value = os.getenv(name)
        if value is None:
            value = os.getenv(name.lower())
        if value is None:
            value = default
        if required and (value is None or value == ""):
            raise ValueError(f"Missing required environment variable: {name}")
        return value if value is not None else ""


    class Settings:
        """Application settings loaded from environment variables."""

        def __init__(self):
            _load_env_file(".env")

            self.slack_webhook_url: str = _env_value("SLACK_WEBHOOK_URL", required=True)
            self.watchlist: str = self._normalize_watchlist(
                _env_value("WATCHLIST", default="AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,DIS")
            )

            self.poll_interval_seconds: int = self._parse_int(
                _env_value("POLL_INTERVAL_SECONDS", default="60"), "POLL_INTERVAL_SECONDS", 10, 300
            )
            self.alert_cooldown_minutes: int = self._parse_int(
                _env_value("ALERT_COOLDOWN_MINUTES", default="60"), "ALERT_COOLDOWN_MINUTES", 1, 1440
            )
            self.signal_confidence_threshold: float = self._parse_float(
                _env_value("SIGNAL_CONFIDENCE_THRESHOLD", default="0.75"),
                "SIGNAL_CONFIDENCE_THRESHOLD",
                0.0,
                1.0,
            )
            self.min_confirmations: int = self._parse_int(
                _env_value("MIN_CONFIRMATIONS", default="3"), "MIN_CONFIRMATIONS", 1, 10
            )
            self.backtest_days: int = self._parse_int(
                _env_value("BACKTEST_DAYS", default="30"), "BACKTEST_DAYS", 7, 365
            )
            self.paper_trading_starting_balance: float = self._parse_float(
                _env_value("PAPER_TRADING_STARTING_BALANCE", default="100000.0"),
                "PAPER_TRADING_STARTING_BALANCE",
                100.0,
                None,
            )
            self.paper_trading_max_trade_value: float = self._parse_float(
                _env_value("PAPER_TRADING_MAX_TRADE_VALUE", default="5000.0"),
                "PAPER_TRADING_MAX_TRADE_VALUE",
                100.0,
                None,
            )
            self.log_level: str = _env_value("LOG_LEVEL", default="INFO").upper()

        @staticmethod
        def _normalize_watchlist(value: str) -> str:
            tickers = [t.strip().upper() for t in value.split(",") if t.strip()]
            if not tickers:
                raise ValueError("Watchlist must contain at least one ticker")
            return ",".join(tickers)

        @staticmethod
        def _parse_int(value: str, name: str, minimum: int, maximum: int | None) -> int:
            try:
                parsed = int(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be an integer") from e

            if parsed < minimum or (maximum is not None and parsed > maximum):
                if maximum is None:
                    raise ValueError(f"{name} must be >= {minimum}")
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
            return parsed

        @staticmethod
        def _parse_float(value: str, name: str, minimum: float, maximum: float | None) -> float:
            try:
                parsed = float(value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be a number") from e

            if parsed < minimum or (maximum is not None and parsed > maximum):
                if maximum is None:
                    raise ValueError(f"{name} must be >= {minimum}")
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
            return parsed

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
