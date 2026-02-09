"""Slack alert system with cooldown management.

Sends trading signal alerts to Slack with per-ticker cooldowns
to avoid notification spam.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

import requests

from app.models.schemas import AlertRecord, Signal, SignalDirection

logger = logging.getLogger(__name__)


class AlertManager:
    """Manages Slack alerts with cooldown tracking."""

    def __init__(self, webhook_url: str, cooldown_minutes: int = 60):
        """Initialize the alert manager.

        Args:
            webhook_url: Slack webhook URL
            cooldown_minutes: Minutes to wait between alerts for same ticker
        """
        self.webhook_url = webhook_url
        self.cooldown_minutes = cooldown_minutes
        self._cooldowns: Dict[str, datetime] = {}
        self._alert_history: list[AlertRecord] = []

    def is_cooled_down(self, ticker: str) -> bool:
        """Check if ticker is past its cooldown period.

        Args:
            ticker: Stock symbol

        Returns:
            True if alert can be sent, False if in cooldown
        """
        ticker = ticker.upper()
        if ticker not in self._cooldowns:
            return True

        cooldown_expires = self._cooldowns[ticker] + timedelta(minutes=self.cooldown_minutes)
        return datetime.utcnow() >= cooldown_expires

    def send_alert(self, signal: Signal) -> bool:
        """Send a signal alert to Slack if not in cooldown.

        Args:
            signal: Trading signal to alert

        Returns:
            True if alert was sent, False if skipped or failed
        """
        ticker = signal.ticker.upper()

        # Check cooldown
        if not self.is_cooled_down(ticker):
            remaining = self._cooldowns[ticker] + timedelta(minutes=self.cooldown_minutes)
            remaining_mins = (remaining - datetime.utcnow()).total_seconds() / 60
            logger.debug(f"Skipping alert for {ticker}, {remaining_mins:.1f} min cooldown remaining")
            return False

        # Format and send message
        message = self._format_message(signal)
        success = self._send_to_slack(message)

        # Record alert
        record = AlertRecord(
            ticker=ticker,
            signal=signal,
            sent_at=datetime.utcnow(),
            slack_response_ok=success,
        )
        self._alert_history.append(record)

        # Update cooldown
        if success:
            self._cooldowns[ticker] = datetime.utcnow()
            logger.info(f"Alert sent for {ticker}: {signal.direction.value}")

        return success

    def _format_message(self, signal: Signal) -> dict:
        """Format signal as Slack message payload.

        Args:
            signal: Trading signal

        Returns:
            Slack message payload dict
        """
        # Emoji and color based on direction
        if signal.direction == SignalDirection.BUY:
            emoji = "🟢"
            color = "#34d399"  # Green
        else:
            emoji = "🔴"
            color = "#f87171"  # Red

        # Format indicators
        indicator_lines = []
        for key, value in signal.indicators.items():
            if value is not None:
                if isinstance(value, float):
                    indicator_lines.append(f"• {key}: {value:.2f}")
                else:
                    indicator_lines.append(f"• {key}: {value}")

        indicators_text = "\n".join(indicator_lines) if indicator_lines else "N/A"

        # Format patterns
        pattern_lines = []
        for pattern in signal.patterns[:3]:  # Limit to 3 patterns
            pattern_lines.append(f"• {pattern.description}")
        patterns_text = "\n".join(pattern_lines) if pattern_lines else "None detected"

        # Build Slack message with blocks
        return {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} {signal.direction.value} Signal: {signal.ticker}",
                                "emoji": True,
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Strategy:*\n{signal.strategy_name}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Confidence:*\n{signal.confidence:.0%}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Price:*\n${signal.current_price:.2f}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Confirmations:*\n{signal.confirmations}",
                                },
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Reason:*\n{signal.reason}",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Indicators:*\n{indicators_text}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Patterns:*\n{patterns_text}",
                                },
                            ],
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"⏰ {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC",
                                },
                            ],
                        },
                    ],
                }
            ]
        }

    def _send_to_slack(self, payload: dict) -> bool:
        """Send message to Slack webhook.

        Args:
            payload: Slack message payload

        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def clear_cooldown(self, ticker: str):
        """Clear cooldown for a specific ticker.

        Args:
            ticker: Stock symbol
        """
        ticker = ticker.upper()
        if ticker in self._cooldowns:
            del self._cooldowns[ticker]

    def clear_all_cooldowns(self):
        """Clear all cooldowns."""
        self._cooldowns.clear()

    def get_cooldown_status(self) -> Dict[str, float]:
        """Get remaining cooldown time for all tickers.

        Returns:
            Dict mapping ticker -> minutes remaining (0 if cooled down)
        """
        now = datetime.utcnow()
        status = {}
        for ticker, last_alert in self._cooldowns.items():
            expires = last_alert + timedelta(minutes=self.cooldown_minutes)
            remaining = (expires - now).total_seconds() / 60
            status[ticker] = max(0, remaining)
        return status

    def get_alert_history(self, limit: int = 50) -> list[AlertRecord]:
        """Get recent alert history.

        Args:
            limit: Max number of records to return

        Returns:
            List of AlertRecord, most recent first
        """
        return list(reversed(self._alert_history[-limit:]))


# Singleton instance (initialized lazily)
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create the alert manager singleton.

    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        from app.config import settings

        _alert_manager = AlertManager(
            webhook_url=settings.slack_webhook_url,
            cooldown_minutes=settings.alert_cooldown_minutes,
        )
    return _alert_manager
