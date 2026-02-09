"""NYSE market hours detection and scheduling.

Handles:
- Market open/close times (9:30 AM - 4:00 PM ET)
- Weekend detection
- Holiday calendar
- Time until next open
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Optional

import pytz

logger = logging.getLogger(__name__)

# NYSE timezone
ET = pytz.timezone("America/New_York")

# Market hours (Eastern Time)
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# 2024-2026 NYSE holidays (market closed)
NYSE_HOLIDAYS = {
    # 2024
    datetime(2024, 1, 1),   # New Year's Day
    datetime(2024, 1, 15),  # MLK Day
    datetime(2024, 2, 19),  # Presidents Day
    datetime(2024, 3, 29),  # Good Friday
    datetime(2024, 5, 27),  # Memorial Day
    datetime(2024, 6, 19),  # Juneteenth
    datetime(2024, 7, 4),   # Independence Day
    datetime(2024, 9, 2),   # Labor Day
    datetime(2024, 11, 28), # Thanksgiving
    datetime(2024, 12, 25), # Christmas
    # 2025
    datetime(2025, 1, 1),   # New Year's Day
    datetime(2025, 1, 20),  # MLK Day
    datetime(2025, 2, 17),  # Presidents Day
    datetime(2025, 4, 18),  # Good Friday
    datetime(2025, 5, 26),  # Memorial Day
    datetime(2025, 6, 19),  # Juneteenth
    datetime(2025, 7, 4),   # Independence Day
    datetime(2025, 9, 1),   # Labor Day
    datetime(2025, 11, 27), # Thanksgiving
    datetime(2025, 12, 25), # Christmas
    # 2026
    datetime(2026, 1, 1),   # New Year's Day
    datetime(2026, 1, 19),  # MLK Day
    datetime(2026, 2, 16),  # Presidents Day
    datetime(2026, 4, 3),   # Good Friday
    datetime(2026, 5, 25),  # Memorial Day
    datetime(2026, 6, 19),  # Juneteenth
    datetime(2026, 7, 3),   # Independence Day (observed)
    datetime(2026, 9, 7),   # Labor Day
    datetime(2026, 11, 26), # Thanksgiving
    datetime(2026, 12, 25), # Christmas
}


def get_et_now() -> datetime:
    """Get current time in Eastern timezone."""
    return datetime.now(ET)


def is_holiday(dt: Optional[datetime] = None) -> bool:
    """Check if date is an NYSE holiday.

    Args:
        dt: Datetime to check (default: now in ET)

    Returns:
        True if holiday, False otherwise
    """
    if dt is None:
        dt = get_et_now()

    # Convert to date for comparison
    if hasattr(dt, 'date'):
        check_date = datetime(dt.year, dt.month, dt.day)
    else:
        check_date = dt

    return check_date in NYSE_HOLIDAYS


def is_weekend(dt: Optional[datetime] = None) -> bool:
    """Check if date is a weekend.

    Args:
        dt: Datetime to check (default: now in ET)

    Returns:
        True if Saturday or Sunday
    """
    if dt is None:
        dt = get_et_now()
    return dt.weekday() >= 5  # Saturday = 5, Sunday = 6


def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if NYSE market is currently open.

    Args:
        dt: Datetime to check (default: now in ET)

    Returns:
        True if market is open
    """
    if dt is None:
        dt = get_et_now()

    # Check weekend
    if is_weekend(dt):
        return False

    # Check holiday
    if is_holiday(dt):
        return False

    # Check time
    current_time = dt.time()
    return MARKET_OPEN <= current_time < MARKET_CLOSE


def time_until_open(dt: Optional[datetime] = None) -> timedelta:
    """Calculate time until market opens.

    Args:
        dt: Datetime to check from (default: now in ET)

    Returns:
        Timedelta until market opens (zero if already open)
    """
    if dt is None:
        dt = get_et_now()

    if is_market_open(dt):
        return timedelta(0)

    # Find next market open
    next_open = _find_next_open(dt)
    return next_open - dt


def _find_next_open(dt: datetime) -> datetime:
    """Find the next market open datetime.

    Args:
        dt: Starting datetime

    Returns:
        Datetime of next market open
    """
    # Start with today's open
    candidate = dt.replace(
        hour=MARKET_OPEN.hour,
        minute=MARKET_OPEN.minute,
        second=0,
        microsecond=0,
    )

    # If we're past today's open, start tomorrow
    if dt.time() >= MARKET_OPEN:
        candidate += timedelta(days=1)

    # Find next valid trading day
    while is_weekend(candidate) or is_holiday(candidate):
        candidate += timedelta(days=1)

    return candidate


def time_until_close(dt: Optional[datetime] = None) -> Optional[timedelta]:
    """Calculate time until market closes.

    Args:
        dt: Datetime to check from (default: now in ET)

    Returns:
        Timedelta until market closes, or None if market is closed
    """
    if dt is None:
        dt = get_et_now()

    if not is_market_open(dt):
        return None

    close_dt = dt.replace(
        hour=MARKET_CLOSE.hour,
        minute=MARKET_CLOSE.minute,
        second=0,
        microsecond=0,
    )
    return close_dt - dt


def get_market_status() -> dict:
    """Get current market status information.

    Returns:
        Dict with status, times, and countdown
    """
    now = get_et_now()
    is_open = is_market_open(now)

    status = {
        "is_open": is_open,
        "current_time_et": now.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "is_weekend": is_weekend(now),
        "is_holiday": is_holiday(now),
    }

    if is_open:
        until_close = time_until_close(now)
        if until_close:
            hours, remainder = divmod(until_close.seconds, 3600)
            minutes = remainder // 60
            status["time_until_close"] = f"{hours}h {minutes}m"
    else:
        until_open = time_until_open(now)
        hours, remainder = divmod(int(until_open.total_seconds()), 3600)
        minutes = remainder // 60
        status["time_until_open"] = f"{hours}h {minutes}m"

    return status


def wait_for_market_open():
    """Block until market opens. Use in sync context."""
    import time as time_module

    while not is_market_open():
        until_open = time_until_open()
        logger.info(f"Market closed. Opens in {until_open}")

        # Sleep for shorter intervals near open time
        if until_open.total_seconds() < 300:  # < 5 min
            time_module.sleep(30)
        elif until_open.total_seconds() < 3600:  # < 1 hour
            time_module.sleep(60)
        else:
            time_module.sleep(300)  # 5 min


async def async_wait_for_market_open():
    """Async version: wait until market opens."""
    import asyncio

    while not is_market_open():
        until_open = time_until_open()
        logger.info(f"Market closed. Opens in {until_open}")

        if until_open.total_seconds() < 300:
            await asyncio.sleep(30)
        elif until_open.total_seconds() < 3600:
            await asyncio.sleep(60)
        else:
            await asyncio.sleep(300)
