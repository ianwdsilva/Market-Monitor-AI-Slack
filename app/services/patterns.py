"""Chart pattern detection for trading signals.

Detects:
- Golden/Death Cross (SMA 50/200)
- MACD Crossover
- RSI Divergence
- Bollinger Squeeze
- Support/Resistance levels
- VWAP Cross
- Parabolic SAR Flip
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd

from app.models.schemas import PatternSignal, PatternType


def detect_all_patterns(df: pd.DataFrame) -> List[PatternSignal]:
    """Detect all chart patterns in the indicator DataFrame.

    Args:
        df: DataFrame with indicators from build_indicator_frame()

    Returns:
        List of detected patterns at the most recent bar
    """
    if df.empty or len(df) < 2:
        return []

    patterns = []
    timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()
    close = df["AdjClose"].iloc[-1] if "AdjClose" in df.columns else df["Close"].iloc[-1]

    # Detect each pattern type
    pattern_funcs = [
        detect_golden_death_cross,
        detect_macd_crossover,
        detect_rsi_divergence,
        detect_bollinger_squeeze,
        detect_support_resistance,
        detect_vwap_cross,
        detect_sar_flip,
    ]

    for func in pattern_funcs:
        result = func(df, timestamp, close)
        if result:
            patterns.extend(result if isinstance(result, list) else [result])

    return patterns


def detect_golden_death_cross(
    df: pd.DataFrame, timestamp: datetime, close: float
) -> List[PatternSignal]:
    """Detect Golden Cross (SMA50 > SMA200) or Death Cross (SMA50 < SMA200)."""
    patterns = []

    if "SMA50" not in df.columns or "SMA200" not in df.columns:
        return patterns
    if len(df) < 2:
        return patterns

    sma50_curr = df["SMA50"].iloc[-1]
    sma50_prev = df["SMA50"].iloc[-2]
    sma200_curr = df["SMA200"].iloc[-1]
    sma200_prev = df["SMA200"].iloc[-2]

    # Golden Cross: SMA50 crosses above SMA200
    if sma50_prev <= sma200_prev and sma50_curr > sma200_curr:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.GOLDEN_CROSS,
                timestamp=timestamp,
                price=close,
                description="SMA50 crossed above SMA200 (bullish)",
                strength=0.85,
            )
        )

    # Death Cross: SMA50 crosses below SMA200
    if sma50_prev >= sma200_prev and sma50_curr < sma200_curr:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.DEATH_CROSS,
                timestamp=timestamp,
                price=close,
                description="SMA50 crossed below SMA200 (bearish)",
                strength=0.85,
            )
        )

    return patterns


def detect_macd_crossover(
    df: pd.DataFrame, timestamp: datetime, close: float
) -> List[PatternSignal]:
    """Detect MACD line crossing signal line."""
    patterns = []

    if "MACD" not in df.columns or "MACDSignal" not in df.columns:
        return patterns
    if len(df) < 2:
        return patterns

    macd_curr = df["MACD"].iloc[-1]
    macd_prev = df["MACD"].iloc[-2]
    signal_curr = df["MACDSignal"].iloc[-1]
    signal_prev = df["MACDSignal"].iloc[-2]

    # Bullish crossover: MACD crosses above signal
    if macd_prev <= signal_prev and macd_curr > signal_curr:
        # Stronger signal if both are below zero (early reversal)
        strength = 0.75 if macd_curr < 0 else 0.65
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.MACD_BULLISH_CROSS,
                timestamp=timestamp,
                price=close,
                description="MACD crossed above signal line (bullish momentum)",
                strength=strength,
            )
        )

    # Bearish crossover: MACD crosses below signal
    if macd_prev >= signal_prev and macd_curr < signal_curr:
        strength = 0.75 if macd_curr > 0 else 0.65
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.MACD_BEARISH_CROSS,
                timestamp=timestamp,
                price=close,
                description="MACD crossed below signal line (bearish momentum)",
                strength=strength,
            )
        )

    return patterns


def detect_rsi_divergence(
    df: pd.DataFrame, timestamp: datetime, close: float, lookback: int = 14
) -> List[PatternSignal]:
    """Detect RSI divergence (price vs RSI direction mismatch).

    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    """
    patterns = []

    if "RSI14" not in df.columns:
        return patterns
    if len(df) < lookback + 1:
        return patterns

    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
    recent = df.iloc[-lookback:]

    price = recent[price_col]
    rsi = recent["RSI14"]

    # Find local extremes
    price_min_idx = price.idxmin()
    price_max_idx = price.idxmax()
    rsi_min_idx = rsi.idxmin()
    rsi_max_idx = rsi.idxmax()

    curr_price = price.iloc[-1]
    curr_rsi = rsi.iloc[-1]

    # Bullish divergence: price at/near recent low, but RSI higher than its low
    price_near_low = curr_price <= price.min() * 1.02  # Within 2% of low
    rsi_higher_low = curr_rsi > rsi.min() + 5  # RSI at least 5 points above its low

    if price_near_low and rsi_higher_low and curr_rsi < 40:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.RSI_BULLISH_DIVERGENCE,
                timestamp=timestamp,
                price=close,
                description="Price at low but RSI showing strength (bullish divergence)",
                strength=0.70,
            )
        )

    # Bearish divergence: price at/near recent high, but RSI lower than its high
    price_near_high = curr_price >= price.max() * 0.98
    rsi_lower_high = curr_rsi < rsi.max() - 5

    if price_near_high and rsi_lower_high and curr_rsi > 60:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.RSI_BEARISH_DIVERGENCE,
                timestamp=timestamp,
                price=close,
                description="Price at high but RSI showing weakness (bearish divergence)",
                strength=0.70,
            )
        )

    return patterns


def detect_bollinger_squeeze(
    df: pd.DataFrame, timestamp: datetime, close: float, percentile: float = 10
) -> List[PatternSignal]:
    """Detect Bollinger Band squeeze (low volatility, breakout imminent)."""
    patterns = []

    if "BB_BW" not in df.columns:
        return patterns
    if len(df) < 50:  # Need history for percentile calculation
        return patterns

    bandwidth = df["BB_BW"]
    current_bw = bandwidth.iloc[-1]

    # Check if current bandwidth is in lowest percentile
    threshold = np.percentile(bandwidth.dropna(), percentile)

    if current_bw <= threshold:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.BOLLINGER_SQUEEZE,
                timestamp=timestamp,
                price=close,
                description=f"Bollinger Bands contracting (volatility squeeze, breakout likely)",
                strength=0.60,
            )
        )

    return patterns


def detect_support_resistance(
    df: pd.DataFrame, timestamp: datetime, close: float, lookback: int = 20
) -> List[PatternSignal]:
    """Detect bounces off support or rejection at resistance."""
    patterns = []

    if len(df) < lookback + 1:
        return patterns

    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
    recent = df.iloc[-lookback:]
    price = recent[price_col]

    # Find support (recent low) and resistance (recent high)
    support = price.min()
    resistance = price.max()

    curr_price = close
    prev_price = df[price_col].iloc[-2]

    # Support bounce: price was near support and now moving up
    near_support = prev_price <= support * 1.01
    bouncing_up = curr_price > prev_price

    if near_support and bouncing_up:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.SUPPORT_BOUNCE,
                timestamp=timestamp,
                price=close,
                description=f"Price bouncing off support at ${support:.2f}",
                strength=0.65,
            )
        )

    # Resistance rejection: price was near resistance and now moving down
    near_resistance = prev_price >= resistance * 0.99
    rejecting_down = curr_price < prev_price

    if near_resistance and rejecting_down:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.RESISTANCE_REJECTION,
                timestamp=timestamp,
                price=close,
                description=f"Price rejected at resistance at ${resistance:.2f}",
                strength=0.65,
            )
        )

    return patterns


def detect_vwap_cross(
    df: pd.DataFrame, timestamp: datetime, close: float
) -> List[PatternSignal]:
    """Detect price crossing above/below VWAP."""
    patterns = []

    if "VWAP" not in df.columns:
        return patterns
    if len(df) < 2:
        return patterns

    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"

    price_curr = df[price_col].iloc[-1]
    price_prev = df[price_col].iloc[-2]
    vwap_curr = df["VWAP"].iloc[-1]
    vwap_prev = df["VWAP"].iloc[-2]

    # Cross above VWAP
    if price_prev <= vwap_prev and price_curr > vwap_curr:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.VWAP_CROSS_ABOVE,
                timestamp=timestamp,
                price=close,
                description="Price crossed above VWAP (bullish intraday)",
                strength=0.55,
            )
        )

    # Cross below VWAP
    if price_prev >= vwap_prev and price_curr < vwap_curr:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.VWAP_CROSS_BELOW,
                timestamp=timestamp,
                price=close,
                description="Price crossed below VWAP (bearish intraday)",
                strength=0.55,
            )
        )

    return patterns


def detect_sar_flip(
    df: pd.DataFrame, timestamp: datetime, close: float
) -> List[PatternSignal]:
    """Detect Parabolic SAR flip (trend reversal signal)."""
    patterns = []

    if "SAR" not in df.columns:
        return patterns
    if len(df) < 2:
        return patterns

    price_col = "AdjClose" if "AdjClose" in df.columns else "Close"

    price_curr = df[price_col].iloc[-1]
    price_prev = df[price_col].iloc[-2]
    sar_curr = df["SAR"].iloc[-1]
    sar_prev = df["SAR"].iloc[-2]

    # Bullish flip: SAR was above price, now below
    sar_was_above = sar_prev > price_prev
    sar_now_below = sar_curr < price_curr

    if sar_was_above and sar_now_below:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.SAR_BULLISH_FLIP,
                timestamp=timestamp,
                price=close,
                description="Parabolic SAR flipped bullish (potential uptrend)",
                strength=0.70,
            )
        )

    # Bearish flip: SAR was below price, now above
    sar_was_below = sar_prev < price_prev
    sar_now_above = sar_curr > price_curr

    if sar_was_below and sar_now_above:
        patterns.append(
            PatternSignal(
                pattern_type=PatternType.SAR_BEARISH_FLIP,
                timestamp=timestamp,
                price=close,
                description="Parabolic SAR flipped bearish (potential downtrend)",
                strength=0.70,
            )
        )

    return patterns


def get_bullish_patterns() -> List[PatternType]:
    """Get list of bullish pattern types."""
    return [
        PatternType.GOLDEN_CROSS,
        PatternType.MACD_BULLISH_CROSS,
        PatternType.RSI_BULLISH_DIVERGENCE,
        PatternType.SUPPORT_BOUNCE,
        PatternType.VWAP_CROSS_ABOVE,
        PatternType.SAR_BULLISH_FLIP,
    ]


def get_bearish_patterns() -> List[PatternType]:
    """Get list of bearish pattern types."""
    return [
        PatternType.DEATH_CROSS,
        PatternType.MACD_BEARISH_CROSS,
        PatternType.RSI_BEARISH_DIVERGENCE,
        PatternType.RESISTANCE_REJECTION,
        PatternType.VWAP_CROSS_BELOW,
        PatternType.SAR_BEARISH_FLIP,
    ]


def count_bullish_patterns(patterns: List[PatternSignal]) -> int:
    """Count bullish patterns in list."""
    bullish = get_bullish_patterns()
    return sum(1 for p in patterns if p.pattern_type in bullish)


def count_bearish_patterns(patterns: List[PatternSignal]) -> int:
    """Count bearish patterns in list."""
    bearish = get_bearish_patterns()
    return sum(1 for p in patterns if p.pattern_type in bearish)
