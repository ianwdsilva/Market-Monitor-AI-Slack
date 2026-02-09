"""Trading signal strategies with multi-indicator confirmation.

Implements 5 conservative strategies:
A) Momentum Reversal - RSI/Stochastic oversold/overbought with MACD confirmation
B) Trend Following - VWAP/ADX/MACD alignment
C) Breakout - Bollinger Band break with volume confirmation
D) Mean Reversion - Extreme readings with trend filter
E) Ensemble - Multi-strategy agreement
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Type

import pandas as pd

from app.models.schemas import PatternSignal, Signal, SignalDirection
from app.services.patterns import (
    count_bearish_patterns,
    count_bullish_patterns,
    detect_all_patterns,
)


class Strategy(ABC):
    """Base class for trading strategies."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        """Check if current data generates a signal.

        Args:
            df: DataFrame with indicators from build_indicator_frame()
            ticker: Stock symbol
            patterns: Optional pre-detected patterns

        Returns:
            Signal if conditions are met, None otherwise
        """
        pass

    def _get_current_values(self, df: pd.DataFrame) -> dict:
        """Extract current indicator values from DataFrame."""
        if df.empty:
            return {}

        row = df.iloc[-1]
        price_col = "AdjClose" if "AdjClose" in df.columns else "Close"

        return {
            "close": row[price_col],
            "rsi": row.get("RSI14"),
            "macd": row.get("MACD"),
            "macd_signal": row.get("MACDSignal"),
            "macd_hist": row.get("MACDHist"),
            "stoch_k": row.get("STOCHK"),
            "stoch_d": row.get("STOCHD"),
            "bb_upper": row.get("BB_Up"),
            "bb_lower": row.get("BB_Low"),
            "bb_mid": row.get("BB_Mid"),
            "adx": row.get("ADX"),
            "plus_di": row.get("PLUS_DI"),
            "minus_di": row.get("MINUS_DI"),
            "vwap": row.get("VWAP"),
            "ema20": row.get("EMA20"),
            "sma200": row.get("SMA200"),
            "mfi": row.get("MFI14"),
            "willr": row.get("WILLR14"),
            "vol_ratio": row.get("VolRatio"),
            "sar": row.get("SAR"),
        }

    def _prev_macd_hist(self, df: pd.DataFrame) -> Optional[float]:
        """Get previous MACD histogram value."""
        if len(df) < 2 or "MACDHist" not in df.columns:
            return None
        return df["MACDHist"].iloc[-2]


class MomentumReversalStrategy(Strategy):
    """Strategy A: Momentum Reversal

    BUY when: RSI < 30 AND Stochastic < 20 AND price near BB lower AND MACD hist turning positive
    SELL when: RSI > 70 AND Stochastic > 80 AND price near BB upper AND MACD hist turning negative
    """

    name = "momentum_reversal"
    description = "Identifies oversold/overbought conditions with momentum confirmation"

    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        if df.empty or len(df) < 2:
            return None

        v = self._get_current_values(df)
        prev_hist = self._prev_macd_hist(df)
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()

        # Check for required indicators
        required = ["rsi", "stoch_k", "bb_lower", "bb_upper", "macd_hist", "close"]
        if any(v.get(k) is None for k in required) or prev_hist is None:
            return None

        confirmations = 0
        reasons = []

        # === BUY CONDITIONS ===
        buy_confirmations = 0

        # RSI oversold
        if v["rsi"] < 30:
            buy_confirmations += 1
            reasons.append(f"RSI({v['rsi']:.1f}) oversold")

        # Stochastic oversold
        if v["stoch_k"] < 20:
            buy_confirmations += 1
            reasons.append(f"Stochastic({v['stoch_k']:.1f}) oversold")

        # Price near lower Bollinger Band
        bb_position = (v["close"] - v["bb_lower"]) / (v["bb_upper"] - v["bb_lower"])
        if bb_position < 0.15:
            buy_confirmations += 1
            reasons.append("Price near BB lower")

        # MACD histogram turning positive
        if prev_hist < 0 and v["macd_hist"] > prev_hist:
            buy_confirmations += 1
            reasons.append("MACD momentum improving")

        if buy_confirmations >= 3:
            confidence = min(0.5 + buy_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                current_price=v["close"],
                indicators={"rsi": v["rsi"], "stoch_k": v["stoch_k"], "macd_hist": v["macd_hist"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=buy_confirmations,
            )

        # === SELL CONDITIONS ===
        sell_confirmations = 0
        reasons = []

        # RSI overbought
        if v["rsi"] > 70:
            sell_confirmations += 1
            reasons.append(f"RSI({v['rsi']:.1f}) overbought")

        # Stochastic overbought
        if v["stoch_k"] > 80:
            sell_confirmations += 1
            reasons.append(f"Stochastic({v['stoch_k']:.1f}) overbought")

        # Price near upper Bollinger Band
        if bb_position > 0.85:
            sell_confirmations += 1
            reasons.append("Price near BB upper")

        # MACD histogram turning negative
        if prev_hist > 0 and v["macd_hist"] < prev_hist:
            sell_confirmations += 1
            reasons.append("MACD momentum weakening")

        if sell_confirmations >= 3:
            confidence = min(0.5 + sell_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=confidence,
                current_price=v["close"],
                indicators={"rsi": v["rsi"], "stoch_k": v["stoch_k"], "macd_hist": v["macd_hist"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=sell_confirmations,
            )

        return None


class TrendFollowingStrategy(Strategy):
    """Strategy B: Trend Following

    BUY when: Price > VWAP AND ADX > 25 AND MACD bullish crossover AND price > EMA 20
    SELL when: Price < VWAP AND ADX > 25 AND MACD bearish crossover AND price < EMA 20
    """

    name = "trend_following"
    description = "Follows established trends with ADX and VWAP confirmation"

    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        if df.empty or len(df) < 2:
            return None

        v = self._get_current_values(df)
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()

        # Check required indicators
        required = ["close", "vwap", "adx", "macd", "macd_signal", "ema20"]
        if any(v.get(k) is None for k in required):
            return None

        # Get previous MACD values for crossover detection
        if len(df) < 2:
            return None
        prev_macd = df["MACD"].iloc[-2]
        prev_signal = df["MACDSignal"].iloc[-2]

        confirmations = 0
        reasons = []

        # === BUY CONDITIONS ===
        buy_confirmations = 0

        # Price above VWAP
        if v["close"] > v["vwap"]:
            buy_confirmations += 1
            reasons.append("Price > VWAP")

        # Strong trend (ADX > 25)
        if v["adx"] > 25:
            buy_confirmations += 1
            reasons.append(f"ADX({v['adx']:.1f}) strong trend")

        # MACD bullish crossover
        if prev_macd <= prev_signal and v["macd"] > v["macd_signal"]:
            buy_confirmations += 1
            reasons.append("MACD bullish crossover")

        # Price above EMA20
        if v["close"] > v["ema20"]:
            buy_confirmations += 1
            reasons.append("Price > EMA20")

        if buy_confirmations >= 3:
            confidence = min(0.5 + buy_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                current_price=v["close"],
                indicators={"adx": v["adx"], "vwap": v["vwap"], "macd": v["macd"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=buy_confirmations,
            )

        # === SELL CONDITIONS ===
        sell_confirmations = 0
        reasons = []

        # Price below VWAP
        if v["close"] < v["vwap"]:
            sell_confirmations += 1
            reasons.append("Price < VWAP")

        # Strong trend
        if v["adx"] > 25:
            sell_confirmations += 1
            reasons.append(f"ADX({v['adx']:.1f}) strong trend")

        # MACD bearish crossover
        if prev_macd >= prev_signal and v["macd"] < v["macd_signal"]:
            sell_confirmations += 1
            reasons.append("MACD bearish crossover")

        # Price below EMA20
        if v["close"] < v["ema20"]:
            sell_confirmations += 1
            reasons.append("Price < EMA20")

        if sell_confirmations >= 3:
            confidence = min(0.5 + sell_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=confidence,
                current_price=v["close"],
                indicators={"adx": v["adx"], "vwap": v["vwap"], "macd": v["macd"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=sell_confirmations,
            )

        return None


class BreakoutStrategy(Strategy):
    """Strategy C: Breakout

    BUY when: Price breaks above BB upper AND volume > 2x average AND ADX rising AND RSI < 70
    SELL when: Price breaks below BB lower AND volume > 2x average AND ADX rising AND RSI > 30
    """

    name = "breakout"
    description = "Identifies breakouts with volume and momentum confirmation"

    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        if df.empty or len(df) < 2:
            return None

        v = self._get_current_values(df)
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()

        required = ["close", "bb_upper", "bb_lower", "vol_ratio", "adx", "rsi"]
        if any(v.get(k) is None for k in required):
            return None

        # ADX rising check
        prev_adx = df["ADX"].iloc[-2] if len(df) >= 2 else v["adx"]
        adx_rising = v["adx"] > prev_adx

        confirmations = 0
        reasons = []

        # === BUY CONDITIONS (Bullish Breakout) ===
        buy_confirmations = 0

        # Price breaks above BB upper
        if v["close"] > v["bb_upper"]:
            buy_confirmations += 1
            reasons.append("Price broke above BB upper")

        # High volume
        if v["vol_ratio"] > 2.0:
            buy_confirmations += 1
            reasons.append(f"Volume {v['vol_ratio']:.1f}x average")

        # ADX rising (momentum building)
        if adx_rising:
            buy_confirmations += 1
            reasons.append("ADX rising")

        # RSI not overbought (room to run)
        if v["rsi"] < 70:
            buy_confirmations += 1
            reasons.append(f"RSI({v['rsi']:.1f}) has room")

        if buy_confirmations >= 3:
            confidence = min(0.5 + buy_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                current_price=v["close"],
                indicators={"vol_ratio": v["vol_ratio"], "adx": v["adx"], "rsi": v["rsi"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=buy_confirmations,
            )

        # === SELL CONDITIONS (Bearish Breakout) ===
        sell_confirmations = 0
        reasons = []

        # Price breaks below BB lower
        if v["close"] < v["bb_lower"]:
            sell_confirmations += 1
            reasons.append("Price broke below BB lower")

        # High volume
        if v["vol_ratio"] > 2.0:
            sell_confirmations += 1
            reasons.append(f"Volume {v['vol_ratio']:.1f}x average")

        # ADX rising
        if adx_rising:
            sell_confirmations += 1
            reasons.append("ADX rising")

        # RSI not oversold
        if v["rsi"] > 30:
            sell_confirmations += 1
            reasons.append(f"RSI({v['rsi']:.1f}) has room")

        if sell_confirmations >= 3:
            confidence = min(0.5 + sell_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=confidence,
                current_price=v["close"],
                indicators={"vol_ratio": v["vol_ratio"], "adx": v["adx"], "rsi": v["rsi"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=sell_confirmations,
            )

        return None


class MeanReversionStrategy(Strategy):
    """Strategy D: Mean Reversion

    BUY when: Price < BB lower AND MFI < 20 AND Williams %R < -80 AND price above SMA 200
    SELL when: Price > BB upper AND MFI > 80 AND Williams %R > -20 AND price below SMA 200
    """

    name = "mean_reversion"
    description = "Identifies extreme conditions likely to revert to mean"

    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        if df.empty:
            return None

        v = self._get_current_values(df)
        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()

        required = ["close", "bb_lower", "bb_upper", "mfi", "willr", "sma200"]
        if any(v.get(k) is None for k in required):
            return None

        confirmations = 0
        reasons = []

        # === BUY CONDITIONS ===
        buy_confirmations = 0

        # Price below BB lower
        if v["close"] < v["bb_lower"]:
            buy_confirmations += 1
            reasons.append("Price below BB lower")

        # MFI oversold
        if v["mfi"] < 20:
            buy_confirmations += 1
            reasons.append(f"MFI({v['mfi']:.1f}) oversold")

        # Williams %R oversold
        if v["willr"] < -80:
            buy_confirmations += 1
            reasons.append(f"Williams %R({v['willr']:.1f}) oversold")

        # Price above SMA200 (long-term uptrend filter)
        if v["close"] > v["sma200"]:
            buy_confirmations += 1
            reasons.append("Above SMA200 (uptrend)")

        if buy_confirmations >= 3:
            confidence = min(0.5 + buy_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                current_price=v["close"],
                indicators={"mfi": v["mfi"], "willr": v["willr"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=buy_confirmations,
            )

        # === SELL CONDITIONS ===
        sell_confirmations = 0
        reasons = []

        # Price above BB upper
        if v["close"] > v["bb_upper"]:
            sell_confirmations += 1
            reasons.append("Price above BB upper")

        # MFI overbought
        if v["mfi"] > 80:
            sell_confirmations += 1
            reasons.append(f"MFI({v['mfi']:.1f}) overbought")

        # Williams %R overbought
        if v["willr"] > -20:
            sell_confirmations += 1
            reasons.append(f"Williams %R({v['willr']:.1f}) overbought")

        # Price below SMA200 (long-term downtrend filter)
        if v["close"] < v["sma200"]:
            sell_confirmations += 1
            reasons.append("Below SMA200 (downtrend)")

        if sell_confirmations >= 3:
            confidence = min(0.5 + sell_confirmations * 0.1, 0.95)
            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=confidence,
                current_price=v["close"],
                indicators={"mfi": v["mfi"], "willr": v["willr"]},
                patterns=patterns or [],
                reason=" | ".join(reasons),
                confirmations=sell_confirmations,
            )

        return None


class EnsembleStrategy(Strategy):
    """Strategy E: Ensemble

    Alert only when 2+ individual strategies agree on direction.
    Highest confidence signals.
    """

    name = "ensemble"
    description = "Multi-strategy agreement for highest confidence signals"

    def __init__(self):
        self.sub_strategies: List[Strategy] = [
            MomentumReversalStrategy(),
            TrendFollowingStrategy(),
            BreakoutStrategy(),
            MeanReversionStrategy(),
        ]

    def check_signal(
        self,
        df: pd.DataFrame,
        ticker: str,
        patterns: Optional[List[PatternSignal]] = None,
    ) -> Optional[Signal]:
        if df.empty:
            return None

        # Collect signals from all sub-strategies
        buy_signals: List[Signal] = []
        sell_signals: List[Signal] = []

        for strategy in self.sub_strategies:
            signal = strategy.check_signal(df, ticker, patterns)
            if signal:
                if signal.direction == SignalDirection.BUY:
                    buy_signals.append(signal)
                else:
                    sell_signals.append(signal)

        timestamp = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else datetime.utcnow()
        price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
        close = df[price_col].iloc[-1]

        # Need 2+ strategies to agree
        if len(buy_signals) >= 2:
            # Aggregate confidence
            avg_confidence = sum(s.confidence for s in buy_signals) / len(buy_signals)
            confidence = min(avg_confidence + 0.1, 0.95)  # Bonus for agreement

            # Combine reasons
            strategy_names = [s.strategy_name for s in buy_signals]
            reasons = f"Ensemble: {len(buy_signals)} strategies agree (BUY) - {', '.join(strategy_names)}"

            # Merge indicators
            indicators = {}
            for s in buy_signals:
                indicators.update(s.indicators)

            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.BUY,
                confidence=confidence,
                current_price=close,
                indicators=indicators,
                patterns=patterns or [],
                reason=reasons,
                confirmations=len(buy_signals),
            )

        if len(sell_signals) >= 2:
            avg_confidence = sum(s.confidence for s in sell_signals) / len(sell_signals)
            confidence = min(avg_confidence + 0.1, 0.95)

            strategy_names = [s.strategy_name for s in sell_signals]
            reasons = f"Ensemble: {len(sell_signals)} strategies agree (SELL) - {', '.join(strategy_names)}"

            indicators = {}
            for s in sell_signals:
                indicators.update(s.indicators)

            return Signal(
                ticker=ticker,
                timestamp=timestamp,
                strategy_name=self.name,
                direction=SignalDirection.SELL,
                confidence=confidence,
                current_price=close,
                indicators=indicators,
                patterns=patterns or [],
                reason=reasons,
                confirmations=len(sell_signals),
            )

        return None


# Registry of all available strategies
STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "momentum_reversal": MomentumReversalStrategy,
    "trend_following": TrendFollowingStrategy,
    "breakout": BreakoutStrategy,
    "mean_reversion": MeanReversionStrategy,
    "ensemble": EnsembleStrategy,
}


def get_strategy(name: str) -> Strategy:
    """Get a strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_REGISTRY.keys())}")
    return STRATEGY_REGISTRY[name]()


def get_all_strategies() -> List[Strategy]:
    """Get instances of all available strategies."""
    return [cls() for cls in STRATEGY_REGISTRY.values()]
