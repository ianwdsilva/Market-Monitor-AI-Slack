"""Machine learning signal scorer.

Trains a model on historical backtest data to predict signal quality,
enhancing confidence calculations.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from app.models.schemas import (
    BacktestResult,
    PatternSignal,
    Signal,
    SignalDirection,
    StrategyPerformance,
    TradeOutcome,
)

logger = logging.getLogger(__name__)


# Feature names for the model
INDICATOR_FEATURES = [
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "stoch_k",
    "stoch_d",
    "adx",
    "plus_di",
    "minus_di",
    "mfi",
    "willr",
    "vol_ratio",
    "bb_position",  # Computed: (close - bb_lower) / (bb_upper - bb_lower)
]

PATTERN_FEATURES = [
    "golden_cross",
    "death_cross",
    "macd_bullish_cross",
    "macd_bearish_cross",
    "rsi_bullish_divergence",
    "rsi_bearish_divergence",
    "bollinger_squeeze",
    "support_bounce",
    "resistance_rejection",
    "vwap_cross_above",
    "vwap_cross_below",
    "sar_bullish_flip",
    "sar_bearish_flip",
]

META_FEATURES = [
    "confirmations",
    "direction_buy",  # 1 for BUY, 0 for SELL
    "hour_of_day",
    "day_of_week",
]


class MLScorer:
    """Machine learning model for scoring trading signals."""

    def __init__(self, model_dir: str = "data/ml_models"):
        """Initialize ML scorer.

        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = model_dir
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = INDICATOR_FEATURES + PATTERN_FEATURES + META_FEATURES
        self.is_trained = False

        # Historical accuracy tracking per strategy
        self.strategy_accuracy: Dict[str, Dict[str, float]] = {}  # strategy -> {ticker: win_rate}

        os.makedirs(model_dir, exist_ok=True)

    def train(self, backtest_result: BacktestResult) -> bool:
        """Train the model on backtest data.

        Args:
            backtest_result: Backtest results containing trade outcomes

        Returns:
            True if training was successful
        """
        logger.info("Preparing training data from backtest results...")

        # Extract training data from all strategies and tickers
        X_data = []
        y_data = []

        for strategy_name, ticker_performances in backtest_result.strategies.items():
            for ticker, performance in ticker_performances.items():
                # Track strategy accuracy
                if strategy_name not in self.strategy_accuracy:
                    self.strategy_accuracy[strategy_name] = {}
                self.strategy_accuracy[strategy_name][ticker] = performance.win_rate_1h

                # Extract features from each outcome
                for outcome in performance.outcomes:
                    features = self._extract_outcome_features(outcome, strategy_name, ticker)
                    if features is not None:
                        X_data.append(features)
                        # Target: was the trade profitable at 1h?
                        y_data.append(1 if outcome.is_winner_1h else 0)

        if len(X_data) < 50:
            logger.warning(f"Insufficient training data ({len(X_data)} samples), skipping ML training")
            return False

        X = np.array(X_data)
        y = np.array(y_data)

        logger.info(f"Training on {len(X)} samples, {sum(y)} winners ({100*sum(y)/len(y):.1f}%)")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train gradient boosting classifier
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # Log feature importances
        importances = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("Top 5 feature importances:")
        for name, importance in importances[:5]:
            logger.info(f"  {name}: {importance:.3f}")

        self.is_trained = True
        self._save_model()
        return True

    def _extract_outcome_features(
        self,
        outcome: TradeOutcome,
        strategy_name: str,
        ticker: str,
    ) -> Optional[List[float]]:
        """Extract feature vector from a trade outcome.

        Note: Since we don't have full indicator data stored with outcomes,
        we use simplified features based on what's available.
        """
        features = []

        # For indicator features, use placeholders (0) since we don't have them stored
        # In a production system, you'd store these with each outcome
        for _ in INDICATOR_FEATURES:
            features.append(0.0)

        # Pattern features (placeholders)
        for _ in PATTERN_FEATURES:
            features.append(0.0)

        # Meta features
        features.append(3)  # Default confirmations
        features.append(1.0 if outcome.direction == SignalDirection.BUY else 0.0)
        features.append(outcome.entry_time.hour)
        features.append(outcome.entry_time.weekday())

        return features

    def score_signal(
        self,
        signal: Signal,
        indicators: Dict[str, float],
        patterns: List[PatternSignal],
    ) -> float:
        """Score a signal using the ML model.

        Args:
            signal: Trading signal to score
            indicators: Current indicator values
            patterns: Detected patterns

        Returns:
            ML confidence score (0-1), or 0.5 if model not trained
        """
        if not self.is_trained or self.model is None:
            return 0.5  # Neutral score if not trained

        features = self._extract_signal_features(signal, indicators, patterns)
        X = np.array([features])
        X_scaled = self.scaler.transform(X)

        # Get probability of winning trade
        proba = self.model.predict_proba(X_scaled)[0][1]
        return float(proba)

    def _extract_signal_features(
        self,
        signal: Signal,
        indicators: Dict[str, float],
        patterns: List[PatternSignal],
    ) -> List[float]:
        """Extract feature vector from a signal."""
        features = []

        # Indicator features
        for name in INDICATOR_FEATURES:
            if name == "bb_position":
                # Compute BB position from indicators if available
                bb_up = indicators.get("bb_upper", 0)
                bb_low = indicators.get("bb_lower", 0)
                close = signal.current_price
                if bb_up > bb_low:
                    bb_pos = (close - bb_low) / (bb_up - bb_low)
                else:
                    bb_pos = 0.5
                features.append(bb_pos)
            else:
                features.append(indicators.get(name, 0.0) or 0.0)

        # Pattern features (binary: 1 if pattern present, 0 otherwise)
        pattern_types = {p.pattern_type.value for p in patterns}
        for pattern_name in PATTERN_FEATURES:
            features.append(1.0 if pattern_name in pattern_types else 0.0)

        # Meta features
        features.append(signal.confirmations)
        features.append(1.0 if signal.direction == SignalDirection.BUY else 0.0)
        features.append(signal.timestamp.hour)
        features.append(signal.timestamp.weekday())

        return features

    def get_strategy_historical_accuracy(
        self,
        strategy_name: str,
        ticker: str,
    ) -> float:
        """Get historical win rate for a strategy-ticker combination.

        Args:
            strategy_name: Name of the strategy
            ticker: Stock symbol

        Returns:
            Historical win rate (0-1), or 0.5 if unknown
        """
        if strategy_name in self.strategy_accuracy:
            return self.strategy_accuracy[strategy_name].get(ticker.upper(), 0.5)
        return 0.5

    def update_from_backtest(self, backtest_result: BacktestResult):
        """Update strategy accuracy from new backtest results."""
        for strategy_name, ticker_performances in backtest_result.strategies.items():
            if strategy_name not in self.strategy_accuracy:
                self.strategy_accuracy[strategy_name] = {}
            for ticker, performance in ticker_performances.items():
                self.strategy_accuracy[strategy_name][ticker] = performance.win_rate_1h

        # Retrain model with new data
        self.train(backtest_result)

    def _save_model(self):
        """Save model and scaler to disk."""
        if not self.is_trained:
            return

        model_path = os.path.join(self.model_dir, "signal_model.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        accuracy_path = os.path.join(self.model_dir, "strategy_accuracy.joblib")

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.strategy_accuracy, accuracy_path)

        logger.info(f"Saved ML model to {model_path}")

    def load_model(self) -> bool:
        """Load model from disk.

        Returns:
            True if model was loaded successfully
        """
        model_path = os.path.join(self.model_dir, "signal_model.joblib")
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        accuracy_path = os.path.join(self.model_dir, "strategy_accuracy.joblib")

        if not all(os.path.exists(p) for p in [model_path, scaler_path]):
            logger.info("No existing ML model found")
            return False

        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            if os.path.exists(accuracy_path):
                self.strategy_accuracy = joblib.load(accuracy_path)
            self.is_trained = True
            logger.info("Loaded ML model from disk")
            return True
        except Exception as e:
            logger.error(f"Error loading ML model: {e}")
            return False


def compute_enhanced_confidence(
    base_confidence: float,
    ml_score: float,
    historical_accuracy: float,
    pattern_strength: float,
    indicator_extremity: float,
) -> float:
    """Compute enhanced confidence score combining multiple factors.

    Args:
        base_confidence: Original rule-based confidence
        ml_score: ML model prediction (0-1)
        historical_accuracy: Strategy historical win rate (0-1)
        pattern_strength: Average strength of detected patterns (0-1)
        indicator_extremity: How extreme current readings are (0-1)

    Returns:
        Enhanced confidence score (0-1)
    """
    # Weighted combination of factors
    weights = {
        "base": 0.30,
        "ml": 0.25,
        "historical": 0.20,
        "pattern": 0.15,
        "extremity": 0.10,
    }

    enhanced = (
        weights["base"] * base_confidence
        + weights["ml"] * ml_score
        + weights["historical"] * historical_accuracy
        + weights["pattern"] * pattern_strength
        + weights["extremity"] * indicator_extremity
    )

    # Clamp to valid range
    return min(max(enhanced, 0.0), 0.95)


def compute_indicator_extremity(indicators: Dict[str, float]) -> float:
    """Compute how extreme current indicator readings are.

    Higher values = more extreme readings = stronger signal.
    """
    extremity_scores = []

    # RSI extremity (0-30 or 70-100)
    rsi = indicators.get("rsi")
    if rsi is not None:
        if rsi < 30:
            extremity_scores.append((30 - rsi) / 30)  # 0 at RSI=30, 1 at RSI=0
        elif rsi > 70:
            extremity_scores.append((rsi - 70) / 30)  # 0 at RSI=70, 1 at RSI=100
        else:
            extremity_scores.append(0.0)

    # Stochastic extremity
    stoch = indicators.get("stoch_k")
    if stoch is not None:
        if stoch < 20:
            extremity_scores.append((20 - stoch) / 20)
        elif stoch > 80:
            extremity_scores.append((stoch - 80) / 20)
        else:
            extremity_scores.append(0.0)

    # Williams %R extremity (-100 to -80 or -20 to 0)
    willr = indicators.get("willr")
    if willr is not None:
        if willr < -80:
            extremity_scores.append((-80 - willr) / 20)
        elif willr > -20:
            extremity_scores.append((willr + 20) / 20)
        else:
            extremity_scores.append(0.0)

    # MFI extremity
    mfi = indicators.get("mfi")
    if mfi is not None:
        if mfi < 20:
            extremity_scores.append((20 - mfi) / 20)
        elif mfi > 80:
            extremity_scores.append((mfi - 80) / 20)
        else:
            extremity_scores.append(0.0)

    # ADX strength (higher = stronger trend)
    adx = indicators.get("adx")
    if adx is not None:
        if adx > 25:
            extremity_scores.append(min((adx - 25) / 50, 1.0))
        else:
            extremity_scores.append(0.0)

    if not extremity_scores:
        return 0.5

    return sum(extremity_scores) / len(extremity_scores)


def compute_pattern_strength(patterns: List[PatternSignal]) -> float:
    """Compute average pattern strength from detected patterns."""
    if not patterns:
        return 0.5

    return sum(p.strength for p in patterns) / len(patterns)


# Singleton instance
_ml_scorer: Optional[MLScorer] = None


def get_ml_scorer() -> MLScorer:
    """Get or create the ML scorer singleton."""
    global _ml_scorer
    if _ml_scorer is None:
        _ml_scorer = MLScorer()
        _ml_scorer.load_model()  # Try to load existing model
    return _ml_scorer
