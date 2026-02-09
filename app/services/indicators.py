"""Technical indicators for market analysis.

Includes existing indicators from the sentiment analyzer plus new additions:
- ADX (Average Directional Index)
- CCI (Commodity Channel Index)
- Williams %R
- VWAP (Volume Weighted Average Price)
- Parabolic SAR
- Volume SMA
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# MOVING AVERAGES
# =============================================================================


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence).

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator.

    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return k, d


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos_flow = rmf.where(tp > tp.shift(1), 0.0)
    neg_flow = rmf.where(tp < tp.shift(1), 0.0)
    pos_mf = pos_flow.rolling(window=period, min_periods=period).sum()
    neg_mf = neg_flow.rolling(window=period, min_periods=period).sum()
    mr = pos_mf / (neg_mf.replace(0, np.nan))
    return 100 - (100 / (1 + mr))


def roc(series: pd.Series, period: int) -> pd.Series:
    """Rate of Change."""
    return series.pct_change(periods=period)


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index.

    CCI measures the difference between a security's price change and its
    average price change. High positive readings indicate the price is well
    above its average, which is a show of strength. Low negative readings
    indicate the price is well below its average, which is a show of weakness.
    """
    tp = (high + low + close) / 3.0
    tp_sma = sma(tp, period)
    mean_dev = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    return (tp - tp_sma) / (0.015 * mean_dev)


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R.

    Williams %R is a momentum indicator that measures overbought/oversold levels.
    Values range from -100 to 0:
    - Above -20: Overbought
    - Below -80: Oversold
    """
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index.

    ADX measures trend strength (not direction). Values:
    - 0-25: Absent or weak trend
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend

    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    # True Range
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Smoothed averages
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    )

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx_val, plus_di, minus_di


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================


def bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns:
        Tuple of (middle, upper, lower, bandwidth)
    """
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    bandwidth = (upper - lower).replace(0, np.nan)
    return mid, upper, lower, bandwidth


def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def rolling_vol(
    returns: pd.Series, window: int = 21, annualization: int = 252
) -> pd.Series:
    """Rolling volatility (annualized)."""
    vol = returns.rolling(window=window, min_periods=window).std(ddof=0)
    return vol * np.sqrt(annualization)


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """Parabolic SAR (Stop and Reverse).

    SAR trails price as a trend matures. When price crosses SAR,
    it signals a potential reversal.
    """
    length = len(close)
    sar = np.zeros(length)
    ep = np.zeros(length)  # Extreme Point
    af = np.zeros(length)  # Acceleration Factor
    trend = np.ones(length)  # 1 = uptrend, -1 = downtrend

    # Initialize
    sar[0] = low.iloc[0]
    ep[0] = high.iloc[0]
    af[0] = af_start

    for i in range(1, length):
        # Calculate SAR
        sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

        # Uptrend
        if trend[i - 1] == 1:
            # SAR cannot be above prior two lows
            sar[i] = min(sar[i], low.iloc[i - 1])
            if i > 1:
                sar[i] = min(sar[i], low.iloc[i - 2])

            # Check for reversal
            if low.iloc[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep[i - 1]
                ep[i] = low.iloc[i]
                af[i] = af_start
            else:
                trend[i] = 1
                # Update EP and AF
                if high.iloc[i] > ep[i - 1]:
                    ep[i] = high.iloc[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]
        # Downtrend
        else:
            # SAR cannot be below prior two highs
            sar[i] = max(sar[i], high.iloc[i - 1])
            if i > 1:
                sar[i] = max(sar[i], high.iloc[i - 2])

            # Check for reversal
            if high.iloc[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep[i - 1]
                ep[i] = high.iloc[i]
                af[i] = af_start
            else:
                trend[i] = -1
                # Update EP and AF
                if low.iloc[i] < ep[i - 1]:
                    ep[i] = low.iloc[i]
                    af[i] = min(af[i - 1] + af_increment, af_max)
                else:
                    ep[i] = ep[i - 1]
                    af[i] = af[i - 1]

    return pd.Series(sar, index=close.index)


# =============================================================================
# VOLUME INDICATORS
# =============================================================================


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume.fillna(0)).cumsum()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return sma(volume, period)


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume relative to its moving average.

    Values > 1 indicate above-average volume, < 1 below-average.
    """
    vol_ma = volume_sma(volume, period)
    return volume / vol_ma


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume Weighted Average Price.

    VWAP is calculated cumulatively for each trading session.
    For intraday, this resets each day.
    """
    tp = (high + low + close) / 3.0
    tp_volume = tp * volume
    cumulative_tp_vol = tp_volume.cumsum()
    cumulative_vol = volume.cumsum()
    return cumulative_tp_vol / cumulative_vol


def vwap_intraday(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Intraday VWAP that resets each trading day."""
    tp = (high + low + close) / 3.0
    tp_volume = tp * volume

    # Group by date and calculate cumulative VWAP
    df = pd.DataFrame(
        {"tp_vol": tp_volume, "volume": volume, "date": index.date}, index=index
    )

    df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
    df["cum_vol"] = df.groupby("date")["volume"].cumsum()

    return df["cum_tp_vol"] / df["cum_vol"]


# =============================================================================
# TREND / DRAWDOWN
# =============================================================================


def max_drawdown(series: pd.Series) -> pd.Series:
    """Maximum drawdown from peak."""
    cummax = series.cummax()
    dd = (series / cummax) - 1.0
    return dd


# =============================================================================
# BUILD INDICATOR FRAME
# =============================================================================


def build_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a DataFrame with all technical indicators.

    Args:
        df: DataFrame with columns: Open, High, Low, Close, Volume, AdjClose

    Returns:
        DataFrame enriched with all indicators
    """
    out = df.copy()
    c = out["AdjClose"] if "AdjClose" in out.columns else out["Close"]
    h, low_col, o, v = out["High"], out["Low"], out["Open"], out["Volume"]

    # Moving averages
    out["SMA20"] = sma(c, 20)
    out["SMA50"] = sma(c, 50)
    out["SMA200"] = sma(c, 200)
    out["EMA12"] = ema(c, 12)
    out["EMA20"] = ema(c, 20)
    out["EMA26"] = ema(c, 26)
    out["EMA50"] = ema(c, 50)

    # MACD
    out["MACD"], out["MACDSignal"], out["MACDHist"] = macd(c)

    # RSI
    out["RSI14"] = rsi(c, 14)

    # Bollinger Bands
    bb_mid, bb_up, bb_low, bb_bw = bollinger(c, 20, 2.0)
    out["BB_Mid"], out["BB_Up"], out["BB_Low"], out["BB_BW"] = bb_mid, bb_up, bb_low, bb_bw

    # ATR
    out["ATR14"] = atr(h, low_col, c, 14)

    # Stochastic
    out["STOCHK"], out["STOCHD"] = stochastic(h, low_col, c)

    # OBV, MFI
    out["OBV"] = obv(c, v)
    out["MFI14"] = mfi(h, low_col, c, v, 14)

    # ROC
    out["ROC12"] = roc(c, 12)
    out["ROC24"] = roc(c, 24)

    # --- NEW INDICATORS ---

    # ADX (trend strength)
    out["ADX"], out["PLUS_DI"], out["MINUS_DI"] = adx(h, low_col, c, 14)

    # CCI (momentum)
    out["CCI20"] = cci(h, low_col, c, 20)

    # Williams %R
    out["WILLR14"] = williams_r(h, low_col, c, 14)

    # VWAP (use intraday if datetime index, else cumulative)
    if isinstance(out.index, pd.DatetimeIndex):
        out["VWAP"] = vwap_intraday(h, low_col, c, v, out.index)
    else:
        out["VWAP"] = vwap(h, low_col, c, v)

    # Parabolic SAR
    out["SAR"] = parabolic_sar(h, low_col, c)

    # Volume indicators
    out["VolSMA20"] = volume_sma(v, 20)
    out["VolRatio"] = volume_ratio(v, 20)

    # Volatility and drawdown
    if "Return" in out.columns:
        out["Vol21"] = rolling_vol(out["Return"], 21)
    out["Drawdown"] = max_drawdown(c)

    # Price derivatives
    out["dC"] = np.gradient(c.ffill().values)
    out["ddC"] = np.gradient(out["dC"].fillna(0).values)

    # Drop rows with NaN from indicator warmup
    out = out.dropna()
    return out
