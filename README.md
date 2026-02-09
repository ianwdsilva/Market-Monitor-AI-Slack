# Market Monitor

Real-time NYSE stock monitoring with technical analysis and Slack alerts.

## Features

- **Live Monitoring**: Runs during NYSE market hours (9:30 AM - 4:00 PM ET)
- **Multi-Indicator Analysis**: 20+ technical indicators including RSI, MACD, Bollinger Bands, ADX, VWAP, and more
- **5 Trading Strategies**:
  - Momentum Reversal (RSI/Stochastic oversold/overbought)
  - Trend Following (VWAP/ADX/MACD alignment)
  - Breakout (Bollinger Band break with volume)
  - Mean Reversion (extreme readings with trend filter)
  - Ensemble (multi-strategy agreement)
- **Pattern Detection**: Golden/Death cross, MACD crossover, RSI divergence, Bollinger squeeze, support/resistance
- **Backtesting**: Auto-selects best strategy per ticker based on historical performance
- **Slack Alerts**: Rich notifications with 1-hour per-ticker cooldown
- **Conservative Signals**: Requires 3+ indicator confirmations and 75%+ confidence

## Quick Start

### 1. Clone and Setup

```bash
cd /Users/iandsilva/Warpfp/market-monitor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```env
# Alpaca API (get keys at https://app.alpaca.markets/)
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Slack Webhook (create at https://api.slack.com/messaging/webhooks)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/xxx/xxx

# Tickers to monitor
WATCHLIST=AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AMD,NFLX,DIS
```

### 3. Run

```bash
python -m app.main
```

The monitor will:
1. Run initial backtest to select best strategies
2. Wait for market open (if closed)
3. Poll every 60 seconds during market hours
4. Send Slack alerts when signals are detected

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WATCHLIST` | 10 major stocks | Comma-separated tickers |
| `POLL_INTERVAL_SECONDS` | 60 | Seconds between analysis cycles |
| `ALERT_COOLDOWN_MINUTES` | 60 | Min wait between alerts per ticker |
| `SIGNAL_CONFIDENCE_THRESHOLD` | 0.75 | Minimum confidence to alert |
| `MIN_CONFIRMATIONS` | 3 | Required indicator confirmations |
| `BACKTEST_DAYS` | 30 | Historical days for backtesting |

## Project Structure

```
market-monitor/
├── app/
│   ├── main.py              # Entry point, monitoring loop
│   ├── config.py            # Settings management
│   ├── services/
│   │   ├── alpaca_client.py # Alpaca data fetching
│   │   ├── indicators.py    # 20+ technical indicators
│   │   ├── patterns.py      # Chart pattern detection
│   │   ├── strategies.py    # 5 trading strategies
│   │   ├── backtest.py      # Strategy evaluation
│   │   ├── alerts.py        # Slack notifications
│   │   └── market_hours.py  # NYSE schedule
│   └── models/
│       └── schemas.py       # Pydantic data models
├── data/
│   ├── cache/               # Data cache
│   └── backtest_results/    # Performance logs
├── requirements.txt
├── .env.example
└── README.md
```

## Strategies

### A) Momentum Reversal
Identifies oversold/overbought conditions with momentum confirmation.

**BUY**: RSI < 30 + Stochastic < 20 + Price near BB lower + MACD improving
**SELL**: RSI > 70 + Stochastic > 80 + Price near BB upper + MACD weakening

### B) Trend Following
Follows established trends with multiple confirmations.

**BUY**: Price > VWAP + ADX > 25 + MACD bullish cross + Price > EMA20
**SELL**: Price < VWAP + ADX > 25 + MACD bearish cross + Price < EMA20

### C) Breakout
Identifies breakouts with volume confirmation.

**BUY**: Price > BB upper + Volume > 2x average + ADX rising + RSI < 70
**SELL**: Price < BB lower + Volume > 2x average + ADX rising + RSI > 30

### D) Mean Reversion
Identifies extreme conditions likely to revert.

**BUY**: Price < BB lower + MFI < 20 + Williams %R < -80 + Price > SMA200
**SELL**: Price > BB upper + MFI > 80 + Williams %R > -20 + Price < SMA200

### E) Ensemble
Alerts only when 2+ strategies agree on direction. Highest confidence.

## Indicators

**Trend**: SMA (20/50/200), EMA (12/20/26/50)
**Momentum**: RSI, MACD, Stochastic, MFI, ROC, CCI, Williams %R
**Volatility**: Bollinger Bands, ATR, Parabolic SAR
**Volume**: OBV, VWAP, Volume Ratio
**Trend Strength**: ADX, +DI, -DI

## Slack Alert Format

```
🟢 BUY Signal: AAPL
━━━━━━━━━━━━━━━━━━
Strategy: momentum_reversal
Confidence: 85%
Price: $178.25
Confirmations: 4

Reason: RSI(28.5) oversold | Stochastic(18.2) oversold | Price near BB lower | MACD momentum improving

Indicators:
• rsi: 28.50
• stoch_k: 18.20
• macd_hist: 0.15

Patterns:
• MACD crossed above signal line (bullish momentum)

⏰ 2024-01-15 10:30:00 UTC
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding a New Strategy

1. Create a new class in `app/services/strategies.py` extending `Strategy`
2. Implement `check_signal()` method
3. Add to `STRATEGY_REGISTRY`
4. The backtest will automatically evaluate it

### Adding a New Indicator

1. Add the function to `app/services/indicators.py`
2. Add it to `build_indicator_frame()` to compute it automatically
3. Use it in strategies via `df["YOUR_INDICATOR"]`

## Disclaimer

**This software is for educational purposes only. It is not financial advice.**

Trading stocks involves risk. Past performance does not guarantee future results. Always do your own research and consider consulting a financial advisor before making investment decisions.

## License

MIT
