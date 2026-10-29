# Market Monitor

A real-time trading signal system for U.S. equities. It scans a watchlist during NYSE market hours, generates buy and sell signals from five rule-based strategies, scores each signal with a machine learning model that keeps learning from new outcomes, sizes positions with the Kelly Criterion, and paper-trades the results while sending alerts to Slack.

## Results

Paper trading from May 18 to June 29, 2026, across a 10-stock watchlist (AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, AMD, NFLX, DIS):

| Metric | Value |
|---|---|
| Closed trades | 170 |
| Profit factor | 1.12 |
| Win rate | 47.6% |
| Net P&L on closed trades | +$524 |

A profit factor above 1 with a win rate below 50% means winning trades were larger on average than losing trades. Results are from simulated trades on live market data, with no commissions or slippage, and are not a guarantee of future performance.

## How it works

1. **Data.** Pulls intraday price data for each ticker with `yfinance`.
2. **Indicators.** Computes 20+ technical indicators: RSI, MACD, Bollinger Bands, ADX, VWAP, Stochastic, MFI, ATR, OBV and more.
3. **Strategies.** Five strategies look for setups, and each signal needs multiple indicator confirmations:
   - **Momentum reversal:** oversold or overbought readings with improving momentum
   - **Trend following:** price, VWAP, ADX and MACD aligned in one direction
   - **Breakout:** Bollinger Band breaks confirmed by volume
   - **Mean reversion:** extreme readings filtered by the long-term trend
   - **Ensemble:** fires only when two or more strategies agree
4. **Backtest-driven strategy selection.** On startup, the system backtests every strategy on every ticker over the past 30 days and picks the best-performing strategy for each ticker.
5. **ML confidence scoring.** A scikit-learn gradient boosting classifier, trained on backtest outcomes, predicts whether a signal will be profitable one hour later. That prediction is blended with the strategy's rule-based confidence, its historical win rate, pattern strength and how extreme the indicator readings are.
6. **Continuous learning.** As trades resolve, the model is updated with the new outcomes, so its scoring adapts to current market conditions.
7. **Position sizing.** Signals above the confidence threshold are sized with half-Kelly, capped at 20% of the portfolio per trade and a fixed dollar maximum.
8. **Execution and alerts.** The paper trader opens and closes simulated positions, tracks P&L, and sends trade alerts and a daily summary to Slack. A live `dashboard.json` reports prices, signals and model status.

Tickers are analyzed concurrently with `asyncio`, so a full watchlist cycle runs in parallel rather than one ticker at a time.

## Tech stack

Python, asyncio, pandas, NumPy, scikit-learn, yfinance, Pydantic, Slack SDK, pandas-market-calendars

## Project structure

```
app/
├── main.py              # Monitoring loop, Kelly sizing, dashboard
├── config.py            # Settings loaded from .env
├── models/              # Pydantic schemas and portfolio models
└── services/
    ├── yfinance_client.py   # Market data
    ├── indicators.py        # Technical indicators
    ├── patterns.py          # Chart pattern detection
    ├── strategies.py        # The five trading strategies
    ├── backtest.py          # Strategy evaluation
    ├── ml_scorer.py         # ML confidence model
    ├── paper_trader.py      # Simulated portfolio
    ├── alerts.py            # Slack notifications
    └── market_hours.py      # NYSE trading calendar
data/ml_models/          # Saved model and scaler
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your Slack webhook URL
python3 -m app.main
```

The monitor runs the startup backtest, waits for the market to open, then analyzes the watchlist every 60 seconds.

## Configuration

Set these in `.env`:

| Variable | Default | Description |
|---|---|---|
| `SLACK_WEBHOOK_URL` | (required) | Slack incoming webhook for alerts |
| `WATCHLIST` | 10 large-cap stocks | Comma-separated tickers |
| `POLL_INTERVAL_SECONDS` | 60 | Seconds between analysis cycles |
| `SIGNAL_CONFIDENCE_THRESHOLD` | 0.75 | Minimum confidence to act on a signal |
| `MIN_CONFIRMATIONS` | 3 | Indicator confirmations required |
| `BACKTEST_DAYS` | 30 | Lookback for strategy selection |
| `PAPER_TRADING_STARTING_BALANCE` | 100000 | Starting paper capital |
| `PAPER_TRADING_MAX_TRADE_VALUE` | 5000 | Dollar cap per trade |

## Disclaimer

This project is for educational and research purposes only and is not financial advice.
