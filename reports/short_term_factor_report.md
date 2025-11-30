# Short-term factor tuning report

This helper-oriented report explains how to surface five resilient short-term factors with in-repo utilities.

## Factors and grids
The tuner sweeps curated parameter grids for these factors:

- Stochastic oscillator crosses
- RSI reversal bands
- MACD histogram flips
- Bollinger Band breakouts
- Range breakouts (recent highs/lows)
- SMA crosses
- Trend filters (double-smoothing)

Each grid stays within short-horizon settings (fast lookbacks, tight thresholds) to emphasize intraday responsiveness.

## Workflow (train/test to avoid decay)
1. Fetch data (e.g., `BTCUSDT 4h`) and split chronologically (default 70% train, 30% forward test).
2. Backtest every parameter set on train and test slices.
3. Compute a decay ratio (`test_return_pct / train_return_pct`) to detect overfit settings.
4. Drop configurations whose decay ratio falls below the stability floor (default `0.5`).
5. Rank remaining candidates by `test_return_pct * decay_ratio`, favoring strong and stable out-of-sample performance.
6. Return the top 5 tuned factors with their parameters and stats (returns, Sharpe, win rate, decay score).

## How to run
```python
from strategies.short_term_factors import fetch_and_tune_top_factors

results = fetch_and_tune_top_factors(
    symbol="BTCUSDT",
    interval="4h",
    top_n=5,
    train_ratio=0.7,
    min_decay=0.6,
)

for res in results:
    print(res)
```

## Interpreting output
- **params**: the recommended short-term parameters for that factor.
- **train/test returns**: directional performance on the split datasets.
- **decay_ratio**: closer to 1 implies similar behavior out-of-sample; values above the `min_decay` guard against rapid fade.
- **score**: `test_return_pct * decay_ratio`; use it to quickly order robust candidates.

Re-run periodically with fresh data to monitor whether the decay ratio remains healthy, signaling that the parameters are not rapidly deteriorating in live conditions.
