# Binance BTCUSDT microstructure alpha pilot

This experiment tests short-horizon trade-flow alpha using Binance USDⓈ-M BTCUSDT daily `aggTrades` archives.

## Design

- Data: 2026-08-10 through 2026-08-16.
- Train/calibration: 2026-08-10 through 2026-08-13.
- Out-of-sample: 2026-08-14 through 2026-08-16.
- Event grid: 500 ms.
- Execution latency proxy: enter one 500 ms bar after the signal.
- Holding periods: 1 s, 5 s, 30 s.
- Non-overlapping positions for each strategy/horizon.
- Execution cost sensitivity: 0, 0.5, 1, 2, 5 bp per side.

## Signals

1. `flow_cont`: trade in the direction of extreme 5-second aggressive notional imbalance. The extreme-flow threshold is the 95th percentile estimated only on the training sample.
2. `absorption_rev`: trade against extreme aggressive flow when 5-second notional is at least median and contemporaneous 5-second price impact is in the low-impact quartile. This is a trade-only absorption proxy, not a true L2 replenishment detector.
3. `flow_persistence`: trade in the direction of three consecutive strong same-sign 1-second flow observations.

## Diagnostics

The run also reports same-sign continuation probability and autocorrelation of 1-second flow imbalance at 0.5, 1, 2.5, 5, and 10-second lags. These are footprint diagnostics only; they cannot identify a specific counterparty or algorithm.

## Important limitation

`aggTrades` does not reconstruct historical bid/ask queues. The price used here is the last traded price in each 500 ms bucket. Therefore this is a first-pass trade-flow study. Any promising signal should next be retested with recorded diff-depth/order-book data and a pessimistic fill/queue model before deployment.

The GitHub Actions workflow writes `results/summary.md`, `metrics.csv`, `daily_oos.csv`, `events.csv`, `footprint.csv`, and `thresholds.json` and uploads them as an artifact.
