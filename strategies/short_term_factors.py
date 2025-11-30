"""Helpers to rank the most useful short-term factors via quick backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from strategies.bb import bb_strategy
from strategies.breakout import breakout_strategy
from strategies.macd import macd_strategy
from strategies.rsi import rsi_strategy
from strategies.sma import sma_strategy
from strategies.stoch import StochasticStrategy
from strategies.trend import trend_strategy


@dataclass
class FactorResult:
    """Aggregated statistics for a single factor backtest."""

    name: str
    total_return_pct: float
    sharpe_ratio: float
    win_rate_pct: float


@dataclass
class ParameterTuningResult:
    """Best-performing parameter set for a factor after stability checks."""

    name: str
    params: Dict
    train_return_pct: float
    test_return_pct: float
    sharpe_ratio: float
    win_rate_pct: float
    decay_ratio: float
    score: float


# Strategy candidates tuned toward short-term signals.
SHORT_TERM_FACTORS: Dict[str, Tuple] = {
    "stochastic_cross": (StochasticStrategy, {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3}),
    "rsi_reversal": (rsi_strategy, {"timeperiod": 14, "buy_threshold": 52, "sell_threshold": 50}),
    "macd_hist_flip": (macd_strategy, {}),
    "bollinger_breakout": (bb_strategy, {"window": 14, "nstd": 2}),
    "range_breakout": (breakout_strategy, {"long_window": 30, "short_window": 30}),
    "sma_cross": (sma_strategy, {"sma1": 21, "sma2": 144}),
    "trend_filter": (trend_strategy, {}),
}


# Parameter grids designed for short-term tuning while guarding against overfitting.
SHORT_TERM_PARAMETER_GRIDS: Dict[str, Tuple] = {
    "stochastic_cross": (
        StochasticStrategy,
        [
            {"fastk_period": 9, "slowk_period": 3, "slowd_period": 3},
            {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
            {"fastk_period": 14, "slowk_period": 5, "slowd_period": 5},
            {"fastk_period": 21, "slowk_period": 3, "slowd_period": 3},
        ],
    ),
    "rsi_reversal": (
        rsi_strategy,
        [
            {"timeperiod": 7, "buy_threshold": 55, "sell_threshold": 45},
            {"timeperiod": 14, "buy_threshold": 52, "sell_threshold": 50},
            {"timeperiod": 14, "buy_threshold": 60, "sell_threshold": 50},
            {"timeperiod": 21, "buy_threshold": 55, "sell_threshold": 45},
        ],
    ),
    "macd_hist_flip": (
        macd_strategy,
        [
            {"fastperiod": 8, "slowperiod": 21, "signalperiod": 5},
            {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            {"fastperiod": 15, "slowperiod": 30, "signalperiod": 9},
        ],
    ),
    "bollinger_breakout": (
        bb_strategy,
        [
            {"window": 10, "nstd": 2},
            {"window": 14, "nstd": 2},
            {"window": 20, "nstd": 1.8},
        ],
    ),
    "range_breakout": (
        breakout_strategy,
        [
            {"long_window": 20, "short_window": 10},
            {"long_window": 30, "short_window": 20},
            {"long_window": 55, "short_window": 21},
        ],
    ),
    "sma_cross": (
        sma_strategy,
        [
            {"sma1": 10, "sma2": 50},
            {"sma1": 21, "sma2": 89},
            {"sma1": 21, "sma2": 144},
        ],
    ),
    "trend_filter": (
        trend_strategy,
        [
            {"n1": 10, "n2": 20},
            {"n1": 20, "n2": 40},
            {"n1": 30, "n2": 60},
        ],
    ),
}


def _first_column(stats: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(stats, pd.DataFrame):
        return stats.iloc[:, 0]
    return stats


def _extract_result(name: str, portfolio) -> FactorResult:
    stats = _first_column(portfolio.stats())

    return FactorResult(
        name=name,
        total_return_pct=float(stats.get("Total Return [%]", float("nan"))),
        sharpe_ratio=float(stats.get("Sharpe Ratio", float("nan"))),
        win_rate_pct=float(stats.get("Win Rate [%]", float("nan"))),
    )


def _train_test_split(ohlcv, train_ratio: float):
    """Chronologically split OHLCV data into train and test slices."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

    cutoff = int(len(ohlcv) * train_ratio)
    if cutoff == 0 or cutoff >= len(ohlcv):
        raise ValueError("train_ratio results in empty train or test set")

    return ohlcv.iloc[:cutoff], ohlcv.iloc[cutoff:]


def _decay_ratio(train_return_pct: float, test_return_pct: float) -> float:
    denom = train_return_pct if abs(train_return_pct) > 1e-9 else 1e-9
    return test_return_pct / denom


def rank_short_term_factors(
    ohlcv,
    freq: str = "4h",
    candidates: Dict[str, Tuple] | None = None,
    top_n: int = 3,
) -> List[FactorResult]:
    """Backtest and rank the strongest short-term factors.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        OHLCV data frame compatible with the strategy backtest API.
    freq : str, optional
        Trading frequency, by default ``"4h"`` for intraday crypto data.
    candidates : dict, optional
        Mapping of factor name to ``(strategy, variables)``. Uses
        :data:`SHORT_TERM_FACTORS` when omitted.
    top_n : int, optional
        Number of top factors to return, by default ``3``.

    Returns
    -------
    list[FactorResult]
        Ranked factor summaries sorted by ``total_return_pct`` descending.
    """

    rankings: List[FactorResult] = []
    for name, (strategy, variables) in (candidates or SHORT_TERM_FACTORS).items():
        portfolio = strategy.backtest(ohlcv, variables=variables, freq=freq, plot=False)
        rankings.append(_extract_result(name, portfolio))

    rankings.sort(key=lambda r: r.total_return_pct, reverse=True)
    return rankings[:top_n]


def tune_short_term_factors(
    ohlcv,
    freq: str = "4h",
    parameter_grids: Dict[str, Tuple] | None = None,
    train_ratio: float = 0.7,
    min_decay: float = 0.5,
    top_n: int = 5,
) -> List[ParameterTuningResult]:
    """Search for robust short-term factor parameters using train/test splits.

    This helper grid-searches parameter sets for each factor, ranks them by
    out-of-sample performance, and filters out unstable configurations where
    test returns collapse relative to the training period.

    Parameters
    ----------
    ohlcv : pandas.DataFrame
        OHLCV data frame compatible with the strategy backtest API.
    freq : str, optional
        Trading frequency passed through to the strategy backtest calls.
    parameter_grids : dict, optional
        Mapping of factor name to ``(strategy, list[parameter_dict])``. Uses
        :data:`SHORT_TERM_PARAMETER_GRIDS` when omitted.
    train_ratio : float, optional
        Portion of samples used for training. The remainder is reserved for
        forward validation.
    min_decay : float, optional
        Minimum acceptable ratio of ``test_return_pct / train_return_pct``.
        Lower decay scores are discarded as likely overfit.
    top_n : int, optional
        Number of tuned factors to return, by default ``5``.

    Returns
    -------
    list[ParameterTuningResult]
        Top-performing parameter sets sorted by stability-aware ``score``.
    """

    train, test = _train_test_split(ohlcv, train_ratio)

    parameter_space = parameter_grids or SHORT_TERM_PARAMETER_GRIDS
    tuned: List[ParameterTuningResult] = []

    for name, (strategy, grid) in parameter_space.items():
        for params in grid:
            train_portfolio = strategy.backtest(train, variables=params, freq=freq, plot=False)
            test_portfolio = strategy.backtest(test, variables=params, freq=freq, plot=False)

            train_result = _extract_result(name, train_portfolio)
            test_result = _extract_result(name, test_portfolio)

            decay = _decay_ratio(train_result.total_return_pct, test_result.total_return_pct)
            if decay < min_decay:
                continue

            score = test_result.total_return_pct * decay
            tuned.append(
                ParameterTuningResult(
                    name=name,
                    params=params,
                    train_return_pct=train_result.total_return_pct,
                    test_return_pct=test_result.total_return_pct,
                    sharpe_ratio=test_result.sharpe_ratio,
                    win_rate_pct=test_result.win_rate_pct,
                    decay_ratio=decay,
                    score=score,
                )
            )

    tuned.sort(key=lambda r: r.score, reverse=True)
    return tuned[:top_n]


def fetch_and_rank_top_factors(symbol: str = "BTCUSDT", interval: str = "4h", top_n: int = 3) -> List[FactorResult]:
    """Convenience wrapper to fetch data then rank short-term factors."""

    import finlab_crypto
    from finlab_crypto import crawler

    finlab_crypto.setup()
    ohlcv = crawler.get_all_binance(symbol, interval)

    return rank_short_term_factors(ohlcv, freq=interval, top_n=top_n)


def fetch_and_tune_top_factors(
    symbol: str = "BTCUSDT",
    interval: str = "4h",
    top_n: int = 5,
    train_ratio: float = 0.7,
    min_decay: float = 0.5,
):
    """Convenience wrapper to fetch data then tune robust short-term factors."""

    import finlab_crypto
    from finlab_crypto import crawler

    finlab_crypto.setup()
    ohlcv = crawler.get_all_binance(symbol, interval)

    return tune_short_term_factors(
        ohlcv,
        freq=interval,
        train_ratio=train_ratio,
        min_decay=min_decay,
        top_n=top_n,
    )
