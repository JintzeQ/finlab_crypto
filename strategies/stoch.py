"""Stochastic Oscillator crossover strategy built with TalibStrategy.

The returned :data:`StochasticStrategy` object exposes the same ``backtest``
API as other strategies in this codebase. See :func:`run_sample_backtest`
below for a minimal runnable example on BTCUSDT 4h data.
"""

from finlab_crypto.talib_strategy import TalibStrategy


StochasticStrategy = TalibStrategy(
    'STOCH',
    entries=lambda ohlcv, stoch: (
        (stoch['slowk'] > stoch['slowd'])
        & (stoch['slowk'].shift(1) <= stoch['slowd'].shift(1))
        & (stoch['slowk'] < 30)
    ),
    exits=lambda ohlcv, stoch: (
        ((stoch['slowk'] < stoch['slowd']) & (stoch['slowk'].shift(1) >= stoch['slowd'].shift(1)))
        | (stoch['slowk'] > 70)
    ),
)


def run_sample_backtest():
    """Fetch data and run a quick backtest of :data:`StochasticStrategy`.

    The example keeps network access and plotting disabled by default to make
    it easier to run in headless environments. Enable plotting by passing
    ``plot=True`` to :meth:`~finlab_crypto.strategy.Strategy.backtest`.
    """

    import finlab_crypto
    from finlab_crypto import crawler

    finlab_crypto.setup()
    ohlcv = crawler.get_all_binance('BTCUSDT', '4h')

    variables = {
        # Talib STOCH parameters: https://mrjbq7.github.io/ta-lib/func_groups/momentum_indicators.html
        'fastk_period': 14,
        'slowk_period': 3,
        'slowd_period': 3,
    }

    return StochasticStrategy.backtest(ohlcv, variables, freq='4h', plot=False)
