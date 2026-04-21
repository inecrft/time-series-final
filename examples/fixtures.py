"""Synthetic price data for examples and benchmarks."""

import numpy as np
import pandas as pd


def make_synthetic_prices(n_days: int = 4000, seed: int = 7) -> pd.Series:
    """Fabricate a price series that mixes drift, vol regimes, and a few
    recurring 'signature' patterns. The recurring patterns give the miner
    something real to find."""
    rng = np.random.default_rng(seed)

    mu = 0.00035
    base_sigma = 0.009
    vol = np.full(n_days, base_sigma)
    for center in [700, 1900, 3100]:
        width = 120
        lo, hi = max(0, center - width), min(n_days, center + width)
        vol[lo:hi] *= 2.5

    eps = rng.standard_normal(n_days)
    rets = mu + vol * eps

    signature = np.concatenate(
        [
            np.linspace(0.0, -0.015, 20),
            np.linspace(-0.015, 0.005, 20),
        ]
    )
    for anchor in [500, 1500, 2700, 3700]:
        rets[anchor : anchor + 40] = signature + 0.002 * rng.standard_normal(40)

    prices = 100 * np.exp(np.cumsum(rets))
    dates = pd.bdate_range(start="2008-01-02", periods=n_days)
    return pd.Series(prices, index=dates, name="close")
