# Time Series Final

Find historical price windows that most closely resemble a query window, then return the price paths that followed them as a distribution (not a point forecast).

## How it works

1. Convert prices to log returns (stationary, scale-invariant)
2. Z-normalize each candidate window so shape dominates over level
3. Score every valid historical window against the query using a distance metric
4. Select the top-K matches with a minimum-separation rule to avoid near-duplicate windows from the same episode
5. Return the forward price paths of each match, rebased for direct comparison

Anti-lookahead is enforced by default: no candidate window whose forward period overlaps the query will ever appear in results.

## Installation

Requires Python 3.12+. Uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone https://github.com/inecrft/time-series-final.git
cd time-series-final
uv sync
```

## Quick start

```python
import pandas as pd
from analog_miner import AnalogMiner

prices: pd.Series = ...  # adjusted close, DatetimeIndex, no NaNs

miner = AnalogMiner(
    window_size=60,       # look-back window in bars
    forecast_horizon=20,  # forward period to return
    metric="euclidean_znorm",
    top_k=10,
)

result = miner.find(prices)

# Distribution of what happened next across all matches
print(result.forward_distribution(quantiles=(0.1, 0.5, 0.9)))

# How much do the matches disagree?
print(result.dispersion())
```

Query at a specific historical date instead of the end of the series:

```python
result = miner.find(prices, query_end="2020-03-15")
```

## Distance metrics

| Key | Description | Best for |
|---|---|---|
| `euclidean_znorm` | Euclidean distance after z-normalization. Equivalent to 1 − Pearson correlation up to a monotone transform. | Default; fast, shape-based |
| `correlation` | 1 − Pearson correlation. Bounded in [0, 2]. | Pure shape match, magnitude ignored |
| `dtw` | Dynamic Time Warping with a Sakoe-Chiba band constraint. | Allows local warping; slower |

```python
miner = AnalogMiner(window_size=40, forecast_horizon=15, metric="dtw", top_k=5)
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `window_size` | — | Length of the query and candidate windows in bars |
| `forecast_horizon` | — | Length of the forward period after each match |
| `top_k` | `10` | Number of matches to return |
| `min_separation` | `window_size // 2` | Minimum bar-distance between match start indices; prevents near-duplicate results |
| `vol_band` | `(0.5, 2.0)` | Prune candidates whose realized vol falls outside `(lo × query_vol, hi × query_vol)`. `None` to disable |
| `lookahead_gap` | `window_size + forecast_horizon` | Minimum bar-distance between any candidate and the query; `None` to disable |

## Result API

```python
result.matches                                    # list[AnalogMatch]
result.forward_paths(normalize=True)              # DataFrame — one column per match
result.forward_distribution((0.1, 0.5, 0.9))     # quantile bands over time
result.dispersion()                               # avg cross-sectional std; high = matches disagree
```

Each `AnalogMatch` exposes:

```python
match.window_prices      # pd.Series — the matched historical window
match.forward_prices     # pd.Series — what happened next
match.distance           # float — distance score (lower = more similar)
match.realized_vol       # float — std of daily returns in the window
match.cumulative_return  # float — total log return over the window
```

## Running the demo

```bash
uv run python examples/demo.py
```

Generates a 4 000-bar synthetic price series with injected recurring patterns and runs the miner with three different metrics.

## Project structure

```
src/
  analog_miner/
    metrics.py   # distance functions and registry
    models.py    # AnalogMatch, AnalogResult dataclasses
    miner.py     # AnalogMiner class
    __init__.py
examples/
  demo.py        # end-to-end demonstration
  fixtures.py    # synthetic price data generator
  reporting.py   # console display helpers
```
