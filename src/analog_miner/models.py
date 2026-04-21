from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd


@dataclass
class AnalogMatch:
    """A single historical analog and its forward period."""

    start_idx: int
    end_idx: int
    forward_start_idx: int
    forward_end_idx: int
    distance: float
    window_prices: pd.Series
    window_returns: pd.Series
    forward_prices: pd.Series
    forward_returns: pd.Series
    realized_vol: float
    cumulative_return: float


@dataclass
class AnalogResult:
    """Full output of a single analog-mining query."""

    query_window: pd.Series
    query_returns: pd.Series
    query_vol: float
    query_cum_return: float
    matches: list[AnalogMatch] = field(default_factory=list)

    def forward_paths(self, normalize: bool = True) -> pd.DataFrame:
        """Stack the forward price paths of all matches as columns.

        When normalize=True each path is rebased so its first bar equals 1.0,
        making paths directly comparable as return curves regardless of the
        absolute price levels they occurred at.
        """
        cols = {}
        for i, m in enumerate(self.matches):
            p = m.forward_prices.values.astype(float)
            if normalize:
                p = p / p[0]
            cols[f"match_{i}_d{m.distance:.3f}"] = p
        return pd.DataFrame(cols)

    def forward_distribution(self, quantiles: Sequence[float] = (0.1, 0.5, 0.9)) -> pd.DataFrame:
        """Aggregate forward paths into quantile bands.

        Wide bands mean history is not speaking clearly at this moment.
        """
        paths = self.forward_paths(normalize=True)
        return paths.quantile(quantiles, axis=1).T

    def dispersion(self) -> float:
        """Average cross-sectional std of normalized forward paths.

        High dispersion means the matches disagree about what comes next.
        """
        paths = self.forward_paths(normalize=True)
        return float(paths.std(axis=1).mean())
