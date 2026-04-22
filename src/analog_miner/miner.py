"""
Given a query window of recent prices, find the K most similar historical
windows and return the price paths that followed them. Output is framed as
a distribution of reference cases, not a point forecast.

Design notes:
- Works on log returns, not raw prices: returns are approximately stationary
  and scale-invariant, which is what we actually want to match on.
- Z-normalization is applied per-window inside the distance function so
  shape dominates over level/scale differences between eras.
- An anti-lookahead gap of (L + H) bars is enforced around the query.
- Top-K selection uses a minimum-separation rule to avoid returning near-
  duplicate overlapping windows from the same historical episode.
"""

from __future__ import annotations

import heapq

import numpy as np
import pandas as pd

from .metrics import METRIC_REGISTRY, MetricName, _dtw_envelope, _znorm, lb_keogh
from .models import AnalogMatch, AnalogResult


class AnalogMiner:
    """Find historical analogs to a query window of price data.

    Parameters
    ----------
    window_size : int
        Length L of the query and candidate windows, in bars.
    forecast_horizon : int
        Length H of the forward period returned after each match.
    metric : str
        Key into METRIC_REGISTRY. Default "euclidean_znorm" is the right
        choice for daily equity data: fast, shape-based, scale-invariant.
    top_k : int
        Number of analog matches to return.
    min_separation : int or None
        Minimum bar-distance between selected matches' start indices. If
        None, defaults to window_size // 2. Prevents the top-K from being
        filled with near-duplicate overlapping windows.
    vol_band : tuple or None
        (low, high) multiplicative band on realized vol relative to the
        query. Candidates outside this band are pruned before the distance
        metric runs. None disables the filter.
    lookahead_gap : int or None
        Minimum bar-distance between any candidate window and the query
        window (and its forward period). None defaults to window_size +
        forecast_horizon, which is the safe choice for backtesting.
    """

    def __init__(
        self,
        window_size: int,
        forecast_horizon: int,
        metric: MetricName = "euclidean_znorm",
        top_k: int = 10,
        min_separation: int | None = None,
        vol_band: tuple[float, float] | None = (0.5, 2.0),
        lookahead_gap: int | None = None,
    ):
        if window_size < 5:
            raise ValueError("window_size must be >= 5 to be meaningful")
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"unknown metric {metric!r}; options: {list(METRIC_REGISTRY)}")

        self.L = window_size
        self.H = forecast_horizon
        self.metric_name = metric
        self.metric_fn = METRIC_REGISTRY[metric]
        self.top_k = top_k
        self.min_separation = min_separation if min_separation is not None else window_size // 2
        self.vol_band = vol_band
        self.lookahead_gap = lookahead_gap if lookahead_gap is not None else window_size + forecast_horizon

    # ---- core entry point -------------------------------------------------

    def find(
        self,
        prices: pd.Series,
        query_end: pd.Timestamp | str | int | None = None,
    ) -> AnalogResult:
        """Run the full pipeline and return matches.

        `prices` must be adjusted close (splits & dividends applied).
        `query_end` is the last bar of the query window. If None, uses the
        final bar of `prices`. String dates and integer positions are both
        accepted for convenience.
        """
        prices = self._validate_series(prices)

        # Convert to "log returns"
        returns = np.log(prices / prices.shift(1)).dropna()

        q_end_idx = self._resolve_index(returns, query_end)
        q_start_idx = q_end_idx - self.L + 1
        if q_start_idx < 0:
            raise ValueError("query window extends before start of data")

        query_returns = returns.iloc[q_start_idx : q_end_idx + 1]
        query_prices = prices.loc[query_returns.index]
        q_arr = query_returns.values
        q_vol = float(q_arr.std())
        q_cumret = float(q_arr.sum())

        # Build candidate windows
        candidate_starts = self._candidate_start_indices(
            n_returns=len(returns),
            q_start_idx=q_start_idx,
            q_end_idx=q_end_idx,
        )

        # Filter to only candidate window with standard deviation within certain range of the query window
        if self.vol_band is not None:
            lo, hi = self.vol_band
            candidate_starts = [
                s for s in candidate_starts if lo * q_vol <= returns.values[s : s + self.L].std() <= hi * q_vol
            ]

        # Score the surviving candidates
        scored: list[tuple[float, int]] = []
        ret_vals = returns.values

        # For DTW, precompute the query envelope once and use LB_Keogh to skip
        # candidates whose lower bound already exceeds the current k-th best.
        use_lb = self.metric_name == "dtw"
        if use_lb:
            band = max(1, int(0.1 * self.L))  # must match band_frac in dtw_constrained
            U_env, L_env = _dtw_envelope(_znorm(q_arr), band)

        # Max-heap (negated distances) tracking the k-th best distance seen so
        # far — used as the LB_Keogh pruning threshold.
        heap: list[tuple[float, int]] = []
        kth_best = float("inf")

        for s in candidate_starts:
            c_arr = ret_vals[s : s + self.L]
            if use_lb and lb_keogh(_znorm(c_arr), U_env, L_env) >= kth_best:
                continue
            d = self.metric_fn(q_arr, c_arr)
            scored.append((d, s))
            # Keep a max-heap of size top_k to maintain the pruning threshold.
            if len(heap) < self.top_k:
                heapq.heappush(heap, (-d, s))
                if len(heap) == self.top_k:
                    kth_best = -heap[0][0]
            elif d < kth_best:
                heapq.heapreplace(heap, (-d, s))
                kth_best = -heap[0][0]

        scored.sort(key=lambda t: t[0])

        # Find Top-K with minimum separation
        selected = self._select_with_separation(scored)
        matches = [self._build_match(s, d, prices, returns) for d, s in selected]

        return AnalogResult(
            query_window=query_prices,
            query_returns=query_returns,
            query_vol=q_vol,
            query_cum_return=q_cumret,
            matches=matches,
        )

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _validate_series(prices: pd.Series) -> pd.Series:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pandas Series")
        if prices.isna().any():
            raise ValueError("prices contains NaN; clean before passing in")
        if (prices <= 0).any():
            raise ValueError("prices must be strictly positive")
        if not prices.index.is_monotonic_increasing:
            raise ValueError("price index must be sorted ascending")
        return prices.astype(float)

    @staticmethod
    def _resolve_index(returns: pd.Series, query_end) -> int:
        """Resolve a user-supplied query-end spec to an integer index into the returns series."""
        if query_end is None:
            return len(returns) - 1
        if isinstance(query_end, (int, np.integer)):
            return int(query_end)
        ts = pd.Timestamp(query_end)
        if ts not in returns.index:
            # Pick the most recent bar at or before the requested date
            pos = returns.index.searchsorted(ts, side="right") - 1
            if pos < 0:
                raise ValueError(f"query_end {query_end} before data starts")
            return int(pos)
        return int(returns.index.get_loc(ts))

    def _candidate_start_indices(self, n_returns: int, q_start_idx: int, q_end_idx: int) -> list[int]:
        """All valid start indices whose window AND forward period fit in the data
        and are separated from the query by at least `lookahead_gap`."""
        last_valid_start = n_returns - self.L - self.H
        if last_valid_start < 0:
            return []
        gap = self.lookahead_gap
        starts = []
        for s in range(0, last_valid_start + 1):
            end = s + self.L - 1
            # Make sure candidate window is not too close to the query. Separate by `gap` at the least.
            if end >= q_start_idx - gap and s <= q_end_idx + gap:
                continue
            starts.append(s)
        return starts

    def _select_with_separation(self, scored: list[tuple[float, int]]) -> list[tuple[float, int]]:
        """Greedy top-K selection enforcing `min_separation` between start point of selected windows."""
        selected: list[tuple[float, int]] = []
        for d, s in scored:
            if all(abs(s - s2) >= self.min_separation for _, s2 in selected):
                selected.append((d, s))
                if len(selected) >= self.top_k:
                    break
        return selected

    def _window_bounds(self, start_idx: int) -> tuple[int, int, int, int]:
        """Compute (start_idx, end_idx, fwd_start, fwd_end) for a candidate window."""
        end_idx = start_idx + self.L - 1
        fwd_start = end_idx + 1
        fwd_end = fwd_start + self.H - 1
        return start_idx, end_idx, fwd_start, fwd_end

    def _build_match(
        self,
        start_idx: int,
        distance: float,
        prices: pd.Series,
        returns: pd.Series,
    ) -> AnalogMatch:
        """Assemble an AnalogMatch from a scored candidate.

        returns[t] is the return from prices[t] to prices[t+1] under the
        log(P/P.shift(1)).dropna() convention, so the price at position
        `start_idx` in `returns` is the price at position `start_idx + 1`
        in the original prices series.
        """
        _, end_idx, fwd_start, fwd_end = self._window_bounds(start_idx)

        window_returns = returns.iloc[start_idx : end_idx + 1]
        forward_returns = returns.iloc[fwd_start : fwd_end + 1]

        # Map returns-indexed slices back to price timestamps
        window_prices = prices.loc[window_returns.index]
        forward_prices = prices.loc[forward_returns.index]

        return AnalogMatch(
            start_idx=start_idx,
            end_idx=end_idx,
            forward_start_idx=fwd_start,
            forward_end_idx=fwd_end,
            distance=distance,
            window_prices=window_prices,
            window_returns=window_returns,
            forward_prices=forward_prices,
            forward_returns=forward_returns,
            realized_vol=float(window_returns.std()),
            cumulative_return=float(window_returns.sum()),
        )
