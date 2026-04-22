"""
Each metric takes two 1D arrays of equal length
and returns a non-negative scalar distance. Smaller = more similar
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np


def _znorm(x: np.ndarray) -> np.ndarray:
    """Z-normalize a 1D array. Guards against zero-variance windows."""
    mu = x.mean()
    sd = x.std()
    if sd < 1e-12:
        return x - mu  # avoid division by ~0
    return (x - mu) / sd


def euclidean_znorm(q: np.ndarray, c: np.ndarray) -> float:
    """Euclidean distance between z-normalized return series.
    Equivalent (up to a monotone transform) to 1 - Pearson correlation;
    matches shape while ignoring scale and level."""
    qz, cz = _znorm(q), _znorm(c)
    return float(np.linalg.norm(qz - cz))


def correlation_distance(q: np.ndarray, c: np.ndarray) -> float:
    """1 - Pearson correlation. Bounded in [0, 2]. Pure shape match."""
    if q.std() < 1e-12 or c.std() < 1e-12:
        return 2.0
    rho = np.corrcoef(q, c)[0, 1]
    return float(1.0 - rho)


def _dtw_envelope(q_znormed: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    """Precompute the Sakoe-Chiba upper/lower envelope of a z-normalized query.

    Used by lb_keogh to avoid redundant recomputation across candidates.
    `band` must match the band used in dtw_constrained.
    """
    n = len(q_znormed)
    U = np.array([q_znormed[max(0, i - band) : i + band + 1].max() for i in range(n)])
    L = np.array([q_znormed[max(0, i - band) : i + band + 1].min() for i in range(n)])
    return U, L


def lb_keogh(c_znormed: np.ndarray, U: np.ndarray, L: np.ndarray) -> float:
    """LB_Keogh lower bound on DTW distance. O(L), no warping path needed.

    Returns a value guaranteed to be <= dtw_constrained(q, c). If this lower
    bound already exceeds the current k-th best distance, the full DTW can be
    skipped entirely.
    """
    above = np.maximum(c_znormed - U, 0.0)
    below = np.maximum(L - c_znormed, 0.0)
    return float(np.sqrt(np.sum((above + below) ** 2)))


def dtw_constrained(q: np.ndarray, c: np.ndarray, band_frac: float = 0.1) -> float:
    """Dynamic Time Warping with a Sakoe-Chiba band constraint.

    The band restricts warping to |i - j| <= band, cutting cost from O(L^2)
    to O(L * band).
    """
    qz, cz = _znorm(q), _znorm(c)
    n, m = len(qz), len(cz)
    band = max(1, int(band_frac * max(n, m)))

    # Construct full cost matrix with only entries inside global constriant band being filled.
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0  # Base case for recurrence
    for i in range(1, n + 1):
        j_lo = max(1, i - band)
        j_hi = min(m, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = (qz[i - 1] - cz[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])  # main recursive function
    return float(np.sqrt(D[n, m]))


MetricName = Literal["euclidean_znorm", "correlation", "dtw"]

METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "euclidean_znorm": euclidean_znorm,
    "correlation": correlation_distance,
    "dtw": dtw_constrained,
}
