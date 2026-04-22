"""
Speed benchmark for AnalogMiner configurations.

Compares query time across metrics and with/without vol_band filtering.
Runs N timed find() calls per configuration and reports median latency
and relative speedup vs the baseline (euclidean_znorm, no vol_band).

Usage:
    uv run python examples/benchmark_speed.py

Data: examples/data/SP500.csv  (columns: observation_date, SP500)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_miner import AnalogMiner

# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "data" / "SP500.csv"
DATE_COL = "observation_date"
PRICE_COL = "SP500"

WINDOW_SIZE = 60
FORECAST_HORIZON = 20
TOP_K = 15
N_QUERIES = 50  # number of timed find() calls per config
STRIDE = 20  # spacing between query points
# ---------------------------------------------------------------------------


@dataclass
class Config:
    name: str
    metric: str
    vol_band: tuple[float, float] | None


CONFIGS: list[Config] = [
    Config("baseline  (euclidean, no vol filter)", "euclidean_znorm", None),
    Config("euclid  + vol_band (0.7–1.5)", "euclidean_znorm", (0.7, 1.5)),
    Config("correlation, no vol filter", "correlation", None),
    Config("correlation + vol_band (0.7–1.5)", "correlation", (0.7, 1.5)),
    Config("dtw,        no vol filter", "dtw", None),
    Config("dtw       + vol_band (0.7–1.5)", "dtw", (0.7, 1.5)),
]


def load_prices(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.dropna(subset=[DATE_COL, PRICE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    return pd.Series(df[PRICE_COL].values.astype(float), index=df[DATE_COL])


def pick_query_indices(prices: pd.Series, n: int, stride: int) -> list[int]:
    t_min = 3 * WINDOW_SIZE + FORECAST_HORIZON
    t_max = len(prices) - FORECAST_HORIZON - 2
    candidates = list(range(t_min, t_max + 1, stride))
    # Take up to n evenly spaced from the candidate list
    if len(candidates) <= n:
        return candidates
    step = len(candidates) // n
    return [candidates[i * step] for i in range(n)]


def time_config(cfg: Config, prices: pd.Series, query_indices: list[int]) -> list[float]:
    miner = AnalogMiner(
        window_size=WINDOW_SIZE,
        forecast_horizon=FORECAST_HORIZON,
        metric=cfg.metric,
        top_k=TOP_K,
        vol_band=cfg.vol_band,
    )
    times: list[float] = []
    for t in query_indices:
        history = prices.iloc[: t + 1]
        t0 = time.perf_counter()
        try:
            miner.find(history)
        except Exception:
            continue
        times.append(time.perf_counter() - t0)
    return times


def report(results: list[tuple[Config, list[float]]], baseline_median: float) -> None:
    print("=" * 70)
    print("  AnalogMiner — Speed Benchmark")
    print(f"  L={WINDOW_SIZE}  H={FORECAST_HORIZON}  K={TOP_K}  N={N_QUERIES} queries each")
    print("=" * 70)
    print(f"  {'Configuration':<42} {'median ms':>9}  {'vs baseline':>12}")
    print("  " + "-" * 66)
    for cfg, times in results:
        if not times:
            print(f"  {cfg.name:<42}  {'n/a':>9}  {'n/a':>12}")
            continue
        med = float(np.median(times)) * 1000
        ratio = med / (baseline_median * 1000)
        rel = f"{ratio:.2f}x" if ratio >= 1 else f"{ratio:.2f}x (faster)"
        print(f"  {cfg.name:<42}  {med:>8.1f}ms  {rel:>12}")
    print("=" * 70)


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}")
        print("Place SP500.csv at examples/data/SP500.csv")
        sys.exit(1)

    print(f"Loading prices from {DATA_PATH} ...")
    prices = load_prices(DATA_PATH)
    print(f"  {len(prices)} bars  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    print()

    query_indices = pick_query_indices(prices, N_QUERIES, STRIDE)
    print(f"Timing {len(query_indices)} queries per configuration ...\n")

    results: list[tuple[Config, list[float]]] = []
    for cfg in CONFIGS:
        print(f"  [{cfg.name}] ...", end=" ", flush=True)
        times = time_config(cfg, prices, query_indices)
        med_ms = float(np.median(times)) * 1000 if times else float("nan")
        print(f"{med_ms:.1f} ms median")
        results.append((cfg, times))

    baseline_times = results[0][1]
    baseline_median = float(np.median(baseline_times)) if baseline_times else 1.0

    print()
    report(results, baseline_median)
