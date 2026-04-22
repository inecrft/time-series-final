"""
LLM tool-call interface for analog_miner.

Exposes `find_pattern_analogs` as a self-contained function whose parameters
are all JSON-serializable (strings, ints, floats) and whose return value is a
plain dict that can be passed directly back to an LLM as a tool result.

Also exports `ANALOG_TOOL_SCHEMA`, the Anthropic tool-use JSON schema that
tells the model what parameters the function accepts.
"""

from __future__ import annotations

import pandas as pd

from .metrics import MetricName
from .miner import AnalogMiner

ANALOG_TOOL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "find_pattern_analogs",
        "description": (
            "Find historical price patterns similar to a specified date range and return "
            "their subsequent forward price distributions. Use this when the user asks to "
            "find patterns similar to a given time period, e.g. 'find analogs to Jan–Apr 2018'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prices_csv": {
                    "type": "string",
                    "description": "Path to a CSV file with a date index and a price column.",
                },
                "query_start": {
                    "type": "string",
                    "description": "Start date of the query pattern window, e.g. '2018-01-01' or 'Jan 2018'.",
                },
                "query_end": {
                    "type": "string",
                    "description": "End date of the query pattern window, e.g. '2018-04-30' or 'April 2018'.",
                },
                "forecast_horizon": {
                    "type": "integer",
                    "description": "Number of bars to look forward after each matched pattern.",
                    "default": 20,
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of analog matches to return.",
                    "default": 10,
                },
                "metric": {
                    "type": "string",
                    "enum": ["euclidean_znorm", "correlation", "dtw"],
                    "description": (
                        "Distance metric for pattern similarity. "
                        "'euclidean_znorm' (default) is fast and shape-based. "
                        "'correlation' ignores amplitude. "
                        "'dtw' allows local time warping but is slower."
                    ),
                    "default": "euclidean_znorm",
                },
                "price_column": {
                    "type": "string",
                    "description": (
                        "Name of the price column in the CSV. "
                        "If omitted, the function prefers 'Adj Close', then 'Close', "
                        "then the first numeric column."
                    ),
                },
            },
            "required": ["prices_csv", "query_start", "query_end"],
        },
    },
}


def find_pattern_analogs(
    prices_csv: str,
    query_start: str,
    query_end: str,
    forecast_horizon: int = 20,
    top_k: int = 15,
    metric: MetricName = "dtw",
    price_column: str | None = None,
    vol_band: tuple[float, float] | None = (0.7, 1.5),
) -> dict:
    """Find historical analogs to the price pattern between *query_start* and *query_end*.

    Parameters
    ----------
    prices_csv : str
        Path to a CSV with a date index and a price column.
    query_start : str
        Start date of the pattern to match, e.g. ``"2018-01-01"`` or ``"Jan 2018"``.
    query_end : str
        End date of the pattern to match, e.g. ``"2018-04-30"`` or ``"April 2018"``.
    forecast_horizon : int
        Bars to examine after each matched analog window.
    top_k : int
        Maximum number of matches to return.
    metric : str
        One of ``"euclidean_znorm"``, ``"correlation"``, or ``"dtw"``.
    price_column : str or None
        Column in the CSV to use as the price series. Auto-detected if None.
    vol_band : tuple or None
        ``(lo, hi)`` multiplicative band on realized volatility relative to the
        query window. Candidates outside this band are skipped.

    Returns
    -------
    dict
        Fully JSON-serializable result with keys:
        ``query``, ``matches``, ``forward_distribution``, ``dispersion``,
        ``parameters``.
    """
    prices = _load_prices(prices_csv, price_column)

    q_start_ts = pd.Timestamp(query_start)
    q_end_ts = pd.Timestamp(query_end)

    mask = (prices.index >= q_start_ts) & (prices.index <= q_end_ts)
    window_size = int(mask.sum()) - 1
    if window_size < 5:
        raise ValueError(
            f"Date range {query_start!r} – {query_end!r} yields only {window_size} "
            "return bars in the data (need at least 5). Check your dates or CSV."
        )

    miner = AnalogMiner(
        window_size=window_size,
        forecast_horizon=forecast_horizon,
        metric=metric,
        top_k=top_k,
        vol_band=vol_band,
    )
    result = miner.find(prices, query_end=query_end)

    # --- serialize ----------------------------------------------------------
    actual_end = str(result.query_window.index[-1].date())
    actual_start = str(result.query_window.index[0].date())

    matches_list = []
    for rank, m in enumerate(result.matches, start=1):
        forward_return_pct = round(float(m.forward_returns.sum()) * 100, 4)
        matches_list.append(
            {
                "rank": rank,
                "distance": round(float(m.distance), 6),
                "window_start": str(m.window_prices.index[0].date()),
                "window_end": str(m.window_prices.index[-1].date()),
                "forward_start": str(m.forward_prices.index[0].date()),
                "forward_end": str(m.forward_prices.index[-1].date()),
                "realized_vol": round(float(m.realized_vol), 6),
                "cumulative_return_pct": round(float(m.cumulative_return) * 100, 4),
                "forward_return_pct": forward_return_pct,
            }
        )

    fwd_dist = result.forward_distribution((0.1, 0.5, 0.9))
    forward_distribution: dict = {"step": list(range(len(fwd_dist)))}
    for col, key in zip(fwd_dist.columns, ["p10", "p50", "p90"]):
        forward_distribution[key] = [round(float(v) * 100, 4) for v in fwd_dist[col]]

    return {
        "query": {
            "start": actual_start,
            "end": actual_end,
            "window_bars": window_size,
            "volatility": round(float(result.query_vol), 6),
            "cumulative_return_pct": round(float(result.query_cum_return) * 100, 4),
        },
        "matches": matches_list,
        "forward_distribution": forward_distribution,
        "dispersion": round(float(result.dispersion()), 6),
        "parameters": {
            "window_bars": window_size,
            "forecast_horizon": forecast_horizon,
            "top_k": top_k,
            "metric": metric,
            "matches_found": len(result.matches),
        },
    }


def _load_prices(prices_csv: str, price_column: str | None) -> pd.Series:
    df = pd.read_csv(prices_csv, index_col=0, parse_dates=True)
    df = df.sort_index()

    if price_column is not None:
        if price_column not in df.columns:
            raise ValueError(f"Column {price_column!r} not found. Available: {list(df.columns)}")
        return df[price_column].dropna().astype(float)

    for candidate in ("Adj Close", "Close", "close", "adj_close", "price", "Price"):
        if candidate in df.columns:
            return df[candidate].dropna().astype(float)

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        raise ValueError(f"No numeric columns found in {prices_csv!r}")
    return df[numeric_cols[0]].dropna().astype(float)
