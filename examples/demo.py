"""
Generates a long GBM-like price series with deliberately injected recurring
patterns, then queries the miner and reports what it found. Runs without
any external data or network access.
"""

from analog_miner import AnalogMiner
from fixtures import make_synthetic_prices
from reporting import summarize


def main() -> None:
    prices = make_synthetic_prices()
    print(f"Synthetic price series: {len(prices)} bars from {prices.index[0].date()} to {prices.index[-1].date()}")

    miner = AnalogMiner(window_size=60, forecast_horizon=20, metric="euclidean_znorm", top_k=8)
    summarize(miner.find(prices), "Euclidean (default), L=60 H=20")

    miner_corr = AnalogMiner(window_size=60, forecast_horizon=20, metric="correlation", top_k=8)
    summarize(miner_corr.find(prices), "Correlation, L=60 H=20")

    miner_dtw = AnalogMiner(
        window_size=40,
        forecast_horizon=15,
        metric="dtw",
        top_k=5,
        vol_band=(0.3, 3.0),
    )
    summarize(miner_dtw.find(prices), "DTW (constrained), L=40 H=15")

    miner_hist = AnalogMiner(window_size=60, forecast_horizon=20, top_k=5)
    summarize(miner_hist.find(prices, query_end="2018-06-15"), "Default metric, query ending 2018-06-15")


if __name__ == "__main__":
    main()
