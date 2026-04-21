"""Console reporting helpers for AnalogResult."""

from analog_miner import AnalogResult


def summarize(result: AnalogResult, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"Query window: {result.query_window.index[0].date()} to {result.query_window.index[-1].date()}")
    print(f"Query realized vol: {result.query_vol:.4f} | Query cum return: {result.query_cum_return:+.4f}")
    print(f"Found {len(result.matches)} matches (top-K, min-separated):")
    for i, m in enumerate(result.matches):
        print(
            f"  [{i}] {m.window_prices.index[0].date()} -> "
            f"{m.window_prices.index[-1].date()} | "
            f"dist={m.distance:.4f} | vol={m.realized_vol:.4f} | "
            f"cum_ret={m.cumulative_return:+.4f} | "
            f"fwd_ret={m.forward_returns.sum():+.4f}"
        )
    print(f"Forward dispersion across matches: {result.dispersion():.4f}")
    print("Forward quantile bands (normalized to 1.0 at t=0):")
    bands = result.forward_distribution(quantiles=(0.1, 0.5, 0.9))
    print(bands.iloc[[0, len(bands) // 2, -1]].round(4))
