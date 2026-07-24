"""
Synthetic aggTrades generator.

Real Binance archives are the point of the bot, but for tests, demos, and CI
we need a tape that exists without a download. This produces a frame in the
exact shape `footprint.load_agg_trades` returns, driven by a random walk with
a little microstructure (clustered aggressor runs, volume that thins at the
extremes). It is NOT a market simulator and carries no edge — do not read
backtest results on synthetic data as evidence of anything. It exists to
prove the plumbing runs end to end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_agg_trades(
    n_trades: int = 200_000,
    start_ms: int = 1_700_000_000_000,
    start_price: float = 50_000.0,
    seed: int = 42,
    mean_dt_ms: float = 250.0,
    tick: float = 0.1,
) -> pd.DataFrame:
    """Generate a synthetic aggTrades frame.

    Columns match the normalised loader output:
    price, quantity, transact_time (ms), is_buyer_maker (bool).
    """
    rng = np.random.default_rng(seed)

    # Timestamps: exponential inter-arrival, monotonically increasing.
    dt = rng.exponential(mean_dt_ms, size=n_trades)
    ts = start_ms + np.cumsum(dt).astype(np.int64)

    # Aggressor side with short-run persistence (order-flow clustering).
    flips = rng.random(n_trades) < 0.15
    side = np.empty(n_trades, dtype=bool)  # True = seller aggressed (is_buyer_maker)
    cur = rng.random() < 0.5
    for i in range(n_trades):
        if flips[i]:
            cur = not cur
        side[i] = cur

    # Price: signed random walk nudged by the aggressor side, in ticks.
    step = np.where(side, -1.0, 1.0) * rng.random(n_trades)
    noise = rng.normal(0, 1.0, n_trades)
    price = start_price + np.cumsum((step + noise) * tick)
    price = np.round(price / tick) * tick
    price = np.maximum(price, tick)

    # Volume: lognormal, independent of side.
    qty = np.round(rng.lognormal(mean=-2.0, sigma=1.0, size=n_trades), 3)
    qty = np.maximum(qty, 0.001)

    return pd.DataFrame(
        {
            "price": price.astype("float64"),
            "quantity": qty.astype("float64"),
            "transact_time": ts,
            "is_buyer_maker": side,
        }
    )


def write_csv(path: str, **kwargs) -> str:
    """Write a headerless 7-column futures-layout CSV, like a real dump."""
    df = generate_agg_trades(**kwargs)
    out = pd.DataFrame(
        {
            "agg_trade_id": np.arange(1, len(df) + 1),
            "price": df["price"],
            "quantity": df["quantity"],
            "first_trade_id": np.arange(1, len(df) + 1),
            "last_trade_id": np.arange(1, len(df) + 1),
            "transact_time": df["transact_time"],
            "is_buyer_maker": np.where(df["is_buyer_maker"], "true", "false"),
        }
    )
    out.to_csv(path, header=False, index=False)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Write a synthetic aggTrades CSV")
    p.add_argument("--out", default="synth-aggTrades.csv")
    p.add_argument("--n", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    path = write_csv(args.out, n_trades=args.n, seed=args.seed)
    print(f"wrote {path}")
