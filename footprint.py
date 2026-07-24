"""
Footprint aggregator for Binance aggTrades data.

Turns a raw aggregated-trade tape into the two tables every order-flow
strategy reads from:

    levels : bar_start, price_level, buy_volume, sell_volume, delta, ...
    bars   : bar_start, open/high/low/close, volume, delta, cum_delta,
             poc, value-area high/low, imbalance counts

AGGRESSOR CONVENTION (the single most important detail in this file)
-------------------------------------------------------------------
Binance gives `is_buyer_maker`. It is counterintuitive:

    is_buyer_maker == True   -> the BUYER was resting passively,
                                so an aggressive SELLER hit the bid.
                                Volume belongs in sell_volume.

    is_buyer_maker == False  -> the aggressor was the BUYER,
                                lifting the ask.
                                Volume belongs in buy_volume.

Getting this backwards inverts every delta reading in the system and
produces a bot that looks plausible and is exactly wrong.
See test_footprint.py::test_aggressor_mapping.

Usage
-----
    python footprint.py --file BTCUSDT-aggTrades-2025-01.csv \
        --interval 5min --tick 10 --out-dir out

    python footprint.py --file ... --inspect "2025-01-15 13:05"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Binance aggTrades dumps: futures have 7 columns, spot has 8.
FUTURES_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]
SPOT_COLUMNS = FUTURES_COLUMNS + ["is_best_match"]


@dataclass
class FootprintConfig:
    """Every knob is a parameter, never a constant buried in the code.

    These all become sensitivity-test dimensions later, so they live in
    one place from day one.
    """

    bar_interval: str = "5min"      # pandas offset alias
    tick_size: float = 10.0         # price bucket width, in quote units
    price_scale: int = 100          # integer units per 1.0 of price (cents)
    imbalance_ratio: float = 3.0    # diagonal imbalance threshold
    min_level_volume: float = 0.05  # volume floor for imbalance eligibility
    stack_length: int = 3           # consecutive levels to call it "stacked"
    value_area_pct: float = 0.70    # fraction of bar volume in the value area

    def tick_int(self) -> int:
        return int(round(self.tick_size * self.price_scale))


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def load_agg_trades(path: str | Path, chunksize: int | None = None) -> pd.DataFrame:
    """Read a Binance aggTrades CSV into a normalised frame.

    Handles the variations you actually hit in the wild: header row
    present or absent, 7-column futures or 8-column spot layout,
    millisecond or microsecond timestamps, and is_buyer_maker as a bool
    or as the strings 'true'/'false'.

    Returns columns: price (float), quantity (float),
    transact_time (int64 ms), is_buyer_maker (bool).
    """
    path = Path(path)

    with open(path, "r") as fh:
        first_line = fh.readline().strip()

    has_header = "price" in first_line.lower()
    n_cols = len(first_line.split(","))

    if n_cols >= 8:
        names = SPOT_COLUMNS[:n_cols]
    else:
        names = FUTURES_COLUMNS[:n_cols]

    read_kwargs = dict(
        usecols=lambda c: str(c).lower()
        in {"price", "quantity", "transact_time", "is_buyer_maker"},
    )

    if has_header:
        df = pd.read_csv(path, **read_kwargs)
        df.columns = [c.lower() for c in df.columns]
    else:
        df = pd.read_csv(path, header=None, names=names, **read_kwargs)

    missing = {"price", "quantity", "transact_time", "is_buyer_maker"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")

    df["price"] = df["price"].astype("float64")
    df["quantity"] = df["quantity"].astype("float64")

    # Timestamp units: Binance moved some feeds to microseconds.
    ts = df["transact_time"].astype("int64")
    if ts.iloc[0] > 1e15:
        ts = ts // 1000
    df["transact_time"] = ts

    ibm = df["is_buyer_maker"]
    if ibm.dtype != bool:
        df["is_buyer_maker"] = (
            ibm.astype(str).str.strip().str.lower().isin({"true", "1"})
        )

    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# Core aggregation
# ----------------------------------------------------------------------


def build_levels(trades: pd.DataFrame, cfg: FootprintConfig) -> pd.DataFrame:
    """Bucket trades into (bar, price level) and split volume by aggressor.

    Price bucketing is done in integer arithmetic. Float floor() scatters
    volume across adjacent buckets at the boundaries and silently
    distorts the ladder.
    """
    tick_int = cfg.tick_int()
    if tick_int <= 0:
        raise ValueError("tick_size * price_scale must be >= 1")

    price_int = np.round(trades["price"].to_numpy() * cfg.price_scale).astype(np.int64)
    level_int = (price_int // tick_int) * tick_int

    ts = pd.to_datetime(trades["transact_time"], unit="ms", utc=True)
    bar_start = ts.dt.floor(cfg.bar_interval)

    qty = trades["quantity"].to_numpy()
    seller_aggressed = trades["is_buyer_maker"].to_numpy()  # see module docstring

    frame = pd.DataFrame(
        {
            "bar_start": bar_start.to_numpy(),
            "price_level_int": level_int,
            "buy_volume": np.where(~seller_aggressed, qty, 0.0),
            "sell_volume": np.where(seller_aggressed, qty, 0.0),
            "trades": 1,
        }
    )

    levels = (
        frame.groupby(["bar_start", "price_level_int"], as_index=False, sort=True)
        .sum()
        .sort_values(["bar_start", "price_level_int"])
        .reset_index(drop=True)
    )

    levels["price_level"] = levels["price_level_int"] / cfg.price_scale
    levels["total_volume"] = levels["buy_volume"] + levels["sell_volume"]
    levels["delta"] = levels["buy_volume"] - levels["sell_volume"]
    return levels


def _dense_ladder(bar_levels: pd.DataFrame, tick_int: int) -> pd.DataFrame:
    """Reindex one bar onto a gapless tick ladder so shifts are exact.

    Without this, a missing price level makes shift(1) compare
    non-adjacent prices and the imbalance logic quietly reads garbage.
    """
    lo = int(bar_levels["price_level_int"].min())
    hi = int(bar_levels["price_level_int"].max())
    full = pd.RangeIndex(lo, hi + tick_int, tick_int)
    out = (
        bar_levels.set_index("price_level_int")
        .reindex(full)
        .fillna({"buy_volume": 0.0, "sell_volume": 0.0, "trades": 0})
    )
    out.index.name = "price_level_int"
    return out.reset_index()


def add_imbalances(levels: pd.DataFrame, cfg: FootprintConfig) -> pd.DataFrame:
    """Flag diagonal imbalances and stacked runs.

    Imbalance is DIAGONAL, not horizontal — this is the part most
    implementations get wrong:

        buy imbalance  at P  :  buy_volume[P]  vs sell_volume[P - 1 tick]
        sell imbalance at P  :  sell_volume[P] vs buy_volume[P + 1 tick]

    The min_level_volume floor is applied to the numerator. Without it,
    thin levels manufacture constant fake imbalances. A zero denominator
    counts as an imbalance provided the numerator clears the floor.
    """
    tick_int = cfg.tick_int()
    out = []

    for bar, grp in levels.groupby("bar_start", sort=True):
        lad = _dense_ladder(
            grp[["price_level_int", "buy_volume", "sell_volume", "trades"]], tick_int
        )

        buy = lad["buy_volume"].to_numpy()
        sell = lad["sell_volume"].to_numpy()

        sell_below = np.roll(sell, 1)
        sell_below[0] = np.nan          # no level below the bar low
        buy_above = np.roll(buy, -1)
        buy_above[-1] = np.nan          # no level above the bar high

        with np.errstate(divide="ignore", invalid="ignore"):
            buy_ratio = np.where(sell_below > 0, buy / sell_below, np.inf)
            sell_ratio = np.where(buy_above > 0, sell / buy_above, np.inf)

        buy_ratio = np.where(np.isnan(sell_below), np.nan, buy_ratio)
        sell_ratio = np.where(np.isnan(buy_above), np.nan, sell_ratio)

        lad["buy_imbalance"] = (
            (buy >= cfg.min_level_volume) & (buy_ratio >= cfg.imbalance_ratio)
        )
        lad["sell_imbalance"] = (
            (sell >= cfg.min_level_volume) & (sell_ratio >= cfg.imbalance_ratio)
        )
        lad["buy_imb_ratio"] = buy_ratio
        lad["sell_imb_ratio"] = sell_ratio

        lad["buy_stacked"] = _mark_runs(
            lad["buy_imbalance"].to_numpy(), cfg.stack_length
        )
        lad["sell_stacked"] = _mark_runs(
            lad["sell_imbalance"].to_numpy(), cfg.stack_length
        )

        lad["bar_start"] = bar
        out.append(lad)

    dense = pd.concat(out, ignore_index=True)
    dense["price_level"] = dense["price_level_int"] / cfg.price_scale
    dense["total_volume"] = dense["buy_volume"] + dense["sell_volume"]
    dense["delta"] = dense["buy_volume"] - dense["sell_volume"]

    cols = [
        "bar_start",
        "price_level",
        "price_level_int",
        "buy_volume",
        "sell_volume",
        "total_volume",
        "delta",
        "trades",
        "buy_imbalance",
        "sell_imbalance",
        "buy_imb_ratio",
        "sell_imb_ratio",
        "buy_stacked",
        "sell_stacked",
    ]
    return dense[cols].sort_values(["bar_start", "price_level_int"]).reset_index(
        drop=True
    )


def _mark_runs(flags: np.ndarray, min_len: int) -> np.ndarray:
    """True for every element inside a run of >= min_len consecutive Trues."""
    marked = np.zeros(len(flags), dtype=bool)
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                marked[start:i] = True
            start = None
    if start is not None and len(flags) - start >= min_len:
        marked[start:] = True
    return marked


def _value_area(
    prices: np.ndarray, volumes: np.ndarray, poc_idx: int, pct: float
) -> tuple[float, float]:
    """Greedy single-level expansion outward from the POC.

    Steps toward whichever adjacent level holds more volume until the
    enclosed volume reaches `pct` of the bar total.
    """
    total = volumes.sum()
    if total <= 0:
        return prices[poc_idx], prices[poc_idx]

    target = total * pct
    lo = hi = poc_idx
    acc = volumes[poc_idx]

    while acc < target and (lo > 0 or hi < len(prices) - 1):
        below = volumes[lo - 1] if lo > 0 else -1.0
        above = volumes[hi + 1] if hi < len(prices) - 1 else -1.0
        if above >= below:
            hi += 1
            acc += volumes[hi]
        else:
            lo -= 1
            acc += volumes[lo]

    return float(prices[lo]), float(prices[hi])


def build_bars(
    trades: pd.DataFrame, levels: pd.DataFrame, cfg: FootprintConfig
) -> pd.DataFrame:
    """Per-bar OHLC plus the order-flow summary metrics."""
    ts = pd.to_datetime(trades["transact_time"], unit="ms", utc=True)
    ohlc = (
        trades.assign(bar_start=ts.dt.floor(cfg.bar_interval))
        .groupby("bar_start")["price"]
        .agg(["first", "max", "min", "last"])
        .rename(columns={"first": "open", "max": "high", "min": "low", "last": "close"})
    )

    rows = []
    for bar, grp in levels.groupby("bar_start", sort=True):
        prices = grp["price_level"].to_numpy()
        vols = grp["total_volume"].to_numpy()
        poc_idx = int(np.argmax(vols))
        va_low, va_high = _value_area(prices, vols, poc_idx, cfg.value_area_pct)

        rows.append(
            {
                "bar_start": bar,
                "volume": float(vols.sum()),
                "buy_volume": float(grp["buy_volume"].sum()),
                "sell_volume": float(grp["sell_volume"].sum()),
                "delta": float(grp["delta"].sum()),
                "trades": int(grp["trades"].sum()),
                "poc": float(prices[poc_idx]),
                "va_low": va_low,
                "va_high": va_high,
                "n_levels": len(prices),
                "buy_imb_count": int(grp.get("buy_imbalance", pd.Series(dtype=bool)).sum()),
                "sell_imb_count": int(grp.get("sell_imbalance", pd.Series(dtype=bool)).sum()),
                "has_buy_stack": bool(grp.get("buy_stacked", pd.Series(dtype=bool)).any()),
                "has_sell_stack": bool(grp.get("sell_stacked", pd.Series(dtype=bool)).any()),
            }
        )

    bars = pd.DataFrame(rows).set_index("bar_start")
    bars = ohlc.join(bars, how="inner")
    bars["cum_delta"] = bars["delta"].cumsum()
    bars["delta_pct"] = bars["delta"] / bars["volume"].replace(0, np.nan)
    return bars.reset_index()


def process(path: str | Path, cfg: FootprintConfig) -> dict[str, pd.DataFrame]:
    """Full pipeline: raw CSV -> levels (with imbalances) + bars."""
    trades = load_agg_trades(path)
    levels = build_levels(trades, cfg)
    levels = add_imbalances(levels, cfg)
    bars = build_bars(trades, levels, cfg)
    return {"trades": trades, "levels": levels, "bars": bars}


# ----------------------------------------------------------------------
# Validation helper — print one bar's ladder to eyeball against a chart
# ----------------------------------------------------------------------


def format_ladder(levels: pd.DataFrame, bars: pd.DataFrame, bar_start) -> str:
    """Render one bar as a text footprint ladder.

    This exists for the validation gate: load the same bar in any
    reference footprint tool and check these numbers match. Do not write
    strategy code until they do.
    """
    bar_ts = pd.Timestamp(bar_start)
    bar_ts = (
        bar_ts.tz_localize("UTC") if bar_ts.tzinfo is None else bar_ts.tz_convert("UTC")
    )
    grp = levels[levels["bar_start"] == bar_ts]
    if grp.empty:
        avail = levels["bar_start"].unique()[:5]
        return f"No bar at {bar_ts}. First few bars: {list(avail)}"

    row = bars[bars["bar_start"] == bar_ts].iloc[0]
    grp = grp.sort_values("price_level_int", ascending=False)

    lines = [
        f"Bar {bar_ts}",
        f"  O {row['open']:.2f}  H {row['high']:.2f}  "
        f"L {row['low']:.2f}  C {row['close']:.2f}",
        f"  volume {row['volume']:.4f}   delta {row['delta']:+.4f}   "
        f"trades {int(row['trades'])}",
        f"  POC {row['poc']:.2f}   value area {row['va_low']:.2f} - {row['va_high']:.2f}",
        "",
        f"{'price':>12} {'sell(bid)':>12} {'buy(ask)':>12} {'delta':>12}   flags",
        "-" * 68,
    ]

    for _, r in grp.iterrows():
        flags = ""
        if r["buy_imbalance"]:
            flags += " B"
        if r["sell_imbalance"]:
            flags += " S"
        if r["buy_stacked"]:
            flags += " [BSTACK]"
        if r["sell_stacked"]:
            flags += " [SSTACK]"
        marker = "  <-- POC" if abs(r["price_level"] - row["poc"]) < 1e-9 else ""
        lines.append(
            f"{r['price_level']:>12.2f} {r['sell_volume']:>12.4f} "
            f"{r['buy_volume']:>12.4f} {r['delta']:>+12.4f}  {flags}{marker}"
        )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--file", required=True, help="Binance aggTrades CSV")
    p.add_argument("--interval", default="5min", help="bar interval, e.g. 1min, 5min")
    p.add_argument("--tick", type=float, default=10.0, help="price bucket width")
    p.add_argument("--imbalance-ratio", type=float, default=3.0)
    p.add_argument("--min-level-volume", type=float, default=0.05)
    p.add_argument("--stack-length", type=int, default=3)
    p.add_argument("--value-area-pct", type=float, default=0.70)
    p.add_argument("--out-dir", default=None, help="write levels.parquet / bars.parquet")
    p.add_argument("--inspect", default=None, help='bar to print, e.g. "2025-01-15 13:05"')
    args = p.parse_args(argv)

    cfg = FootprintConfig(
        bar_interval=args.interval,
        tick_size=args.tick,
        imbalance_ratio=args.imbalance_ratio,
        min_level_volume=args.min_level_volume,
        stack_length=args.stack_length,
        value_area_pct=args.value_area_pct,
    )

    print(f"config: {asdict(cfg)}", file=sys.stderr)
    res = process(args.file, cfg)
    trades, levels, bars = res["trades"], res["levels"], res["bars"]

    print(
        f"loaded {len(trades):,} trades -> {len(bars):,} bars, "
        f"{len(levels):,} price levels",
        file=sys.stderr,
    )

    if args.inspect:
        print(format_ladder(levels, bars, args.inspect))
        return 0

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        levels.to_parquet(out / "levels.parquet", index=False)
        bars.to_parquet(out / "bars.parquet", index=False)
        print(f"wrote {out/'levels.parquet'} and {out/'bars.parquet'}", file=sys.stderr)
    else:
        cols = ["bar_start", "open", "high", "low", "close", "volume", "delta",
                "cum_delta", "poc", "has_buy_stack", "has_sell_stack"]
        print(bars[cols].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
