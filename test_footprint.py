"""
Tests for the footprint aggregator.

Run:  python -m pytest test_footprint.py -v
      (or just: python test_footprint.py)

The first test is the one that matters most. If the aggressor mapping is
inverted, every delta reading in the system is backwards and the bot will
look plausible while being exactly wrong.
"""

import numpy as np
import pandas as pd

from footprint import (
    FootprintConfig,
    build_levels,
    build_bars,
    add_imbalances,
    _mark_runs,
    _value_area,
    load_agg_trades,
)

# Anchored to an exact 5-minute boundary (2023-11-14 22:10:00 UTC) so that
# bar-assignment assertions are unambiguous.
BASE_MS = 1_699_999_800_000


def make_trades(rows):
    """rows: list of (price, qty, is_buyer_maker, seconds_offset)"""
    return pd.DataFrame(
        {
            "price": [r[0] for r in rows],
            "quantity": [r[1] for r in rows],
            "is_buyer_maker": [r[2] for r in rows],
            "transact_time": [BASE_MS + r[3] * 1000 for r in rows],
        }
    )


def test_aggressor_mapping():
    """is_buyer_maker=True means an aggressive SELLER hit a resting bid."""
    cfg = FootprintConfig(tick_size=10.0, bar_interval="5min")
    trades = make_trades(
        [
            (50_000.0, 1.0, True, 0),   # seller aggressed -> sell_volume
            (50_000.0, 3.0, False, 1),  # buyer aggressed  -> buy_volume
        ]
    )
    lv = build_levels(trades, cfg)
    assert len(lv) == 1
    row = lv.iloc[0]
    assert row["sell_volume"] == 1.0, "is_buyer_maker=True must land in sell_volume"
    assert row["buy_volume"] == 3.0, "is_buyer_maker=False must land in buy_volume"
    assert row["delta"] == 2.0
    print("ok  aggressor mapping")


def test_integer_price_bucketing():
    """Prices must floor into ticks exactly, with no float boundary leakage."""
    cfg = FootprintConfig(tick_size=10.0)
    trades = make_trades(
        [
            (50_000.00, 1.0, False, 0),  # -> 50000
            (50_009.99, 1.0, False, 1),  # -> 50000
            (50_010.00, 1.0, False, 2),  # -> 50010  (boundary)
            (49_999.99, 1.0, False, 3),  # -> 49990
        ]
    )
    lv = build_levels(trades, cfg)
    got = sorted(lv["price_level"].tolist())
    assert got == [49_990.0, 50_000.0, 50_010.0], got
    at_50k = lv.loc[lv["price_level"] == 50_000.0, "buy_volume"].iloc[0]
    assert at_50k == 2.0, "both sub-tick prices must land in the same bucket"
    print("ok  integer price bucketing")


def test_bar_assignment_is_left_closed():
    """Left-closed, right-open bars, in UTC."""
    cfg = FootprintConfig(bar_interval="5min", tick_size=10.0)
    trades = make_trades(
        [
            (50_000.0, 1.0, False, 0),
            (50_000.0, 1.0, False, 299),  # same bar
            (50_000.0, 1.0, False, 300),  # next bar
        ]
    )
    lv = build_levels(trades, cfg)
    bars = sorted(lv["bar_start"].unique())
    assert len(bars) == 2, bars
    first = lv[lv["bar_start"] == bars[0]]["buy_volume"].sum()
    assert first == 2.0
    print("ok  bar assignment")


def test_diagonal_imbalance_direction():
    """Buy imbalance at P compares buy[P] against sell[P - 1 tick]."""
    cfg = FootprintConfig(
        tick_size=10.0, imbalance_ratio=3.0, min_level_volume=0.05, stack_length=3
    )
    # buy at 50010 = 10.0 ; sell at 50000 = 1.0 -> ratio 10 -> buy imbalance
    trades = make_trades(
        [
            (50_000.0, 1.0, True, 0),    # sell volume at 50000
            (50_010.0, 10.0, False, 1),  # buy volume at 50010
        ]
    )
    lv = add_imbalances(build_levels(trades, cfg), cfg)
    top = lv[lv["price_level"] == 50_010.0].iloc[0]
    bottom = lv[lv["price_level"] == 50_000.0].iloc[0]
    assert bool(top["buy_imbalance"]) is True, "expected buy imbalance at the upper level"
    assert bool(bottom["buy_imbalance"]) is False
    print("ok  diagonal imbalance direction")


def test_min_volume_floor_suppresses_thin_levels():
    cfg = FootprintConfig(tick_size=10.0, imbalance_ratio=3.0, min_level_volume=1.0)
    trades = make_trades(
        [
            (50_000.0, 0.001, True, 0),
            (50_010.0, 0.010, False, 1),  # ratio is 10 but volume is tiny
        ]
    )
    lv = add_imbalances(build_levels(trades, cfg), cfg)
    assert not lv["buy_imbalance"].any(), "thin levels must not manufacture imbalances"
    print("ok  min volume floor")


def test_dense_ladder_fills_gaps():
    """A skipped price level must not make shift() compare non-adjacent prices."""
    cfg = FootprintConfig(tick_size=10.0)
    trades = make_trades(
        [
            (50_000.0, 1.0, True, 0),
            (50_030.0, 1.0, False, 1),  # 50010 and 50020 never traded
        ]
    )
    lv = add_imbalances(build_levels(trades, cfg), cfg)
    got = sorted(lv["price_level"].tolist())
    assert got == [50_000.0, 50_010.0, 50_020.0, 50_030.0], got
    print("ok  dense ladder fills gaps")


def test_mark_runs():
    f = np.array([False, True, True, False, True, True, True, False])
    assert _mark_runs(f, 3).tolist() == [
        False, False, False, False, True, True, True, False
    ]
    assert _mark_runs(np.array([True, True, True]), 3).all()
    assert not _mark_runs(np.array([True, True]), 3).any()
    print("ok  stacked run detection")


def test_value_area_covers_target():
    prices = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    vols = np.array([1.0, 2.0, 10.0, 2.0, 1.0])
    lo, hi = _value_area(prices, vols, poc_idx=2, pct=0.70)
    assert lo <= 120.0 <= hi
    inside = vols[(prices >= lo) & (prices <= hi)].sum()
    assert inside >= 0.70 * vols.sum()
    print("ok  value area")


def test_bars_delta_matches_levels():
    cfg = FootprintConfig(tick_size=10.0, bar_interval="5min")
    rng = np.random.default_rng(7)
    rows = [
        (50_000 + float(rng.integers(-50, 50)), float(rng.random()),
         bool(rng.integers(0, 2)), int(i))
        for i in range(500)
    ]
    trades = make_trades(rows)
    lv = add_imbalances(build_levels(trades, cfg), cfg)
    bars = build_bars(trades, lv, cfg)
    assert np.isclose(bars["delta"].sum(), lv["delta"].sum())
    assert np.isclose(bars["volume"].sum(), trades["quantity"].sum())
    assert np.isclose(bars["cum_delta"].iloc[-1], bars["delta"].sum())
    print("ok  bar totals reconcile with level totals")


def test_loader_handles_headerless_and_string_bools(tmp_path=None):
    import tempfile, os

    d = tempfile.mkdtemp()
    # headerless, 7-column futures layout, string booleans
    p1 = os.path.join(d, "headerless.csv")
    with open(p1, "w") as fh:
        fh.write(f"1,50000.0,1.5,10,11,{BASE_MS},true\n")
        fh.write(f"2,50010.0,2.5,12,13,{BASE_MS + 1000},false\n")
    df = load_agg_trades(p1)
    assert df["is_buyer_maker"].tolist() == [True, False]
    assert df["quantity"].tolist() == [1.5, 2.5]

    # with header, 8-column spot layout
    p2 = os.path.join(d, "header.csv")
    with open(p2, "w") as fh:
        fh.write(
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
            "transact_time,is_buyer_maker,is_best_match\n"
        )
        fh.write(f"1,50000.0,1.5,10,11,{BASE_MS},True,True\n")
    df2 = load_agg_trades(p2)
    assert df2["is_buyer_maker"].tolist() == [True]

    # microsecond timestamps get normalised to ms
    p3 = os.path.join(d, "micros.csv")
    with open(p3, "w") as fh:
        fh.write(f"1,50000.0,1.5,10,11,{BASE_MS * 1000},true\n")
    df3 = load_agg_trades(p3)
    assert df3["transact_time"].iloc[0] == BASE_MS
    print("ok  loader variants")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} tests passed")
