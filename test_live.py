"""
Tests for the online aggregator and paper trader.

The key property: the online aggregator must produce the same closed-bar
numbers as the offline batch aggregator on the same ticks. If those drift,
the live bot is trading a different instrument than the one you backtested.

Run:  python test_live.py
"""

import numpy as np
import pandas as pd

from footprint import FootprintConfig, build_levels, add_imbalances, build_bars
from strategy import StrategyConfig
from backtest import CostConfig
from live import (
    Tick,
    OnlineFootprint,
    PaperBroker,
    LiveTrader,
    LiveConfig,
    RealBroker,
    replay_feed,
    run,
)
from synth import generate_agg_trades, write_csv

BASE_MS = 1_700_000_000_000


def _ticks_from_frame(df):
    for row in df.itertuples(index=False):
        yield Tick(
            price=float(row.price),
            qty=float(row.quantity),
            ts_ms=int(row.transact_time),
            is_buyer_maker=bool(row.is_buyer_maker),
        )


def test_online_matches_offline():
    """Online closed bars must equal offline bars, decimal for decimal."""
    cfg = FootprintConfig(tick_size=10.0, bar_interval="1min")
    trades = generate_agg_trades(n_trades=40_000, seed=5)

    # offline reference
    levels = add_imbalances(build_levels(trades, cfg), cfg)
    offline = build_bars(trades, levels, cfg)

    # online: feed tick by tick, collect finalised bars
    agg = OnlineFootprint(cfg)
    for t in _ticks_from_frame(trades):
        agg.add(t)
    online = agg.bars

    # the online aggregator has not finalised the still-open last bar, so
    # compare on the bars they share.
    n = min(len(offline), len(online))
    assert n > 5
    for col in ["open", "high", "low", "close", "volume", "delta", "poc"]:
        a = offline[col].to_numpy()[:n]
        b = online[col].to_numpy()[:n]
        assert np.allclose(a, b, equal_nan=True), f"mismatch in {col}"
    print(f"ok  online matches offline on {n} bars")


def test_paper_broker_stop_and_target():
    costs = CostConfig(enabled=False, tick_size=1.0)
    b = PaperBroker(costs=costs, equity=10_000.0)
    b.enter(direction=1, ref_price=100.0, stop_dist=5.0, target_dist=10.0,
            ts=BASE_MS, setup="t")
    assert b.position is not None
    assert b.position.stop_px == 95.0 and b.position.target_px == 110.0
    # price ticks up to target -> exit with profit
    assert b.on_price(109.0, BASE_MS + 1) is None
    fill = b.on_price(110.0, BASE_MS + 2)
    assert fill is not None and fill["reason"] == "target"
    assert fill["gross_pnl"] > 0
    assert b.position is None
    print("ok  paper broker stop/target")


def test_paper_broker_stop_loss():
    costs = CostConfig(enabled=False, tick_size=1.0)
    b = PaperBroker(costs=costs, equity=10_000.0)
    b.enter(direction=-1, ref_price=100.0, stop_dist=5.0, target_dist=10.0,
            ts=BASE_MS, setup="t")
    # short: stop is above entry
    assert b.position.stop_px == 105.0
    fill = b.on_price(105.0, BASE_MS + 1)
    assert fill["reason"] == "stop"
    assert fill["gross_pnl"] < 0
    print("ok  paper broker stop loss")


def test_real_broker_refuses():
    rb = RealBroker()
    try:
        rb.enter(1, 100.0, 1.0, 2.0, 0, "x")
    except NotImplementedError:
        print("ok  real broker refuses to trade")
        return
    raise AssertionError("RealBroker must refuse")


def test_live_trader_runs_and_is_flat_or_consistent(tmp_path=None):
    cfg = FootprintConfig(tick_size=10.0, bar_interval="1min")
    trades = generate_agg_trades(n_trades=60_000, seed=9)
    broker = PaperBroker(costs=CostConfig(tick_size=10.0), equity=10_000.0)
    trader = LiveTrader(
        cfg, StrategyConfig(setup="delta_divergence", min_delta_pct=0.05),
        broker, LiveConfig(warmup_bars=20, verbose=False),
    )
    summary = run(_ticks_from_frame(trades), trader)
    assert summary["ticks"] == 60_000
    assert summary["bars"] > 20
    # every recorded fill must be internally consistent
    for f in summary["fills"]:
        assert f["exit_ts"] >= f["entry_ts"]
        assert f["reason"] in {"stop", "target"}
    print(f"ok  live trader ran: {summary['bars']} bars, {summary['n_fills']} fills")


def test_replay_feed_roundtrip():
    import tempfile, os

    d = tempfile.mkdtemp()
    path = os.path.join(d, "synth.csv")
    write_csv(path, n_trades=5_000, seed=3)
    ticks = list(replay_feed(path))
    assert len(ticks) == 5_000
    # timestamps must be non-decreasing
    ts = [t.ts_ms for t in ticks]
    assert ts == sorted(ts)
    print("ok  replay feed roundtrip")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} tests passed")
