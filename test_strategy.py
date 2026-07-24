"""
Tests for the signal, study, and backtest layers.

The backtest tests are the important ones: they pin down the no-lookahead
contract, the cost model, and the pessimistic-fill rule. If those slip, the
equity curve lies.

Run:  python test_strategy.py   (or: python -m pytest test_strategy.py -v)
"""

import numpy as np
import pandas as pd

from strategy import StrategyConfig, generate_signals, true_range, add_features
from study import StudyConfig, run_study, verdict, forward_returns
from backtest import (
    BacktestConfig,
    CostConfig,
    run_backtest,
    run_with_and_without_costs,
    _funding_events,
    max_drawdown,
)

BASE = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def make_bars(rows, interval_min=5):
    """rows: list of dicts with at least open/high/low/close/volume/delta.

    Fills in the footprint columns the strategy/study code reads, with
    sensible defaults so tests only specify what they exercise.
    """
    out = []
    for i, r in enumerate(rows):
        bar = {
            "bar_start": BASE + pd.Timedelta(minutes=interval_min * i),
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r.get("volume", 100.0),
            "buy_volume": r.get("buy_volume", 50.0),
            "sell_volume": r.get("sell_volume", 50.0),
            "delta": r.get("delta", 0.0),
            "trades": r.get("trades", 100),
            "poc": r.get("poc", r["close"]),
            "va_low": r.get("va_low", r["low"]),
            "va_high": r.get("va_high", r["high"]),
            "n_levels": r.get("n_levels", 5),
            "buy_imb_count": r.get("buy_imb_count", 0),
            "sell_imb_count": r.get("sell_imb_count", 0),
            "has_buy_stack": r.get("has_buy_stack", False),
            "has_sell_stack": r.get("has_sell_stack", False),
        }
        out.append(bar)
    df = pd.DataFrame(out)
    df["cum_delta"] = df["delta"].cumsum()
    df["delta_pct"] = df["delta"] / df["volume"].replace(0, np.nan)
    return df


# ----------------------------------------------------------------------
# strategy
# ----------------------------------------------------------------------


def test_true_range_basic():
    bars = make_bars(
        [
            {"open": 100, "high": 110, "low": 90, "close": 105},
            {"open": 105, "high": 120, "low": 104, "close": 118},
        ]
    )
    tr = true_range(bars)
    assert tr.iloc[0] == 20  # first bar: high-low
    # second bar: max(120-104, |120-105|, |104-105|) = 16
    assert tr.iloc[1] == 16
    print("ok  true range")


def test_no_signal_during_atr_warmup():
    """Before ATR has a full window, signals must be forced flat."""
    rows = [
        {"open": 100, "high": 101, "low": 99, "close": 100.5,
         "has_buy_stack": True, "delta": 50, "volume": 100}
        for _ in range(10)
    ]
    bars = make_bars(rows)
    cfg = StrategyConfig(setup="stacked_continuation", atr_window=14)
    sig = generate_signals(bars, cfg)
    assert (sig["signal"] == 0).all(), "no signal allowed before ATR warms up"
    print("ok  atr warmup suppresses signals")


def test_stacked_continuation_direction():
    # 20 warmup bars then a clean buy-stack bar closing near its high.
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    rows.append(
        {"open": 100, "high": 105, "low": 100, "close": 104.9,
         "has_buy_stack": True, "delta": 80, "volume": 100}
    )
    bars = make_bars(rows)
    cfg = StrategyConfig(setup="stacked_continuation", atr_window=14, min_delta_pct=0.1)
    sig = generate_signals(bars, cfg)
    assert sig["signal"].iloc[-1] == 1, "buy stack closing high should be long"
    assert sig["stop_dist"].iloc[-1] > 0
    assert np.isclose(
        sig["target_dist"].iloc[-1], cfg.reward_risk * sig["stop_dist"].iloc[-1]
    )
    print("ok  stacked continuation direction")


def test_delta_divergence_direction():
    # Rising highs, but the breakout bar has strongly negative delta.
    rows = [
        {"open": 100 + i, "high": 101 + i, "low": 99 + i, "close": 100 + i}
        for i in range(20)
    ]
    rows.append(
        {"open": 120, "high": 130, "low": 119, "close": 125,
         "delta": -90, "volume": 100}  # new high, delta negative
    )
    bars = make_bars(rows)
    cfg = StrategyConfig(setup="delta_divergence", atr_window=14,
                         divergence_lookback=10, min_delta_pct=0.1)
    sig = generate_signals(bars, cfg)
    assert sig["signal"].iloc[-1] == -1, "new high on negative delta should fade short"
    print("ok  delta divergence direction")


# ----------------------------------------------------------------------
# study
# ----------------------------------------------------------------------


def test_forward_returns_alignment():
    bars = make_bars(
        [{"open": 100, "high": 100, "low": 100, "close": c}
         for c in [100, 110, 121, 133.1]]
    )
    fwd = forward_returns(bars, 1)
    assert np.isclose(fwd[0], np.log(110 / 100))
    assert np.isnan(fwd[-1]), "last bar has no 1-bar future"
    print("ok  forward returns alignment")


def test_study_runs_and_verdict_shape():
    rng = np.random.default_rng(0)
    rows = []
    price = 100.0
    for i in range(300):
        price *= np.exp(rng.normal(0, 0.01))
        rows.append(
            {"open": price, "high": price * 1.001, "low": price * 0.999,
             "close": price, "delta": rng.normal(0, 20), "volume": 100,
             "has_buy_stack": bool(rng.random() < 0.1)}
        )
    bars = make_bars(rows)
    df = run_study(bars, StudyConfig(horizons=(3, 12), n_bootstrap=300))
    assert not df.empty
    assert {"name", "horizon", "separation", "p_value", "aligned"} <= set(df.columns)
    v = verdict(df, StudyConfig(horizons=(3, 12)))
    assert "survivors" in v and "dead" in v
    print("ok  study runs and produces a verdict")


# ----------------------------------------------------------------------
# backtest — the load-bearing tests
# ----------------------------------------------------------------------


def test_no_lookahead_entry_on_next_open():
    """A signal on bar i must fill at bar i+1's open, not bar i's close."""
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    # signal bar: buy stack; its own close is 104.9
    rows.append({"open": 100, "high": 105, "low": 100, "close": 104.9,
                 "has_buy_stack": True, "delta": 80, "volume": 100})
    # next bar opens at a distinct price 200 so we can see where the fill lands
    rows.append({"open": 200, "high": 205, "low": 199, "close": 201})
    bars = make_bars(rows)
    sig = generate_signals(bars, StrategyConfig(atr_window=14, min_delta_pct=0.1))
    res = run_backtest(sig, BacktestConfig(),
                       CostConfig(enabled=False))
    tr = res["trades"]
    assert len(tr) >= 1
    # entry must be at the NEXT bar's open (200), never the signal bar close.
    assert abs(tr.iloc[0]["entry_price"] - 200) < 1e-9, tr.iloc[0]["entry_price"]
    print("ok  no lookahead: entry at next bar open")


def test_costs_reduce_pnl():
    """Costs-on net pnl must be <= costs-off for the same signals."""
    rng = np.random.default_rng(3)
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    price = 100.0
    for i in range(200):
        price *= np.exp(rng.normal(0, 0.01))
        rows.append(
            {"open": price, "high": price * 1.01, "low": price * 0.99,
             "close": price, "has_buy_stack": bool(rng.random() < 0.3),
             "has_sell_stack": bool(rng.random() < 0.3),
             "delta": rng.normal(0, 40), "volume": 100}
        )
    bars = make_bars(rows)
    sig = generate_signals(bars, StrategyConfig(atr_window=14, min_delta_pct=0.05))
    res = run_with_and_without_costs(sig, BacktestConfig(),
                                     CostConfig(tick_size=1.0, taker_bps=5))
    off = res["without_costs"]["metrics"]["net_pnl"]
    on = res["with_costs"]["metrics"]["net_pnl"]
    if res["with_costs"]["metrics"]["n_trades"] > 0:
        assert on <= off + 1e-6, (on, off)
    print("ok  costs reduce pnl")


def test_stop_and_target_same_bar_is_pessimistic():
    """If a bar straddles both stop and target, the stop must win."""
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    rows.append({"open": 100, "high": 101, "low": 99, "close": 100.9,
                 "has_buy_stack": True, "delta": 80, "volume": 100})
    # entry next bar at open=100; make that bar span far above and below.
    rows.append({"open": 100, "high": 140, "low": 60, "close": 100})
    bars = make_bars(rows)
    sig = generate_signals(bars, StrategyConfig(atr_window=14, min_delta_pct=0.1,
                                                atr_stop_mult=1.5, reward_risk=2.0))
    res = run_backtest(sig, BacktestConfig(), CostConfig(enabled=False))
    tr = res["trades"]
    assert len(tr) == 1
    assert tr.iloc[0]["reason"] == "stop", tr.iloc[0]["reason"]
    assert tr.iloc[0]["net_pnl"] < 0
    print("ok  pessimistic fill: stop before target")


def test_time_stop_exits():
    """A position that never hits stop or target exits at max_hold_bars."""
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    rows.append({"open": 100, "high": 105, "low": 100, "close": 104.9,
                 "has_buy_stack": True, "delta": 80, "volume": 100})
    # many flat bars, tight range, so neither stop nor target triggers.
    rows += [{"open": 100, "high": 100.1, "low": 99.9, "close": 100} for _ in range(10)]
    bars = make_bars(rows)
    sig = generate_signals(bars, StrategyConfig(atr_window=14, min_delta_pct=0.1,
                                                atr_stop_mult=5.0, reward_risk=5.0))
    res = run_backtest(sig, BacktestConfig(max_hold_bars=3), CostConfig(enabled=False))
    tr = res["trades"]
    assert len(tr) >= 1
    assert tr.iloc[0]["reason"] == "time"
    assert tr.iloc[0]["bars_held"] == 3
    print("ok  time stop exits at max_hold_bars")


def test_funding_events_counts_stamps():
    a = pd.Timestamp("2024-01-01 07:00:00", tz="UTC")
    b = pd.Timestamp("2024-01-01 09:00:00", tz="UTC")
    assert _funding_events(a, b) == 1  # 08:00 stamp
    c = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    d = pd.Timestamp("2024-01-01 23:59:00", tz="UTC")
    assert _funding_events(c, d) == 2  # 08:00 and 16:00 (00:00 excluded at start)
    print("ok  funding stamp counting")


def test_max_drawdown():
    eq = np.array([100.0, 120.0, 90.0, 110.0, 80.0])
    dd = max_drawdown(eq)
    assert np.isclose(dd, (80 - 120) / 120)
    print("ok  max drawdown")


def test_short_disabled_blocks_short_entries():
    rows = [{"open": 100, "high": 101, "low": 99, "close": 100} for _ in range(20)]
    rows.append({"open": 100, "high": 100, "low": 95, "close": 95.1,
                 "has_sell_stack": True, "delta": -80, "volume": 100})
    rows.append({"open": 95, "high": 96, "low": 90, "close": 92})
    bars = make_bars(rows)
    sig = generate_signals(bars, StrategyConfig(atr_window=14, min_delta_pct=0.1))
    res = run_backtest(sig, BacktestConfig(allow_short=False), CostConfig(enabled=False))
    assert len(res["trades"]) == 0, "shorts must be blocked when allow_short=False"
    print("ok  short disabled blocks short entries")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} tests passed")
