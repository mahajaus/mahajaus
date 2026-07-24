"""
Event-driven backtest for the footprint bot — with the costs that decide it.

For a perpetual future, execution realism is the line between a backtest that
means something and one that flatters you. This engine is built around three
commitments:

1. NO LOOKAHEAD. A signal computed from bar i is acted on at bar i+1's open,
   never at bar i's close. The loop physically cannot see a bar before it
   would have existed. This is the single most common way a footprint bot
   ends up with a beautiful, fictional equity curve.

2. COSTS ON OR OFF, BY A FLAG. Taker fees, funding, and slippage are all
   modelled and can be switched off together. The gap between costs-on and
   costs-off equity is how much fragility the strategy is carrying — if the
   edge lives entirely in that gap, there is no edge.

3. PESSIMISTIC FILLS. When a bar's range straddles both stop and target, the
   stop is assumed to fill first. Slippage always works against you.

The engine takes the output of `strategy.generate_signals` and produces a
trade blotter, an equity curve, and the metrics that actually decide it —
expectancy net of costs, drawdown, and above all trade count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CostConfig:
    """Binance USDⓈ-M perp cost model. Defaults are deliberately not rosy."""

    taker_bps: float = 4.5          # per side, in basis points of notional
    slippage_ticks: float = 1.0     # ticks against you on entry and on exit
    tick_size: float = 10.0
    funding_rate: float = 0.0001    # per 8h stamp; applied while in a position
    apply_funding: bool = True
    enabled: bool = True            # master switch: False = frictionless run


@dataclass
class BacktestConfig:
    equity0: float = 10_000.0       # starting equity, quote currency
    position_frac: float = 1.0      # notional as a fraction of equity (1 = 1x)
    max_hold_bars: int = 48         # time stop
    allow_short: bool = True
    one_position: bool = True       # no pyramiding; flat before re-entry


# Binance USDT-perp funding stamps, in UTC hours.
FUNDING_HOURS = (0, 8, 16)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    qty: float
    bars_held: int
    gross_pnl: float
    fees: float
    funding: float
    net_pnl: float
    reason: str            # "target" | "stop" | "time"
    setup: str


def _funding_events(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Count funding stamps strictly inside (start, end].

    A position open across a funding timestamp pays (or receives) once per
    stamp. We count stamps in the half-open interval so a position entered
    exactly on a stamp is not charged for that instant.
    """
    if end <= start:
        return 0
    rng = pd.date_range(
        start.floor("h"), end.ceil("h"), freq="h", tz="UTC", inclusive="both"
    )
    count = 0
    for t in rng:
        if t.hour in FUNDING_HOURS and start < t <= end:
            count += 1
    return count


def run_backtest(
    signals: pd.DataFrame,
    bt: BacktestConfig | None = None,
    costs: CostConfig | None = None,
) -> dict:
    """Run the event-driven backtest over a signals frame.

    `signals` must be the output of strategy.generate_signals: one row per
    bar, sorted by time, with open/high/low/close, `signal`, `stop_dist`,
    `target_dist`, and `setup`.
    """
    bt = bt or BacktestConfig()
    costs = costs or CostConfig()

    rows = signals.reset_index(drop=True)
    n = len(rows)
    times = pd.to_datetime(rows["bar_start"]).to_numpy()
    opens = rows["open"].to_numpy(dtype=float)
    highs = rows["high"].to_numpy(dtype=float)
    lows = rows["low"].to_numpy(dtype=float)
    closes = rows["close"].to_numpy(dtype=float)
    sig = rows["signal"].to_numpy(dtype=int)
    stop_dist = rows["stop_dist"].to_numpy(dtype=float)
    target_dist = rows["target_dist"].to_numpy(dtype=float)
    setups = rows["setup"].astype(str).to_numpy()

    slip = costs.slippage_ticks * costs.tick_size if costs.enabled else 0.0
    fee_rate = (costs.taker_bps / 1e4) if costs.enabled else 0.0

    equity = bt.equity0
    equity_curve = np.empty(n)
    trades: list[Trade] = []

    # Open-position state
    in_pos = False
    direction = 0
    entry_price = 0.0
    qty = 0.0
    stop_px = 0.0
    target_px = 0.0
    entry_i = -1
    entry_setup = ""
    funding_accrued = 0.0

    def close_position(exit_i: int, exit_px_raw: float, reason: str) -> None:
        nonlocal equity, in_pos, funding_accrued
        # slippage on exit works against the position direction
        exit_px = exit_px_raw - direction * slip
        gross = direction * (exit_px - entry_price) * qty
        exit_fee = abs(exit_px) * qty * fee_rate
        entry_fee = abs(entry_price) * qty * fee_rate
        fees = entry_fee + exit_fee
        net = gross - fees - funding_accrued
        equity += net
        trades.append(
            Trade(
                entry_time=pd.Timestamp(times[entry_i]),
                exit_time=pd.Timestamp(times[exit_i]),
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_px,
                qty=qty,
                bars_held=exit_i - entry_i,
                gross_pnl=gross,
                fees=fees,
                funding=funding_accrued,
                net_pnl=net,
                reason=reason,
                setup=entry_setup,
            )
        )
        in_pos = False
        funding_accrued = 0.0

    def manage(i: int, is_entry_bar: bool) -> None:
        """Resolve stop/target/time against bar i's range.

        Stop and target are checked on every bar the position is open —
        including the entry bar, because a position filled at the open can be
        stopped out or hit its target later in that same bar. Funding and the
        time stop only apply on bars *after* entry.
        """
        nonlocal funding_accrued
        if not is_entry_bar and costs.enabled and costs.apply_funding:
            stamps = _funding_events(
                pd.Timestamp(times[i - 1]), pd.Timestamp(times[i])
            )
            if stamps:
                notional = abs(entry_price) * qty
                # long pays when funding positive; short receives
                funding_accrued += direction * costs.funding_rate * notional * stamps

        hit_stop = lows[i] <= stop_px <= highs[i] if direction > 0 else (
            highs[i] >= stop_px >= lows[i]
        )
        hit_target = highs[i] >= target_px >= lows[i] if direction > 0 else (
            lows[i] <= target_px <= highs[i]
        )

        if hit_stop and hit_target:
            close_position(i, stop_px, "stop")   # pessimistic: stop first
        elif hit_stop:
            close_position(i, stop_px, "stop")
        elif hit_target:
            close_position(i, target_px, "target")
        elif not is_entry_bar and i - entry_i >= bt.max_hold_bars:
            close_position(i, closes[i], "time")

    for i in range(n):
        # 1) Manage a position opened on an earlier bar against THIS bar.
        if in_pos:
            manage(i, is_entry_bar=False)

        # 2) Act on the PREVIOUS bar's signal at this bar's open.
        if not in_pos and i > 0 and sig[i - 1] != 0:
            d = sig[i - 1]
            if d < 0 and not bt.allow_short:
                pass
            elif np.isnan(stop_dist[i - 1]) or stop_dist[i - 1] <= 0:
                pass
            else:
                raw_entry = opens[i]
                entry_price = raw_entry + d * slip   # slippage against entry
                notional = equity * bt.position_frac
                qty = notional / entry_price if entry_price > 0 else 0.0
                if qty > 0:
                    direction = d
                    stop_px = entry_price - d * stop_dist[i - 1]
                    target_px = entry_price + d * target_dist[i - 1]
                    entry_i = i
                    entry_setup = setups[i - 1]
                    funding_accrued = 0.0
                    in_pos = True
                    # 3) The entry bar itself can hit stop or target after the
                    #    open — check it before moving on.
                    manage(i, is_entry_bar=True)

        equity_curve[i] = equity + _open_mark(
            in_pos, direction, entry_price, closes[i], qty
        )

    # Force-close anything still open at the last bar's close.
    if in_pos:
        close_position(n - 1, closes[n - 1], "time")
        equity_curve[n - 1] = equity

    blotter = pd.DataFrame([t.__dict__ for t in trades])
    curve = pd.DataFrame({"bar_start": rows["bar_start"], "equity": equity_curve})
    return {
        "trades": blotter,
        "equity_curve": curve,
        "final_equity": float(equity),
        "metrics": compute_metrics(blotter, curve, bt),
    }


def _open_mark(in_pos, direction, entry_price, close, qty) -> float:
    """Mark-to-market of an open position for the equity curve only."""
    if not in_pos:
        return 0.0
    return direction * (close - entry_price) * qty


# ----------------------------------------------------------------------
# Metrics — the numbers that actually decide it
# ----------------------------------------------------------------------


def max_drawdown(equity: np.ndarray) -> float:
    """Largest peak-to-trough decline as a fraction of the running peak."""
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


def compute_metrics(
    blotter: pd.DataFrame, curve: pd.DataFrame, bt: BacktestConfig
) -> dict:
    """The decision metrics, with trade count front and centre.

    Win rate is deliberately not the headline. A 65% win rate over 40 trades
    is indistinguishable from luck; expectancy net of costs over hundreds of
    trades is what to trust. The `net_pnl_drop5` field re-reads the result
    with the five best trades removed — if the edge vanishes, it was five
    lucky days, not a strategy.
    """
    n = len(blotter)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": float("nan"),
            "expectancy": float("nan"),
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "total_fees": 0.0,
            "total_funding": 0.0,
            "profit_factor": float("nan"),
            "max_drawdown": max_drawdown(curve["equity"].to_numpy()),
            "return_pct": 0.0,
            "net_pnl_drop5": 0.0,
            "avg_bars_held": float("nan"),
        }

    net = blotter["net_pnl"].to_numpy()
    wins = net[net > 0]
    losses = net[net < 0]
    gross_win = wins.sum()
    gross_loss = -losses.sum()

    drop5 = np.sort(net)[:-5].sum() if n > 5 else 0.0

    return {
        "n_trades": int(n),
        "win_rate": float((net > 0).mean()),
        "expectancy": float(net.mean()),
        "net_pnl": float(net.sum()),
        "gross_pnl": float(blotter["gross_pnl"].sum()),
        "total_fees": float(blotter["fees"].sum()),
        "total_funding": float(blotter["funding"].sum()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "max_drawdown": max_drawdown(curve["equity"].to_numpy()),
        "return_pct": float(curve["equity"].iloc[-1] / bt.equity0 - 1.0),
        "net_pnl_drop5": float(drop5),
        "avg_bars_held": float(blotter["bars_held"].mean()),
    }


def format_metrics(m: dict, label: str = "") -> str:
    if m["n_trades"] == 0:
        return f"[{label}] no trades taken."
    return (
        f"[{label}]\n"
        f"  trades           {m['n_trades']}\n"
        f"  win rate         {m['win_rate']:.1%}\n"
        f"  expectancy/trade {m['expectancy']:+.2f}\n"
        f"  net pnl          {m['net_pnl']:+.2f}  (gross {m['gross_pnl']:+.2f}, "
        f"fees {m['total_fees']:.2f}, funding {m['total_funding']:+.2f})\n"
        f"  return           {m['return_pct']:+.1%}\n"
        f"  profit factor    {m['profit_factor']:.2f}\n"
        f"  max drawdown     {m['max_drawdown']:.1%}\n"
        f"  avg bars held    {m['avg_bars_held']:.1f}\n"
        f"  net pnl w/o best5 {m['net_pnl_drop5']:+.2f}   "
        f"(edge is fragile if this collapses)"
    )


def run_with_and_without_costs(
    signals: pd.DataFrame,
    bt: BacktestConfig | None = None,
    costs: CostConfig | None = None,
) -> dict:
    """Run the same signals twice — frictionless and with costs.

    The gap between the two equity curves is the fragility budget. If the
    frictionless run is a winner and the costs-on run is a loser, the edge
    was never real at the horizon you are trading.
    """
    bt = bt or BacktestConfig()
    costs = costs or CostConfig()
    costs_off = CostConfig(**{**costs.__dict__, "enabled": False})

    with_costs = run_backtest(signals, bt, costs)
    without_costs = run_backtest(signals, bt, costs_off)
    return {"with_costs": with_costs, "without_costs": without_costs}


def main(argv=None) -> int:
    import argparse

    from strategy import StrategyConfig, generate_signals

    p = argparse.ArgumentParser(description="Backtest footprint signals over bars.parquet")
    p.add_argument("--bars", required=True, help="bars.parquet from footprint.py")
    p.add_argument("--setup", default="stacked_continuation")
    p.add_argument("--tick", type=float, default=10.0)
    p.add_argument("--taker-bps", type=float, default=4.5)
    p.add_argument("--rr", type=float, default=2.0)
    p.add_argument("--atr-stop-mult", type=float, default=1.5)
    args = p.parse_args(argv)

    bars = pd.read_parquet(args.bars)
    scfg = StrategyConfig(
        setup=args.setup, reward_risk=args.rr, atr_stop_mult=args.atr_stop_mult
    )
    signals = generate_signals(bars, scfg)
    costs = CostConfig(taker_bps=args.taker_bps, tick_size=args.tick)

    res = run_with_and_without_costs(signals, BacktestConfig(), costs)
    print(format_metrics(res["without_costs"]["metrics"], "frictionless"))
    print()
    print(format_metrics(res["with_costs"]["metrics"], "with costs"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
