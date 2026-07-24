"""
End-to-end demo: aggregator -> study -> backtest, on synthetic data.

Runs the whole pipeline without a download so you can confirm the plumbing
works and see the shape of every output. The numbers are meaningless — the
input is a random walk with no edge — but the pipeline is exactly the one you
will point at real Binance archives.

    python demo.py

The order it runs in is the order the project insists on:
  1. aggregate ticks into bars/levels
  2. ask whether any bucket carries forward information (the gate)
  3. only then, backtest a strategy with costs on and off
"""

from __future__ import annotations

from footprint import FootprintConfig, build_levels, add_imbalances, build_bars
from strategy import StrategyConfig, generate_signals
from study import StudyConfig, run_study, format_report
from backtest import (
    BacktestConfig,
    CostConfig,
    run_with_and_without_costs,
    format_metrics,
)
from synth import generate_agg_trades


def main() -> int:
    print("1) generating synthetic tape and aggregating ...")
    trades = generate_agg_trades(n_trades=400_000, seed=1)
    cfg = FootprintConfig(tick_size=10.0, bar_interval="1min")
    levels = add_imbalances(build_levels(trades, cfg), cfg)
    bars = build_bars(trades, levels, cfg)
    print(f"   {len(trades):,} trades -> {len(bars):,} bars, {len(levels):,} levels\n")

    print("2) signal-existence study (the gate) ...")
    scfg = StudyConfig(horizons=(3, 12, 48), n_bootstrap=1000)
    study_df = run_study(bars, scfg)
    print(format_report(study_df, scfg))
    print()

    print("3) backtest, costs off vs on (synthetic — numbers are noise) ...")
    signals = generate_signals(
        bars, StrategyConfig(setup="delta_divergence", min_delta_pct=0.05)
    )
    res = run_with_and_without_costs(
        signals, BacktestConfig(), CostConfig(tick_size=10.0)
    )
    print(format_metrics(res["without_costs"]["metrics"], "frictionless"))
    print()
    print(format_metrics(res["with_costs"]["metrics"], "with costs"))
    print(
        "\nreminder: on a random walk the expected verdict is 'dead' and the "
        "expected\nbacktest edge is negative once costs are on. That is the "
        "pipeline working."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
