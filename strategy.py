"""
Signal layer for the footprint bot.

This module turns the aggregator's `bars` table into per-bar trading
signals. It contains *mechanics only* — no claim that any of these setups
has an edge. Whether a signal carries forward information is a question for
`study.py`, and whether it survives costs is a question for `backtest.py`.
The honest sequence is: build the signal, measure it, and only then believe
it. This file deliberately makes that easy and makes no promises.

TIMING CONTRACT (read this before touching the backtest)
-------------------------------------------------------
A signal on bar i is computed *only* from data through bar i's close. It may
never be acted on before bar i+1's open. The backtest enforces the one-bar
delay; this module's job is simply to never peek forward. Every feature here
is either from the current closed bar or a trailing window ending at it.

OUTPUT
------
`generate_signals(bars, cfg)` returns the `bars` frame with added columns:

    signal      : +1 long, -1 short, 0 flat  (intended for NEXT bar's open)
    stop_dist   : stop distance in price units, set at the signal bar
    target_dist : target distance in price units
    setup       : name of the setup that fired (or "")

Distances are returned rather than absolute levels because the entry price
is not known until the next bar opens; the backtest anchors them to the
actual fill.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Signal-generation knobs. All are sensitivity-test dimensions.

    Keep the search space small and pre-declared — the kill criteria allow
    three configurations, not thirty.
    """

    setup: str = "stacked_continuation"  # or "delta_divergence"
    atr_window: int = 14                 # bars for the true-range average
    atr_stop_mult: float = 1.5           # stop distance = mult * ATR
    reward_risk: float = 2.0             # target distance = rr * stop distance
    divergence_lookback: int = 20        # bars for the new-high/new-low test
    min_delta_pct: float = 0.10          # |bar delta| / volume floor to act
    exclude_edge_levels: int = 1         # reserved: trim outermost levels
    # A bullish setup wants price accepting higher; require close in the
    # upper `accept_frac` of the bar range to confirm.
    accept_frac: float = 0.5


# ----------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------


def true_range(bars: pd.DataFrame) -> pd.Series:
    """Classic true range: max of the three high/low/prev-close spans."""
    prev_close = bars["close"].shift(1)
    a = bars["high"] - bars["low"]
    b = (bars["high"] - prev_close).abs()
    c = (bars["low"] - prev_close).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)


def add_features(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Attach trailing features used by the setups. No forward peeking.

    ATR uses only bars up to and including the current one (min_periods
    equal to the window, so early bars are NaN and produce no signal).
    """
    out = bars.copy()
    tr = true_range(out)
    out["atr"] = tr.rolling(cfg.atr_window, min_periods=cfg.atr_window).mean()

    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_pos"] = (out["close"] - out["low"]) / rng  # 0=low, 1=high

    # Rolling extremes over the lookback, *excluding* the current bar so the
    # "new high" test is genuinely against prior bars only.
    lb = cfg.divergence_lookback
    out["prior_high"] = out["high"].shift(1).rolling(lb, min_periods=lb).max()
    out["prior_low"] = out["low"].shift(1).rolling(lb, min_periods=lb).min()
    return out


# ----------------------------------------------------------------------
# Setups
# ----------------------------------------------------------------------


def _stacked_continuation(bars: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    """Stacked diagonal imbalances in the direction of the close.

    Hypothesis: a bar that prints a stack of buy imbalances AND closes in
    the upper part of its range is showing aggressive buyers with follow
    through — a continuation-up tell. Symmetric for sell stacks.
    """
    sig = np.zeros(len(bars), dtype=int)
    long_ok = (
        bars["has_buy_stack"].to_numpy()
        & (bars["close_pos"].to_numpy() >= cfg.accept_frac)
        & (bars["delta_pct"].to_numpy() >= cfg.min_delta_pct)
    )
    short_ok = (
        bars["has_sell_stack"].to_numpy()
        & (bars["close_pos"].to_numpy() <= (1.0 - cfg.accept_frac))
        & (bars["delta_pct"].to_numpy() <= -cfg.min_delta_pct)
    )
    sig[long_ok] = 1
    sig[short_ok] = -1
    return sig


def _delta_divergence(bars: pd.DataFrame, cfg: StrategyConfig) -> np.ndarray:
    """Price extends but delta disagrees — an exhaustion / reversal tell.

    Hypothesis: a bar that makes a new high over the lookback while its
    delta is negative shows the up-move is being sold into. Fade it short.
    Symmetric for a new low with positive delta.
    """
    sig = np.zeros(len(bars), dtype=int)
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    prior_high = bars["prior_high"].to_numpy()
    prior_low = bars["prior_low"].to_numpy()
    dpct = bars["delta_pct"].to_numpy()

    new_high = high > prior_high
    new_low = low < prior_low

    short_ok = new_high & (dpct <= -cfg.min_delta_pct)
    long_ok = new_low & (dpct >= cfg.min_delta_pct)
    sig[short_ok] = -1
    sig[long_ok] = 1
    return sig


SETUPS = {
    "stacked_continuation": _stacked_continuation,
    "delta_divergence": _delta_divergence,
}


def generate_signals(bars: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    """Compute per-bar signals and stop/target distances.

    Returns a copy of `bars` with `signal`, `stop_dist`, `target_dist`,
    and `setup` columns added. Rows without a valid ATR (warm-up period)
    are forced flat.
    """
    if cfg.setup not in SETUPS:
        raise ValueError(f"unknown setup {cfg.setup!r}; choose from {list(SETUPS)}")

    out = add_features(bars, cfg)
    sig = SETUPS[cfg.setup](out, cfg)

    atr = out["atr"].to_numpy()
    valid = ~np.isnan(atr)
    sig = np.where(valid, sig, 0)

    stop_dist = cfg.atr_stop_mult * np.where(valid, atr, np.nan)
    target_dist = cfg.reward_risk * stop_dist

    out["signal"] = sig
    out["stop_dist"] = np.where(sig != 0, stop_dist, np.nan)
    out["target_dist"] = np.where(sig != 0, target_dist, np.nan)
    out["setup"] = np.where(sig != 0, cfg.setup, "")
    return out
