"""
Signal-existence study — the step that runs before any strategy is believed.

The question every strategy silently assumes an answer to: does the metric
carry any forward information at all? This module answers it directly and
cheaply, by bucketing bars on a condition and comparing each bucket's
forward-return distribution against the baseline of all bars.

If a bucket separates from baseline across every horizon, there may be
something there. If it doesn't, the metric is dead and no amount of strategy
tuning will resurrect it — you have learned that in an afternoon instead of
in six months of iteration.

This is a measurement tool, not a backtest. It applies no costs and takes no
positions. Costs come later, in `backtest.py`, and only for signals that
survive here.

FORWARD RETURN CONVENTION
-------------------------
For horizon N, the forward return of bar i is the log return of close from
bar i to bar i+N:  log(close[i+N] / close[i]). Bars without a full N-bar
future are dropped from that horizon. Forward returns are, by construction,
information from the future — they are the thing being predicted, never a
feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class StudyConfig:
    horizons: tuple[int, ...] = (3, 12, 48)
    # A bucket "separates" if its median forward return differs from baseline
    # by at least this many multiples of the baseline's per-bar dispersion.
    separation_k: float = 0.15
    n_bootstrap: int = 2000
    seed: int = 7


@dataclass
class BucketResult:
    name: str
    direction: int          # +1 bullish expectation, -1 bearish, 0 neutral
    horizon: int
    n: int
    baseline_median: float
    bucket_median: float
    bucket_mean: float
    separation: float       # (bucket_median - baseline_median) / baseline_std
    p_value: float          # bootstrap two-sided, bucket mean vs baseline
    aligned: bool           # does the sign of the move match `direction`?


def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """log(close[i+N]/close[i]); NaN where the future is incomplete."""
    close = bars["close"].to_numpy(dtype=float)
    fwd = np.full(len(close), np.nan)
    if horizon < len(close):
        fwd[:-horizon] = np.log(close[horizon:] / close[:-horizon])
    return fwd


def define_buckets(bars: pd.DataFrame) -> dict[str, tuple[np.ndarray, int]]:
    """Map bucket name -> (boolean mask over bars, directional expectation).

    Directional expectation encodes what a *bullish* or *bearish* reading
    would predict, so we can check whether the realised move aligns. It is
    not an assertion that it does — that is exactly what the study tests.
    """
    b = bars
    close = b["close"].to_numpy()
    poc = b["poc"].to_numpy()
    vah = b["va_high"].to_numpy()
    val = b["va_low"].to_numpy()

    buckets = {
        "buy_stack": (b["has_buy_stack"].to_numpy(), +1),
        "sell_stack": (b["has_sell_stack"].to_numpy(), -1),
        "close_above_vah": (close > vah, +1),
        "close_below_val": (close < val, -1),
        "close_above_poc": (close > poc, +1),
        "close_below_poc": (close < poc, -1),
        "pos_delta": (b["delta"].to_numpy() > 0, +1),
        "neg_delta": (b["delta"].to_numpy() < 0, -1),
    }
    return buckets


def _bootstrap_p(
    bucket_vals: np.ndarray, baseline_vals: np.ndarray, n: int, rng: np.random.Generator
) -> float:
    """Two-sided bootstrap p-value for bucket mean vs baseline mean.

    Resample the baseline at the bucket's size many times and ask how often
    the resampled mean is at least as extreme as the observed bucket mean.
    """
    if len(bucket_vals) == 0 or len(baseline_vals) == 0:
        return 1.0
    obs = bucket_vals.mean()
    base_mean = baseline_vals.mean()
    size = len(bucket_vals)
    draws = rng.choice(baseline_vals, size=(n, size), replace=True).mean(axis=1)
    # centre on baseline mean so we test deviation, not level
    extreme = np.abs(draws - base_mean) >= np.abs(obs - base_mean)
    return float((extreme.sum() + 1) / (n + 1))


def run_study(bars: pd.DataFrame, cfg: StudyConfig | None = None) -> pd.DataFrame:
    """Run the conditional-return study over all buckets and horizons.

    Returns a tidy DataFrame of BucketResult rows, sorted so the strongest,
    correctly-aligned separations come first.
    """
    cfg = cfg or StudyConfig()
    rng = np.random.default_rng(cfg.seed)
    buckets = define_buckets(bars)

    rows: list[BucketResult] = []
    for horizon in cfg.horizons:
        fwd = forward_returns(bars, horizon)
        complete = ~np.isnan(fwd)
        baseline = fwd[complete]
        if len(baseline) == 0:
            continue
        base_median = float(np.median(baseline))
        base_std = float(np.std(baseline)) or np.nan

        for name, (mask, direction) in buckets.items():
            m = mask & complete
            vals = fwd[m]
            if len(vals) == 0:
                continue
            b_median = float(np.median(vals))
            b_mean = float(np.mean(vals))
            sep = (b_median - base_median) / base_std if base_std else 0.0
            p = _bootstrap_p(vals, baseline, cfg.n_bootstrap, rng)
            aligned = np.sign(b_median - base_median) == np.sign(direction)
            rows.append(
                BucketResult(
                    name=name,
                    direction=direction,
                    horizon=horizon,
                    n=int(len(vals)),
                    baseline_median=base_median,
                    bucket_median=b_median,
                    bucket_mean=b_mean,
                    separation=float(sep),
                    p_value=p,
                    aligned=bool(aligned),
                )
            )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df
    df["abs_sep"] = df["separation"].abs()
    df = df.sort_values(["abs_sep", "aligned"], ascending=[False, False])
    return df.drop(columns="abs_sep").reset_index(drop=True)


def verdict(study_df: pd.DataFrame, cfg: StudyConfig | None = None) -> dict:
    """Apply the kill criteria to a study result.

    A bucket "survives" if, for every horizon, it separates from baseline by
    at least `separation_k` in the direction its label predicts, with a
    bootstrap p-value under 0.05. This is intentionally strict: order flow
    has a strong mythology and a lax gate lets noise through.
    """
    cfg = cfg or StudyConfig()
    if study_df.empty:
        return {"survivors": [], "dead": True, "reason": "no data"}

    survivors = []
    for name, grp in study_df.groupby("name"):
        ok = True
        for _, r in grp.iterrows():
            passes = (
                r["aligned"]
                and abs(r["separation"]) >= cfg.separation_k
                and r["p_value"] < 0.05
            )
            if not passes:
                ok = False
                break
        if ok and grp["horizon"].nunique() == len(set(cfg.horizons)):
            survivors.append(name)

    return {
        "survivors": survivors,
        "dead": len(survivors) == 0,
        "reason": (
            "no bucket separated from baseline across all horizons"
            if not survivors
            else f"{len(survivors)} bucket(s) cleared the gate"
        ),
    }


def format_report(study_df: pd.DataFrame, cfg: StudyConfig | None = None) -> str:
    """Human-readable study report plus the verdict."""
    cfg = cfg or StudyConfig()
    if study_df.empty:
        return "study: no complete bars for any horizon — need more data."

    lines = ["conditional-return study", "=" * 72]
    header = (
        f"{'bucket':>16} {'dir':>4} {'H':>4} {'n':>7} "
        f"{'base_med':>10} {'buck_med':>10} {'sep':>7} {'p':>7} {'aln':>4}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in study_df.iterrows():
        lines.append(
            f"{r['name']:>16} {r['direction']:>+4d} {r['horizon']:>4d} "
            f"{r['n']:>7d} {r['baseline_median']:>10.6f} "
            f"{r['bucket_median']:>10.6f} {r['separation']:>7.2f} "
            f"{r['p_value']:>7.3f} {str(bool(r['aligned'])):>4}"
        )

    v = verdict(study_df, cfg)
    lines.append("")
    lines.append(f"verdict: {v['reason']}")
    if v["survivors"]:
        lines.append(f"survivors: {', '.join(v['survivors'])}")
    else:
        lines.append("KILL CRITERION MET — the metric is dead on this data. Stop.")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Signal-existence study over bars.parquet")
    p.add_argument("--bars", required=True, help="bars.parquet from footprint.py --out-dir")
    p.add_argument("--horizons", default="3,12,48", help="comma-separated bar horizons")
    args = p.parse_args(argv)

    bars = pd.read_parquet(args.bars)
    cfg = StudyConfig(horizons=tuple(int(h) for h in args.horizons.split(",")))
    df = run_study(bars, cfg)
    print(format_report(df, cfg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
