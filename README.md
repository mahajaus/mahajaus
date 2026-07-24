# Footprint order-flow bot — BTCUSDT perp

An order-flow bot built on Binance aggregated-trade data. It turns a raw
trade tape into a footprint (bar × price-level bid/ask volume), asks whether
any footprint signal actually carries forward information, backtests the ones
that do with realistic perpetual-futures costs, and runs the survivors on
paper.

The pipeline is built in the order the project insists on, and each stage is
a gate on the next:

```
ticks ─▶ aggregator ─▶ validation ─▶ signal-existence study ─▶ backtest ─▶ paper
        footprint.py                 study.py        strategy.py+backtest.py  live.py
```

The discipline is the point. Order flow has a strong mythology; the code is
arranged so you measure before you believe — the study can return "dead", the
backtest shows costs-on vs costs-off side by side, and live trading is paper
by default with real order routing left as an inert stub.

## Files

| file | purpose |
|---|---|
| `footprint.py` | aggregator: loader, bar/level volume split, delta, POC, value area, imbalances |
| `strategy.py` | signal mechanics (stacked-imbalance continuation, delta divergence) — no edge claimed |
| `study.py` | signal-existence study: does a bucket separate from baseline? applies the kill criteria |
| `backtest.py` | event-driven, no-lookahead backtest with taker fees, funding, and slippage |
| `live.py` | online aggregator + paper trader; replay a CSV or stream the Binance websocket |
| `synth.py` | synthetic tape generator for tests and demos (no download needed) |
| `demo.py` | runs the whole pipeline end to end on synthetic data |
| `test_footprint.py` | aggregator tests, including the aggressor-mapping test |
| `test_strategy.py` | strategy, study, and backtest tests (no-lookahead, cost model, pessimistic fills) |
| `test_live.py` | online-vs-offline consistency and paper-broker tests |
| `README.md` | this file |

## Quick start

```bash
pip install pandas numpy pyarrow           # websockets too, only for live streaming
python test_footprint.py && python test_strategy.py && python test_live.py
python demo.py                             # whole pipeline on synthetic data
```

`demo.py` is the fastest way to see the shape of every stage. On its random-walk
input the study correctly returns "dead" and costs turn the backtest negative —
that is the pipeline working, not failing.

## Install

```bash
pip install pandas numpy pyarrow
```

`pyarrow` is only needed if you use `--out-dir` to write parquet.

## Get the data

Binance publishes free public archives of aggregated trades — no account, no
API key. Browse to `data.binance.vision` and take the futures (um) monthly
`aggTrades` dump for `BTCUSDT`. Each monthly file unzips to a CSV.

The loader handles the layout variations you will actually hit: header row
present or absent, 7-column futures or 8-column spot layout, millisecond or
microsecond timestamps, and `is_buyer_maker` as a bool or as `true`/`false`
strings.

## Run

```bash
# summary of the first bars
python footprint.py --file BTCUSDT-aggTrades-2025-01.csv --interval 5min --tick 10

# write the two tables for downstream work
python footprint.py --file BTCUSDT-aggTrades-2025-01.csv --out-dir out

# print one bar's ladder
python footprint.py --file BTCUSDT-aggTrades-2025-01.csv --inspect "2025-01-15 13:05"
```

Roughly 200k trades process in about a second, so a full month is a couple of
minutes.

## Output

**`levels`** — one row per (bar, price level):
`bar_start, price_level, buy_volume, sell_volume, total_volume, delta, trades,
buy_imbalance, sell_imbalance, buy_imb_ratio, sell_imb_ratio, buy_stacked,
sell_stacked`

**`bars`** — one row per bar:
`bar_start, open, high, low, close, volume, buy_volume, sell_volume, delta,
cum_delta, delta_pct, trades, poc, va_low, va_high, n_levels, buy_imb_count,
sell_imb_count, has_buy_stack, has_sell_stack`

## The aggressor convention

Binance gives `is_buyer_maker`, and it reads backwards from what you expect:

- `True` → the **buyer** was resting passively, so an aggressive **seller**
  hit the bid → the volume is **sell_volume**.
- `False` → the **buyer** was the aggressor, lifting the ask →
  **buy_volume**.

Invert this and every delta reading in the system is backwards, while the
equity curve still looks plausible. `test_aggressor_mapping` pins it down.

## Design decisions worth knowing

**Integer price bucketing.** Prices are scaled to integer cents before
flooring into ticks. Float `floor()` scatters volume across adjacent buckets
at the boundaries and silently distorts the ladder.

**Dense ladder.** Each bar is reindexed onto a gapless tick ladder before
imbalances are computed. Without it, a price level that never traded makes
`shift(1)` compare non-adjacent prices.

**Imbalance is diagonal, not horizontal.** This is the part most
implementations get wrong:

```
buy  imbalance at P :  buy_volume[P]  vs sell_volume[P - 1 tick]
sell imbalance at P :  sell_volume[P] vs buy_volume[P + 1 tick]
```

**Volume floor.** `min_level_volume` is applied to the numerator. Without it,
thin levels manufacture constant fake imbalances. A zero denominator counts
as an imbalance provided the numerator clears the floor.

**Value area.** Greedy single-level expansion outward from the POC, stepping
toward whichever adjacent level holds more volume, until `value_area_pct` of
bar volume is enclosed. Other tools use a two-level comparison and may differ
by a tick — worth knowing before you blame your code during validation.

## Known caveat

Volume naturally thins toward the extremes of a bar's range, so the top and
bottom levels generate imbalances more readily than the middle. Consider
excluding the outermost level or two before you treat imbalance counts as
meaningful. Test this rather than assuming it.

## The validation gate

**Do not write strategy code until this passes.**

1. Pick one specific day and one specific 5-minute bar.
2. Run `--inspect` on it.
3. Load the same symbol, same day, same interval, same tick size in a
   reference footprint tool.
4. Compare POC, bar delta, and the per-level buy/sell split.

They should match. If they don't, reconcile before going further. This is the
step that separates a working project from a year spent on broken math.

Run the tests first:

```bash
python test_footprint.py
```

## Kill criteria

Write these down before you look at any results, because thresholds are
continuous and you can always find a setting that looks good on data you have
already seen.

- If no signal bucket's median forward return separates from baseline across
  all tested horizons, the metric is dead and I stop.
- If separation exists in one month and vanishes in another, it is noise and
  I stop.
- I get three threshold configurations. Not thirty.

## The signal-existence study (`study.py`)

Runs *before* any strategy is believed. It buckets bars on a condition —
buy/sell stacks, close outside the value area, delta sign — and compares each
bucket's forward-return distribution against the baseline of all bars, over
several horizons, with a bootstrap p-value.

```bash
python footprint.py --file BTCUSDT-aggTrades-2025-01.csv --out-dir out
python study.py --bars out/bars.parquet --horizons 3,12,48
```

The report ends in a verdict against the kill criteria. If no bucket separates
from baseline across *every* horizon, aligned with the direction its label
predicts and with p < 0.05, the metric is dead — and you have learned that in
an afternoon instead of six months. Only survivors earn a backtest.

## The strategy layer (`strategy.py`)

Signal mechanics only, no claim of edge. Two documented setups, selectable and
fully parameterised:

- `stacked_continuation` — a stack of diagonal buy (sell) imbalances with the
  close in the upper (lower) part of the bar range: aggressive participation
  with follow-through.
- `delta_divergence` — a new high (low) over the lookback while bar delta
  disagrees: the move is being faded. An exhaustion tell.

Each signal on bar *i* is computed only from data through bar *i*'s close and
carries a stop and target distance sized off ATR. Whether either setup is worth
anything is a question for the study and the backtest, not for this file.

## The backtest (`backtest.py`)

Event-driven, bar by bar, built around three commitments:

1. **No lookahead.** A signal on bar *i* fills at bar *i+1*'s open. The loop
   cannot see a bar before it would have existed.
2. **Costs on or off, by a flag.** Taker fees (~4.5 bps/side), funding every
   8h while in a position, and slippage in ticks. The gap between the
   costs-on and costs-off equity curves is the fragility budget.
3. **Pessimistic fills.** When a bar straddles both stop and target, the stop
   fills first; slippage always works against you.

```bash
python backtest.py --bars out/bars.parquet --setup delta_divergence --rr 2.0
```

It prints the frictionless and with-costs results together. The metrics put
**trade count** and net-of-cost **expectancy** first — win rate is the least
informative number — and include `net_pnl_drop5`, the net result with the five
best trades removed. If the edge disappears with them, it was five lucky days.

## Live / paper trading (`live.py`)

Paper by default, and deliberately hard to make otherwise. The online
aggregator reuses the exact batch footprint code, so live closed-bar numbers
match the backtest bar-for-bar (`test_live.py` asserts this). Real order
routing is an inert stub that refuses to trade.

```bash
# offline: stream a CSV through the live path (no network, deterministic)
python live.py --replay BTCUSDT-aggTrades-2025-01.csv --interval 1min --tick 10

# live: Binance USDⓈ-M futures websocket (needs `pip install websockets`)
python live.py --live BTCUSDT --interval 1min --tick 10
```

Execution is last for a reason: a signal has to survive the study and the
backtest before it earns the right to touch an exchange.

## Workflow, end to end

1. `footprint.py --out-dir out` — aggregate a month of real aggTrades.
2. `footprint.py --inspect "<bar>"` — pass the validation gate against a
   reference chart. **Do not proceed until this matches.**
3. `study.py --bars out/bars.parquet` — is there any signal at all? Obey the
   verdict.
4. `backtest.py --bars out/bars.parquet` — does a survivor clear costs on a
   development month? Hold out a separate month before you believe it.
5. `live.py --replay ...` then `--live ...` — paper trade only.
