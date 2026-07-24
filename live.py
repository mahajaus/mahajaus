"""
Live runner for the footprint bot — paper by default, and deliberately so.

Execution comes last in this project, and it comes wrapped in safety. This
module streams trades, aggregates them into bars online with the SAME
footprint code the backtest uses, generates signals on each closed bar, and
manages a single position. By default it is a paper trader: it places no real
orders and moves no real money. Real order routing is a stub that raises
unless you deliberately wire in a broker — a footprint edge has to survive
the study and the backtest before it earns the right to touch an exchange.

TWO INPUT SOURCES
-----------------
  --replay FILE : stream a Binance aggTrades CSV through the online path.
                  Needs no network, runs deterministically, and is how the
                  live path is tested. Start here.
  --live SYMBOL : connect to the Binance USDⓈ-M futures aggTrade websocket.
                  Requires the `websockets` package and network access.

The online aggregator emits a bar the moment a trade crosses into the next
interval, so signals are produced on genuinely closed bars only — the same
no-lookahead contract the backtest enforces.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional

import numpy as np
import pandas as pd

from footprint import (
    FootprintConfig,
    build_levels,
    add_imbalances,
    build_bars,
)
from strategy import StrategyConfig, generate_signals
from backtest import CostConfig


# ----------------------------------------------------------------------
# Online aggregation
# ----------------------------------------------------------------------


@dataclass
class Tick:
    price: float
    qty: float
    ts_ms: int
    is_buyer_maker: bool


class OnlineFootprint:
    """Accumulate ticks into the current bar; finalise on rollover.

    Reuses the batch footprint functions on each completed bar's tick
    buffer, so the online numbers are identical to the offline ones by
    construction — no second, drifting implementation of the maths.
    """

    def __init__(self, cfg: FootprintConfig):
        self.cfg = cfg
        self._buf: list[Tick] = []
        self._cur_bar: Optional[pd.Timestamp] = None
        self.bars = pd.DataFrame()

    def _bar_of(self, ts_ms: int) -> pd.Timestamp:
        return pd.Timestamp(ts_ms, unit="ms", tz="UTC").floor(self.cfg.bar_interval)

    def add(self, tick: Tick) -> Optional[pd.DataFrame]:
        """Feed one tick. Returns the freshly closed bar row, or None."""
        bar = self._bar_of(tick.ts_ms)
        finalized = None
        if self._cur_bar is None:
            self._cur_bar = bar
        elif bar > self._cur_bar:
            finalized = self._finalize()
            self._cur_bar = bar
            self._buf = []
        self._buf.append(tick)
        return finalized

    def _finalize(self) -> Optional[pd.DataFrame]:
        if not self._buf:
            return None
        trades = pd.DataFrame(
            {
                "price": [t.price for t in self._buf],
                "quantity": [t.qty for t in self._buf],
                "transact_time": [t.ts_ms for t in self._buf],
                "is_buyer_maker": [t.is_buyer_maker for t in self._buf],
            }
        )
        levels = add_imbalances(build_levels(trades, self.cfg), self.cfg)
        bar = build_bars(trades, levels, self.cfg)
        self.bars = (
            pd.concat([self.bars, bar], ignore_index=True) if len(self.bars) else bar
        )
        # cum_delta is a running sum across the whole session
        self.bars["cum_delta"] = self.bars["delta"].cumsum()
        return bar


# ----------------------------------------------------------------------
# Paper execution
# ----------------------------------------------------------------------


@dataclass
class Position:
    direction: int
    entry_price: float
    qty: float
    stop_px: float
    target_px: float
    entry_ts: int
    setup: str


@dataclass
class PaperBroker:
    """A minimal paper broker. Fills at the given price plus slippage.

    It is intentionally simple and clearly not an exchange. It exists so the
    live path can be exercised and logged end to end without risk.
    """

    costs: CostConfig
    equity: float = 10_000.0
    position: Optional[Position] = None
    fills: list = field(default_factory=list)

    def _slip(self) -> float:
        return self.costs.slippage_ticks * self.costs.tick_size if self.costs.enabled else 0.0

    def _fee(self, notional: float) -> float:
        return notional * (self.costs.taker_bps / 1e4) if self.costs.enabled else 0.0

    def enter(self, direction, ref_price, stop_dist, target_dist, ts, setup) -> None:
        if self.position is not None:
            return
        entry = ref_price + direction * self._slip()
        qty = self.equity / entry if entry > 0 else 0.0
        if qty <= 0:
            return
        self.position = Position(
            direction=direction,
            entry_price=entry,
            qty=qty,
            stop_px=entry - direction * stop_dist,
            target_px=entry + direction * target_dist,
            entry_ts=ts,
            setup=setup,
        )
        self.equity -= self._fee(abs(entry) * qty)

    def on_price(self, price: float, ts: int) -> Optional[dict]:
        """Check stop/target against a live price. Returns a fill on exit."""
        p = self.position
        if p is None:
            return None
        hit_stop = price <= p.stop_px if p.direction > 0 else price >= p.stop_px
        hit_target = price >= p.target_px if p.direction > 0 else price <= p.target_px
        if not (hit_stop or hit_target):
            return None
        # pessimistic: if a single print somehow satisfies both, take the stop
        exit_ref = p.stop_px if hit_stop else p.target_px
        reason = "stop" if hit_stop else "target"
        exit_px = exit_ref - p.direction * self._slip()
        gross = p.direction * (exit_px - p.entry_price) * p.qty
        self.equity += gross - self._fee(abs(exit_px) * p.qty)
        fill = {
            "entry_ts": p.entry_ts,
            "exit_ts": ts,
            "direction": p.direction,
            "entry_price": p.entry_price,
            "exit_price": exit_px,
            "reason": reason,
            "gross_pnl": gross,
            "equity": self.equity,
            "setup": p.setup,
        }
        self.fills.append(fill)
        self.position = None
        return fill


class RealBroker:
    """Placeholder for a real exchange connection. Intentionally inert.

    Wiring this up is a deliberate, separate decision that should only happen
    after a signal has survived the study and the backtest. It raises on
    every call so 'live real' can never happen by accident or default.
    """

    def enter(self, *a, **k):
        raise NotImplementedError(
            "Real order routing is not wired up. This bot trades on paper "
            "until you deliberately implement RealBroker against your exchange "
            "account — and not before a signal has earned it in backtest."
        )

    def on_price(self, *a, **k):
        raise NotImplementedError("RealBroker is a stub; use PaperBroker.")


# ----------------------------------------------------------------------
# The trader
# ----------------------------------------------------------------------


@dataclass
class LiveConfig:
    warmup_bars: int = 30      # don't trade until this many bars exist
    verbose: bool = True


class LiveTrader:
    """Glue: ticks -> online bars -> signal on close -> paper position.

    On each closed bar it recomputes signals over the accumulated history
    (the strategy is cheap) and, if flat and the latest closed bar fired a
    signal, arms an entry to be filled on the next incoming trade — the live
    analogue of the backtest's next-bar-open fill.
    """

    def __init__(
        self,
        fp_cfg: FootprintConfig,
        st_cfg: StrategyConfig,
        broker: PaperBroker,
        live_cfg: LiveConfig | None = None,
    ):
        self.agg = OnlineFootprint(fp_cfg)
        self.st_cfg = st_cfg
        self.broker = broker
        self.cfg = live_cfg or LiveConfig()
        self._pending: Optional[tuple] = None  # (direction, stop_dist, target_dist, setup)

    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(msg, file=sys.stderr)

    def on_tick(self, tick: Tick) -> None:
        # 1) manage any open position against the live price first
        self.broker.on_price(tick.price, tick.ts_ms)

        # 2) fill a pending entry at this trade's price (the "next open")
        if self._pending is not None and self.broker.position is None:
            d, sd, td, setup = self._pending
            self.broker.enter(d, tick.price, sd, td, tick.ts_ms, setup)
            if self.broker.position is not None:
                self._log(
                    f"ENTER {'LONG' if d>0 else 'SHORT'} @ {tick.price:.2f} "
                    f"stop {self.broker.position.stop_px:.2f} "
                    f"tgt {self.broker.position.target_px:.2f} [{setup}]"
                )
            self._pending = None

        # 3) feed the aggregator; if a bar just closed, re-evaluate the signal
        closed = self.agg.add(tick)
        if closed is None:
            return
        bars = self.agg.bars
        if len(bars) < self.cfg.warmup_bars:
            return

        sig = generate_signals(bars, self.st_cfg)
        last = sig.iloc[-1]
        if (
            self.broker.position is None
            and self._pending is None
            and int(last["signal"]) != 0
            and last["stop_dist"] > 0
        ):
            self._pending = (
                int(last["signal"]),
                float(last["stop_dist"]),
                float(last["target_dist"]),
                str(last["setup"]),
            )
            self._log(
                f"SIGNAL {last['setup']} dir={int(last['signal'])} on bar "
                f"{last['bar_start']} — armed for next trade"
            )


# ----------------------------------------------------------------------
# Feeds
# ----------------------------------------------------------------------


def replay_feed(path: str) -> Iterator[Tick]:
    """Stream a Binance aggTrades CSV as ticks, in timestamp order."""
    from footprint import load_agg_trades

    df = load_agg_trades(path).sort_values("transact_time")
    for row in df.itertuples(index=False):
        yield Tick(
            price=float(row.price),
            qty=float(row.quantity),
            ts_ms=int(row.transact_time),
            is_buyer_maker=bool(row.is_buyer_maker),
        )


def binance_ws_feed(symbol: str) -> Iterator[Tick]:
    """Stream live aggTrades from Binance USDⓈ-M futures. Needs `websockets`.

    Yields ticks forever. This is the only path that touches the network,
    and it is import-guarded so the rest of the module works without the
    dependency installed.
    """
    try:
        import asyncio
        import websockets
    except ImportError as e:  # pragma: no cover - network/dep path
        raise RuntimeError(
            "live streaming needs the 'websockets' package: pip install websockets"
        ) from e

    url = f"wss://fstream.binance.com/ws/{symbol.lower()}@aggTrade"

    async def _run(queue: "asyncio.Queue") -> None:  # pragma: no cover
        async with websockets.connect(url) as ws:
            async for raw in ws:
                m = json.loads(raw)
                await queue.put(
                    Tick(
                        price=float(m["p"]),
                        qty=float(m["q"]),
                        ts_ms=int(m["T"]),
                        is_buyer_maker=bool(m["m"]),
                    )
                )

    # Bridge the async socket to a blocking generator.
    import asyncio  # pragma: no cover

    loop = asyncio.new_event_loop()  # pragma: no cover
    queue: "asyncio.Queue" = asyncio.Queue()  # pragma: no cover
    task = loop.create_task(_run(queue))  # pragma: no cover
    try:  # pragma: no cover
        while True:
            tick = loop.run_until_complete(queue.get())
            yield tick
    finally:  # pragma: no cover
        task.cancel()
        loop.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def run(feed: Iterator[Tick], trader: LiveTrader, max_ticks: int | None = None) -> dict:
    """Drive a trader from any tick feed. Returns a run summary."""
    n = 0
    for tick in feed:
        trader.on_tick(tick)
        n += 1
        if max_ticks is not None and n >= max_ticks:
            break
    b = trader.broker
    return {
        "ticks": n,
        "bars": len(trader.agg.bars),
        "fills": b.fills,
        "n_fills": len(b.fills),
        "equity": b.equity,
        "open_position": b.position,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Live/paper footprint runner")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="Binance aggTrades CSV to stream offline")
    src.add_argument("--live", help="symbol for the Binance futures websocket, e.g. BTCUSDT")
    p.add_argument("--interval", default="1min")
    p.add_argument("--tick", type=float, default=10.0)
    p.add_argument("--setup", default="delta_divergence")
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument(
        "--real",
        action="store_true",
        help="attempt REAL order routing (stubbed — will refuse). Paper is default.",
    )
    args = p.parse_args(argv)

    fp_cfg = FootprintConfig(bar_interval=args.interval, tick_size=args.tick)
    st_cfg = StrategyConfig(setup=args.setup)
    costs = CostConfig(tick_size=args.tick)

    if args.real:
        print(
            "refusing: real order routing is not implemented. Running paper.",
            file=sys.stderr,
        )
    broker = PaperBroker(costs=costs)
    trader = LiveTrader(fp_cfg, st_cfg, broker, LiveConfig(warmup_bars=args.warmup))

    if args.replay:
        feed = replay_feed(args.replay)
    else:
        feed = binance_ws_feed(args.live)

    summary = run(feed, trader, max_ticks=args.max_ticks)
    print(
        f"\nran {summary['ticks']:,} ticks -> {summary['bars']} bars, "
        f"{summary['n_fills']} paper fills, equity {summary['equity']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
