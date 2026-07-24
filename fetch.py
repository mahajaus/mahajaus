"""
Fetch Binance aggTrades archives — run this on YOUR machine.

The public archives at data.binance.vision are free and need no account, but
they are a plain file download, so this must run somewhere with open internet.
It will not work inside a locked-down CI/agent sandbox (the host may block the
domain); that is expected. On a normal machine it is one command.

Uses the standard library only — no pip install needed.

    # one day of BTCUSDT USDⓈ-M futures aggTrades
    python fetch.py --symbol BTCUSDT --date 2025-01-15

    # a whole month
    python fetch.py --symbol BTCUSDT --month 2025-01

    # a date range (inclusive)
    python fetch.py --symbol BTCUSDT --from 2025-01-01 --to 2025-01-07

Downloads land in ./data as unzipped CSVs, ready for:

    python footprint.py --file data/BTCUSDT-aggTrades-2025-01-15.csv --inspect "..."
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.binance.vision/data"

# market -> URL path segment
MARKETS = {
    "um": "futures/um",   # USDⓈ-M futures (BTCUSDT perp) — the project default
    "cm": "futures/cm",   # COIN-M futures
    "spot": "spot",       # spot
}


def _daily_url(market: str, symbol: str, day: dt.date) -> str:
    seg = MARKETS[market]
    fname = f"{symbol}-aggTrades-{day:%Y-%m-%d}.zip"
    return f"{BASE}/{seg}/daily/aggTrades/{symbol}/{fname}"


def _monthly_url(market: str, symbol: str, year: int, month: int) -> str:
    seg = MARKETS[market]
    fname = f"{symbol}-aggTrades-{year:04d}-{month:02d}.zip"
    return f"{BASE}/{seg}/monthly/aggTrades/{symbol}/{fname}"


def _download_and_unzip(url: str, out_dir: Path) -> Path | None:
    """Download a .zip and extract its single CSV. Returns the CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / Path(url).name
    try:
        print(f"  GET {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        zip_path.write_bytes(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  404 not found (no archive for that date yet): {url}", file=sys.stderr)
        else:
            print(f"  HTTP {e.code} for {url}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        print(
            f"  network error: {e}\n"
            f"  (this domain is commonly blocked in sandboxes — run on your own machine)",
            file=sys.stderr,
        )
        return None

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        zf.extractall(out_dir)
    zip_path.unlink()  # keep only the CSV
    csv_path = out_dir / names[0]
    print(f"  -> {csv_path} ({csv_path.stat().st_size/1e6:.1f} MB)", file=sys.stderr)
    return csv_path


def _date_range(a: dt.date, b: dt.date):
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download Binance aggTrades archives")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--market", default="um", choices=list(MARKETS))
    p.add_argument("--date", help="single day, YYYY-MM-DD")
    p.add_argument("--month", help="whole month, YYYY-MM")
    p.add_argument("--from", dest="date_from", help="range start, YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", help="range end, YYYY-MM-DD (inclusive)")
    p.add_argument("--out-dir", default="data")
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    got: list[Path] = []

    if args.month:
        year, month = (int(x) for x in args.month.split("-"))
        url = _monthly_url(args.market, args.symbol, year, month)
        r = _download_and_unzip(url, out_dir)
        if r:
            got.append(r)
    elif args.date:
        day = dt.date.fromisoformat(args.date)
        r = _download_and_unzip(_daily_url(args.market, args.symbol, day), out_dir)
        if r:
            got.append(r)
    elif args.date_from and args.date_to:
        a = dt.date.fromisoformat(args.date_from)
        b = dt.date.fromisoformat(args.date_to)
        for day in _date_range(a, b):
            r = _download_and_unzip(_daily_url(args.market, args.symbol, day), out_dir)
            if r:
                got.append(r)
    else:
        p.error("specify one of --date, --month, or --from/--to")

    if not got:
        print("nothing downloaded.", file=sys.stderr)
        return 1
    print(f"\ndownloaded {len(got)} file(s) to {out_dir}/")
    print("next: python footprint.py --file "
          f"{got[0]} --inspect \"<a bar in this file>\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
