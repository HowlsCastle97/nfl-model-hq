import argparse
import sqlite3
import time
import sys
from datetime import datetime, timezone

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    ts_utc TEXT NOT NULL,
    ticker TEXT NOT NULL,
    event_ticker TEXT,
    series_ticker TEXT,
    title TEXT,
    yes_bid REAL,
    yes_ask REAL,
    last_price REAL,
    volume REAL,
    open_interest REAL,
    status TEXT,
    yes_sub_title TEXT,
    floor_strike TEXT
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON snapshots (ticker, ts_utc);
"""


def get_json(session, path, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(BASE + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)


def price(market, dollars_key, cents_key):
    v = market.get(dollars_key)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = market.get(cents_key)
    return None if v is None else float(v) / 100.0


def fetch_series_markets(session, series_ticker, status="open"):
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series_ticker, "limit": 200}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        data = get_json(session, "/markets", params)
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            return markets


def snapshot(db_path, series_list):
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    session = requests.Session()
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for col in ("yes_sub_title TEXT", "floor_strike TEXT"):
        try:
            con.execute(f"ALTER TABLE snapshots ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    total = 0
    for series in series_list:
        try:
            markets = fetch_series_markets(session, series)
        except Exception as e:
            print(f"[{ts}] {series}: FAILED ({e})", file=sys.stderr)
            continue
        rows = [(
            ts,
            m.get("ticker"),
            m.get("event_ticker"),
            series,
            m.get("title"),
            price(m, "yes_bid_dollars", "yes_bid"),
            price(m, "yes_ask_dollars", "yes_ask"),
            price(m, "last_price_dollars", "last_price"),
            float(m["volume_fp"]) if m.get("volume_fp") is not None
            else m.get("volume"),
            float(m["open_interest_fp"]) if m.get("open_interest_fp") is not None
            else m.get("open_interest"),
            m.get("status"),
            m.get("yes_sub_title") or m.get("subtitle"),
            None if m.get("floor_strike") is None else str(m.get("floor_strike")),
        ) for m in markets]
        con.executemany(
            "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        total += len(rows)
        print(f"[{ts}] {series}: {len(rows)} markets logged")
    con.commit()
    con.close()
    return total


CANDIDATE_SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL", "KXNFLPOINTS",
                    "KXNFLTOTALPOINTS", "KXNFLMARGIN", "KXNFLPROP"]


def discover(candidates=None):
    session = requests.Session()
    candidates = candidates or CANDIDATE_SERIES
    found = []
    for series in candidates:
        counts = {}
        cursor = None
        try:
            while True:
                params = {"series_ticker": series, "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                data = get_json(session, "/markets", params)
                for m in data.get("markets", []):
                    counts[m.get("status", "?")] = counts.get(m.get("status", "?"), 0) + 1
                cursor = data.get("cursor")
                if not cursor:
                    break
        except Exception as e:
            print(f"  {series}: error ({e})")
            continue
        if counts:
            found.append(series)
            print(f"  {series}: {counts}")
        else:
            print(f"  {series}: no markets (series likely does not exist)")
    print("\nSeries with any markets:", found or "none")
    print("If open counts are zero everywhere, markets simply have not opened "
          "yet; keep the scheduled logger running and it will catch them.")
    return found


def main():
    ap = argparse.ArgumentParser(description="Kalshi NFL market price logger")
    ap.add_argument("--db", default="kalshi_prices.db")
    ap.add_argument("--series", nargs="+", default=DEFAULT_SERIES)
    ap.add_argument("--interval-min", type=float, default=10.0)
    ap.add_argument("--once", action="store_true",
                    help="take one snapshot and exit (use with Task Scheduler)")
    ap.add_argument("--discover", nargs="*", default=None, metavar="SERIES",
                    help="probe candidate series tickers (default list if none "
                         "given), report market counts by status, then exit")
    args = ap.parse_args()

    if args.discover is not None:
        discover(args.discover or None)
        return

    if args.once:
        n = snapshot(args.db, args.series)
        sys.exit(0 if n > 0 else 1)

    print(f"Logging {args.series} to {args.db} every {args.interval_min} min. "
          "Ctrl+C to stop.")
    while True:
        try:
            snapshot(args.db, args.series)
        except Exception as e:
            print(f"snapshot failed: {e}", file=sys.stderr)
        time.sleep(args.interval_min * 60)


if __name__ == "__main__":
    main()
