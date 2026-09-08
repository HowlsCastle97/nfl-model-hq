"""Fast layer: one Kalshi snapshot to a small JSON the published page fetches.

Deliberately tiny. No torch, no pandas, no model, no database: this runs every
few minutes on a GitHub runner, so it imports nothing but the standard library
plus requests, and finishes in a couple of seconds.

The slow layer (website.py) bakes the model's probabilities into the page at
build time. This layer publishes only what actually moves minute to minute, the
market side, and the page recombines the two in the browser. Emitted shape:

    {"ts": "2026-09-07T23:05:00+00:00",
     "events": {"SEASF": {"SEA": {"bid": .41, "ask": .43},
                          "SF":  {"bid": .57, "ask": .59}}}}

Event keys are the Kalshi event ticker's team half, matching the keys
rundown.latest_prices builds from the logged database, so the page can look up
a game with the same alias candidates the build used.
"""
import argparse
import json
import re
import sys
from datetime import datetime, timezone

import requests

BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = "KXNFLGAME"
# KXNFLGAME-25DEC28SEASF -> ("25DEC28", "SEASF"); same pattern as rundown.py.
TICKER_RE = re.compile(r"^KXNFLGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)$")


def price(market, dollars_key, cents_key):
    """Kalshi returns dollar strings on some fields and integer cents on others."""
    v = market.get(dollars_key)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    v = market.get(cents_key)
    return None if v is None else float(v) / 100.0


def fetch_markets(session, series=SERIES, status="open", retries=3):
    markets, cursor = [], None
    while True:
        params = {"series_ticker": series, "limit": 200}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        for attempt in range(retries):
            r = session.get(BASE + "/markets", params=params, timeout=30)
            if r.status_code == 429:
                continue
            r.raise_for_status()
            break
        data = r.json()
        markets.extend(data.get("markets", []))
        cursor = data.get("cursor")
        if not cursor:
            return markets


def build_payload(markets):
    events = {}
    for m in markets:
        key = TICKER_RE.match(m.get("event_ticker") or "")
        if not key:
            continue
        team = (m.get("ticker") or "").rsplit("-", 1)[-1]
        if not team:
            continue
        bid = price(m, "yes_bid_dollars", "yes_bid")
        ask = price(m, "yes_ask_dollars", "yes_ask")
        if bid is None and ask is None:
            continue
        events.setdefault(key.group(2), {})[team] = {"bid": bid, "ask": ask}
    return {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "events": events}


def main():
    ap = argparse.ArgumentParser(description="Publish current Kalshi NFL prices as JSON")
    ap.add_argument("--out", default="prices.json")
    ap.add_argument("--series", default=SERIES)
    args = ap.parse_args()
    try:
        markets = fetch_markets(requests.Session(), args.series)
    except Exception as e:
        # Never leave a stale file looking fresh, and never fail the whole
        # workflow: an outage should just mean the page keeps its baked prices.
        print(f"kalshi fetch failed: {e}", file=sys.stderr)
        return 1
    payload = build_payload(markets)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"), sort_keys=True)
    n = sum(len(v) for v in payload["events"].values())
    print(f"{args.out}: {len(payload['events'])} events, {n} markets, ts {payload['ts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
