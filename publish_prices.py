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
import os
import re
import subprocess
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


def publish_to_branch(path, ts, branch="prices", remote="origin", repo=None):
    """Force-push one file to a single-commit orphan branch, via plumbing only.

    The GitHub workflow publishes by checking out an orphan branch, which is
    fine on a throwaway runner. It would be unacceptable here: this runs on the
    development machine every 10 minutes, and a checkout that raced a real edit
    could lose work. So the blob, tree and commit are built directly and only
    the remote ref moves. HEAD, the index and the working tree are never read
    or written, and the branch stays exactly one commit deep because
    commit-tree is given no parent.

    Both publishers force-push, so whichever ran last simply wins; there is no
    conflict to resolve and no history to accumulate.
    """
    repo = repo or os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ,
               # In a Task Scheduler service context there is no console, so a
               # credential prompt would hang until the task timed out. Fail
               # fast and loudly instead.
               GIT_TERMINAL_PROMPT="0",
               GCM_INTERACTIVE="never")

    def git(*args, stdin=None):
        # stdin is passed as bytes on purpose. In text mode Python translates
        # "\n" to os.linesep on write, which on Windows put a carriage return
        # inside the mktree entry and published the file as "prices.json\r".
        # The branch looked fine and the raw URL 404'd.
        r = subprocess.run(("git",) + args, cwd=repo, input=stdin, env=env,
                           capture_output=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: "
                               f"{r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode(errors="replace").strip()

    blob = git("hash-object", "-w", os.path.abspath(path))
    # mktree wants: <mode> SP <type> SP <sha> TAB <name> LF, and that LF must
    # survive as a bare LF, hence the explicit encode.
    tree = git("mktree", stdin=f"100644 blob {blob}\tprices.json\n".encode())
    commit = git("commit-tree", tree, "-m", f"prices {ts}")
    git("push", "--force", remote, f"{commit}:refs/heads/{branch}")
    return commit


def main():
    ap = argparse.ArgumentParser(description="Publish current Kalshi NFL prices as JSON")
    ap.add_argument("--out", default="prices.json")
    ap.add_argument("--series", default=SERIES)
    ap.add_argument("--push", action="store_true",
                    help="force-push the result to the single-commit prices "
                         "branch (used by the local scheduled task; the GitHub "
                         "workflow does its own equivalent push)")
    ap.add_argument("--branch", default="prices")
    ap.add_argument("--remote", default="origin")
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
    if not n:
        # An empty feed would blank every market bar on the page. Better to
        # leave the last good prices standing than to publish nothing.
        print("refusing to publish an empty feed", file=sys.stderr)
        return 1
    if args.push:
        try:
            commit = publish_to_branch(args.out, payload["ts"],
                                       branch=args.branch, remote=args.remote)
            print(f"published {commit[:9]} to {args.remote}/{args.branch}")
        except Exception as e:
            print(f"publish failed: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
