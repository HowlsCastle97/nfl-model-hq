"""Drive every failure branch of healthcheck.py with synthetic data.

A watchdog that has only ever been seen passing is not a watchdog.
"""
import json, os, sqlite3, sys, tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import healthcheck as H

fail = []
def ok(c, m):
    if not c: fail.append(m)

def iso(minutes_ago=0, days_ago=0):
    t = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago, days=days_ago)
    return t.isoformat(timespec="seconds")

# ---------- price history ----------
def make_db(path, ages):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE snapshots (ts_utc TEXT, series_ticker TEXT)")
    for series, mins in ages.items():
        con.execute("INSERT INTO snapshots VALUES (?,?)", (iso(minutes_ago=mins), series))
    con.commit(); con.close()

with tempfile.TemporaryDirectory() as d:
    fresh = os.path.join(d, "fresh.db")
    make_db(fresh, {"KXNFLGAME": 5, "KXNFLSPREAD": 5, "KXNFLTOTAL": 5})
    good, detail = H.check_price_history(fresh)
    ok(good, f"fresh db should pass: {detail}")

    stale = os.path.join(d, "stale.db")
    make_db(stale, {"KXNFLGAME": 900, "KXNFLSPREAD": 900, "KXNFLTOTAL": 900})
    good, detail = H.check_price_history(stale)
    ok(not good, "a db 15 hours stale must fail")
    print("stale db  ->", detail[:80])

    # The case that actually matters: a Kalshi ticker rename drops one series
    # while the other two keep flowing and everything looks alive.
    part = os.path.join(d, "part.db")
    make_db(part, {"KXNFLGAME": 5, "KXNFLSPREAD": 5, "KXNFLTOTAL": 4000})
    good, detail = H.check_price_history(part)
    ok(not good, "one dead series must fail even when the others are fresh")
    ok("KXNFLTOTAL" in detail, "the failing series should be named")
    print("one series->", detail[:80])

    missing = os.path.join(d, "none.db")
    make_db(missing, {"KXNFLGAME": 5})
    good, detail = H.check_price_history(missing)
    ok(not good and "never been logged" in detail, "never-logged series must fail")

    good, detail = H.check_price_history(os.path.join(d, "absent.db"))
    ok(not good, "a missing database must fail")

# ---------- stub the network ----------
class Resp:
    def __init__(self, payload=None, text="", status=200):
        self._p, self.text, self.status = payload, text, status
    def raise_for_status(self):
        if self.status >= 400: raise RuntimeError(f"http {self.status}")
    def json(self): return self._p

def stub(feed=None, site=None, workflows=None, runs=None):
    def get(url, **kw):
        if "prices.json" in url: return Resp(payload=feed)
        if url.startswith(H.SITE): return Resp(text=site)
        if "actions/workflows" in url: return Resp(payload={"workflows": workflows})
        if "actions/runs" in url: return Resp(payload={"workflow_runs": runs})
        raise AssertionError(url)
    H.requests.get = get

SITE_OK = '<div class="card gcard">x</div> generated ' + \
          (datetime.now(timezone.utc)).strftime("%Y-%m-%d")
SITE_OLD = '<div class="card gcard">x</div> generated ' + \
           (datetime.now(timezone.utc) - timedelta(days=20)).strftime("%Y-%m-%d")
EVENTS = {"ARILAC": {"ARI": {"bid": .1, "ask": .2}}}

stub(feed={"ts": iso(3), "events": EVENTS})
good, detail = H.check_feed()
ok(good, f"a 3 minute feed should pass: {detail}")

stub(feed={"ts": iso(57), "events": EVENTS})
good, detail = H.check_feed()
ok(not good, "the 57 minute gap seen on 2026-09-07 must fail")
print("stale feed->", detail[:80])

stub(feed={"ts": iso(2), "events": {}})
good, detail = H.check_feed()
ok(not good, "an empty feed must fail even when it is fresh")

stub(site=SITE_OK)
good, detail = H.check_site()
ok(good, f"a fresh site should pass: {detail}")

stub(site=SITE_OLD)
good, detail = H.check_site()
ok(not good and "weekly rebuild is not running" in detail,
   "a 20 day old build must fail")
print("old build ->", detail[:80])

stub(site='<html>no cards here</html>')
good, detail = H.check_site()
ok(not good, "a site with no game cards must fail")

# ---------- actions ----------
WF_OK = [{"id": 1, "name": "Publish prices", "path": ".github/workflows/prices.yml"},
         {"id": 2, "name": "Weekly site rebuild", "path": ".github/workflows/weekly-site.yml"}]
WF_BROKEN = [{"id": 2, "name": ".github/workflows/weekly-site.yml",
              "path": ".github/workflows/weekly-site.yml"}]

stub(workflows=WF_OK, runs=[{"workflow_id": 1, "status": "completed",
                             "conclusion": "success", "created_at": iso(5)},
                            {"workflow_id": 2, "status": "completed",
                             "conclusion": "success", "created_at": iso(60)}])
good, detail = H.check_actions()
ok(good, f"all green should pass: {detail}")

stub(workflows=WF_BROKEN, runs=[])
good, detail = H.check_actions()
ok(not good and "cannot parse" in detail, "an unparseable workflow must fail")
print("unparsed  ->", detail[:80])

stub(workflows=WF_OK, runs=[{"workflow_id": 2, "status": "completed",
                             "conclusion": "failure", "created_at": iso(30)}])
good, detail = H.check_actions()
ok(not good and "FAILED" in detail, "a failed run must fail")

# The regression that made me rewrite this: a pre-fix failure carries the old
# file-path name, so grouping by name would stick forever. Grouped by id, a
# later success clears it.
stub(workflows=WF_OK,
     runs=[{"workflow_id": 2, "status": "completed", "conclusion": "success",
            "created_at": iso(5)},
           {"workflow_id": 2, "status": "completed", "conclusion": "failure",
            "created_at": iso(600)}])
good, detail = H.check_actions()
ok(good, f"a newer success must clear an older failure: {detail}")

stub(workflows=WF_OK, runs=[{"workflow_id": 1, "status": "in_progress",
                             "conclusion": None, "created_at": iso(1)}])
good, detail = H.check_actions()
ok(good, "a run in progress is not a failure")

# ---------- a crashing check must not take the run down ----------
def boom(*a, **k): raise RuntimeError("network on fire")
H.requests.get = boom
good, detail = H.check_feed()
ok(good is False, "an exploding fetch should be a failure, not a crash")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else
              "PASS: every watchdog branch fires"))
sys.exit(1 if fail else 0)
