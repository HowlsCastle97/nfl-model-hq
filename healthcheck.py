"""Watchdog for the layers of this system that fail silently.

Three separate failures on 2026-09-07 shared one shape: every one of them looked
fine from the outside. The price logger had been dead for eleven days because
nothing was scheduling it. weekly-site.yml had never run at all, because GitHub
could not parse it and reported that only as a run named after the file path.
The prices workflow was being throttled to roughly every thirty minutes while
the page cheerfully said "live". All three were found by accident.

None of them would have survived an hour if anything had been watching. This is
that thing.

It checks delivered data, not configuration. That distinction is the whole
point: an enabled job that is not delivering is exactly the case that keeps
getting missed, and "the workflow exists" was true in all three failures above.

Exit code is the number of failing checks, so Task Scheduler records a nonzero
LastTaskResult and the watchdog becomes visible if it starts failing itself.
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import requests

REPO = "HowlsCastle97/nfl-model-hq"
SITE = "https://howlscastle97.github.io/nfl-model-hq/"
FEED = f"https://raw.githubusercontent.com/{REPO}/prices/prices.json"
API = f"https://api.github.com/repos/{REPO}"
TASK = "Kalshi NFL price logger"
SERIES = ("KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL")

# The logger runs every 10 minutes. 25 leaves room for one missed run plus a
# slow Kalshi response before anyone is bothered.
HISTORY_MAX_MIN = 25
FEED_MAX_MIN = 25
# The site rebuilds weekly, on Wednesdays. Nine days is one missed rebuild.
SITE_MAX_DAYS = 9
# Once something is known broken, nagging every 30 minutes teaches you to
# dismiss the alert unread. Re-raise twice a day instead.
REALERT_HOURS = 12

HERE = os.path.dirname(os.path.abspath(__file__))


def now():
    return datetime.now(timezone.utc)


def age_min(ts):
    return (now() - ts).total_seconds() / 60.0


def parse_ts(s):
    """Tolerate both the trailing Z and the +00:00 offset."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def check_price_history(db_path):
    """The irreplaceable half. Checked per series, because a Kalshi ticker
    rename would silently drop one series while the other two look healthy."""
    if not os.path.exists(db_path):
        return False, f"no database at {db_path}"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = dict(con.execute(
            "SELECT series_ticker, MAX(ts_utc) FROM snapshots GROUP BY 1"))
        con.close()
    except Exception as e:
        return False, f"cannot read the database: {e}"
    problems, ages = [], []
    for s in SERIES:
        if s not in rows:
            problems.append(f"{s} has never been logged")
            continue
        a = age_min(parse_ts(rows[s]))
        ages.append(f"{s} {a:.0f}m")
        if a > HISTORY_MAX_MIN:
            problems.append(f"{s} last logged {a:.0f} min ago")
    if problems:
        return False, "; ".join(problems)
    return True, "newest snapshot per series: " + ", ".join(ages)


def check_feed():
    """What the published page actually reads."""
    try:
        r = requests.get(FEED, timeout=20, headers={"Cache-Control": "no-cache"})
        r.raise_for_status()
        d = r.json()
    except Exception as e:
        return False, f"feed unreachable: {e}"
    if not d.get("events"):
        return False, "feed published with no events"
    a = age_min(parse_ts(d["ts"]))
    n = sum(len(v) for v in d["events"].values())
    if a > FEED_MAX_MIN:
        return False, (f"feed is {a:.0f} min old; the local publisher is not "
                       f"running and the Actions backstop has not covered it")
    return True, f"{a:.0f} min old, {len(d['events'])} events, {n} markets"


def check_site():
    """The page itself, and how long ago the model behind it was refit."""
    try:
        r = requests.get(SITE, timeout=30)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        return False, f"site unreachable: {e}"
    if "gcard" not in html:
        return False, "site responded but has no game cards"
    m = re.search(r"generated (\d{4}-\d{2}-\d{2})", html)
    if not m:
        return False, "site has no build stamp"
    built = datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = (now() - built).days
    if days > SITE_MAX_DAYS:
        return False, (f"built {m.group(1)}, {days} days ago: the weekly rebuild "
                       f"is not running")
    return True, f"built {m.group(1)}, {days} days ago"


def check_actions():
    """Two different failures, deliberately told apart.

    A workflow GitHub cannot parse is listed under its file path instead of its
    name: that is how weekly-site.yml hid, and it is answered by the workflows
    endpoint, which reflects the file as it stands right now.

    A workflow that parses but whose last run failed is answered by the runs
    endpoint, grouped by workflow id rather than by name. Grouping by name would
    strand a pre-fix failure forever, since the run carries the old file-path
    name and nothing will ever supersede it under that key.
    """
    try:
        wf = requests.get(f"{API}/actions/workflows", timeout=20)
        wf.raise_for_status()
        workflows = {w["id"]: w for w in wf.json().get("workflows", [])
                     if w.get("path", "").startswith(".github/")}
        rn = requests.get(f"{API}/actions/runs", params={"per_page": 40}, timeout=20)
        rn.raise_for_status()
        runs = rn.json().get("workflow_runs", [])
    except Exception as e:
        return None, f"cannot reach the GitHub API: {e}"
    if not workflows:
        return False, "no workflows found"

    bad, good = [], []
    for w in workflows.values():
        if w["name"] == w["path"]:
            bad.append(f"{w['path']} is listed under its file path, so GitHub "
                       f"cannot parse it and it will never run")
    latest = {}
    for run in runs:
        latest.setdefault(run.get("workflow_id"), run)
    for wid, w in sorted(workflows.items(), key=lambda kv: kv[1]["name"]):
        run = latest.get(wid)
        name = w["name"]
        if run is None:
            good.append(f"{name} no recent run")
        elif run.get("status") == "in_progress":
            good.append(f"{name} running")
        elif run.get("conclusion") == "failure":
            bad.append(f"{name} last run FAILED at {run['created_at']}")
        else:
            good.append(f"{name} {run.get('conclusion')}")
    if bad:
        return False, "; ".join(bad)
    return True, ", ".join(good)


def check_task():
    """Windows only, and advisory: the data checks above are the real signal."""
    if os.name != "nt":
        return None, "not Windows, skipped"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-ScheduledTaskInfo -TaskName '{TASK}').LastTaskResult"],
            capture_output=True, text=True, timeout=60)
    except Exception as e:
        return None, f"could not query the task: {e}"
    val = out.stdout.strip()
    if not val:
        return False, f'scheduled task "{TASK}" is not registered'
    if val != "0":
        return False, f"last run exited {val}; see logs/kalshi_logger.log"
    return True, "last run exited 0"


def notify(lines):
    """Raise a dialog the desktop cannot swallow.

    A toast was the first choice and it was the wrong one. The WinRT call
    returned success and nothing appeared, because a toast needs a registered
    AppUserModelID and can be suppressed by Focus Assist or per-app notification
    settings, all of it invisible to the caller. An alerting channel that fails
    silently is worse than none: it is the failure mode this file exists to
    prevent, reproduced inside the thing meant to prevent it.

    A message box is crude, but it is a window. It appears or it does not, and
    it cannot be turned off behind your back. It is launched detached so the
    check exits immediately rather than blocking until someone clicks OK, which
    would otherwise be killed by the task's execution time limit.
    """
    body = "\n".join(f"- {ln}" for ln in lines)[:900]
    script = (
        "Add-Type -AssemblyName System.Windows.Forms\n"
        "[System.Windows.Forms.MessageBox]::Show(@'\n"
        f"{body}\n"
        "\nRun `python healthcheck.py` for the full report.\n"
        "'@, 'NFL Model HQ: health check failed', 0, 48) | Out-Null\n")
    try:
        tmp = os.path.join(os.environ.get("TEMP", HERE), "nflhq_alert.ps1")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(script)
        subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                          "-WindowStyle", "Hidden", "-File", tmp])
    except Exception as e:
        print(f"could not raise an alert: {e}", file=sys.stderr)


def load_prev(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def should_alert(prev, failing):
    """Alert on the transition into failure, then at most once every 12 hours.

    Every 30 minutes forever would train you to dismiss it without reading,
    which is the same as having no alert at all.
    """
    if not failing:
        return False
    if not prev.get("failing"):
        return True
    last = prev.get("last_alert")
    if not last:
        return True
    try:
        return age_min(parse_ts(last)) > REALERT_HOURS * 60
    except Exception:
        return True


def main():
    ap = argparse.ArgumentParser(description="Check every layer that fails quietly")
    ap.add_argument("--db", default=os.path.join(HERE, "kalshi_prices.db"))
    ap.add_argument("--json", dest="json_out",
                    default=os.path.join(HERE, "logs", "health.json"),
                    help="where the result and the alert state are kept")
    ap.add_argument("--notify", action="store_true",
                    help="raise a Windows toast if any check fails")
    ap.add_argument("--quiet", action="store_true",
                    help="print only when something is wrong")
    args = ap.parse_args()

    checks = [
        ("price history", lambda: check_price_history(args.db)),
        ("price feed", check_feed),
        ("published site", check_site),
        ("github actions", check_actions),
        ("scheduled task", check_task),
    ]
    results, failed = [], []
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:                      # a check must never crash the run
            ok, detail = False, f"check itself errored: {e}"
        results.append({"check": name, "ok": ok, "detail": detail})
        if ok is False:
            failed.append(f"{name}: {detail}")

    if failed or not args.quiet:
        stamp = now().isoformat(timespec="seconds")
        print(f"health check {stamp}")
        for r in results:
            mark = "ok  " if r["ok"] else "SKIP" if r["ok"] is None else "FAIL"
            print(f"  [{mark}] {r['check']}: {r['detail']}")
        print(f"  {len(failed)} failing" if failed else "  all clear")

    prev = load_prev(args.json_out)
    alerting = args.notify and should_alert(prev, len(failed))
    if alerting:
        notify(failed)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"ts": now().isoformat(timespec="seconds"),
                       "failing": len(failed), "checks": results,
                       "last_alert": (now().isoformat(timespec="seconds")
                                      if alerting else prev.get("last_alert"))},
                      f, indent=1)
    return len(failed)


if __name__ == "__main__":
    sys.exit(main())
