"""The weekly rollover, which fires once a week and would fail quietly.

The page shows one NFL week. Getting that wrong is not subtle to a reader, they
see last week's finished games or next week's fixtures too early, but it is
invisible to every other test, because it only changes at a specific moment on a
specific day. So it is exercised here against a synthetic schedule instead of
being trusted.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import website as W

fail = []
def ok(c, m):
    if not c:
        fail.append(m)


def slate(rows):
    """rows: (season, week, gameday, away, home, result)."""
    df = pd.DataFrame(rows, columns=["season", "week", "gameday", "away_team",
                                     "home_team", "result"])
    df["gameday"] = pd.to_datetime(df["gameday"])
    return df


def future_of(df, today):
    """The same filter build_site applies before calling select_week."""
    t = pd.Timestamp(today)
    return df[df["result"].isna() & (df["gameday"] >= t)]


WEEK1 = [(2026, 1, "2026-09-13", "CHI", "CAR", None),
         (2026, 1, "2026-09-13", "ARI", "LAC", None),
         (2026, 1, "2026-09-14", "DEN", "KC", None)]
WEEK2 = [(2026, 2, "2026-09-17", "DET", "BUF", None),
         (2026, 2, "2026-09-20", "CAR", "ATL", None)]

# 1. Before kickoff: week 1 only, and week 2 must not leak in.
wk, note = W.select_week(future_of(slate(WEEK1 + WEEK2), "2026-09-09"))
print("before kickoff :", note.replace("<p class=\"sub\">", "").replace("</p>", ""))
ok(set(wk["week"]) == {1}, f"expected week 1 only, got {sorted(set(wk['week']))}")
ok(len(wk) == 3, f"expected 3 games, got {len(wk)}")
ok("Week 1 of the 2026 season, 3 games" in note, f"caption wrong: {note}")

# 2. Mid week, after Sunday but before Monday night: still week 1, because the
#    Monday game has not been played.
played = [(s, w, d, a, h, 7.0) if d == "2026-09-13" else (s, w, d, a, h, r)
          for s, w, d, a, h, r in WEEK1]
wk, note = W.select_week(future_of(slate(played + WEEK2), "2026-09-14"))
ok(set(wk["week"]) == {1} and len(wk) == 1,
   f"Monday should still show week 1's last game, got {len(wk)} games")

# 3. Tuesday, every week 1 game graded: rolls to week 2. This is what the new
#    Tuesday rebuild exists to catch.
allplayed = [(s, w, d, a, h, 7.0) for s, w, d, a, h, _ in WEEK1]
wk, note = W.select_week(future_of(slate(allplayed + WEEK2), "2026-09-15"))
print("tuesday        :", note.replace("<p class=\"sub\">", "").replace("</p>", ""))
ok(set(wk["week"]) == {2}, f"Tuesday should roll to week 2, got {sorted(set(wk['week']))}")

# 4. Tuesday, but nflverse has NOT posted results yet. The dates have passed, so
#    the page must still advance rather than presenting finished games as
#    upcoming. This is the failure the gameday filter quietly prevents.
wk, note = W.select_week(future_of(slate(WEEK1 + WEEK2), "2026-09-15"))
ok(set(wk["week"]) == {2},
   f"must roll over even with results unposted, got {sorted(set(wk['week']))}")
print("results unposted:", note.replace("<p class=\"sub\">", "").replace("</p>", ""))

# 5. Season boundary: an unplayed week 22 of the old season outranks week 1 of
#    the new one, since sorting is on (season, week) and not week alone.
rows = [(2026, 22, "2027-02-07", "SEA", "NE", None),
        (2027, 1, "2027-09-09", "CHI", "CAR", None)]
wk, note = W.select_week(future_of(slate(rows), "2027-02-01"))
ok(int(wk.iloc[0]["season"]) == 2026 and set(wk["week"]) == {22},
   f"season boundary mishandled: {note}")

# 6. Nothing left to play at all, the offseason, must not raise.
wk, note = W.select_week(future_of(slate(allplayed), "2026-09-15"))
ok(len(wk) == 0 and note == "", "empty future should give an empty slate")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else
              "PASS: the slate rolls over on its own, and only when it should"))
sys.exit(1 if fail else 0)
