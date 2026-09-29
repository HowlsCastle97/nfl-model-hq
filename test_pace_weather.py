"""Pace features and the weather columns.

Two things worth a test. Pace is the first feature block added since the leakage
rules were written down, so it gets the same treatment the quarterback rating got:
tamper with one game and prove the feature for that game does not move. Weather is
the first input that is sometimes unknown at prediction time, so each of its three
cases is checked, including that a game day override actually reaches the model.
"""
from collections import namedtuple

import numpy as np
import pandas as pd

from features import (NEUTRAL_TEMP, NEUTRAL_WIND, PACE_PRIOR, TOTAL_COLS,
                      build_features, weather_columns)

fail = []


def ok(cond, msg):
    if not cond:
        fail.append(msg)


Row = namedtuple("Row", ["off_epa_pass", "off_epa_rush", "cpoe", "def_epa_pass",
                         "def_epa_rush", "off_plays", "off_drives", "off_points",
                         "def_plays", "def_drives", "def_points"])


def frame(weeks=4, temp=50.0, wind=10.0, roof="outdoors"):
    rows = []
    for wk in range(1, weeks + 1):
        rows.append({
            "game_id": f"2024_{wk:02d}_BBB_AAA", "season": 2024, "week": wk,
            "gameday": pd.Timestamp(f"2024-09-{wk:02d}"),
            "home_team": "AAA", "away_team": "BBB",
            "home_score": 24, "away_score": 17, "result": 7.0,
            "home_rest": 7, "away_rest": 7, "div_game": 0, "roof": roof,
            "temp": temp, "wind": wind, "home_qb_id": "H", "away_qb_id": "A",
        })
    return pd.DataFrame(rows)


def stats(plays_by_week):
    out = {}
    for wk, plays in plays_by_week.items():
        gid = f"2024_{wk:02d}_BBB_AAA"
        for team in ("AAA", "BBB"):
            out[(gid, team)] = Row(0.1, 0.0, 1.0, 0.1, 0.0,
                                   plays, 11.0, 21.0, plays, 11.0, 21.0)
    return out


base = {1: 60.0, 2: 60.0, 3: 60.0, 4: 60.0}
df = build_features(frame(), stats(base))

# Week 1 sits on the prior, because nothing has been played.
w1 = df[df.week == 1].iloc[0]
ok(abs(w1.plays_sum - 2 * PACE_PRIOR["off_plays"]) < 1e-9,
   f"week 1 plays_sum {w1.plays_sum} is not two priors")
ok(w1.drives_sum > 0, "drives_sum started at zero, which would divide by zero")
ok(abs(w1.off_ppd_sum - 2 * PACE_PRIOR["off_points"] / PACE_PRIOR["off_drives"]) < 1e-9,
   f"week 1 points per drive {w1.off_ppd_sum} is not the prior ratio")

# Tamper with week 2 only. Week 2's own features must be untouched, and week 3's
# must move, which is the whole content of "pre-game values only".
tampered = dict(base)
tampered[2] = 95.0
dt = build_features(frame(), stats(tampered))
same2 = abs(df[df.week == 2].iloc[0].plays_sum - dt[dt.week == 2].iloc[0].plays_sum)
diff3 = abs(df[df.week == 3].iloc[0].plays_sum - dt[dt.week == 3].iloc[0].plays_sum)
ok(same2 < 1e-12, f"week 2's own pace feature moved by {same2}: leakage")
ok(diff3 > 1.0, f"week 3 did not see week 2's pace, moved {diff3}")
print(f"pace: week 2 unchanged ({same2:.1e}), week 3 moved {diff3:.2f} plays")

# Every totals column is finite on every row, including the first.
bad = [c for c in TOTAL_COLS if not np.isfinite(df[c]).all()]
ok(not bad, f"non finite totals features: {bad}")

# A stats file from before pace existed leaves the prior in place instead of
# raising, since the margin path must keep working through the upgrade.
Old = namedtuple("Old", ["off_epa_pass", "off_epa_rush", "cpoe",
                         "def_epa_pass", "def_epa_rush"])
old_lookup = {k: Old(0.1, 0.0, 1.0, 0.1, 0.0) for k in stats(base)}
try:
    do = build_features(frame(), old_lookup)
    ok(abs(do[do.week == 4].iloc[0].plays_sum - 2 * PACE_PRIOR["off_plays"]) < 1e-9,
       "an old stats file moved the pace features somehow")
    ok(abs(do[do.week == 4].iloc[0].off_pass_diff
           - df[df.week == 4].iloc[0].off_pass_diff) < 1e-12,
       "an old stats file changed the margin features")
except Exception as e:
    fail.append(f"a stats file without pace columns raised {e!r}")
print("a pre-pace stats file still builds, with the pace prior untouched")

# Weather, case by case.
ind = build_features(frame(roof="dome"), stats(base))
ok((ind["game_temp"] == NEUTRAL_TEMP).all() and (ind["game_wind"] == NEUTRAL_WIND).all(),
   "a domed game did not get neutral weather")
ok((ind["indoor"] == 1).all(), "the indoor flag did not follow the roof")

out = build_features(frame(temp=28.0, wind=19.0), stats(base))
ok((out["game_temp"] == 28.0).all() and (out["game_wind"] == 19.0).all(),
   "an outdoor reading was not used as given")

# Unknown weather, which is every game that has not kicked off. September's own
# median is 60 degrees in this frame, so an unknown September game gets 60.
mixed = frame(temp=60.0, wind=8.0)
mixed.loc[mixed.index[-1], ["temp", "wind"]] = np.nan
mixed.loc[mixed.index[-1], ["result", "home_score", "away_score"]] = np.nan
unk = build_features(mixed, stats(base))
last = unk.iloc[-1]
ok(abs(last.game_temp - 60.0) < 1e-9, f"unknown temp fell back to {last.game_temp}")
ok(abs(last.game_wind - 8.0) < 1e-9, f"unknown wind fell back to {last.game_wind}")
ok(pd.isna(last.y_total), "an unplayed game was handed a total")

over = build_features(mixed, stats(base),
                      weather_override={mixed.iloc[-1].game_id: {"temp": 12, "wind": 26}})
ok(abs(over.iloc[-1].game_temp - 12) < 1e-9, "the override did not reach temp")
ok(abs(over.iloc[-1].game_wind - 26) < 1e-9, "the override did not reach wind")
ok(abs(over.iloc[0].game_temp - 60.0) < 1e-9, "the override touched another game")
print("weather: dome neutral, reading used, unknown falls back to the month, "
      "override wins")

# weather_columns on a frame with no weather columns at all must not raise.
bare = pd.DataFrame({"game_id": ["g"], "roof": ["outdoors"],
                     "gameday": [pd.Timestamp("2024-12-01")], "result": [3.0]})
t, w = weather_columns(bare)
ok(np.isfinite(t).all() and np.isfinite(w).all(),
   "a frame without temp or wind produced non finite weather")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail
              else "PASS: pace is leakage safe and weather has no missing case"))
raise SystemExit(1 if fail else 0)
