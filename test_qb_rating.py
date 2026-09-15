"""The quarterback rating must never see the game it is rating.

A leak here would be invisible in every other test: the rating would simply look
predictive, because it would contain the answer. So the property is attacked
directly. Rewrite a game's own quarterback stats to something absurd and the
rating going into that game must not move. Rewrite an earlier game's and it must.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as F

fail = []
def ok(c, m):
    if not c:
        fail.append(m)

PRIOR = -0.076
games = pd.DataFrame([
    {"game_id": f"2025_{w:02d}_BBB_AAA", "season": 2025, "week": w,
     "gameday": pd.Timestamp("2025-09-07") + pd.Timedelta(days=7 * (w - 1)),
     "home_team": "AAA", "away_team": "BBB", "result": 3.0,
     "home_qb_id": "QB-HOME", "away_qb_id": "QB-AWAY", "roof": "outdoors",
     "home_rest": 7, "away_rest": 7, "div_game": 0}
    for w in (1, 2, 3)])


def ratings(lookup):
    out = F.build_features(games, stats_lookup=None, qb_lookup=lookup,
                           qb_half_life_dropbacks=600.0, qb_prior_n=150.0,
                           qb_prior_mean=PRIOR)
    return out.set_index("week")["qb_rating_home"]


base = {g: [("QB-HOME", 35, 0.10 * 35), ("QB-AWAY", 35, 0.0)]
        for g in games["game_id"]}
r = ratings(base)
print("home QB rating going into weeks 1-3:", r.round(4).tolist())

# 1. With no prior games, a quarterback starts at the prior, not at zero.
ok(abs(r.loc[1] - PRIOR) < 1e-12, f"week 1 should be the prior {PRIOR}, got {r.loc[1]}")

# 2. The game's own stats must not reach its own rating.
tampered = dict(base)
tampered["2025_02_BBB_AAA"] = [("QB-HOME", 35, 50.0 * 35), ("QB-AWAY", 35, 0.0)]
r2 = ratings(tampered)
ok(r2.loc[2] == r.loc[2], f"week 2's own stats leaked into week 2: {r.loc[2]} -> {r2.loc[2]}")
print("tampering week 2's stats changes week 2's rating:", r2.loc[2] != r.loc[2])

# 3. But they must reach the next game, or the rating is not learning at all.
ok(r2.loc[3] > r.loc[3] + 1.0, "week 2's stats never reached week 3")
print("...and does change week 3's:", r2.loc[3] > r.loc[3])

# 4. A game with no result is future: its stats, if any, update nothing.
future = games.copy()
future.loc[future.week == 2, "result"] = np.nan
out = F.build_features(future, qb_lookup=tampered, qb_half_life_dropbacks=600.0,
                       qb_prior_n=150.0, qb_prior_mean=PRIOR).set_index("week")
ok(out.loc[3, "qb_rating_home"] == r.loc[2],
   "an unplayed game updated the rating")

# 5. Shrinkage: one good game should move a rating only part of the way.
ok(PRIOR < r.loc[2] < 0.10, f"one game should land between prior and observed, got {r.loc[2]}")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else
              "PASS: no game informs its own rating, and every earlier game does"))
sys.exit(1 if fail else 0)
