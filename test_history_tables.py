"""Track Record seasons: auto-inclusion, the empty guard, and the live flag.

These only show their edge cases at season boundaries, which is to say about
once a year and never on the day anyone is looking. A synthetic schedule forces
every one of them now: a season fully played, a season in progress, a season
scheduled with nothing played yet, and a frame with no results at all.
"""
import os
import sys

import numpy as np
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rundown as rd
import website as W
from models import LinearGaussianModel

# The season logic under test does not depend on which model is graded, and the
# deployed ensemble would make this suite take minutes. Swap in the linear model
# here; the default factory itself is checked separately at the bottom.
FAST = lambda: LinearGaussianModel(lam=rd.LIN_LAM)

fail = []
def ok(c, m):
    if not c:
        fail.append(m)

rng = np.random.default_rng(0)
TEAMS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]


def season(year, weeks, played_weeks):
    """Four games a week; weeks after played_weeks have no result yet."""
    rows = []
    for wk in range(1, weeks + 1):
        for g in range(4):
            home, away = TEAMS[2 * g], TEAMS[2 * g + 1]
            x = rng.normal(0, 1, len(rd.V3_COLS))
            played = wk <= played_weeks
            margin = float(np.round(3 * x[0] + rng.normal(0, 10))) if played else np.nan
            rows.append({"game_id": f"{year}_{wk:02d}_{away}_{home}",
                         "season": year, "week": wk, "home_team": home,
                         "away_team": away, "result": margin,
                         "spread_line": float(np.round(3 * x[0] * 2) / 2),
                         "home_score": 20 + max(margin, 0) if played else np.nan,
                         "away_score": 20 + max(-margin, 0) if played else np.nan,
                         **dict(zip(rd.V3_COLS, x))})
    return rows


frame = pd.DataFrame(
    season(2019, 17, 17) + season(2020, 17, 17)   # before the Track Record, training only
    + season(2025, 17, 17)                        # fully played
    + season(2026, 17, 2)                         # in progress: weeks 1-2 played
    + season(2027, 17, 0))                        # scheduled, nothing played
frame["y"] = frame["result"].astype(float)

hist, by_season, calib = W.history_tables(frame, model_factory=FAST)
print(by_season[["games", "winner_pct", "live"]].to_string())

# Auto-inclusion: nothing hardcoded, so a season after 2025 joins on its own.
ok(2026 in by_season.index, "an in-progress season after 2025 was not included")
ok(2025 in by_season.index, "a completed season was dropped")

# Seasons before the Track Record never appear, even though they have results.
ok(not ({2019, 2020} & set(by_season.index)), "a pre-2021 season was scored")

# A season with nothing played is skipped, not passed to walk_forward, which
# would hand pd.concat an empty list.
ok(2027 not in by_season.index, "a season with zero completed games was included")

# The live flag: in progress means both played and unplayed games.
ok(bool(by_season.loc[2026, "live"]), "the in-progress season is not marked live")
ok(not bool(by_season.loc[2025, "live"]), "a completed season is marked live")

# Only completed games are scored, including inside the live season's test weeks.
ok(not hist["y"].isna().any(), "an unplayed game was scored")
ok(int(by_season.loc[2026, "games"]) == 8,
   f"expected 8 scored 2026 games (2 weeks x 4), got {by_season.loc[2026, 'games']}")

# The guard proper: a frame with no results anywhere must not raise.
unplayed = frame.copy()
unplayed["result"] = np.nan
unplayed["y"] = np.nan
try:
    h0, s0, c0 = W.history_tables(unplayed, model_factory=FAST)
    ok(len(h0) == 0 and len(s0) == 0, "an all-unplayed frame should give empty tables")
    print("\nall-unplayed frame: empty tables, no exception")
except Exception as e:
    fail.append(f"an all-unplayed frame raised {type(e).__name__}: {e}")

# Explicit seasons still work, and a requested season with no results is skipped.
h1, s1, _ = W.history_tables(frame, seasons=[2025, 2027], model_factory=FAST)
ok(list(s1.index) == [2025], f"explicit seasons mishandled: {list(s1.index)}")

# The default must be the deployed model, not the linear one it used to grade.
# Checked without fitting anything, by intercepting the factory walk_forward is
# handed and substituting the fast model for the actual fit.
seen = []
real_wf = W.walk_forward
def spy(df, cols, season, **kw):
    seen.append(kw.get("model_factory"))
    return real_wf(df, cols, season, half_life_seasons=kw.get("half_life_seasons"),
                   model_factory=FAST)
W.walk_forward = spy
try:
    W.history_tables(frame, seasons=[2025])
finally:
    W.walk_forward = real_wf
ok(bool(seen) and seen[0] is rd.DeployedModel,
   f"history_tables grades {seen[0] if seen else None}, not rd.DeployedModel")

# And the deployed model must publish exactly what the page used to compute by
# hand: RECAL_SCALE times the root of aleatoric plus epistemic variance.
Xs = frame[frame["result"].notna()][rd.V3_COLS].values[:200]
ys = frame[frame["result"].notna()]["y"].values[:200]
dm = rd.DeployedModel().fit(Xs, ys)
mu_d, sig_d = dm.predict_dist(Xs[:20])
mu_s, ale, epi = dm.predict_split(Xs[:20])
ok(np.allclose(mu_d, mu_s), "predict_dist and predict_split disagree on mu")
ok(np.allclose(sig_d, rd.RECAL_SCALE * np.sqrt(ale + epi)),
   "published sigma is not RECAL_SCALE * sqrt(aleatoric + epistemic)")
ok(dm.ens.hidden == 16 and dm.ens.epochs == 200 and dm.ens.weight_decay == 1e-2,
   "DeployedModel is not using the tuned ensemble hyperparameters")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else
              "PASS: seasons join on their own, empty seasons are skipped, live is live"))
sys.exit(1 if fail else 0)
