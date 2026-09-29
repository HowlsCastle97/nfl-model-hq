"""The recommendation record: does the Track Record grade bets the way a book settles them.

Five hand built games, each chosen for one settlement rule: a won moneyline at plus
money, a lost one, a tie that voids, a spread push that voids, and a game where the
price is too rich for either side so nothing is recommended at all. Every total
below was worked out by hand before the code ran.
"""
import numpy as np
import pandas as pd

from website import SPREAD_JUICE, rec_records, rec_rows

fail = []


def ok(cond, msg):
    if not cond:
        fail.append(msg)


# mu, sigma, spread_line (positive means the home team is favoured), final margin,
# home and away moneylines, and the model's own home win probability.
games = [
    # Model loves the home dog and the home dog wins: +150 pays 1.5.
    ("g1", 10.0, 13.0, 3.0, 7.0, 150, -180, 0.70),
    # Model takes the home side on price, the home side loses by 4. The spread
    # pick that game is the away team getting 3, which does cover.
    ("g2", -2.0, 13.0, 3.0, -4.0, 170, -200, 0.40),
    # Both prices above the model's numbers: no moneyline pick. And a 53% cover
    # is below the 58% the card requires, so no spread pick either.
    ("g3", 1.0, 13.0, 0.0, 1.0, -200, 120, 0.60),
    # Final margin lands exactly on the line: the spread pick pushes, while the
    # moneyline still wins.
    ("g4", 10.0, 13.0, 3.0, 3.0, 150, -180, 0.70),
    # A tie voids the moneyline.
    ("g5", 0.5, 13.0, 0.0, 0.0, 110, -130, 0.55),
]
hist = pd.DataFrame([{
    "game_id": g, "season": 2025, "week": 1, "home_team": "AAA",
    "away_team": "BBB", "mu": mu, "sigma": sg, "spread_line": sl, "y": y,
    "p_home": ph} for g, mu, sg, sl, y, _, _, ph in games])
df = pd.DataFrame([{"game_id": g, "home_moneyline": h, "away_moneyline": a}
                   for g, _, _, _, _, h, a, _ in games])

by_week, by_season = rec_records(hist, df)
r = by_week.iloc[0]

ok(int(r.games) == 5, f"graded {r.games} games, expected 5")

# Every ML pick: the model's own favourite in all five games, at that side's price,
# value or not. g1 and g4 favour the home dog at +150 and win; g2 favours away at
# -200 and the away side does win by 4; g3 favours home at -200 and home wins by 1;
# g5 is a tie and voids. So 4-0 on four settled bets.
ok(int(r.all_n) == 4, f"every-pick settled {r.all_n} bets, expected 4")
ok(int(r.all_w) == 4 and int(r.all_l) == 0,
   f"every-pick record {r.all_w}-{r.all_l}, expected 4-0")
ok(int(r.all_void) == 1, f"{r.all_void} voids, expected 1 for the tie")
# 1.5 + 1.5 from the two plus money winners, 0.5 from -200, 0.5 from -200.
ok(abs(r.all_roi - 100 * 4.0 / 4) < 1e-9,
   f"every-pick ROI {r.all_roi:.3f}%, expected {100.0:.3f}%")
# The two are different populations on purpose: the value filter drops picks.
ok(int(r.all_n) > int(r.ml_n),
   "every ML pick should be a superset of the value picks")
ok(int(r.ml_n) == 3, f"moneyline settled {r.ml_n} bets, expected 3 (one void)")
ok(int(r.ml_w) == 2 and int(r.ml_l) == 1, f"moneyline record {r.ml_w}-{r.ml_l}, expected 2-1")
ok(int(r.ml_void) == 1, f"{r.ml_void} voids, expected 1 for the tie")
# +150 twice is 3.0 profit, one loss is -1.0, so 2.0 on 3 units staked.
ok(abs(r.ml_roi - 100 * 2.0 / 3) < 1e-9,
   f"moneyline ROI {r.ml_roi:.3f}%, expected {100 * 2 / 3:.3f}%")

# Unfiltered, every game with a line is a bet. Two land exactly on the number and
# void: g4 at 3 on a 3, and g5 at 0 on a pick'em. The other three all cover.
ok(int(r.asp_n) == 3, f"every-spread settled {r.asp_n} bets, expected 3")
ok(int(r.asp_w) == 3 and int(r.asp_l) == 0,
   f"every-spread record {r.asp_w}-{r.asp_l}, expected 3-0")
ok(int(r.asp_push) == 2, f"{r.asp_push} pushes, expected 2")
ok(int(r.asp_n) > int(r.sp_n),
   "every spread pick should be a superset of the ones the card printed")

ok(int(r.sp_n) == 2, f"spread settled {r.sp_n} bets, expected 2")
ok(int(r.sp_w) == 2 and int(r.sp_l) == 0, f"spread record {r.sp_w}-{r.sp_l}, expected 2-0")
ok(int(r.sp_push) == 1, f"{r.sp_push} pushes, expected 1")
ok(abs(r.sp_roi - 100 * 2 * (SPREAD_JUICE - 1) / 2) < 1e-9,
   f"spread ROI {r.sp_roi:.3f}%, expected {100 * (SPREAD_JUICE - 1):.3f}%")
print(f"moneyline {int(r.ml_w)}-{int(r.ml_l)} ({int(r.ml_void)} void) "
      f"ROI {r.ml_roi:+.1f}%   spread {int(r.sp_w)}-{int(r.sp_l)} "
      f"({int(r.sp_push)} push) ROI {r.sp_roi:+.1f}%")

# The all seasons row is the same arithmetic over everything, not an average of
# averages, which would weight a thin week like a full season.
allrow = by_season[by_season["season"] == "All seasons"].iloc[0]
ok(abs(float(allrow.ml_roi) - float(r.ml_roi)) < 1e-9,
   "the all seasons row disagrees with the only week there is")
ok(len(by_season) == 2, f"by_season has {len(by_season)} rows, expected season plus total")

# A period where nothing was recommended reads as a dash, not as 0-0 and 0%.
none_hist = hist[hist.game_id == "g3"]
nw, _ = rec_records(none_hist, df)
html = rec_rows(nw, "week", lambda w: f"Week {int(w)}")
ok(html.count("&mdash;") == 4, f"a week with no picks rendered {html}")
ok("0-0" not in html, "a week with no picks claimed a 0-0 record")

# Sign drives the colour, so a losing week cannot print green.
good = rec_rows(by_week, "week", lambda w: f"Week {int(w)}")
ok('class="hit"' in good and 'class="miss"' not in good,
   f"a winning week was not coloured as one: {good}")
losing = by_week.copy()
losing.loc[:, ["all_roi", "ml_roi", "asp_roi", "sp_roi"]] = -12.5
bad = rec_rows(losing, "week", lambda w: f"Week {int(w)}")
ok('class="miss"' in bad and 'class="hit"' not in bad,
   f"a losing week was not coloured as one: {bad}")
ok("-12.5%" in bad, "the loss lost its sign")

# An empty history returns empty frames rather than raising, the way the rest of
# the page already behaves in the preseason.
ew, es = rec_records(hist.iloc[:0], df)
ok(ew.empty and es.empty, "an empty history did not come back empty")
ok("Nothing graded yet" in rec_rows(ew, "week"), "no fallback row for an empty table")
print("empty history, no pick weeks, and colour by sign all behave")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail
              else "PASS: recommendations settle the way a book would settle them"))
raise SystemExit(1 if fail else 0)
