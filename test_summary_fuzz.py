"""Fuzz the generated prose, and check its factual claims against real data.

Every bug that reached the published page in this project's first week was in
generated copy, not in the model: a plural that disagreed with its own number, a
present-tense sentence about a season that had not started, and a claim that a
team was starting a backup when it was not. None of them were caught by the
handwritten tests, because those assert the cases I thought of, and prose has
more states than I can think of: sign of every contribution, which theme wins,
whether a quarterback is listed, how far into a season it is.

So this does two things a handwritten test cannot.

It fuzzes. Thousands of random contribution vectors across every stage of a
season, with and without quarterback names, asserting properties that must hold
for any of them.

And it checks the claims against reality. The prose asserts things about the
world, like "the pass defence has been the leakier one", which are only true if
def_epa_pass really is EPA allowed. That is verified against actual scorelines
rather than against my reading of a column name.
"""
import html as H
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rundown as rd
import website as W

COLS = rd.V3_COLS
fail = []
def ok(c, m):
    if not c:
        fail.append(m)


# ---------------------------------------------------------------- fuzz the copy
rng = np.random.default_rng(0)
seen_themes = set()
TRIALS = 4000
for _ in range(TRIALS):
    con = rng.normal(0, 1.2, len(COLS))
    con[rng.random(len(COLS)) < 0.4] = 0.0          # many inputs do nothing
    val = rng.normal(0, 1.0, len(COLS))
    played = int(rng.choice([0, 1, 3, 6, 12]))
    named = bool(rng.random() < 0.7)
    hqb, aqb = ("Joe Burrow", "Baker Mayfield") if named else ("", "")
    mu = float(rng.normal(0, 6))
    html = W.plain_summary(val, con, "CIN", "TB", mu, hqb, aqb, played,
                           fam_h=float(rng.random()), fam_a=float(rng.random()))
    t = H.unescape(re.sub(r"<[^>]+>", "", html))
    ctx = t[:130]

    ok("  " not in t, f"double space: {ctx}")
    ok(" ," not in t and " ." not in t, f"space before punctuation: {ctx}")
    ok(".." not in t, f"double period: {ctx}")
    ok(t.strip().endswith("."), f"no final period: {ctx}")
    ok(": ." not in t and ":." not in t, f"empty clause: {ctx}")
    ok("None" not in t, f"a None leaked into the prose: {ctx}")
    ok("-0.0" not in t, f"negative zero: {ctx}")
    ok(not re.search(r"\b0\.0 points?\b", t), f"claims a zero-point effect: {ctx}")
    ok(not re.search(r"\bCIN CIN\b|\bTB TB\b", t), f"team repeated: {ctx}")

    # Plural must follow the number printed beside it.
    for num, word in re.findall(r"about (\d+\.\d) (points?)", t):
        want = "point" if num == "1.0" else "points"
        ok(word == want, f"'{num} {word}' should be '{num} {want}': {ctx}")

    # Tense must follow the evidence, in both directions.
    if played == 0:
        ok("Nothing has been played yet" in t, f"week 1 lacks its disclaimer: {ctx}")
        ok("have been" not in t, f"present tense in week 1: {ctx}")
    else:
        ok("Nothing has been played yet" not in t,
           f"week 1 disclaimer mid season: {ctx}")

    # The claim that cost the most trust: qb_fam_diff is a share of recent
    # starts, so it can never be rendered as somebody being benched.
    ok("usual man under centre" not in t, f"implies a benching: {ctx}")
    ok(" do not" not in t, f"asserts a team lacks its starter: {ctx}")

    for key, phrase in (("ground", "On the ground"), ("air", "Through the air"),
                        ("class", "the better side"), ("qb", "has seen more of"),
                        ("spot", "Situationally"), ("form", "outscor")):
        if phrase in t:
            seen_themes.add(key)

ok(len(seen_themes) == 6,
   f"fuzzing never exercised {set(['ground','air','class','qb','spot','form']) - seen_themes}")
print(f"fuzzed {TRIALS} summaries; all six themes exercised: "
      f"{sorted(seen_themes)}")


# ------------------------------------------------- check the claims against data
# The prose says "the pass defence has been the leakier one" off the sign of
# def_epa_pass. That is a statement about the world, and it is only true if a
# higher def_epa_pass really does mean more points conceded.
here = os.path.dirname(os.path.abspath(__file__))
stats = pd.read_csv(os.path.join(here, "team_game_stats.csv"))
games = pd.read_csv(os.path.join(here, "games.csv"),
                    usecols=["game_id", "home_team", "away_team",
                             "home_score", "away_score"]).dropna()
rows = []
for r in games.itertuples():
    rows.append((r.game_id, r.home_team, r.away_score, r.home_score))
    rows.append((r.game_id, r.away_team, r.home_score, r.away_score))
pts = pd.DataFrame(rows, columns=["game_id", "team", "allowed", "scored"])
m = stats.merge(pts, on=["game_id", "team"], how="inner")
ok(len(m) > 5000, f"only {len(m)} team-games matched, cannot verify signs")

for col, target, claim in (
        ("def_epa_pass", "allowed", "higher def_epa_pass means a leakier pass defence"),
        ("def_epa_rush", "allowed", "higher def_epa_rush means a leakier run defence"),
        ("off_epa_pass", "scored", "higher off_epa_pass means a better pass offence"),
        ("off_epa_rush", "scored", "higher off_epa_rush means a better run offence"),
        ("cpoe", "scored", "higher cpoe goes with scoring more")):
    r = float(np.corrcoef(m[col], m[target])[0, 1])
    print(f"  corr({col}, {target}) = {r:+.3f}")
    ok(r > 0.15, f"the page's claim is unsupported: {claim} (corr {r:+.3f})")

print("\n" + ("FAIL:\n - " + "\n - ".join(dict.fromkeys(fail)) if fail else
              "PASS: the prose holds up under fuzzing and its claims match the data"))
sys.exit(1 if fail else 0)
