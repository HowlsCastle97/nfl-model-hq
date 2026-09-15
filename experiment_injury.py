"""Does a weighted injury feature improve the model? Pre-registered, honest split.

Hypothesis, stated before any result: adding injury_points_diff, the points of
starters each side is expected to be missing, improves one-step predictive
log-likelihood on the deployed model.

Motivation is measured rather than assumed. Against the deployed ensemble's own
2021-2025 walk-forward misses, each non-QB regular starter listed Out or Doubtful
moved his team -0.50 points against prediction (95% CI -1.04 to +0.03), steadily
across 0, 1, 2 and 3+ starters out. That was a flat count; this weights it.

Derivation, every fitted quantity confined to data before 2024:

  absence     P(misses | designation), from how often players with each final
              designation actually took a snap, seasons 2012-2023
  importance  average snap share over the player's last eight games strictly
              before the report week (built in prep_injuries.py)
  missing_g   per team and week, sum of absence times importance within each
              position group: OL, REC, RB, DL, LB, DB
  weights     points per full starter missing, by group, from an ordinary least
              squares fit of the deployed ensemble's walk-forward residuals on
              home-minus-away missing_g, seasons 2021-2023 only
  feature     injury_points_diff = sum over groups of weight times difference

Protocol: the selection seasons are reported but are optimistic by construction,
since the weights were fitted on them. The decision is made on 2024-2025, which
never touched a weight or a rate: ship only if the paired interval on
rd.DeployedModel excludes zero in the right direction. 2026 stays untouched.

Out of scope for this test, deliberately: quarterbacks, which the rating covers
and which did not ship; and live timing, since the final report lands on Friday
and the current rebuilds cannot see it. Seasons before 2012 have no snap counts,
so the feature is zero there; by 2024 those seasons carry about 1% of training
weight under the two-season decay.
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

import rundown as rd
from experiment_qb import line, paired, score
from features import build_features, load_games, load_team_game_stats
from kalman import TeamKalman
from models import LinearGaussianModel

GROUPS = ["OL", "REC", "RB", "DL", "LB", "DB"]
FIT = (2021, 2022, 2023)
HELDOUT = (2024, 2025)


def main():
    games = load_games("games.csv", first_season=2010, keep_unplayed=True)
    stats = load_team_game_stats("team_game_stats.csv")
    df = TeamKalman(**rd.KALMAN_PARAMS).run(
        build_features(games, stats, form_half_life_games=8))

    inj = pd.read_csv("injury_player_weeks.csv")
    pre = inj[inj.season <= 2023]
    absence = (1 - pre.groupby("report_status").played.mean()).to_dict()
    print("absence rate by designation, 2012-2023:",
          {k: round(v, 3) for k, v in sorted(absence.items())})

    inj["missing"] = inj.report_status.map(absence) * inj.importance
    per = (inj.groupby(["season", "week", "team", "group"]).missing.sum()
           .unstack("group").reindex(columns=GROUPS).fillna(0.0).reset_index())

    def side(team_col, suffix):
        return per.rename(columns={"team": team_col,
                                   **{g: f"{g}_{suffix}" for g in GROUPS}})
    df = (df.merge(side("home_team", "h"), on=["season", "week", "home_team"], how="left")
            .merge(side("away_team", "a"), on=["season", "week", "away_team"], how="left"))
    for g in GROUPS:
        df[f"{g}_h"] = df[f"{g}_h"].fillna(0.0)
        df[f"{g}_a"] = df[f"{g}_a"].fillna(0.0)
        df[f"d_{g}"] = df[f"{g}_h"] - df[f"{g}_a"]
    dcols = [f"d_{g}" for g in GROUPS]

    print("\nfitting weights on the deployed ensemble's 2021-2023 walk-forward residuals")
    base_cols = list(rd.V3_COLS)
    res = score(df, base_cols, FIT, rd.DeployedModel)
    fit = df.set_index("game_id").loc[res.index]
    X, r = fit[dcols].values, (res["y"] - res["mu"]).values
    w = np.linalg.lstsq(X, r, rcond=None)[0]
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(r), len(r))
        boots.append(np.linalg.lstsq(X[i], r[i], rcond=None)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    print(f"  {len(r)} games.  points per full starter missing (home minus away):")
    for g, wi, l, h in zip(GROUPS, w, lo, hi):
        print(f"    {g:4} {wi:+.2f}  [{l:+.2f}, {h:+.2f}]")

    df["injury_points_diff"] = df[dcols].values @ w
    inj_cols = base_cols + ["injury_points_diff"]
    lin = lambda: LinearGaussianModel(lam=rd.LIN_LAM)

    print("\nselection 2021-2023, linear  (optimistic: the weights were fitted here)")
    b, c = score(df, base_cols, FIT, lin), score(df, inj_cols, FIT, lin)
    print(line("full season", *paired(b, c)))

    for name, factory in (("linear", lin), ("DEPLOYED ENSEMBLE", rd.DeployedModel)):
        print(f"\nheld out 2024-2025, {name}  (injury minus shipped, negative is better)")
        b, c = score(df, base_cols, HELDOUT, factory), score(df, inj_cols, HELDOUT, factory)
        print(f"  shipped NLL {b['nll'].mean():.4f} RMSE {np.sqrt(b['se'].mean()):.4f}   "
              f"injury NLL {c['nll'].mean():.4f} RMSE {np.sqrt(c['se'].mean()):.4f}")
        print(line("full season", *paired(b, c)))
        print(line("weeks 1-6", *paired(b, c, 6)))


if __name__ == "__main__":
    main()
