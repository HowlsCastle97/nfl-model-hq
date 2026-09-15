"""Does a per-quarterback rating improve the model? Pre-registered, honest split.

Hypothesis, stated before any results were seen: adding qb_rating_diff to the
twelve deployed inputs improves one-step predictive log-likelihood.

The team features already carry quarterback play indirectly, through team EPA
and CPOE, and qb_fam_diff says whether the listed starter is the usual one. What
nothing says is how good that starter is. A backup and a star with the same
share of recent starts look identical to the current model, and team EPA takes
weeks to notice a starter change. The rating follows the player, so it should
know on the day he takes over.

Protocol, the same one the EPA offseason experiment used and for the same
reason:

  selection   2021-2023: choose the rating's half-life and prior strength on a
              small grid, linear model, since it is the screen
  held out    2024-2025: never consulted until the choice is fixed. Report the
              paired per-game difference against the shipped model with a
              bootstrap interval, full season and weeks 1-6 separately
  ensemble    the chosen settings again on 2024-2025 with rd.DeployedModel, the
              model the site actually publishes
  2026        untouched. It is the only season that did not exist while the
              model was built and it is not spent on a tuning decision

The prior mean is not searched. It is estimated from data, 2012-2023 only: the
EPA per dropback of quarterbacks over their first 150 career dropbacks, which is
what a quarterback with no history actually produces.

Ships only if the held-out interval excludes zero in the right direction on the
deployed model. A monotone pattern or a nice point estimate is not enough; the
EPA experiment showed how a pooled pattern can reverse on an honest split.
"""
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

import rundown as rd
from features import (build_features, load_games, load_qb_game_stats,
                      load_team_game_stats)
from kalman import TeamKalman
from models import LinearGaussianModel, gaussian_nll
from walkforward import walk_forward

SELECT = (2021, 2022, 2023)
HELDOUT = (2024, 2025)
EARLY_WEEKS = 6
HALF_LIVES = (300.0, 600.0, 1200.0)
PRIOR_NS = (150.0, 300.0, 600.0)


def prior_mean(qb_path, first=2012, last=2023, window=150):
    """Empirical-Bayes prior: EPA per dropback over a debut window."""
    q = pd.read_csv(qb_path)
    q["season"] = q["game_id"].str[:4].astype(int)
    q = q.sort_values("game_id")
    debut = q.groupby("passer_id")["season"].transform("min")
    q = q[(debut >= first) & (q["season"] <= last)].copy()
    q["before"] = q.groupby("passer_id")["dropbacks"].cumsum() - q["dropbacks"]
    early = q[q["before"] < window]
    return float(early["qb_epa"].sum() / early["dropbacks"].sum())


def frame(games, stats, qb=None, **kw):
    df = build_features(games, stats, form_half_life_games=8, qb_lookup=qb, **kw)
    return TeamKalman(**rd.KALMAN_PARAMS).run(df)


def score(df, cols, seasons, factory):
    done = df[df["result"].notna()]
    out = []
    for s in seasons:
        out.append(walk_forward(done, cols, s, half_life_seasons=rd.DECAY_HL,
                                model_factory=factory))
    p = pd.concat(out, ignore_index=True)
    p["nll"] = gaussian_nll(p["y"].values, p["mu"].values, p["sigma"].values)
    p["se"] = (p["y"] - p["mu"]) ** 2
    return p.set_index("game_id")


def paired(base, other, weeks=None, boots=6000, seed=0):
    idx = base.index.intersection(other.index)
    if weeks is not None:
        idx = idx[base.loc[idx, "week"] <= weeks]
    d = (other.loc[idx, "nll"] - base.loc[idx, "nll"]).values
    means = np.random.default_rng(seed).choice(d, size=(boots, len(d))).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(d)


def line(label, m, lo, hi, n):
    verdict = ("better" if hi < 0 else "worse" if lo > 0 else "no detectable difference")
    return f"  {label:12} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={n:4}  {verdict}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--skip-ensemble", action="store_true",
                    help="stop after the linear held-out result")
    args = ap.parse_args()

    games = load_games("games.csv", first_season=2010, keep_unplayed=True)
    stats = load_team_game_stats("team_game_stats.csv")
    qb = load_qb_game_stats("qb_game_stats.csv")
    mu0 = prior_mean("qb_game_stats.csv")
    lin = lambda: LinearGaussianModel(lam=rd.LIN_LAM)
    base_cols = list(rd.V3_COLS)
    qb_cols = base_cols + ["qb_rating_diff"]
    print(f"prior mean (empirical, 2012-2023 debuts): {mu0:+.4f} EPA per dropback\n")

    base_df = frame(games, stats)
    print("selection, 2021-2023, linear")
    base_sel = score(base_df, base_cols, SELECT, lin)
    print(f"  shipped model                     NLL {base_sel['nll'].mean():.4f}")
    grid = []
    for hl in HALF_LIVES:
        for n0 in PRIOR_NS:
            df = frame(games, stats, qb, qb_half_life_dropbacks=hl,
                       qb_prior_n=n0, qb_prior_mean=mu0)
            nll = score(df, qb_cols, SELECT, lin)["nll"].mean()
            grid.append((nll, hl, n0))
            print(f"  + rating  half-life {hl:5.0f}  prior {n0:4.0f}   NLL {nll:.4f}")
    _, hl, n0 = min(grid)
    print(f"\nchosen before looking at 2024-2025: half-life {hl:.0f} dropbacks, "
          f"prior {n0:.0f} dropbacks")

    qb_df = frame(games, stats, qb, qb_half_life_dropbacks=hl,
                  qb_prior_n=n0, qb_prior_mean=mu0)
    print("\nheld out, 2024-2025, linear  (rating minus shipped, negative is better)")
    b, c = score(base_df, base_cols, HELDOUT, lin), score(qb_df, qb_cols, HELDOUT, lin)
    print(f"  shipped NLL {b['nll'].mean():.4f} RMSE {np.sqrt(b['se'].mean()):.4f}   "
          f"rating NLL {c['nll'].mean():.4f} RMSE {np.sqrt(c['se'].mean()):.4f}")
    print(line("full season", *paired(b, c)))
    print(line("weeks 1-6", *paired(b, c, EARLY_WEEKS)))

    if args.skip_ensemble:
        return
    print("\nheld out, 2024-2025, deployed ensemble")
    b, c = (score(base_df, base_cols, HELDOUT, rd.DeployedModel),
            score(qb_df, qb_cols, HELDOUT, rd.DeployedModel))
    print(f"  shipped NLL {b['nll'].mean():.4f} RMSE {np.sqrt(b['se'].mean()):.4f}   "
          f"rating NLL {c['nll'].mean():.4f} RMSE {np.sqrt(c['se'].mean()):.4f}")
    print(line("full season", *paired(b, c)))
    print(line("weeks 1-6", *paired(b, c, EARLY_WEEKS)))


if __name__ == "__main__":
    main()
