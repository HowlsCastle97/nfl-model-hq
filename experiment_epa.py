"""Does treating EPA's offseason carry differently actually help?

The question, asked properly. EPA state carries across the offseason at full
strength, while kalman.py reverts team ratings 0.7 toward the league mean and
inflates their variance. That asymmetry is accidental, but "accidental" is not
an argument for changing it, because the model was trained with the same carry
every year and has already learned what September EPA is worth.

Two candidate answers, and a baseline:

  shrink   pull EPA toward the league mean at each season boundary, the same
           shape of treatment the ratings already get
  games    leave EPA alone and instead hand the model epa_games, how many games
           this season back the numbers, so it can learn the discount itself.
           kalman_var is precedent: an estimate paired with its uncertainty

Two things this reports that a normal walk-forward would not.

Early season separately. The effect lives in roughly weeks 1 to 6, and a season
long average dilutes it toward nothing, so a real improvement could be reported
as no change.

A paired confidence interval, not just two numbers. The same games are scored by
both models, so the honest comparison is the per-game difference and how wide
its interval is. An NLL gap of 0.002 across 1400 games is noise, and reporting it
as an improvement is how a model ships a change that does nothing.
"""
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd

import rundown as rd
from features import load_games, load_team_game_stats, build_features
from kalman import TeamKalman
from models import gaussian_nll
from walkforward import walk_forward

EARLY_WEEKS = 6
TEST_SEASONS = (2021, 2022, 2023, 2024, 2025)


def frame(games, stats, epa_revert=1.0, form_revert=1.0):
    df = build_features(games, stats, form_half_life_games=8,
                        epa_season_revert=epa_revert,
                        form_season_revert=form_revert)
    return TeamKalman(**rd.KALMAN_PARAMS).run(df)


def run(df, cols, seasons, ensemble=False):
    """Walk forward each season, returning one row per predicted game."""
    factory = None
    if ensemble:
        from ensemble import DeepEnsemble
        factory = lambda: DeepEnsemble(n_members=5, hidden=16,
                                       weight_decay=1e-2, epochs=200, seed=0)
    out = []
    for s in seasons:
        p = walk_forward(df, cols, s, lam=rd.LIN_LAM,
                         half_life_seasons=rd.DECAY_HL, model_factory=factory)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p["nll"] = gaussian_nll(p["y"].values, p["mu"].values, p["sigma"].values)
    p["se"] = (p["y"] - p["mu"]) ** 2
    return p


def summarise(p, label):
    early = p[p["week"] <= EARLY_WEEKS]
    return {
        "variant": label,
        "n": len(p),
        "nll": p["nll"].mean(),
        "rmse": np.sqrt(p["se"].mean()),
        "n_early": len(early),
        "nll_early": early["nll"].mean(),
        "rmse_early": np.sqrt(early["se"].mean()),
    }


def paired(base, other, col="nll", weeks=None, boots=4000, seed=0):
    """Mean per-game difference (other minus base) with a bootstrap interval.

    Negative is better for both nll and squared error. The interval is what
    decides whether a difference is worth believing.
    """
    a = base.set_index("game_id")
    b = other.set_index("game_id")
    idx = a.index.intersection(b.index)
    if weeks is not None:
        idx = idx[a.loc[idx, "week"] <= weeks]
    d = (b.loc[idx, col] - a.loc[idx, col]).values
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(boots, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), len(d)


def main():
    ap = argparse.ArgumentParser(description="EPA offseason-carry experiments")
    ap.add_argument("--ensemble", action="store_true",
                    help="use the deployed deep ensemble instead of the fast "
                         "linear proxy (slow; the linear run is for screening)")
    ap.add_argument("--seasons", type=int, nargs="+", default=list(TEST_SEASONS))
    args = ap.parse_args()

    games = load_games("games.csv", first_season=2010, keep_unplayed=True)
    stats = load_team_game_stats("team_game_stats.csv")
    base_cols = list(rd.V3_COLS)

    variants = [
        ("baseline", dict(), base_cols),
        ("shrink 0.7", dict(epa_revert=0.7), base_cols),
        ("shrink 0.5", dict(epa_revert=0.5), base_cols),
        ("shrink 0.0 (full reset)", dict(epa_revert=0.0), base_cols),
        ("epa_games feature", dict(), base_cols + ["epa_games"]),
        ("shrink 0.7 + epa_games", dict(epa_revert=0.7), base_cols + ["epa_games"]),
        ("form shrink 0.7 too", dict(epa_revert=0.7, form_revert=0.7), base_cols),
    ]

    model = "deep ensemble" if args.ensemble else "linear (screening)"
    print(f"model: {model}   test seasons: {args.seasons}   "
          f"early = weeks 1-{EARLY_WEEKS}\n")

    preds, rows = {}, []
    for label, kw, cols in variants:
        df = frame(games, stats, **kw)
        p = run(df, cols, args.seasons, ensemble=args.ensemble)
        preds[label] = p
        rows.append(summarise(p, label))
        print(f"  ran {label}")

    table = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\npaired against baseline, negative is better")
    print(f"{'variant':26} {'full season NLL':>26} {'early season NLL':>26}")
    b = preds["baseline"]
    for label, _, _ in variants[1:]:
        m, lo, hi, n = paired(b, preds[label])
        me, loe, hie, ne = paired(b, preds[label], weeks=EARLY_WEEKS)
        # Excludes zero when both ends share a sign. Written the other way round
        # first, which marked every straddling interval as significant.
        sig = "*" if lo * hi > 0 else " "
        sige = "*" if loe * hie > 0 else " "
        print(f"{label:26} {m:+.4f} [{lo:+.4f},{hi:+.4f}]{sig} "
              f"{me:+.4f} [{loe:+.4f},{hie:+.4f}]{sige}")
    print("\n* means the interval excludes zero. Anything else is noise, "
          "however good the point estimate looks.")
    print(f"baseline for reference: NLL {table.iloc[0]['nll']:.4f}, "
          f"RMSE {table.iloc[0]['rmse']:.4f} "
          f"(the pinned 2025 numbers are 3.980 / 12.94)")


if __name__ == "__main__":
    main()
