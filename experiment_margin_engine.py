"""Pre-registered: should the deployed margin model stay a deep ensemble?

Written 2026-09-29, before the 2026 season finished, precisely so the rule cannot
be chosen after the answer is known. Run it when 2026 is complete.

WHY THIS EXISTS

Chasing a reader's complaint about one card (a NE at BUF pick'em against a BUF -7
line) turned into a comparison nobody had run: the deployed deep ensemble against
the plain ridge model it replaced, walk-forward over every season the Track Record
covers. The ensemble lost, slightly, nearly everywhere.

    season   n    ridge NLL   ensemble NLL   difference
    2021    272     4.0614       4.0678       +0.0063
    2022    271     3.8926       3.8977       +0.0051
    2023    272     4.0218       4.0456       +0.0238
    2024    272     3.9769       3.9718       -0.0051
    2025    272     3.9799       3.9971       +0.0172
    2026     48     4.0782       4.0680       -0.0102

    pooled NLL  ensemble minus ridge  +0.0088 [-0.0009, +0.0188]  n=1407
    pooled squared error                +1.897 [-0.685, +4.492]
    against the closing spread   ridge 48.9%   ensemble 48.8%   (n=1371)
    straight up                  ridge 63.9%   ensemble 63.8%   (n=1403)

Worse in four seasons of six, indistinguishable where it counts, and the interval
touches zero. That is not grounds to swap, and it is not grounds to keep paying
for either: the ensemble costs minutes of build time per run, carries its own
recalibration constant, and its one distinguishing feature, a per-game sigma, is
exactly what NLL rewards and NLL does not reward it.

Note what is NOT evidence here. Those six seasons include the ones the ensemble's
hyperparameters were chosen on, so they flatter it if anything, and they have been
looked at now, which spends them. 2026 is the only season untouched by this
question. Hence this file.

THE RULE, FIXED NOW

Population: every completed 2026 regular season game the walk-forward can grade,
weeks 1 onward, both engines refit weekly on the same features, same season decay.
Primary measure: mean per game NLL, paired bootstrap, 4000 resamples.
Secondary, reported but not decisive: RMSE, ATS hit rate, straight up hit rate.

    SWAP to ridge if the 2026 interval on (ensemble minus ridge) NLL lies entirely
    above zero, meaning the ensemble is detectably worse on a season that never
    informed it.

    KEEP the ensemble if the interval lies entirely below zero.

    If the interval contains zero, which is the likely outcome given 272 games and
    an effect this small, the tie is broken on cost and not on accuracy: SWAP,
    because a closed form solve that runs in under a second and needs no variance
    recalibration is the better engineering when the accuracy is the same. This is
    the same rule already applied to the totals engine on 2026-09-29, where ridge
    shipped over TabPFN on exactly these grounds. Applying it to totals and not to
    margins would be choosing by taste.

WHAT A SWAP WOULD COST, ACKNOWLEDGED IN ADVANCE SO IT IS NOT A SURPRISE

    1. Every card's plus-or-minus becomes the same number. The ensemble gives a
       per-game sigma (12 here, 16 there) and readers have asked about it twice.
       Ridge gives one fitted sigma for every game. The honest reading of the NLL
       result is that the varying number was not carrying information, but it is
       still a visible loss and the Bayesian 101 copy would need rewriting.
    2. The Bayesian 101 tab is built around the room of five networks, which is
       the subject of the project this grew out of. That is a real reason to keep
       it that has nothing to do with accuracy, and if it is the deciding reason
       then the site should say so plainly rather than implying the ensemble wins
       on merit.
    3. Track Record numbers all change, because the tab grades the deployed model.

Run: python experiment_margin_engine.py --season 2026
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd

import rundown as rd
from models import LinearGaussianModel, gaussian_nll
from walkforward import walk_forward

BOOTS = 4000


def run(season, games_path="games.csv", stats_path="team_game_stats.csv"):
    df = rd.build_frame(games_path, stats_path)
    done = df[df["result"].notna()]
    if not (done["season"] == season).any():
        print(f"{season} has no completed games yet; nothing to decide.")
        return None

    preds = {}
    for name, factory in (("ridge", lambda: LinearGaussianModel(lam=rd.LIN_LAM)),
                          ("ensemble", rd.DeployedModel)):
        p = walk_forward(done, rd.V3_COLS, season, half_life_seasons=rd.DECAY_HL,
                         model_factory=factory)
        p["nll"] = gaussian_nll(p["y"].values, p["mu"].values, p["sigma"].values)
        p["sq"] = (p["y"] - p["mu"]) ** 2
        preds[name] = p.set_index("game_id")
        print(f"{name}: {len(p)} games, NLL {p['nll'].mean():.4f}, "
              f"RMSE {np.sqrt(p['sq'].mean()):.3f}")

    idx = preds["ridge"].index.intersection(preds["ensemble"].index)
    d = (preds["ensemble"].loc[idx, "nll"] - preds["ridge"].loc[idx, "nll"]).values
    rng = np.random.default_rng(0)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTS)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\nensemble minus ridge, NLL: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"n={len(idx)}")

    line = df[["game_id", "spread_line"]].set_index("game_id")
    for name, p in preds.items():
        j = p.join(line)
        j = j[j["spread_line"].notna() & (j["y"] != j["spread_line"])]
        ats = ((j["mu"] > j["spread_line"]) == (j["y"] > j["spread_line"])).mean()
        su = p[p["y"] != 0]
        su = ((su["mu"] > 0) == (su["y"] > 0)).mean()
        print(f"  {name:9} ATS {100*ats:.1f}%  straight up {100*su:.1f}%")

    if lo > 0:
        verdict = ("SWAP to ridge: the ensemble is detectably worse on a season "
                   "that never informed it.")
    elif hi < 0:
        verdict = ("KEEP the ensemble: it is detectably better on a season that "
                   "never informed it.")
    else:
        verdict = ("SWAP to ridge on the tie-break fixed in advance: equal "
                   "accuracy, and ridge is a closed form solve with no variance "
                   "recalibration and no minutes of build time. See the costs "
                   "listed in this file's docstring before acting.")
    print(f"\nPRE-REGISTERED VERDICT: {verdict}")
    return {"n": len(idx), "diff": float(d.mean()), "lo": float(lo),
            "hi": float(hi), "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    args = ap.parse_args()
    run(args.season, args.games, args.stats)


if __name__ == "__main__":
    main()
