"""Which engine should price totals: the pre-registered comparison.

The question. The baseline in `totals.py` established that the totals features
carry real information (ridge beats a constant by -0.230 CRPS, interval excluding
zero) and that the deep ensemble that wins on margins does not transfer to this
target. So: can a modern tabular method beat ridge on walk-forward CRPS, and can
anything approach the closing line?

The split, fixed before any number was read. Hyperparameters are chosen on 2024
only. 2025 is read once, at the end, for the winner and the baselines. Nothing is
tuned on 2025 and nothing here touches 2026, which stays the only untouched
season for whatever comes next. Note what that costs: after this runs, 2025 has
been spent on the totals question the way 2024-2025 was spent on the injury
question, so a future totals engine needs 2026 or later to be tested honestly.

The decision rule, also fixed in advance. An engine ships only if its held-out
CRPS beats ridge with a paired bootstrap interval that excludes zero. Anything
else stays in the harness as a measured candidate and the site keeps the simpler
model. CRPS rather than NLL is the yardstick because one 70 point game can
dominate a log score, and because CRPS reads in points.

Candidates.
  ridge        the step 2 baseline, a Gaussian with one fitted sigma.
  hetero       ridge mean, ridge log sigma. Added to the tuning stage after the
               trees lost in a way that was monotone in capacity, which pointed at
               the constant sigma rather than at the mean model. Added late and
               said so; the held-out rule below did not move to accommodate it.
  gbm          LightGBM quantile regression on a 23 level grid, four
               hyperparameter settings tried on 2024, none of them large: the
               ensemble already showed what capacity does to this target.
  tabpfn       TabPFN, a transformer that does in-context Bayesian inference over
               a synthetic tabular prior and is not trained on this data at all.
               Version 2 here, because 2.5 and newer are gated behind a Prior
               Labs account and licence; `--tabpfn-version v2.5` runs it once that
               login exists.
  market       the closing Vegas total, as the scale.

Run: python experiment_totals.py --stage tune, then --stage test.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time

import numpy as np
import pandas as pd

import totals as T
from features import TOTAL_COLS
from models import LinearGaussianModel, crps_from_quantiles, crps_gaussian
from walkforward import evaluate, walk_forward

TUNE_SEASON = 2024
TEST_SEASON = 2025
BOOTS = 4000

# Four settings, all small. The grid is deliberately short: every extra candidate
# tried on 2024 is another chance to pick noise, and the selection is not free.
# How hard the spread model is allowed to move. Higher is closer to one constant
# sigma, which is the thing being tested against.
HETERO_GRID = {
    "hetero-firm": dict(sigma_lam=4000.0),
    "hetero-mid": dict(sigma_lam=2000.0),
    "hetero-loose": dict(sigma_lam=500.0),
}

GBM_GRID = {
    "gbm-base": dict(num_leaves=7, min_child_samples=60, n_estimators=250,
                     learning_rate=0.05),
    "gbm-tiny": dict(num_leaves=3, min_child_samples=100, n_estimators=400,
                     learning_rate=0.03),
    "gbm-wide": dict(num_leaves=15, min_child_samples=30, n_estimators=250,
                     learning_rate=0.05),
    "gbm-stub": dict(num_leaves=3, min_child_samples=150, n_estimators=150,
                     learning_rate=0.05),
}


def crps_series(preds):
    """Per game CRPS from whichever distribution the engine actually produced."""
    cols = [f"q{t:g}" for t in T.TAU_GRID]
    if all(c in preds.columns for c in cols):
        return pd.Series(
            crps_from_quantiles(preds["y"].values, preds[cols].values, T.TAU_GRID),
            index=preds["game_id"].values)
    return pd.Series(
        crps_gaussian(preds["y"].values, preds["mu"].values, preds["sigma"].values),
        index=preds["game_id"].values)


def paired(a, b, name_a, name_b, rng):
    """Mean per game CRPS difference a minus b, paired bootstrap interval."""
    m = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (m["a"] - m["b"]).values
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOTS)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    verdict = "REAL" if lo * hi > 0 else "not detectable"
    print(f"  {name_a} minus {name_b}: {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"n={len(d)}  {verdict}")
    return {"comparison": f"{name_a} minus {name_b}", "diff": float(d.mean()),
            "lo": float(lo), "hi": float(hi), "n": int(len(d))}


def score(name, factory, played, season, quantiles=True):
    t0 = time.perf_counter()
    preds = walk_forward(played, TOTAL_COLS, season,
                         half_life_seasons=T.TOTAL_DECAY_HL, model_factory=factory,
                         target="y_total", keep_cols=("total_line",),
                         quantiles=T.TAU_GRID if quantiles else None)
    secs = time.perf_counter() - t0
    g = evaluate(preds)
    q = T.quantile_metrics(preds)
    own = q.get("crps_q", g["crps"])
    rec = T.ou_record_quantiles(preds) or T.ou_record(preds)
    hit = rec.get("ou_hit_q", rec.get("hit", float("nan")))
    print(f"  {name:12} CRPS {own:.4f} (gaussian {g['crps']:.4f})  "
          f"NLL {q.get('nll_q', g['nll']):.4f}  RMSE {g['rmse']:.3f}  "
          f"mu {preds['mu'].mean():.1f}  sd {q.get('sd_q', preds['sigma'].mean()):.2f}  "
          f"O/U {hit:.3f}  [{secs:.0f}s]", flush=True)
    return preds, {"engine": name, "season": season, "crps_own": own,
                   "crps_gaussian": g["crps"], "nll": q.get("nll_q", g["nll"]),
                   "rmse": g["rmse"], "mu": float(preds["mu"].mean()),
                   "sd": q.get("sd_q", float(preds["sigma"].mean())),
                   "ou_hit": hit, "secs": secs,
                   "secs_per_week": secs / max(preds["week"].nunique(), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=["tune", "test"], default="tune")
    ap.add_argument("--tabpfn-version", default="v2",
                    help="v2 downloads without an account; v2.5 needs the licence")
    ap.add_argument("--skip-tabpfn", action="store_true",
                    help="TabPFN is minutes per week on CPU; leave it out")
    ap.add_argument("--engines", default=None,
                    help="comma separated subset to run; ridge is always included "
                         "because it is what everything is measured against")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = T.build_frame()
    played = df[df["y_total"].notna()]
    rng = np.random.default_rng(0)
    rows, crps = [], {}
    pick = None if args.engines is None else set(args.engines.split(","))

    def wanted(name):
        return pick is None or name in pick

    if args.stage == "tune":
        print(f"TUNING on {TUNE_SEASON}. Nothing below is a held-out number.")
        p, r = score("ridge", lambda: LinearGaussianModel(lam=T.TOTAL_LIN_LAM),
                     played, TUNE_SEASON, quantiles=False)
        crps["ridge"], rows_ = crps_series(p), rows.append(r)
        for name, kw in HETERO_GRID.items():
            if not wanted(name):
                continue
            p, r = score(name, lambda kw=kw: T.HeteroLinear(**kw), played,
                         TUNE_SEASON, quantiles=False)
            crps[name] = crps_series(p)
            rows.append(r)
        for name, kw in GBM_GRID.items():
            if not wanted(name):
                continue
            p, r = score(name, lambda kw=kw: T.QuantileGBM(**kw), played, TUNE_SEASON)
            crps[name] = crps_series(p)
            rows.append(r)
        if not args.skip_tabpfn:
            for label, kw in (("tabpfn-all", {}), ("tabpfn-recent", {"min_weight": 0.25})):
                if not wanted(label):
                    continue
                p, r = score(label, lambda kw=kw: T.TotalsTabPFN(
                    version=args.tabpfn_version, **kw), played, TUNE_SEASON)
                crps[label] = crps_series(p)
                rows.append(r)
        print("\nagainst the ridge baseline, on the tuning season:")
        for name in crps:
            if name != "ridge":
                paired(crps[name], crps["ridge"], name, "ridge", rng)
        print("\nPick one winner from the table above, then run --stage test.")
    else:
        print(f"HELD OUT {TEST_SEASON}. Read once.")
        mkt = T.market_baseline(df, TEST_SEASON)
        print(f"  {'market':12} CRPS {evaluate(mkt)['crps']:.4f}")
        crps["market"] = pd.Series(
            crps_gaussian(mkt["y"].values, mkt["mu"].values, mkt["sigma"].values),
            index=mkt["game_id"].values)
        rows.append({"engine": "market", "season": TEST_SEASON,
                     "crps_own": evaluate(mkt)["crps"], "nll": evaluate(mkt)["nll"]})
        p, r = score("ridge", lambda: LinearGaussianModel(lam=T.TOTAL_LIN_LAM),
                     played, TEST_SEASON, quantiles=False)
        crps["ridge"], _ = crps_series(p), rows.append(r)
        for name, kw in HETERO_GRID.items():
            if not wanted(name):
                continue
            p, r = score(name, lambda kw=kw: T.HeteroLinear(**kw), played,
                         TEST_SEASON, quantiles=False)
            crps[name] = crps_series(p)
            rows.append(r)
        for name, kw in GBM_GRID.items():
            if not wanted(name):
                continue
            p, r = score(name, lambda kw=kw: T.QuantileGBM(**kw), played, TEST_SEASON)
            crps[name] = crps_series(p)
            rows.append(r)
        if not args.skip_tabpfn:
            p, r = score("tabpfn", lambda: T.TotalsTabPFN(version=args.tabpfn_version),
                         played, TEST_SEASON)
            crps["tabpfn"] = crps_series(p)
            rows.append(r)
        print("\npaired bootstrap on the held-out season:")
        for name in crps:
            if name != "ridge":
                rows.append({"engine": name, "season": TEST_SEASON,
                             **paired(crps[name], crps["ridge"], name, "ridge", rng)})
        paired(crps["ridge"], crps["market"], "ridge", "market", rng)

    table = pd.DataFrame(rows)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
