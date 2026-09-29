"""The totals model: how many points both teams score together.

Deliberately a separate model from the margin one, and deliberately not required
to be Bayesian. The margin and moneyline side of this project is a Bayesian
predictive distribution because the question there is "how sure are we", and the
answer has to survive being turned into a price. A total is the same kind of
question but a different shape of data: the quantity is bounded below, mildly
right skewed, and piles up on key numbers (41, 44, 47, 51) because football
scores in threes and sevens. What matters is the whole predictive distribution,
so the engine is chosen on walk-forward CRPS and whatever wins, wins.

This file holds the baseline: the same deep ensemble architecture the margin
model uses, pointed at y_total with the totals feature set. It exists to be
beaten, and to be the thing a fancier engine has to beat before it ships.

Run it directly for the walk-forward table:

    python totals.py --seasons 2024 2025
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import time

import numpy as np
import pandas as pd

from ensemble import DeepEnsemble
from features import TOTAL_COLS, load_games, load_team_game_stats, build_features
from models import (LinearGaussianModel, crps_from_quantiles, crps_gaussian,
                    gaussian_nll, moments_from_quantiles, nll_from_quantiles,
                    prob_total_over, prob_total_over_quantiles)
from walkforward import evaluate, season_decay_weights, walk_forward

# Probability levels the quantile engines are asked for. Dense in the middle, and
# reaching to 1% and 99% because CRPS integrates over the whole grid and anything
# outside it is invisible to the score.
TAU_GRID = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                     0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                     0.95, 0.975, 0.99])

# Same season decay as the margin model. Not retuned yet: the point of the
# baseline is to change one thing at a time.
TOTAL_DECAY_HL = 2.0
TOTAL_LIN_LAM = 100.0


class TotalsEnsemble:
    """The margin model's architecture with the totals target.

    Same tuned hyperparameters as rundown.DeployedModel, so any difference in the
    numbers is the target and the features rather than the network. recal is a
    variance recalibration in the shape of RECAL_SCALE and defaults to 1.0,
    because whether totals need one is a question for the walk-forward table
    below, not an assumption to carry over from margins.
    """

    def __init__(self, n_members=5, hidden=16, weight_decay=1e-2, epochs=200,
                 seed=0, recal=1.0):
        self.recal = recal
        self.ens = DeepEnsemble(n_members=n_members, hidden=hidden,
                                weight_decay=weight_decay, epochs=epochs, seed=seed)

    def fit(self, X, y, sample_weight=None):
        self.ens.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_split(self, X):
        return self.ens.predict_split(X)

    def predict_dist(self, X):
        mu, aleatoric, epistemic = self.ens.predict_split(X)
        return mu, self.recal * np.sqrt(aleatoric + epistemic)


class HeteroLinear:
    """Ridge for the mean, ridge for the spread. The smallest step up from the baseline.

    Added after the boosted trees came back worse than ridge in a way that was
    monotone in capacity: every extra leaf made the score worse, which is what a
    linear signal buried in noise looks like. The next question that leaves is not
    "a bigger mean model" but "is the single fitted sigma the weak part": a windy
    December game between two slow teams is not as uncertain as a dome shootout,
    and the baseline gives both the same 13.2.

    So the mean stays a ridge, and log sigma becomes its own ridge on the same
    features, fitted to the mean model's squared residuals. Fitting log of a
    squared residual by least squares is biased low by the mean of log of a chi
    square with one degree of freedom, which is -1.2704, so that constant is added
    back rather than left to quietly shrink every interval.

    Two parameters of capacity, both heavy on purpose: lam for the mean, sigma_lam
    for the spread, the second much larger because a sigma that chases residual
    noise is worse than a constant one.
    """

    def __init__(self, lam=100.0, sigma_lam=2000.0, sigma_floor=0.5):
        self.lam = lam
        self.sigma_lam = sigma_lam
        self.sigma_floor = sigma_floor

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.mean_ = LinearGaussianModel(lam=self.lam).fit(X, y, sample_weight=w)
        r2 = (y - self.mean_.predict_mu(X)) ** 2
        # -1.2704 is E[log chi square with one degree of freedom]: taking logs of a
        # squared residual and fitting by least squares estimates log sigma squared
        # minus that constant, so it is added back here.
        target = np.log(np.maximum(r2, 1e-6)) + 1.2704
        self.spread_ = LinearGaussianModel(lam=self.sigma_lam).fit(
            X, target, sample_weight=w)
        self.base_sigma_ = self.mean_.sigma_
        return self

    def predict_dist(self, X):
        mu = self.mean_.predict_mu(X)
        log_var = self.spread_.predict_mu(X)
        sigma = np.exp(0.5 * np.clip(log_var, -10, 10))
        return mu, np.maximum(sigma, self.sigma_floor)


class QuantileGBM:
    """LightGBM quantile regression, one booster per probability level.

    A gradient boosted tree fitted on the pinball loss estimates one quantile of
    the conditional distribution. Fit a grid of them and the collection is a
    predictive distribution with no shape assumed: it can be skewed, and it can be
    narrow in one part of feature space and wide in another, which a single fitted
    sigma cannot.

    Two prices for that. Separate fits can cross, so each row is sorted, which is
    the standard remedy and costs nothing at these grid sizes. And the tails exist
    only out to the outermost level, so `moments_from_quantiles` slightly
    understates sd and CRPS ignores the last 1% at each end.

    Shallow trees and a high minimum leaf count on purpose: 272 games a season is
    not much data, and the deep ensemble already showed what excess capacity does
    to this target.
    """

    def __init__(self, taus=TAU_GRID, n_estimators=250, learning_rate=0.05,
                 num_leaves=7, min_child_samples=60, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0,
                 random_state=0):
        self.taus = np.asarray(taus, float)
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                           num_leaves=num_leaves, min_child_samples=min_child_samples,
                           feature_fraction=feature_fraction,
                           bagging_fraction=bagging_fraction,
                           bagging_freq=bagging_freq, reg_lambda=reg_lambda,
                           random_state=random_state, verbose=-1, n_jobs=-1)

    def fit(self, X, y, sample_weight=None):
        import lightgbm as lgb
        self.models_ = []
        for tau in self.taus:
            m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau),
                                  **self.params)
            m.fit(X, y, sample_weight=sample_weight)
            self.models_.append(m)
        return self

    def predict_quantiles(self, X, taus=None):
        Q = np.column_stack([m.predict(X) for m in self.models_])
        Q = np.sort(Q, axis=1)
        if taus is None or (len(taus) == len(self.taus)
                            and np.allclose(taus, self.taus)):
            return Q
        taus = np.asarray(taus, float)
        return np.vstack([np.interp(taus, self.taus, row) for row in Q])

    def predict_dist(self, X):
        return moments_from_quantiles(self.predict_quantiles(X), self.taus)


class TotalsTabPFN:
    """TabPFN as a totals engine: a transformer that infers in context.

    It is not trained on this data at all. The network was pretrained on millions
    of synthetic tabular problems and does Bayesian inference over that prior in a
    single forward pass, so "fitting" is handing it the context and prediction is
    one pass over it. That is why it belongs in this comparison: no per-week refit,
    and a predictive distribution that comes out of the model rather than out of an
    assumption.

    Two things it cannot do. It has no sample weighting, so the season decay the
    other engines use has no equivalent; `min_weight` approximates it by dropping
    context rows the harness has already weighted below that level, which is a
    recency cut rather than a taper. And the 2.5 weights and newer are gated behind
    a Prior Labs account and licence, so `version` defaults to v2, the one that
    downloads without an account.

    Two speed notes, both measured on this machine (16 CPU threads, 3,900 row
    context, 16 test games). Left on 'auto', TabPFN builds a preprocessing
    ensemble and one week costs 173 seconds. At n_estimators=1 the same week costs
    25 and the week's CRPS moves by 0.006, which is nothing, so one member is the
    default here. And mean and quantiles come back from a single forward pass:
    asking for them separately doubled the bill for no new information.
    """

    def __init__(self, version="v2", device="cpu", random_state=0, min_weight=None,
                 taus=TAU_GRID, n_estimators=1):
        self.version = version
        self.device = device
        self.random_state = random_state
        self.min_weight = min_weight
        self.n_estimators = n_estimators
        self.taus = np.asarray(taus, float)

    def fit(self, X, y, sample_weight=None):
        from tabpfn import TabPFNRegressor
        from tabpfn.model_loading import ModelVersion
        if self.min_weight is not None and sample_weight is not None:
            keep = np.asarray(sample_weight, float) >= self.min_weight
            X, y = np.asarray(X)[keep], np.asarray(y)[keep]
        self.reg_ = TabPFNRegressor.create_default_for_version(
            ModelVersion(self.version), device=self.device,
            random_state=self.random_state, ignore_pretraining_limits=True,
            n_estimators=self.n_estimators)
        self.reg_.fit(np.asarray(X, float), np.asarray(y, float))
        self._cache = None
        return self

    def _predict(self, X):
        """One forward pass, cached, giving the mean and the whole quantile grid."""
        X = np.asarray(X, float)
        key = (X.shape, float(X[0, 0]), float(X[-1, -1]))
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        out = self.reg_.predict(X, output_type="full",
                                quantiles=[float(t) for t in self.taus])
        mu = np.asarray(out["mean"]).ravel()
        Q = np.sort(np.column_stack(
            [np.asarray(v).ravel() for v in out["quantiles"]]), axis=1)
        self._cache = (key, (mu, Q))
        return mu, Q

    def predict_quantiles(self, X, taus=None):
        _, Q = self._predict(X)
        if taus is None or (len(taus) == len(self.taus)
                            and np.allclose(taus, self.taus)):
            return Q
        taus = np.asarray(taus, float)
        return np.vstack([np.interp(taus, self.taus, row) for row in Q])

    def predict_dist(self, X):
        mu, Q = self._predict(X)
        _, sd = moments_from_quantiles(Q, self.taus)
        return mu, sd


class ConstantModel:
    """Predict the training mean, with the training spread. The floor.

    Any engine that cannot beat this has learned nothing about football, and on a
    quantity as noisy as a game total that is a real possibility worth measuring
    rather than assuming away.
    """

    def fit(self, X, y, sample_weight=None):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.mu_ = float((w * y).sum() / w.sum())
        self.sigma_ = float(np.sqrt((w * (y - self.mu_) ** 2).sum() / w.sum()))
        return self

    def predict_dist(self, X):
        n = len(X)
        return np.full(n, self.mu_), np.full(n, self.sigma_)


class DeployedTotals:
    """The totals distribution the site publishes, defined exactly once.

    Ridge on TOTAL_COLS, the step 2 baseline, because everything tried against it
    lost or could not be separated from it: boosted quantile trees lost by more
    the more capacity they were given, a heteroscedastic sigma moved nothing, and
    TabPFN's in-context inference came out a fraction ahead on the tuning season
    and inside the noise on the held-out one. When the candidates all tie, the one
    that is a closed form solve and takes a second to fit wins on grounds that are
    not accuracy.

    Its sigma is honest as it stands: standardized residuals came back with a
    spread of 0.97 in 2024 and 1.01 in 2025, so unlike the margin ensemble it
    needs no variance recalibration.

    Same interface as rundown.DeployedModel, so anything that wants a published
    total goes through predict_dist here rather than building its own ridge.
    """

    def __init__(self, lam=None):
        self.lam = TOTAL_LIN_LAM if lam is None else lam
        self.lin = LinearGaussianModel(lam=self.lam)

    def fit(self, X, y, sample_weight=None):
        self.lin.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_dist(self, X):
        return self.lin.predict_dist(X)


def fit_totals(df, asof_season, cols=TOTAL_COLS):
    """Fit the deployed totals model on every completed game, decayed by season."""
    train = df[df["y_total"].notna()]
    sw = season_decay_weights(train["season"].values, asof_season, TOTAL_DECAY_HL)
    return DeployedTotals().fit(train[cols].values, train["y_total"].values,
                                sample_weight=sw)


def build_frame(games_path="games.csv", stats_path="team_game_stats.csv",
                first_season=2010, weather_override=None):
    """Feature frame for totals work, unplayed games included."""
    games = load_games(games_path, first_season=first_season, keep_unplayed=True)
    stats = load_team_game_stats(stats_path)
    return build_features(games, stats, form_half_life_games=8,
                          weather_override=weather_override)


def market_baseline(df, test_season):
    """The closing total as a predictive distribution, for scale.

    sigma is fitted on the line's own errors over every completed game before the
    season, which is the fairest reading of "how wrong is Vegas usually". The
    market is not a model this project can beat on accuracy and is not supposed
    to be; it is the number that says whether a CRPS of 9 is good.
    """
    d = df[df["y_total"].notna() & df["total_line"].notna()]
    train = d[d["season"] < test_season]
    test = d[d["season"] == test_season]
    if len(train) < 100 or len(test) == 0:
        return None
    sigma = float(np.sqrt(((train["y_total"] - train["total_line"]) ** 2).mean()))
    out = test[["game_id", "season", "week", "home_team", "away_team"]].copy()
    out["y"] = test["y_total"].values
    out["mu"] = test["total_line"].values
    out["sigma"] = sigma
    return out


def ou_record(preds, line_col="total_line"):
    """Over/under record against the market line, at standard -110.

    Pushes are thrown out rather than counted as half a win, which is how a
    sportsbook settles them. 52.4% is break even at -110, so anything in the
    forties is a losing strategy no matter how good the CRPS looks.
    """
    d = preds[preds[line_col].notna()].copy()
    if d.empty:
        return {"n": 0}
    p_over, _, _ = prob_total_over(d["mu"].values, d["sigma"].values, d[line_col].values)
    d["pick_over"] = p_over > 0.5
    d["result_over"] = d["y"] > d[line_col]
    d["push"] = d["y"] == d[line_col]
    graded = d[~d["push"]]
    hit = (graded["pick_over"] == graded["result_over"]).mean()
    return {"n": int(len(graded)), "pushes": int(d["push"].sum()),
            "hit": float(hit), "roi": float(hit * (100 / 110) - (1 - hit))}


def quantile_metrics(preds, taus=TAU_GRID):
    """Scores from the engine's own predictive shape, not from a fitted Gaussian.

    Returns nothing if the run carried no quantile columns, which is how a
    Gaussian engine falls through this cleanly.
    """
    cols = [f"q{t:g}" for t in taus]
    if any(c not in preds.columns for c in cols):
        return {}
    Q = preds[cols].values
    y = preds["y"].values
    mu, sd = moments_from_quantiles(Q, taus)
    return {"crps_q": float(crps_from_quantiles(y, Q, taus).mean()),
            "nll_q": float(nll_from_quantiles(y, Q, taus).mean()),
            "mu_q": float(mu.mean()), "sd_q": float(sd.mean())}


def ou_record_quantiles(preds, taus=TAU_GRID, line_col="total_line"):
    """The over/under record using the engine's own distribution for P(over)."""
    cols = [f"q{t:g}" for t in taus]
    d = preds[preds[line_col].notna()]
    if d.empty or any(c not in preds.columns for c in cols):
        return {}
    p_over, _, _ = prob_total_over_quantiles(d[cols].values, taus,
                                             d[line_col].values)
    pick_over = p_over > 0.5
    result_over = (d["y"] > d[line_col]).values
    push = (d["y"] == d[line_col]).values
    hit = (pick_over[~push] == result_over[~push]).mean()
    return {"ou_n_q": int((~push).sum()), "ou_hit_q": float(hit),
            "ou_roi_q": float(hit * (100 / 110) - (1 - hit))}


def calibration(preds):
    """Spread of the standardized residuals. 1.0 means sigma is honest."""
    z = (preds["y"] - preds["mu"]) / preds["sigma"]
    return float(z.std(ddof=1))


def run(seasons=(2024, 2025), games_path="games.csv",
        stats_path="team_game_stats.csv", engines=None, preds_out=None):
    df = build_frame(games_path, stats_path)
    played = df[df["y_total"].notna()]
    if engines is None:
        engines = {
            "constant": lambda: ConstantModel(),
            "linear": lambda: LinearGaussianModel(lam=TOTAL_LIN_LAM),
            "ensemble": lambda: TotalsEnsemble(),
        }
    rows = []
    for season in seasons:
        mkt = market_baseline(df, season)
        if mkt is not None:
            m = evaluate(mkt)
            rows.append({"season": season, "engine": "market line", **m,
                         "z_sd": calibration(mkt), "mean_mu": float(mkt["mu"].mean())})
        for name, factory in engines.items():
            t0 = time.perf_counter()
            preds = walk_forward(played, TOTAL_COLS, season,
                                 half_life_seasons=TOTAL_DECAY_HL,
                                 model_factory=factory, target="y_total",
                                 keep_cols=("total_line",), quantiles=TAU_GRID)
            secs = time.perf_counter() - t0
            m = evaluate(preds)
            rec = ou_record(preds)
            qm = quantile_metrics(preds)
            qr = ou_record_quantiles(preds)
            rows.append({"season": season, "engine": name, **m, **qm, **qr,
                         "z_sd": calibration(preds),
                         "mean_mu": float(preds["mu"].mean()),
                         "mean_sigma": float(preds["sigma"].mean()),
                         "ou_n": rec.get("n"), "ou_hit": rec.get("hit"),
                         "ou_roi": rec.get("roi"),
                         "secs": secs, "secs_per_week": secs / max(preds["week"].nunique(), 1)})
            extra = (f" | own shape: CRPS {qm['crps_q']:.4f} NLL {qm['nll_q']:.4f} "
                     f"O/U {qr.get('ou_hit_q', float('nan')):.3f}" if qm else "")
            print(f"{season} {name}: CRPS {m['crps']:.4f} NLL {m['nll']:.4f} "
                  f"RMSE {m['rmse']:.3f} mu {preds['mu'].mean():.1f} "
                  f"sigma {preds['sigma'].mean():.2f} z_sd {calibration(preds):.3f} "
                  f"O/U {rec.get('hit', float('nan')):.3f} on {rec.get('n')} "
                  f"[{secs:.0f}s]{extra}", flush=True)
            if preds_out is not None:
                preds["engine"] = name
                preds_out.append(preds)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    ap.add_argument("--out", default=None, help="write the table to CSV")
    args = ap.parse_args()
    table = run(tuple(args.seasons), args.games, args.stats)
    cols = ["season", "engine", "n", "crps", "crps_q", "nll", "nll_q", "rmse",
            "z_sd", "mean_mu", "mean_sigma", "sd_q", "ou_hit", "ou_hit_q",
            "ou_roi", "secs"]
    print()
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
