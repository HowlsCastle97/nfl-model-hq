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

import numpy as np
import pandas as pd

from ensemble import DeepEnsemble
from features import TOTAL_COLS, load_games, load_team_game_stats, build_features
from models import LinearGaussianModel, crps_gaussian, gaussian_nll, prob_total_over
from walkforward import evaluate, season_decay_weights, walk_forward

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


def calibration(preds):
    """Spread of the standardized residuals. 1.0 means sigma is honest."""
    z = (preds["y"] - preds["mu"]) / preds["sigma"]
    return float(z.std(ddof=1))


def run(seasons=(2024, 2025), games_path="games.csv",
        stats_path="team_game_stats.csv", engines=None):
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
            preds = walk_forward(played, TOTAL_COLS, season,
                                 half_life_seasons=TOTAL_DECAY_HL,
                                 model_factory=factory, target="y_total",
                                 keep_cols=("total_line",))
            m = evaluate(preds)
            rec = ou_record(preds)
            rows.append({"season": season, "engine": name, **m,
                         "z_sd": calibration(preds),
                         "mean_mu": float(preds["mu"].mean()),
                         "mean_sigma": float(preds["sigma"].mean()),
                         "ou_n": rec.get("n"), "ou_hit": rec.get("hit"),
                         "ou_roi": rec.get("roi")})
            print(f"{season} {name}: CRPS {m['crps']:.4f} NLL {m['nll']:.4f} "
                  f"RMSE {m['rmse']:.3f} mu {preds['mu'].mean():.1f} "
                  f"sigma {preds['sigma'].mean():.2f} z_sd {calibration(preds):.3f} "
                  f"O/U {rec.get('hit', float('nan')):.3f} on {rec.get('n')}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    ap.add_argument("--out", default=None, help="write the table to CSV")
    args = ap.parse_args()
    table = run(tuple(args.seasons), args.games, args.stats)
    cols = ["season", "engine", "n", "crps", "nll", "rmse", "mae", "z_sd",
            "mean_mu", "mean_sigma", "ou_n", "ou_hit", "ou_roi"]
    print()
    print(table[[c for c in cols if c in table.columns]].to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
