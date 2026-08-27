import numpy as np
import pandas as pd
from models import LinearGaussianModel, gaussian_nll


def season_decay_weights(train_seasons, asof_season, half_life_seasons):
    age = asof_season - np.asarray(train_seasons, dtype=float)
    if np.isinf(half_life_seasons):
        return np.ones_like(age)
    return 0.5 ** (age / half_life_seasons)


def walk_forward(df, feature_cols, test_season, lam=0.0, half_life_seasons=np.inf,
                 model_factory=None):
    """Refit weekly on all games strictly before each test week; predict that week.

    model_factory() must return an object with fit(X, y, sample_weight) and
    predict_dist(X). Defaults to LinearGaussianModel(lam). Swap in the MLP
    wrapper for Deliverable III.
    """
    if model_factory is None:
        model_factory = lambda: LinearGaussianModel(lam=lam)

    out = []
    test_weeks = sorted(df.loc[df["season"] == test_season, "week"].unique())
    for wk in test_weeks:
        train = df[(df["season"] < test_season) |
                   ((df["season"] == test_season) & (df["week"] < wk))]
        test = df[(df["season"] == test_season) & (df["week"] == wk)]
        if len(train) < 100 or len(test) == 0:
            continue
        sw = season_decay_weights(train["season"].values, test_season, half_life_seasons)
        model = model_factory()
        model.fit(train[feature_cols].values, train["y"].values, sample_weight=sw)
        mu, sigma = model.predict_dist(test[feature_cols].values)
        chunk = test[["game_id", "season", "week", "home_team", "away_team", "y"]].copy()
        chunk["mu"] = mu
        chunk["sigma"] = sigma
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def evaluate(preds):
    r = preds["y"] - preds["mu"]
    return {
        "n": len(preds),
        "nll": float(gaussian_nll(preds["y"].values, preds["mu"].values,
                                  preds["sigma"].values).mean()),
        "rmse": float(np.sqrt((r**2).mean())),
        "mae": float(r.abs().mean()),
    }


def tune(df, feature_cols, val_season, lam_grid, half_life_grid):
    rows = []
    for hl in half_life_grid:
        for lam in lam_grid:
            preds = walk_forward(df, feature_cols, val_season,
                                 lam=lam, half_life_seasons=hl)
            m = evaluate(preds)
            rows.append({"half_life_seasons": hl, "lam": lam, **m})
    table = pd.DataFrame(rows).sort_values("nll").reset_index(drop=True)
    best = table.iloc[0]
    return {"half_life_seasons": float(best["half_life_seasons"]),
            "lam": float(best["lam"])}, table
