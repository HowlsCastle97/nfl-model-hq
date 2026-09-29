import numpy as np
import pandas as pd
from models import LinearGaussianModel, gaussian_nll, crps_gaussian


def season_decay_weights(train_seasons, asof_season, half_life_seasons):
    age = asof_season - np.asarray(train_seasons, dtype=float)
    if np.isinf(half_life_seasons):
        return np.ones_like(age)
    return 0.5 ** (age / half_life_seasons)


def walk_forward(df, feature_cols, test_season, lam=0.0, half_life_seasons=np.inf,
                 model_factory=None, target="y", keep_cols=(), quantiles=None):
    """Refit weekly on all games strictly before each test week; predict that week.

    model_factory() must return an object with fit(X, y, sample_weight) and
    predict_dist(X). Defaults to LinearGaussianModel(lam). Swap in the MLP
    wrapper for Deliverable III.

    target names the column to predict, "y" (home margin) by default and
    "y_total" for the totals model. Rows where it is NaN are dropped from train
    and test both, so a season in progress grades only what has been played. The
    returned frame always calls the truth "y", so evaluate does not need to know
    which target was asked for. keep_cols carries extra test columns through
    untouched, which is how the market's own total rides along for comparison.

    quantiles asks a model that can produce them for a quantile function per game,
    stored as q<tau> columns. A model without predict_quantiles ignores it, so one
    harness can grade a Gaussian and a quantile engine side by side.
    """
    if model_factory is None:
        model_factory = lambda: LinearGaussianModel(lam=lam)

    df = df[df[target].notna()]
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
        model.fit(train[feature_cols].values, train[target].values, sample_weight=sw)
        mu, sigma = model.predict_dist(test[feature_cols].values)
        cols = ["game_id", "season", "week", "home_team", "away_team"] + list(keep_cols)
        chunk = test[cols].copy()
        chunk["y"] = test[target].values
        chunk["mu"] = mu
        chunk["sigma"] = sigma
        if quantiles is not None and hasattr(model, "predict_quantiles"):
            Q = model.predict_quantiles(test[feature_cols].values, quantiles)
            for k, tau in enumerate(quantiles):
                chunk[f"q{tau:g}"] = Q[:, k]
        out.append(chunk)
    return pd.concat(out, ignore_index=True)


def evaluate(preds):
    r = preds["y"] - preds["mu"]
    return {
        "n": len(preds),
        "nll": float(gaussian_nll(preds["y"].values, preds["mu"].values,
                                  preds["sigma"].values).mean()),
        "crps": float(crps_gaussian(preds["y"].values, preds["mu"].values,
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
