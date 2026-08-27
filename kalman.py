import numpy as np
import pandas as pd
from itertools import product


class TeamKalman:
    """Joint Kalman filter over team strength ratings plus home-field advantage.

    Observation per game: margin = r_home - r_away + hfa + noise (hfa dropped
    at neutral sites). Weekly random-walk process noise on team ratings;
    between seasons, ratings revert toward the league mean and variance is
    inflated. Hyperparameters are tuned by maximizing the filter's one-step
    predictive log-likelihood (empirical Bayes).
    """

    def __init__(self, obs_var=170.0, weekly_q=0.05, season_inflate=2.0,
                 season_revert=0.8, init_var=25.0, hfa_q=1e-4, burn_in_seasons=2):
        self.obs_var = obs_var
        self.weekly_q = weekly_q
        self.season_inflate = season_inflate
        self.season_revert = season_revert
        self.init_var = init_var
        self.hfa_q = hfa_q
        self.burn_in_seasons = burn_in_seasons

    def run(self, df):
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        f = n

        x = np.zeros(n + 1)
        x[f] = 2.0
        P = np.eye(n + 1) * self.init_var
        P[f, f] = 4.0

        first_season = int(df["season"].min())
        prev_season, prev_week = None, None
        diff = np.empty(len(df))
        var = np.empty(len(df))
        hfa_track = np.empty(len(df))
        loglik = 0.0
        n_scored = 0

        for i, row in enumerate(df.itertuples(index=False)):
            season, week = int(row.season), int(row.week)
            if prev_season is not None and season != prev_season:
                r = self.season_revert
                x[:n] *= r
                P[:n, :] *= r
                P[:, :n] *= r
                P[:n, :n] += self.season_inflate * np.eye(n)
            elif prev_week is not None and week != prev_week:
                P[:n, :n] += self.weekly_q * np.eye(n)
                P[f, f] += self.hfa_q
            prev_season, prev_week = season, week

            h, a = idx[row.home_team], idx[row.away_team]
            neutral = getattr(row, "location", "Home") == "Neutral"

            ph = P[:, h] - P[:, a] + (0 if neutral else P[:, f])
            hfa_term = 0.0 if neutral else x[f]
            yhat = x[h] - x[a] + hfa_term
            S = ph[h] - ph[a] + (0 if neutral else ph[f]) + self.obs_var

            diff[i] = x[h] - x[a]
            var[i] = P[h, h] + P[a, a] - 2 * P[h, a]
            hfa_track[i] = x[f]

            if pd.isna(row.result):
                continue
            y = float(row.result)
            if season >= first_season + self.burn_in_seasons:
                loglik += -0.5 * np.log(2 * np.pi * S) - (y - yhat) ** 2 / (2 * S)
                n_scored += 1

            K = ph / S
            x = x + K * (y - yhat)
            P = P - np.outer(ph, ph) / S

        out = df.copy()
        out["kalman_diff"] = diff
        out["kalman_var"] = var
        out["kalman_hfa"] = hfa_track
        self.mean_loglik_ = loglik / max(n_scored, 1)
        self.final_ratings_ = pd.Series(x[:n], index=teams).sort_values(ascending=False)
        self.final_hfa_ = x[f]
        return out


def tune_kalman(df, grid=None, train_end_season=2023):
    train = df[df["season"] <= train_end_season]
    if grid is None:
        grid = {
            "obs_var": [140.0, 160.0, 180.0],
            "weekly_q": [0.02, 0.05, 0.1, 0.2],
            "season_inflate": [1.0, 2.0, 4.0],
            "season_revert": [0.6, 0.75, 0.9],
        }
    best, best_ll, rows = None, -np.inf, []
    keys = list(grid)
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo))
        kf = TeamKalman(**params)
        kf.run(train)
        rows.append({**params, "mean_loglik": kf.mean_loglik_})
        if kf.mean_loglik_ > best_ll:
            best_ll, best = kf.mean_loglik_, params
    return best, pd.DataFrame(rows).sort_values("mean_loglik", ascending=False)


def add_kalman_features(df, params):
    return TeamKalman(**params).run(df)
