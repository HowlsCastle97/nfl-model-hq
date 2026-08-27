import numpy as np
import pandas as pd
from features import load_games, load_team_game_stats, build_features, FEATURE_COLS
from kalman import tune_kalman, TeamKalman
from walkforward import walk_forward, evaluate
from ensemble import DeepEnsemble

V2_COLS = ["kalman_diff", "kalman_var"] + FEATURE_COLS + ["qb_fam_diff", "indoor"]

games = load_games("games.csv", first_season=2010)
stats = load_team_game_stats("team_game_stats.csv")
df = build_features(games, stats, form_half_life_games=8)

best_kf, kf_table = tune_kalman(df, grid={
    "obs_var": [150.0, 160.0, 170.0],
    "weekly_q": [0.2, 0.4, 0.8, 1.6],
    "season_inflate": [4.0, 8.0, 16.0],
    "season_revert": [0.7, 0.75, 0.8]}, train_end_season=2023)
print("Kalman hyperparameters (empirical Bayes on <=2023):", best_kf)
kf = TeamKalman(**best_kf)
df = kf.run(df)
print("Estimated HFA:", round(kf.final_hfa_, 2))
print("\nTop 8 team ratings entering 2026:")
print(kf.final_ratings_.head(8).round(2).to_string())

print("\n=== Walk-forward 2025 test ===")
lin_old = walk_forward(df, FEATURE_COLS, 2025, lam=100.0, half_life_seasons=2.0)
lin_new = walk_forward(df, V2_COLS, 2025, lam=100.0, half_life_seasons=2.0)
factory = lambda: DeepEnsemble(n_members=5, hidden=16, weight_decay=1e-2,
                               epochs=200, seed=0)
ens = walk_forward(df, V2_COLS, 2025, half_life_seasons=2.0, model_factory=factory)
print("linear (8 features):        ", evaluate(lin_old))
print("linear + kalman:            ", evaluate(lin_new))
print("deep ensemble (het., beta): ", evaluate(ens))

train = df[df["season"] <= 2025]
model = DeepEnsemble(n_members=5, hidden=16, weight_decay=1e-2, epochs=200, seed=0)
from walkforward import season_decay_weights
sw = season_decay_weights(train["season"].values, 2026, 2.0)
model.fit(train[V2_COLS].values, train["y"].values, sample_weight=sw)
print("\n2026 deployment ensemble fit on all data through 2025: ready")
