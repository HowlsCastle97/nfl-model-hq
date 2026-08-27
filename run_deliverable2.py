import numpy as np
from features import load_games, load_team_game_stats, build_features, FEATURE_COLS
from models import prob_margin_over
from walkforward import walk_forward, evaluate, tune

games = load_games("games.csv", first_season=2010)
stats = load_team_game_stats("team_game_stats.csv")

lam_grid = [0.0, 10.0, 100.0, 1000.0]
hl_grid = [2.0, 3.0, 5.0, np.inf]

results = {}
for hl_games in [8, 12, 16]:
    df = build_features(games, stats, form_half_life_games=hl_games)
    best, table = tune(df, FEATURE_COLS, val_season=2024,
                       lam_grid=lam_grid, half_life_grid=hl_grid)
    results[hl_games] = (df, best, table)
    print(f"form_hl={hl_games}: best {best}, val NLL={table.iloc[0]['nll']:.4f}, "
          f"val RMSE={table.iloc[0]['rmse']:.3f}")

best_hl = min(results, key=lambda k: results[k][2].iloc[0]["nll"])
df, best, table = results[best_hl]
print(f"\nSelected form_hl={best_hl}, {best}")
print("\n=== Validation lambda sweep at selected settings ===")
sel = table[table["half_life_seasons"] == best["half_life_seasons"]]
print(sel.to_string(index=False))

print("\n=== Walk-forward test on 2025 ===")
mle = walk_forward(df, FEATURE_COLS, 2025, lam=0.0,
                   half_life_seasons=best["half_life_seasons"])
map_ = walk_forward(df, FEATURE_COLS, 2025, lam=best["lam"],
                    half_life_seasons=best["half_life_seasons"])
print("MLE:", evaluate(mle))
print("MAP:", evaluate(map_))

base = mle.copy()
hist = df[df["season"] < 2025]
base["mu"] = hist["y"].mean()
base["sigma"] = hist["y"].std()
print("Baseline:", evaluate(base))

print("\n=== Example market probabilities (2025 week 1) ===")
for row in map_[map_["week"] == 1].head(5).itertuples(index=False):
    p = prob_margin_over(row.mu, row.sigma, 0.5)
    print(f"{row.away_team} @ {row.home_team}: mu={row.mu:+.1f}, "
          f"sigma={row.sigma:.1f}, P(home wins by >0.5)={p:.3f}, actual={row.y:+.0f}")
