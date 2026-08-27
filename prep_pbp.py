import os
import urllib.request
import pandas as pd

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "season_type", "posteam", "defteam",
        "epa", "pass", "rush", "cpoe", "wp"]


def build_team_game_stats(first_season=2010, last_season=2025,
                          cache_path="team_game_stats.csv", pbp_dir="pbp_cache",
                          wp_filter=None):
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    os.makedirs(pbp_dir, exist_ok=True)
    frames = []
    for season in range(first_season, last_season + 1):
        fp = os.path.join(pbp_dir, f"pbp_{season}.parquet")
        if not os.path.exists(fp):
            urllib.request.urlretrieve(PBP_URL.format(season), fp)
        pbp = pd.read_parquet(fp, columns=COLS)
        pbp = pbp[pbp["season_type"] == "REG"]
        plays = pbp[pbp["epa"].notna() & ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
        if wp_filter is not None:
            plays = plays[plays["wp"].between(*wp_filter)]

        off = plays.groupby(["game_id", "posteam"]).apply(
            lambda g: pd.Series({
                "off_epa_pass": g.loc[g["pass"] == 1, "epa"].mean(),
                "off_epa_rush": g.loc[g["rush"] == 1, "epa"].mean(),
                "cpoe": g.loc[g["cpoe"].notna(), "cpoe"].mean(),
            }), include_groups=False).reset_index().rename(columns={"posteam": "team"})

        deff = plays.groupby(["game_id", "defteam"]).apply(
            lambda g: pd.Series({
                "def_epa_pass": g.loc[g["pass"] == 1, "epa"].mean(),
                "def_epa_rush": g.loc[g["rush"] == 1, "epa"].mean(),
            }), include_groups=False).reset_index().rename(columns={"defteam": "team"})

        frames.append(off.merge(deff, on=["game_id", "team"], how="outer"))
        print(f"{season}: {len(frames[-1])} team-games")

    stats = pd.concat(frames, ignore_index=True).fillna(0.0)
    stats.to_csv(cache_path, index=False)
    return stats


if __name__ == "__main__":
    build_team_game_stats()
