"""Per-team, per-game EPA and CPOE, aggregated from nflverse play-by-play.

Two modes, and the difference matters.

A full build walks every season from scratch and is a once-ever job: it pulls
about 300 MB of parquet and takes minutes. It short-circuits on the cache file,
which is right for a first run and wrong for everything after, because that is
how team_game_stats.csv sat at 2025 week 18 while games.csv kept moving.

A refresh recomputes exactly one season and splices it into the cache. That is
the weekly job. It always re-downloads, because a season in progress grows every
Sunday and a cached copy of it is precisely the thing that goes stale. The rest
of the file is left untouched, byte for byte, so a refresh is cheap and its diff
shows only the games that were actually added.

Note the REG filter below: postseason games have never carried EPA rows, so team
EPA state coasts through January on regular season values. That is pre-existing
behaviour, left alone here deliberately, since changing it would change the
model's inputs and belongs behind a walk-forward comparison rather than inside a
staleness fix.
"""
import argparse
import os
import urllib.error
import urllib.request

import pandas as pd

PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "season_type", "posteam", "defteam",
        "epa", "pass", "rush", "cpoe", "wp"]


def download_season(season, pbp_dir="pbp_cache", force=False):
    """Path to one season's play-by-play, or None if nflverse has not posted it.

    A 404 is the normal state for a season that has not kicked off, so it is a
    None rather than an error: the weekly job should carry on and build the site.
    """
    os.makedirs(pbp_dir, exist_ok=True)
    fp = os.path.join(pbp_dir, f"pbp_{season}.parquet")
    if os.path.exists(fp) and not force:
        return fp
    tmp = fp + ".part"
    try:
        urllib.request.urlretrieve(PBP_URL.format(season), tmp)
    except urllib.error.HTTPError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        if e.code == 404:
            return None
        raise
    os.replace(tmp, fp)          # never leave a half-written parquet in place
    return fp


def season_team_game_stats(season, pbp_dir="pbp_cache", wp_filter=None,
                           force=False):
    """One season of team-game EPA rows, or None if the season has no data yet."""
    fp = download_season(season, pbp_dir, force)
    if fp is None:
        return None
    pbp = pd.read_parquet(fp, columns=COLS)
    pbp = pbp[pbp["season_type"] == "REG"]
    plays = pbp[pbp["epa"].notna() & ((pbp["pass"] == 1) | (pbp["rush"] == 1))]
    if wp_filter is not None:
        plays = plays[plays["wp"].between(*wp_filter)]
    if plays.empty:
        return None

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

    return off.merge(deff, on=["game_id", "team"], how="outer").fillna(0.0)


def build_team_game_stats(first_season=2010, last_season=2025,
                          cache_path="team_game_stats.csv", pbp_dir="pbp_cache",
                          wp_filter=None):
    """Full rebuild from scratch. Short-circuits on the cache: see refresh()."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    frames = []
    for season in range(first_season, last_season + 1):
        got = season_team_game_stats(season, pbp_dir, wp_filter)
        if got is None:
            print(f"{season}: no play-by-play published, skipped")
            continue
        frames.append(got)
        print(f"{season}: {len(got)} team-games")
    stats = pd.concat(frames, ignore_index=True).fillna(0.0)
    stats.to_csv(cache_path, index=False)
    return stats


def refresh(season, cache_path="team_game_stats.csv", pbp_dir="pbp_cache",
            wp_filter=None):
    """Recompute one season and splice it into the cache. Idempotent.

    Rows for other seasons keep their existing order and values, and the
    refreshed season is appended at the end, which is where the newest season
    already lives. Re-running with no new games rewrites the same file.
    """
    fresh = season_team_game_stats(season, pbp_dir, wp_filter, force=True)
    if fresh is None or fresh.empty:
        print(f"{season}: no play-by-play published yet, nothing to refresh")
        return 0

    # Spliced as text, not through a DataFrame. Reading the whole cache with
    # read_csv and writing it back loses a digit on every untouched row
    # (0.09235475691101869 comes back as 0.0923547569110186), which would put
    # 7740 spurious lines into a weekly commit and bury the games that actually
    # arrived. Other seasons are copied byte for byte instead.
    prefix = f"{season}_"
    header, kept, replaced = None, [], 0
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
            for line in f:
                if line.startswith(prefix):
                    replaced += 1
                else:
                    kept.append(line.rstrip("\n"))
    cols = header.split(",") if header else list(fresh.columns)
    missing = [c for c in cols if c not in fresh.columns]
    if missing:
        raise ValueError(f"refreshed rows are missing columns {missing}")
    body = fresh[cols].to_csv(index=False, header=False).rstrip("\n").split("\n")
    # Explicit LF. The blob is stored with LF and core.autocrlf is true here, so
    # writing platform-native CRLF makes git report all 8351 lines as changed.
    with open(cache_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(",".join(cols) + "\n")
        for line in kept:
            f.write(line + "\n")
        for line in body:
            f.write(line + "\n")
    games = fresh["game_id"].nunique()
    print(f"{season}: {len(fresh)} team-game rows over {games} games "
          f"(replaced {replaced}); {cache_path} now has "
          f"{len(kept) + len(body)} rows")
    return len(fresh)


def latest_season(games_path="games.csv"):
    """Newest season on the schedule, which is the one worth refreshing."""
    return int(pd.read_csv(games_path, usecols=["season"])["season"].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refresh", type=int, metavar="SEASON",
                    help="recompute one season and splice it into the cache")
    ap.add_argument("--refresh-latest", action="store_true",
                    help="refresh the newest season present in games.csv")
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--cache", default="team_game_stats.csv")
    ap.add_argument("--pbp-dir", default="pbp_cache")
    args = ap.parse_args()

    if args.refresh_latest:
        return refresh(latest_season(args.games), args.cache, args.pbp_dir)
    if args.refresh:
        return refresh(args.refresh, args.cache, args.pbp_dir)
    build_team_game_stats(cache_path=args.cache, pbp_dir=args.pbp_dir)
    return 0


if __name__ == "__main__":
    main()
