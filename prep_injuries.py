"""Injury report rows joined to how much each player normally plays.

One row per player on a final NFL injury report: his team and week, position
group, the designation (Out, Doubtful, Questionable), how important he was going
into that week, and whether he actually took a snap.

Raw facts only. Nothing here is fitted: the absence rate per designation and the
points each position group is worth are estimated in experiment_injury.py, where
the rule that 2024 onward cannot influence them is enforced. Keeping estimation
out of prep is what makes that rule checkable.

Importance is the player's average snap share over his most recent eight games
strictly before the report week, across teams and seasons. Strictly before is
the part that matters. A player who misses week W has no snap row in week W, so
any join on the report week silently finds nobody, and an earlier version of the
analysis did exactly that and matched 3 injured starters in five seasons. Across
seasons and teams is what gives a week 1 report, or a player traded in October,
a real number instead of none.

`played` is whether he took an offensive or defensive snap that week. It is
outcome information and is used only to estimate how often each designation
actually misses, on selection seasons, never as a per-game input.

Snap counts begin in 2012, so that is where this starts.
"""
import argparse
import bisect
import os
import urllib.error
import urllib.request
from datetime import date

import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA"}
GROUPS = {
    "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL", "OL": "OL",
    "WR": "REC", "TE": "REC",
    "RB": "RB", "FB": "RB",
    "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL",
    "LB": "LB", "ILB": "LB", "MLB": "LB", "OLB": "LB",
    "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DB": "DB",
}
RECENT_GAMES = 8


def fetch(kind, name, cache):
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, name)
    if os.path.exists(fp):
        return fp
    try:
        urllib.request.urlretrieve(f"{BASE}/{kind}/{name}", fp + ".part")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    os.replace(fp + ".part", fp)
    return fp


def build(first=2012, last=None, cache="nflverse_cache", out="injury_player_weeks.csv"):
    last = last or date.today().year
    players = pd.read_parquet(fetch("players", "players.parquet", cache),
                              columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates("gsis_id")

    snaps, inj = [], []
    for y in range(first, last + 1):
        sp = fetch("snap_counts", f"snap_counts_{y}.parquet", cache)
        ip = fetch("injuries", f"injuries_{y}.parquet", cache)
        if sp is None or ip is None:
            continue
        s = pd.read_parquet(sp)
        snaps.append(s[s["game_type"] == "REG"])
        i = pd.read_parquet(ip)
        inj.append(i[(i["game_type"] == "REG")
                     & i["report_status"].isin(["Out", "Doubtful", "Questionable"])])
    snaps = pd.concat(snaps, ignore_index=True)
    inj = pd.concat(inj, ignore_index=True)

    snaps["share"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    snaps["t"] = snaps["season"] * 100 + snaps["week"]
    snaps = snaps.sort_values(["pfr_player_id", "t"])
    played_ids = set(zip(snaps.loc[(snaps["offense_snaps"] + snaps["defense_snaps"]) > 0,
                                   "pfr_player_id"],
                         snaps.loc[(snaps["offense_snaps"] + snaps["defense_snaps"]) > 0, "t"]))
    history = {pid: (g["t"].tolist(), g["share"].tolist())
               for pid, g in snaps.groupby("pfr_player_id")}

    inj = inj.merge(players, on="gsis_id", how="left")
    crosswalked = inj["pfr_id"].notna().mean()
    inj["team"] = inj["team"].replace(MAP)
    inj["season"] = inj["season"].astype(int)
    inj["week"] = inj["week"].astype(int)
    inj["group"] = inj["position"].map(GROUPS)
    inj = inj[inj["group"].notna()].copy()        # QB, K, P, LS are not in scope
    inj["t"] = inj["season"] * 100 + inj["week"]

    importance, played = [], []
    for pid, t in zip(inj["pfr_id"], inj["t"]):
        ts, sh = history.get(pid, ([], []))
        k = bisect.bisect_left(ts, t)             # games strictly before week t
        recent = sh[max(0, k - RECENT_GAMES):k]
        importance.append(sum(recent) / len(recent) if recent else 0.0)
        played.append((pid, t) in played_ids)
    inj["importance"] = importance
    inj["played"] = played

    cols = ["season", "week", "team", "gsis_id", "position", "group",
            "report_status", "importance", "played"]
    res = inj[cols].sort_values(["season", "week", "team", "gsis_id"])
    res.to_csv(out, index=False)
    print(f"{out}: {len(res)} report rows, seasons {res.season.min()}-{res.season.max()}, "
          f"{crosswalked:.1%} of reports crosswalked to a snap-count id, "
          f"{(res.importance > 0).mean():.1%} with prior snaps")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--first", type=int, default=2012)
    ap.add_argument("--cache", default="nflverse_cache")
    ap.add_argument("--out", default="injury_player_weeks.csv")
    a = ap.parse_args()
    build(first=a.first, cache=a.cache, out=a.out)


if __name__ == "__main__":
    main()
