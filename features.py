import pandas as pd
import numpy as np
from collections import deque

FRANCHISE_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

EPA_STATS = ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush", "cpoe"]

FEATURE_COLS = ["pdiff_ewma_diff", "off_pass_diff", "off_rush_diff",
                "def_pass_diff", "def_rush_diff", "cpoe_diff",
                "rest_diff", "div_game"]


def load_games(path, first_season=2010, reg_only=True, keep_unplayed=False):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["season"] >= first_season]
    if reg_only:
        df = df[df["game_type"] == "REG"]
    if not keep_unplayed:
        df = df[df["result"].notna()]
    df = df.copy()
    df["home_team"] = df["home_team"].replace(FRANCHISE_MAP)
    df["away_team"] = df["away_team"].replace(FRANCHISE_MAP)
    df["gameday"] = pd.to_datetime(df["gameday"])
    df = df.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return df


def load_team_game_stats(path):
    stats = pd.read_csv(path)
    stats["team"] = stats["team"].replace(FRANCHISE_MAP)
    return {(r.game_id, r.team): r for r in stats.itertuples(index=False)}


def build_features(df, stats_lookup=None, form_half_life_games=8, rest_clip=(3, 21),
                   epa_season_revert=1.0, form_season_revert=1.0):
    """Leakage-safe features, chronologically.

    epa_season_revert and form_season_revert are experiment knobs and default to
    1.0, which is no change: state carries across the offseason at full strength,
    exactly as it always has. Below 1.0 the deviation from the league mean is
    multiplied by that factor at each season boundary, the same shape of
    treatment kalman.py already applies to team ratings (season_revert 0.7).

    They exist because that asymmetry is accidental rather than considered: the
    ratings admit a roster turns over and the EPA features do not. Whether
    fixing it helps is a question for walk-forward, not for taste, so the
    default stays where the tuned numbers were measured.
    """
    df = df.copy()
    decay = 0.5 ** (1.0 / form_half_life_games)
    pdiff = {}
    epa_state = {}

    home_form = np.empty(len(df))
    away_form = np.empty(len(df))
    home_qbfam = np.empty(len(df))
    away_qbfam = np.empty(len(df))
    qb_hist = {}
    epa_sides = {s: (np.empty(len(df)), np.empty(len(df))) for s in EPA_STATS}
    # How many games this season already back each side's EPA, counted before
    # the game in question. Zero in week 1, so a model given this can learn how
    # much to discount numbers that are really last season's.
    epa_games = np.empty(len(df))
    season_games = {}
    prev_season = None

    def get_state(team):
        return epa_state.setdefault(team, {s: 0.0 for s in EPA_STATS})

    def familiarity(team, qb_id):
        hist = qb_hist.setdefault(team, deque(maxlen=16))
        if len(hist) == 0:
            return 0.5
        if qb_id is None or (isinstance(qb_id, float) and np.isnan(qb_id)):
            qb_id = hist[-1]
        return sum(1 for q in hist if q == qb_id) / len(hist)

    for i, row in enumerate(df.itertuples(index=False)):
        h, a = row.home_team, row.away_team
        season = int(row.season)
        if prev_season is not None and season != prev_season:
            # A new season. Ratings already get this treatment in kalman.py;
            # these knobs let the same question be asked of the EPA and form
            # states, shrinking each team's deviation from the league mean.
            season_games = {}
            if epa_season_revert != 1.0 and epa_state:
                for s in EPA_STATS:
                    mean_s = np.mean([st[s] for st in epa_state.values()])
                    for st in epa_state.values():
                        st[s] = mean_s + epa_season_revert * (st[s] - mean_s)
            if form_season_revert != 1.0 and pdiff:
                mean_f = np.mean(list(pdiff.values()))
                for k in pdiff:
                    pdiff[k] = mean_f + form_season_revert * (pdiff[k] - mean_f)
        prev_season = season
        epa_games[i] = min(season_games.get(h, 0), season_games.get(a, 0))
        home_form[i] = pdiff.get(h, 0.0)
        away_form[i] = pdiff.get(a, 0.0)
        home_qbfam[i] = familiarity(h, row.home_qb_id)
        away_qbfam[i] = familiarity(a, row.away_qb_id)
        if stats_lookup is not None:
            hs, as_ = get_state(h), get_state(a)
            for s in EPA_STATS:
                epa_sides[s][0][i] = hs[s]
                epa_sides[s][1][i] = as_[s]
        if pd.isna(row.result):
            continue
        margin = float(row.result)
        season_games[h] = season_games.get(h, 0) + 1
        season_games[a] = season_games.get(a, 0) + 1
        pdiff[h] = decay * pdiff.get(h, 0.0) + (1 - decay) * margin
        pdiff[a] = decay * pdiff.get(a, 0.0) + (1 - decay) * (-margin)
        qb_hist[h].append(row.home_qb_id)
        qb_hist[a].append(row.away_qb_id)
        if stats_lookup is not None:
            for team, state in ((h, get_state(h)), (a, get_state(a))):
                obs = stats_lookup.get((row.game_id, team))
                if obs is not None:
                    for s in EPA_STATS:
                        state[s] = decay * state[s] + (1 - decay) * getattr(obs, s)

    df["pdiff_ewma_diff"] = home_form - away_form
    # Not in V3_COLS: an experiment feature, opted into by name.
    df["epa_games"] = epa_games
    df["qb_fam_diff"] = home_qbfam - away_qbfam
    # Each side kept as well as the difference. Only the difference is a model
    # input; these two exist so the site can say what the number actually means
    # ("started 7 of their last 16") instead of inferring a story from the gap.
    df["qb_fam_home"] = home_qbfam
    df["qb_fam_away"] = away_qbfam
    df["indoor"] = df["roof"].isin(["dome", "closed"]).astype(int)
    if stats_lookup is not None:
        df["off_pass_diff"] = epa_sides["off_epa_pass"][0] - epa_sides["off_epa_pass"][1]
        df["off_rush_diff"] = epa_sides["off_epa_rush"][0] - epa_sides["off_epa_rush"][1]
        df["def_pass_diff"] = epa_sides["def_epa_pass"][0] - epa_sides["def_epa_pass"][1]
        df["def_rush_diff"] = epa_sides["def_epa_rush"][0] - epa_sides["def_epa_rush"][1]
        df["cpoe_diff"] = epa_sides["cpoe"][0] - epa_sides["cpoe"][1]

    hr = df["home_rest"].clip(*rest_clip).fillna(7)
    ar = df["away_rest"].clip(*rest_clip).fillna(7)
    df["rest_diff"] = hr - ar
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    df["y"] = df["result"].astype(float)
    return df
