import pandas as pd
import numpy as np
from collections import deque

FRANCHISE_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

EPA_STATS = ["off_epa_pass", "off_epa_rush", "def_epa_pass", "def_epa_rush", "cpoe"]

# Pace and scoring volume, from prep_pbp. Carried per team with the same EWMA as
# the EPA states. These feed the totals model, which needs how often a game gets
# snapped and how much each possession is worth, not who is better.
PACE_STATS = ["off_plays", "off_drives", "off_points",
              "def_plays", "def_drives", "def_points"]

# What a team looks like before it has played a game, so the first row of 2010
# is not a team with zero drives. These are league structural numbers (about 11
# drives and 65 snaps a side, about 21 offensive points), not anything fitted to
# this dataset, and they wash out within a season of the 8 game half-life.
PACE_PRIOR = {"off_plays": 64.0, "off_drives": 11.3, "off_points": 21.0,
              "def_plays": 64.0, "def_drives": 11.3, "def_points": 21.0}

# Weather a covered game gets: the roof is the point, so neutral means calm and
# mild rather than the outdoor average.
NEUTRAL_TEMP, NEUTRAL_WIND = 70.0, 0.0

FEATURE_COLS = ["pdiff_ewma_diff", "off_pass_diff", "off_rush_diff",
                "def_pass_diff", "def_rush_diff", "cpoe_diff",
                "rest_diff", "div_game"]

# Totals inputs are sums, not differences: two good offences and two bad ones
# give the same margin and a very different total, so every matchup term is
# added rather than subtracted.
TOTAL_COLS = ["plays_sum", "plays_allowed_sum", "drives_sum", "drives_allowed_sum",
              "off_ppd_sum", "def_ppd_sum",
              "off_pass_sum", "off_rush_sum", "def_pass_sum", "def_rush_sum",
              "cpoe_sum", "game_temp", "game_wind", "indoor"]


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


def load_qb_game_stats(path):
    """game_id -> [(passer_id, dropbacks, qb_epa), ...] for every QB who dropped back."""
    q = pd.read_csv(path)
    out = {}
    for r in q.itertuples(index=False):
        out.setdefault(r.game_id, []).append((r.passer_id, r.dropbacks, r.qb_epa))
    return out


def weather_columns(df, override=None):
    """Temperature and wind as the model should see them, one value per game.

    Three cases. A covered roof gets NEUTRAL_TEMP and NEUTRAL_WIND, because the
    weather is not in the building and an average outdoor day would be a fiction.
    An outdoor game with a reading uses it. An outdoor game without one, which is
    every game that has not kicked off, gets the median of outdoor games played
    in the same calendar month, so a December game is cold and a September game
    is not, without anybody pretending to know Sunday's forecast.

    Those medians come from completed games only. They are climate, not outcome,
    but taking them from the same rows the model is about to predict would be the
    kind of shortcut that is hard to argue with later.

    override, keyed by game_id, wins over all of it: that is where a real
    forecast goes on game day.
    """
    def col(name):
        """A numeric column, or an all-missing one if this frame has no such column.

        Synthetic frames in the tests carry only the columns under test, and a
        weather reading is exactly the kind of thing they leave out. Missing is
        already a case this function handles, so it should not be an error.
        """
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[name], errors="coerce")

    indoor = (df["roof"].isin(["dome", "closed"]) if "roof" in df.columns
              else pd.Series(False, index=df.index))
    temp, wind = col("temp"), col("wind")
    gd = (pd.to_datetime(df["gameday"], errors="coerce") if "gameday" in df.columns
          else pd.Series(pd.NaT, index=df.index))
    month = gd.dt.month
    played_outdoor = (~indoor) & df["result"].notna()
    ht, hw = played_outdoor & temp.notna(), played_outdoor & wind.notna()
    tn = temp[ht].groupby(month[ht]).median()
    wn = wind[hw].groupby(month[hw]).median()
    temp = temp.fillna(month.map(tn)).fillna(
        float(tn.median()) if len(tn) else NEUTRAL_TEMP)
    wind = wind.fillna(month.map(wn)).fillna(
        float(wn.median()) if len(wn) else NEUTRAL_WIND)
    temp = temp.where(~indoor, NEUTRAL_TEMP)
    wind = wind.where(~indoor, NEUTRAL_WIND)
    if override:
        for gid, w in override.items():
            m = df["game_id"] == gid
            if "temp" in w:
                temp = temp.where(~m, float(w["temp"]))
            if "wind" in w:
                wind = wind.where(~m, float(w["wind"]))
    return temp.astype(float).values, wind.astype(float).values


def build_features(df, stats_lookup=None, form_half_life_games=8, rest_clip=(3, 21),
                   epa_season_revert=1.0, form_season_revert=1.0, qb_lookup=None,
                   qb_half_life_dropbacks=600.0, qb_prior_n=300.0,
                   qb_prior_mean=-0.05, weather_override=None):
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

    weather_override is a game_id keyed dict of {"temp": F, "wind": mph} for
    game day forecasts. An unplayed outdoor game has no weather in games.csv, so
    without an override it gets the seasonal norm for its month, which is a
    deliberate "no information" value rather than a guess at Sunday.
    """
    df = df.copy()
    decay = 0.5 ** (1.0 / form_half_life_games)
    pdiff = {}
    epa_state = {}

    home_form = np.empty(len(df))
    away_form = np.empty(len(df))
    pace_state = {}
    pace_sides = {s: (np.empty(len(df)), np.empty(len(df))) for s in PACE_STATS}
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
    # Quarterback rating, opt-in through qb_lookup and not in V3_COLS until it
    # beats the current model on an honest split. Per player, so it follows a
    # quarterback across teams, which team EPA cannot do.
    #
    # State is a decayed sum of QB EPA and of dropbacks. Decay is per dropback of
    # new evidence rather than per game, so a backup's five garbage-time dropbacks
    # do not age his history as much as a starter's forty. The rating shrinks
    # toward qb_prior_mean with the weight of qb_prior_n dropbacks: a rookie with
    # forty dropbacks gets a modest number, not whatever those forty happened to
    # be. The prior mean sits below league average on purpose, because a
    # quarterback nobody has data on is usually a backup or a rookie.
    home_qbr = np.full(len(df), np.nan)
    away_qbr = np.full(len(df), np.nan)
    qb_state = {}

    def qb_rating(qid):
        if qid is None or (isinstance(qid, float) and np.isnan(qid)):
            return qb_prior_mean
        s = qb_state.get(qid)
        if s is None:
            return qb_prior_mean
        return (s[0] + qb_prior_n * qb_prior_mean) / (s[1] + qb_prior_n)

    def get_state(team):
        return epa_state.setdefault(team, {s: 0.0 for s in EPA_STATS})

    def get_pace(team):
        return pace_state.setdefault(team, dict(PACE_PRIOR))

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
        if qb_lookup is not None:
            home_qbr[i] = qb_rating(row.home_qb_id)
            away_qbr[i] = qb_rating(row.away_qb_id)
        if stats_lookup is not None:
            hs, as_ = get_state(h), get_state(a)
            for s in EPA_STATS:
                epa_sides[s][0][i] = hs[s]
                epa_sides[s][1][i] = as_[s]
            hp, ap = get_pace(h), get_pace(a)
            for s in PACE_STATS:
                pace_sides[s][0][i] = hp[s]
                pace_sides[s][1][i] = ap[s]
        if pd.isna(row.result):
            continue
        margin = float(row.result)
        season_games[h] = season_games.get(h, 0) + 1
        season_games[a] = season_games.get(a, 0) + 1
        pdiff[h] = decay * pdiff.get(h, 0.0) + (1 - decay) * margin
        pdiff[a] = decay * pdiff.get(a, 0.0) + (1 - decay) * (-margin)
        qb_hist[h].append(row.home_qb_id)
        qb_hist[a].append(row.away_qb_id)
        if qb_lookup is not None:
            # Every quarterback who dropped back, backups included, updated only
            # after the game is over, so no game ever informs its own rating.
            for pid, n, e in qb_lookup.get(row.game_id, ()):
                s = qb_state.setdefault(pid, [0.0, 0.0])
                f = 0.5 ** (n / qb_half_life_dropbacks)
                s[0] = f * s[0] + e
                s[1] = f * s[1] + n
        if stats_lookup is not None:
            for team in (h, a):
                obs = stats_lookup.get((row.game_id, team))
                if obs is None:
                    continue
                state = get_state(team)
                for s in EPA_STATS:
                    state[s] = decay * state[s] + (1 - decay) * getattr(obs, s)
                pace = get_pace(team)
                for s in PACE_STATS:
                    # A stats file written before pace existed simply leaves the
                    # prior in place rather than crashing the whole build.
                    v = getattr(obs, s, None)
                    if v is not None and not pd.isna(v):
                        pace[s] = decay * pace[s] + (1 - decay) * float(v)

    df["pdiff_ewma_diff"] = home_form - away_form
    # Not in V3_COLS: an experiment feature, opted into by name.
    df["epa_games"] = epa_games
    df["qb_fam_diff"] = home_qbfam - away_qbfam
    # Each side kept as well as the difference. Only the difference is a model
    # input; these two exist so the site can say what the number actually means
    # ("started 7 of their last 16") instead of inferring a story from the gap.
    df["qb_fam_home"] = home_qbfam
    df["qb_fam_away"] = away_qbfam
    if qb_lookup is not None:
        df["qb_rating_home"] = home_qbr
        df["qb_rating_away"] = away_qbr
        df["qb_rating_diff"] = home_qbr - away_qbr
    df["indoor"] = df["roof"].isin(["dome", "closed"]).astype(int)
    if stats_lookup is not None:
        df["off_pass_diff"] = epa_sides["off_epa_pass"][0] - epa_sides["off_epa_pass"][1]
        df["off_rush_diff"] = epa_sides["off_epa_rush"][0] - epa_sides["off_epa_rush"][1]
        df["def_pass_diff"] = epa_sides["def_epa_pass"][0] - epa_sides["def_epa_pass"][1]
        df["def_rush_diff"] = epa_sides["def_epa_rush"][0] - epa_sides["def_epa_rush"][1]
        df["cpoe_diff"] = epa_sides["cpoe"][0] - epa_sides["cpoe"][1]
        # The same states added instead of subtracted. Nothing here is a model
        # input for margin; these exist for the totals model.
        df["off_pass_sum"] = epa_sides["off_epa_pass"][0] + epa_sides["off_epa_pass"][1]
        df["off_rush_sum"] = epa_sides["off_epa_rush"][0] + epa_sides["off_epa_rush"][1]
        df["def_pass_sum"] = epa_sides["def_epa_pass"][0] + epa_sides["def_epa_pass"][1]
        df["def_rush_sum"] = epa_sides["def_epa_rush"][0] + epa_sides["def_epa_rush"][1]
        df["cpoe_sum"] = epa_sides["cpoe"][0] + epa_sides["cpoe"][1]
        df["plays_sum"] = pace_sides["off_plays"][0] + pace_sides["off_plays"][1]
        df["plays_allowed_sum"] = pace_sides["def_plays"][0] + pace_sides["def_plays"][1]
        df["drives_sum"] = pace_sides["off_drives"][0] + pace_sides["off_drives"][1]
        df["drives_allowed_sum"] = (pace_sides["def_drives"][0]
                                    + pace_sides["def_drives"][1])
        # Points per drive, each side's own EWMA points over its own EWMA drives.
        # Both denominators start at the prior and are averages of positive
        # counts, so neither can reach zero.
        df["off_ppd_sum"] = (pace_sides["off_points"][0] / pace_sides["off_drives"][0]
                             + pace_sides["off_points"][1] / pace_sides["off_drives"][1])
        df["def_ppd_sum"] = (pace_sides["def_points"][0] / pace_sides["def_drives"][0]
                             + pace_sides["def_points"][1] / pace_sides["def_drives"][1])

    temp, wind = weather_columns(df, weather_override)
    df["game_temp"] = temp
    df["game_wind"] = wind

    hr = df["home_rest"].clip(*rest_clip).fillna(7)
    ar = df["away_rest"].clip(*rest_clip).fillna(7)
    df["rest_diff"] = hr - ar
    df["div_game"] = df["div_game"].fillna(0).astype(int)
    df["y"] = df["result"].astype(float)
    # The totals target. NaN wherever the game has not been played, exactly like
    # y, so every filter that already keys on a played game keeps working. Built
    # from the two scores when they are present, since a frame assembled for a
    # margin test may carry neither them nor nflverse's own total.
    if {"home_score", "away_score"} <= set(df.columns):
        df["y_total"] = (pd.to_numeric(df["home_score"], errors="coerce")
                         + pd.to_numeric(df["away_score"], errors="coerce"))
    elif "total" in df.columns:
        df["y_total"] = pd.to_numeric(df["total"], errors="coerce")
    else:
        df["y_total"] = np.nan
    return df
