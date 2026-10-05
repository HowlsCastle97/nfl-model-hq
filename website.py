import argparse
import itertools
import os
from collections import Counter
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import norm

import rundown as rd
import totals as tot
from features import EPA_STATS, FEATURE_COLS, PACE_STATS
from models import prob_total_over
from walkforward import walk_forward

V3 = rd.V3_COLS
SPREAD_JUICE = 1.909


def american(p):
    if p <= 0 or p >= 1:
        return "n/a"
    return f"{-round(100 * p / (1 - p))}" if p >= 0.5 else f"+{round(100 * (1 - p) / p)}"


def dec_from_american(a):
    a = float(a)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def half_point(mu):
    """The model's margin rounded to the nearest half point.

    Lines are quoted in half points, so this is the model's number expressed in
    the unit a bettor actually sees. The headline and the quoted line both come
    from here: deriving them separately is what let a margin of 5.3 print as
    "by 5" and "-5.5" on the same card.
    """
    return round(mu * 2) / 2


# Square-3 rule. A disagreement big enough that the likelier explanation is the
# model missing news, not the market mispricing. Measured in units of the
# model's own predictive standard deviation, which for a Gaussian predictive
# margin is simply the difference of probits, since p = Phi(mu / sigma) on both
# sides. Probability is the wrong yardstick: a 0.15 gap is 0.39 sigma at an even
# price but 1.81 sigma at 0.90, so a fixed probability threshold barely applied
# to lopsided games at all. 0.40 is set to agree with the old 0.15 rule at a
# coin-flip price, which is where that rule was calibrated.
SQUARE3_SIGMAS = 0.40


def square3_gap(p_model, p_market):
    """Model minus market for one side, in the model's own predictive sigmas."""
    lo, hi = 1e-6, 1 - 1e-6
    return float(norm.ppf(min(max(p_model, lo), hi))
                 - norm.ppf(min(max(p_market, lo), hi)))


def fav_line(mu, home, away):
    s = half_point(mu)
    if s == 0:
        return "pick'em"
    team = home if s > 0 else away
    return f"{team} -{abs(s):g}"


# Principle 1: every season before this one shaped the model's settings, so the
# Track Record never quotes them. Everything from here on is walk-forward.
TRACK_FIRST_SEASON = 2021


def history_tables(df, seasons=None, model_factory=None):
    """Walk-forward report card, one row per completed game.

    Graded by rd.DeployedModel unless told otherwise: the same deep ensemble,
    hyperparameters and variance recalibration that make the This Week picks.
    It used to grade the ridge linear model, which was honest walk-forward but
    of a different model from the one a reader was acting on, so every sigma on
    the tab was one constant and the "(live)" row tested the wrong thing. The
    factory is a parameter only so tests can swap in something fast.

    Seasons default to TRACK_FIRST_SEASON onward, through the latest season with
    a completed game, so a new season joins the table the week its first result
    lands instead of waiting for someone to edit a tuple. A season with no
    completed games yet is skipped rather than passed to walk_forward, which
    would hand pd.concat an empty list.

    The walk-forward runs on completed games only. `df` carries the unplayed
    schedule too, because the This Week tab needs features for it, but an
    unplayed row has no result: it cannot be scored, and in the live season it
    would otherwise sit inside a test week. Features were built chronologically
    over the whole frame before this, so dropping future rows here cannot leak
    anything backwards.

    by_season gains a `live` column: a season with both completed and unplayed
    games is still in progress, and the page says so.
    """
    done = df[df["result"].notna()]
    if model_factory is None:
        model_factory = rd.DeployedModel
    if seasons is None:
        last = int(done["season"].max()) if len(done) else TRACK_FIRST_SEASON
        seasons = range(TRACK_FIRST_SEASON, last + 1)
    frames = []
    for season in seasons:
        if not (done["season"] == season).any():
            continue
        p = walk_forward(done, V3, season, half_life_seasons=rd.DECAY_HL,
                         model_factory=model_factory)
        if len(p):
            p["season"] = season
            frames.append(p)
    if not frames:
        empty_hist = pd.DataFrame(columns=[
            "game_id", "season", "week", "home_team", "away_team", "y", "mu",
            "sigma", "spread_line", "p_home", "su_pick", "su_win", "su_correct",
            "ats_pick", "ats_correct", "home_score", "away_score"])
        empty_season = pd.DataFrame(
            columns=["games", "winner_pct", "ats_pct", "avg_miss", "live"])
        empty_calib = pd.DataFrame(
            columns=["games", "model_said", "home_actually_won"])
        return empty_hist, empty_season, empty_calib
    hist = pd.concat(frames, ignore_index=True)
    hist = hist.merge(df[["game_id", "spread_line"]], on="game_id", how="left")
    hist["p_home"] = norm.cdf(hist["mu"] / hist["sigma"])
    hist["su_pick"] = np.where(hist["mu"] > 0, hist["home_team"], hist["away_team"])
    hist["su_win"] = np.where(hist["y"] > 0, hist["home_team"],
                              np.where(hist["y"] < 0, hist["away_team"], "tie"))
    hist["su_correct"] = hist["su_pick"] == hist["su_win"]
    has_line = hist["spread_line"].notna()
    hist["ats_pick"] = np.where(hist["mu"] > hist["spread_line"],
                                hist["home_team"], hist["away_team"])
    home_cover = hist["y"] > hist["spread_line"]
    push = hist["y"] == hist["spread_line"]
    hist["ats_correct"] = np.where(push, np.nan,
        np.where(hist["ats_pick"] == hist["home_team"], home_cover, ~home_cover))
    hist.loc[~has_line, "ats_correct"] = np.nan

    hist = hist.merge(df[["game_id", "home_score", "away_score"]],
                      on="game_id", how="left")

    by_season = hist.groupby("season").apply(lambda g: pd.Series({
        "games": len(g),
        "winner_pct": 100 * g["su_correct"].mean(),
        "ats_pct": 100 * g["ats_correct"].dropna().astype(float).mean(),
        "avg_miss": (g["y"] - g["mu"]).abs().mean(),
    }), include_groups=False).round(1)
    unplayed = set(df.loc[df["result"].isna(), "season"].astype(int))
    by_season["live"] = [int(s) in unplayed for s in by_season.index]

    edges = [0, .35, .45, .55, .65, 1.0]
    labels = ["0-35%", "35-45%", "45-55%", "55-65%", "65-100%"]
    bins = pd.cut(hist["p_home"], edges, labels=labels)
    calib = hist.groupby(bins, observed=True).apply(lambda g: pd.Series({
        "games": len(g),
        "model_said": 100 * g["p_home"].mean(),
        "home_actually_won": 100 * (g["y"] > 0).mean(),
    }), include_groups=False).round(1)
    return hist, by_season, calib


# The card publishes a spread pick only when the model gives its side at least
# this chance of covering. Graded here at the same threshold, because a record of
# picks the site never made would flatter or damn it for no reason.
SPREAD_REC_P = 0.58
REC_COLS = ["games", "all_n", "all_w", "all_l", "all_void", "all_roi",
            "all_hit", "all_imp", "all_said",
            "ml_n", "ml_w", "ml_l", "ml_void", "ml_roi",
            "asp_n", "asp_w", "asp_l", "asp_push", "asp_roi",
            "sp_n", "sp_w", "sp_l", "sp_push", "sp_roi"]


def american_dec(ml):
    """Decimal payout from an American moneyline, NaN safe."""
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))


def rec_records(hist, df):
    """Betting record of the two recommendations the site publishes, by week.

    Two strategies, each graded the way it would have settled.

    The spread is graded twice for the same reason the moneyline is. Every spread
    pick is the model's side against the Vegas closing line in every game it has a
    line for, which is the number the old accuracy table has always reported as a
    percentage and never as money. The filtered one is the subset the card
    actually prints, where the model gives its side a 58% chance or better.

    Both settle at standard -110: a win pays 0.909 per unit risked, a push is
    voided rather than counted as half a win, and a game with no closing line is
    not graded at all.

    The moneyline is graded twice, because the card shows two different things and
    a reader may act on either. Every ML pick is the model's outright call, the
    side it makes the favourite, whatever the price: that is the "ML pick" row, and
    it is on every card whether or not anything is worth buying. The value pick is
    the narrower one, the games where the model's number beat the price. Grading
    only the second would leave out most of the picks the site actually publishes.

    Both settle at the same closing price on the side taken. The value pick is the
    model's win probability The price used is the closing
    Vegas moneyline out of games.csv, because Kalshi prices were only logged from
    2026 and a record that started this September would say nothing. That has two
    consequences worth stating on the page rather than burying: the Vegas price
    carries the book's vig, which makes this a slightly harder test than a Kalshi
    ask, and Kalshi's 7% fee on the win is not in these numbers. So this is the
    record of the model's value rule, not a transcript of what a Kalshi account
    would show. An outright tie voids the bet.

    ROI is profit over amount staked, void and pushed bets excluded from both.
    """
    if hist.empty:
        empty = pd.DataFrame(columns=["season", "week"] + REC_COLS)
        return empty, empty.copy()

    d = hist.merge(df[["game_id", "home_moneyline", "away_moneyline"]],
                   on="game_id", how="left").copy()

    # Spread side and its cover probability, the card's own calculation.
    p_cover_home = pd.Series(
        norm.cdf((d["mu"] - d["spread_line"]) / d["sigma"]), index=d.index)
    home_side = p_cover_home >= 0.5
    sp_p = p_cover_home.where(home_side, 1 - p_cover_home)
    has_line = d["spread_line"].notna()
    d["sp_rec"] = has_line & (sp_p >= SPREAD_REC_P)
    home_cover = d["y"] > d["spread_line"]
    covered = pd.Series(np.where(home_side, home_cover, ~home_cover), index=d.index)
    on_the_number = d["y"] == d["spread_line"]
    d["sp_push"] = d["sp_rec"] & on_the_number
    d["sp_win"] = d["sp_rec"] & ~d["sp_push"] & covered
    d["sp_loss"] = d["sp_rec"] & ~d["sp_push"] & ~d["sp_win"]
    d["sp_profit"] = (d["sp_win"] * (SPREAD_JUICE - 1.0)) - d["sp_loss"] * 1.0

    # The same side in every game with a line, unfiltered.
    d["asp_push"] = has_line & on_the_number
    d["asp_win"] = has_line & ~d["asp_push"] & covered
    d["asp_loss"] = has_line & ~d["asp_push"] & ~d["asp_win"]
    d["asp_profit"] = (d["asp_win"] * (SPREAD_JUICE - 1.0)) - d["asp_loss"] * 1.0

    # Moneyline value side: model probability minus the price's implied
    # probability, taking whichever side is better and only if it is positive.
    dec_h = pd.Series(american_dec(d["home_moneyline"]), index=d.index)
    dec_a = pd.Series(american_dec(d["away_moneyline"]), index=d.index)
    edge_h = d["p_home"] - 1.0 / dec_h
    edge_a = (1 - d["p_home"]) - 1.0 / dec_a
    take_home = edge_h >= edge_a
    edge = edge_h.where(take_home, edge_a)
    dec = dec_h.where(take_home, dec_a)
    d["ml_rec"] = edge.notna() & (edge > 0)
    won = np.where(take_home, d["y"] > 0, d["y"] < 0)
    d["ml_void"] = d["ml_rec"] & (d["y"] == 0)
    d["ml_win"] = d["ml_rec"] & ~d["ml_void"] & won
    d["ml_loss"] = d["ml_rec"] & ~d["ml_void"] & ~d["ml_win"]
    d["ml_profit"] = d["ml_win"] * (dec.fillna(1.0) - 1.0) - d["ml_loss"] * 1.0

    # Every ML pick: the model's own favourite, at that side's closing price,
    # regardless of whether the price left any value in it. Nothing is filtered
    # out here except a game with no posted moneyline.
    fav_home = d["p_home"] >= 0.5
    dec_all = dec_h.where(fav_home, dec_a)
    d["all_rec"] = dec_all.notna()
    all_won = np.where(fav_home, d["y"] > 0, d["y"] < 0)
    d["all_void"] = d["all_rec"] & (d["y"] == 0)
    d["all_win"] = d["all_rec"] & ~d["all_void"] & all_won
    d["all_loss"] = d["all_rec"] & ~d["all_void"] & ~d["all_win"]
    d["all_profit"] = d["all_win"] * (dec_all.fillna(1.0) - 1.0) - d["all_loss"] * 1.0
    # What the prices paid for those picks implied. Carried through the aggregate
    # so the page can say what a 64% hit rate had to clear, instead of leaving a
    # reader to wonder how a winning record loses money.
    d["all_imp"] = (1.0 / dec_all).where(d["all_win"] | d["all_loss"])
    d["all_said"] = d["p_home"].where(fav_home, 1 - d["p_home"]).where(
        d["all_win"] | d["all_loss"])

    def agg(g):
        all_n = int(g["all_win"].sum() + g["all_loss"].sum())
        ml_n = int(g["ml_win"].sum() + g["ml_loss"].sum())
        asp_n = int(g["asp_win"].sum() + g["asp_loss"].sum())
        sp_n = int(g["sp_win"].sum() + g["sp_loss"].sum())
        return pd.Series({
            "games": len(g),
            "all_n": all_n, "all_w": int(g["all_win"].sum()),
            "all_l": int(g["all_loss"].sum()), "all_void": int(g["all_void"].sum()),
            "all_roi": (100 * g["all_profit"].sum() / all_n) if all_n else np.nan,
            "all_hit": (100 * g["all_win"].sum() / all_n) if all_n else np.nan,
            "all_imp": 100 * g["all_imp"].mean(),
            "all_said": 100 * g["all_said"].mean(),
            "ml_n": ml_n, "ml_w": int(g["ml_win"].sum()),
            "ml_l": int(g["ml_loss"].sum()), "ml_void": int(g["ml_void"].sum()),
            "ml_roi": (100 * g["ml_profit"].sum() / ml_n) if ml_n else np.nan,
            "asp_n": asp_n, "asp_w": int(g["asp_win"].sum()),
            "asp_l": int(g["asp_loss"].sum()), "asp_push": int(g["asp_push"].sum()),
            "asp_roi": (100 * g["asp_profit"].sum() / asp_n) if asp_n else np.nan,
            "sp_n": sp_n, "sp_w": int(g["sp_win"].sum()),
            "sp_l": int(g["sp_loss"].sum()), "sp_push": int(g["sp_push"].sum()),
            "sp_roi": (100 * g["sp_profit"].sum() / sp_n) if sp_n else np.nan,
        })

    by_week = d.groupby(["season", "week"]).apply(
        agg, include_groups=False).reset_index()
    by_season = d.groupby("season").apply(agg, include_groups=False).reset_index()
    allrow = agg(d)
    allrow["season"] = "All seasons"
    by_season = pd.concat([by_season, allrow.to_frame().T], ignore_index=True)
    return by_week, by_season


def totals_history(df, seasons=None):
    """Walk-forward scorecard for the totals model, one row per season.

    A separate model from the Bayesian one and therefore a separate scorecard,
    which is the whole reason it gets its own table on the page rather than two
    more columns on the existing one. Graded two ways: how far the predicted total
    landed from the real one, and how often it beat the closing total, which is
    the only question a bettor is asking.

    Pushes are dropped, as a book would settle them. 52.4% is break even at -110.
    """
    played = df[df["y_total"].notna()]
    if seasons is None:
        last = int(played["season"].max()) if len(played) else TRACK_FIRST_SEASON
        seasons = range(TRACK_FIRST_SEASON, last + 1)
    rows = []
    for season in seasons:
        if not (played["season"] == season).any():
            continue
        p = walk_forward(played, tot.TOTAL_COLS, season,
                         half_life_seasons=tot.TOTAL_DECAY_HL,
                         model_factory=tot.DeployedTotals, target="y_total",
                         keep_cols=("total_line",))
        if not len(p):
            continue
        graded = p[p["total_line"].notna() & (p["y"] != p["total_line"])]
        over = graded["mu"] > graded["total_line"]
        hit = (over == (graded["y"] > graded["total_line"])).mean() if len(graded) else np.nan
        rows.append({"season": season, "games": len(p),
                     "avg_miss": float((p["y"] - p["mu"]).abs().mean()),
                     "graded": int(len(graded)),
                     "ou_pct": 100 * float(hit),
                     "mean_total": float(p["mu"].mean()),
                     "mean_actual": float(p["y"].mean())})
    return pd.DataFrame(rows)


def ou_rows(r):
    """The over/under, in the same bold rows as everything else the model says.

    Returns two rows, shaped like the two they sit between. The prediction row
    reads like "Bayesian prediction": the number, then the give or take and the
    market's number beside it. The pick row reads like "ML pick": the side, then
    how often the model gets there.

    It started as one line of small grey text under the card, which set the
    model's total in the typeface of a footnote. A reader acts on a total the same
    way they act on a spread, so it is sized the same.

    The pick is graded on the spread row's thresholds, 58% for value at -110 and
    54.5% for a lean, so the two mean the same thing by the same rule. What they do
    not share is a record: the spread pick has beaten 52.4% in some seasons and the
    totals model never has, which the This Week copy says out loud rather than
    leaving a green tag to imply it.

    Either row can be empty: no totals model at all gives two, and a game with no
    posted total gives a prediction with nothing to pick against.
    """
    mu, sigma = r.get("tot_mu"), r.get("tot_sigma")
    if mu is None or pd.isna(mu):
        return "", ""
    line = r.get("total_line")
    has_line = line is not None and not pd.isna(line)
    note = (f'&plusmn;{sigma:.0f} &middot; Vegas {float(line):g}' if has_line
            else f'&plusmn;{sigma:.0f} &middot; no posted total yet')
    pred = (f'<div class="lrow"><span class="llab">Model total:</span> '
            f'<span class="lval">{mu:.1f}</span> '
            f'<span class="lnote">{note}</span></div>')
    if not has_line:
        return pred, ""
    p_over, p_under, p_push = prob_total_over(mu, sigma, float(line))
    side, p = ("Over", float(p_over)) if p_over >= p_under else ("Under", float(p_under))
    if p >= 0.58:
        tag = ' &middot; <span class="hit">value at a book\'s -110</span>'
    elif p >= 0.545:
        tag = ' &middot; <span style="color:var(--yellow)">slight lean at -110</span>'
    else:
        tag = ' &middot; no edge at -110'
    push = (f' &middot; {p_push*100:.0f}% chance it lands exactly on {float(line):g}'
            if p_push > 0.005 else '')
    pick = (f'<div class="lrow"><span class="llab">O/U pick:</span> '
            f'<span class="lval">{side} {float(line):g}</span> '
            f'<span class="lpct">hits <b>{p*100:.0f}%</b> of the time{tag}{push}'
            f'</span></div>')
    return pred, pick


# Which totals inputs belong together in one sentence, and how to say each one.
TOTAL_THEMES = (
    ("possessions", ("drives_sum", "drives_allowed_sum", "plays_sum",
                     "plays_allowed_sum")),
    ("scoring", ("off_ppd_sum", "def_ppd_sum")),
    ("efficiency", ("off_pass_sum", "off_rush_sum", "def_pass_sum",
                    "def_rush_sum", "cpoe_sum")),
    ("conditions", ("game_wind", "game_temp", "indoor")),
)

TOTAL_LABELS = {
    "drives_sum": "Drives, both offences",
    "drives_allowed_sum": "Drives faced, both defences",
    "plays_sum": "Snaps, both offences",
    "plays_allowed_sum": "Snaps faced, both defences",
    "off_ppd_sum": "Points per drive, both offences",
    "def_ppd_sum": "Points per drive allowed",
    "off_pass_sum": "Passing EPA, both offences",
    "off_rush_sum": "Rushing EPA, both offences",
    "def_pass_sum": "Passing EPA allowed",
    "def_rush_sum": "Rushing EPA allowed",
    "cpoe_sum": "CPOE, both offences",
    "game_wind": "Wind",
    "game_temp": "Temperature",
    "indoor": "Roof",
}


def totals_reason(r):
    """Why the totals model landed where it did, as a sum that actually adds up.

    The margin panel has to warn that its parts do not sum to the whole, because
    a network is not additive. This one does not: the deployed totals engine is a
    weighted least squares fit, so the prediction is exactly the average game's
    total plus one term per input. Every number below is that term, and the
    sentence says so.

    Each clause quotes this game's value against the league average the model
    measured it from, because "the pace is slow" is a claim and "20.4 drives
    against a league 23.2" is the evidence for it.
    """
    mu, sigma = r.get("tot_mu"), r.get("tot_sigma")
    c = r.get("tot_c")
    if mu is None or pd.isna(mu) or c is None or not len(c):
        return ""
    base = float(r.get("tot_base", mu))
    vals = dict(zip(tot.TOTAL_COLS, r.get("tot_x", [])))
    mean = dict(zip(tot.TOTAL_COLS, r.get("tot_mean", [])))
    con = dict(zip(tot.TOTAL_COLS, c))
    home, away = r["home"], r["away"]
    lev = r.get("lev") or {}

    def sgn(x, places=2):
        v = float(x)
        if abs(v) < 0.5 * 10 ** -places:
            v = 0.0
        return f"{v:+.{places}f}"

    def num(fmt, key, side):
        v = lev.get(f"{key}_{side}")
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else fmt.format(v)

    def pace_pair(key, fmt="{:.1f}"):
        a, b = num(fmt, key, "home"), num(fmt, key, "away")
        return (a, b) if a and b else (None, None)

    said = []
    for name, keys in TOTAL_THEMES:
        tot_pts = sum(con.get(k, 0.0) for k in keys)
        if abs(tot_pts) < 0.25:
            continue
        way = "up" if tot_pts > 0 else "down"
        pts = f"{abs(tot_pts):.1f}"
        unit = "point" if pts == "1.0" else "points"
        if name == "possessions":
            h, a = pace_pair("off_drives")
            both = vals.get("drives_sum", float("nan"))
            who = (f"{home} have been averaging {h} drives a game and {away} {a}, "
                   f"{both:.1f} between them" if h and a else
                   f"these two combine for {both:.1f} drives")
            said.append(f"Pace takes it <b>{way} {pts} {unit}</b>: {who}, against "
                        f"a league {mean.get('drives_sum', 0):.1f}.")
        elif name == "scoring":
            hp = num("{:.1f}", "off_points", "home")
            hd = num("{:.1f}", "off_drives", "home")
            ap = num("{:.1f}", "off_points", "away")
            ad = num("{:.1f}", "off_drives", "away")
            if hp and hd and ap and ad:
                hr = float(hp) / max(float(hd), 1e-6)
                ar = float(ap) / max(float(ad), 1e-6)
                who = (f"{home} are scoring {hr:.2f} points a drive and {away} "
                       f"{ar:.2f}, {hr + ar:.2f} between them")
            else:
                who = (f"they combine for "
                       f"{vals.get('off_ppd_sum', float('nan')):.2f} points a drive")
            said.append(f"Scoring rate takes it <b>{way} {pts} {unit}</b>: {who}, "
                        f"against a league {mean.get('off_ppd_sum', 0):.2f}.")
        elif name == "efficiency":
            bits = []
            if abs(con.get("def_pass_sum", 0)) >= 0.2:
                bits.append(f"the two pass defences concede "
                            f"{sgn(vals.get('def_pass_sum', 0))} EPA a dropback "
                            f"between them against {sgn(mean.get('def_pass_sum', 0))}")
            if abs(con.get("off_pass_sum", 0)) >= 0.2:
                bits.append(f"the two passing offences are at "
                            f"{sgn(vals.get('off_pass_sum', 0))} against "
                            f"{sgn(mean.get('off_pass_sum', 0))}")
            if abs(con.get("def_rush_sum", 0)) >= 0.2:
                bits.append(f"the run defences concede "
                            f"{sgn(vals.get('def_rush_sum', 0))} against "
                            f"{sgn(mean.get('def_rush_sum', 0))}")
            body = "; ".join(bits) if bits else "the efficiency numbers lean that way"
            said.append(f"Efficiency takes it <b>{way} {pts} {unit}</b>: {body}.")
        else:
            bits = []
            if vals.get("indoor"):
                bits.append(f"the roof is closed, so no wind and a steady "
                            f"{vals.get('game_temp', 70):.0f} degrees where an "
                            f"outdoor game would average "
                            f"{mean.get('game_wind', 0):.0f} mph")
            else:
                if abs(con.get("game_wind", 0)) >= 0.15:
                    bits.append(f"{vals.get('game_wind', 0):.0f} mph of wind "
                                f"against a typical {mean.get('game_wind', 0):.0f}")
                if abs(con.get("game_temp", 0)) >= 0.15:
                    bits.append(f"{vals.get('game_temp', 0):.0f} degrees against a "
                                f"typical {mean.get('game_temp', 0):.0f}")
            body = "; ".join(bits) if bits else "the conditions"
            said.append(f"Conditions take it <b>{way} {pts} {unit}</b>: {body}.")

    line = r.get("total_line")
    tail = ""
    if line is not None and not pd.isna(line):
        d = mu - float(line)
        tail = (f" The market is at {float(line):g}, so the model is "
                f"{abs(d):.1f} {'above' if d > 0 else 'below'} it.")
    if not said:
        said = ["Nothing about this matchup moves the total far from average."]

    rows = []
    for k, v in sorted(con.items(), key=lambda kv: -abs(kv[1])):
        pull = ('<span class="rnil">no effect</span>' if abs(v) < 0.05 else
                f'<span class="{"rhome" if v > 0 else "raway"}">{abs(v):.1f} '
                f'{"up" if v > 0 else "down"}</span>')
        shown = (f"{vals.get(k, 0):.0f}" if k in ("plays_sum", "plays_allowed_sum",
                                                  "game_temp", "game_wind")
                 else "indoors" if k == "indoor" and vals.get(k)
                 else "outdoors" if k == "indoor"
                 else sgn(vals.get(k, 0)) if k.endswith("_sum") and "pp" not in k
                 else f"{vals.get(k, 0):.2f}")
        rows.append(f'<tr><td class="rpull">{pull}</td>'
                    f'<td class="rname">{TOTAL_LABELS.get(k, k)}'
                    f'<span class="rval">{shown}</span></td>'
                    f'<td class="rdesc">league {mean.get(k, 0):.2f}</td></tr>')

    return (f'<p class="rlead" style="margin-top:14px"><b>Why that total.</b> An '
            f'average matchup prices at {base:.1f} points. This one comes out at '
            f'<b>{mu:.1f}</b> &plusmn;{sigma:.0f}.{tail}</p>'
            f'<p class="rplain">{" ".join(said)}</p>'
            f'<p class="rlead">The same thing as arithmetic. Unlike the margin '
            f'table above, these do add up: {base:.1f} plus every line below is '
            f'exactly {mu:.1f}, because the totals model is a weighted least '
            f'squares fit and not a network.</p>'
            f'<table class="rtab">{"".join(rows)}</table>')


def stamp_text(ts):
    """A UTC instant written out in US Eastern, as the text under the live clock.

    The page's JS rewrites these on the reader's own clock, the same way kickoffs
    are handled. This is what a reader sees with scripting off, so it carries its
    zone explicitly rather than leaving a bare time to be guessed at.
    """
    et = ts.tz_convert(ZoneInfo("America/New_York"))
    return et.strftime("%b %d, %Y at %I:%M %p ET").replace(" 0", " ")


def rec_rows(frame, label_col, label_fmt=str):
    """One HTML row per period of the recommendation record.

    A period with no recommendation of a given kind prints a dash rather than a
    0-0 record and a 0% return, because "we did not advise anything" and "we
    advised things and broke even" are different weeks.
    """
    out = []
    for r in frame.itertuples(index=False):
        cells = [f"<td>{label_fmt(getattr(r, label_col))}</td>",
                 f"<td>{int(r.games)}</td>"]
        for n, w, l, dead, dead_word, roi in (
                (r.all_n, r.all_w, r.all_l, r.all_void, "void", r.all_roi),
                (r.ml_n, r.ml_w, r.ml_l, r.ml_void, "void", r.ml_roi),
                (r.asp_n, r.asp_w, r.asp_l, r.asp_push, "push", r.asp_roi),
                (r.sp_n, r.sp_w, r.sp_l, r.sp_push, "push", r.sp_roi)):
            if not int(n):
                cells += ['<td>&mdash;</td>', '<td>&mdash;</td>']
                continue
            extra = f", {int(dead)} {dead_word}" if int(dead) else ""
            cls = "hit" if roi > 0 else "miss" if roi < 0 else ""
            cells += [f"<td>{int(w)}-{int(l)}{extra}</td>",
                      f'<td class="{cls}">{roi:+.1f}%</td>']
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out) or '<tr><td colspan="10">Nothing graded yet.</td></tr>'


def select_week(future):
    """The earliest NFL week that still has an unplayed game, and its caption.

    `future` is already filtered to games with no result whose gameday has not
    passed, which is what makes this roll over on its own. Two things fall out of
    that and both matter:

    Once every game in a week has a result the week vanishes from `future`, so
    the Tuesday rebuild after Monday Night Football moves the page on without
    anyone deciding it should.

    And the gameday filter covers for nflverse being slow. If Sunday's results
    have not been published yet those games still carry a null result, but their
    date has passed, so they are already excluded and the page still advances
    rather than showing a finished slate as though it were upcoming.

    A rolling window of days did neither, which is how week 2 fixtures appeared
    above week 1 games that had not kicked off.
    """
    if not len(future):
        return future, ""
    nxt = future.sort_values(["season", "week"]).iloc[0]
    week = future[(future["season"] == nxt["season"]) &
                  (future["week"] == nxt["week"])]
    # Monday leaves exactly one game in the week, Monday Night Football, which is
    # how "1 games" reached the page.
    count = f'{len(week)} game{"" if len(week) == 1 else "s"}'
    note = (f'<p class="sub">Week {int(nxt["week"])} of the '
            f'{int(nxt["season"])} season, {count}.</p>')
    return week, note


def build_parlays(upcoming_rows, top_n=10):
    legs = []
    for r in upcoming_rows:
        mu, sigma = r["mu"], r["sigma"]
        # Which team the leg is for is not enough to identify it. An eight day
        # horizon can show the same team twice, and "DET ML" then means either
        # of two games. The opponent disambiguates it in five characters.
        opp = {r["home"]: f" v {r['away']}", r["away"]: f" at {r['home']}"}
        p_home = norm.cdf(mu / sigma)
        if r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home")):
            side, p = (r["home"], p_home) if p_home >= 0.5 else (r["away"], 1 - p_home)
            price = r["mkt_home"] if side == r["home"] else 1 - r["mkt_home"]
            if 0.02 < price < 0.98:
                legs.append({"game": f"{r['away']}@{r['home']}",
                             "team": side, "desc": f"{side} ML{opp[side]}",
                             "p": p, "dec": 1 / (price + rd.kalshi_fee(price)),
                             "wild": square3_gap(p, price) > SQUARE3_SIGMAS})
        sl = r.get("spread_line")
        if sl is not None and not pd.isna(sl):
            p_cover_home = norm.cdf((mu - sl) / sigma)
            side, p = ((r["home"], p_cover_home) if p_cover_home >= 0.5
                       else (r["away"], 1 - p_cover_home))
            line = f"-{abs(sl):g}" if (side == r["home"]) == (sl > 0) else f"+{abs(sl):g}"
            # The line itself is the market's view of the margin, so a fair
            # line implies a 50% cover and the disagreement is just probit(p).
            # This leg used to be hardcoded wild=False, which let the model's
            # most extreme spread opinions into the credible bands unflagged.
            legs.append({"game": f"{r['away']}@{r['home']}",
                         "team": side, "desc": f"{side} {line}{opp[side]}",
                         "p": p, "dec": SPREAD_JUICE,
                         "wild": square3_gap(p, 0.5) > SQUARE3_SIGMAS})
        tmu, tsig, tl = r.get("tot_mu"), r.get("tot_sigma"), r.get("total_line")
        if (tmu is not None and not pd.isna(tmu)
                and tl is not None and not pd.isna(tl)):
            p_over, p_under, _ = prob_total_over(tmu, tsig, float(tl))
            ou, p = (("Over", float(p_over)) if p_over >= p_under
                     else ("Under", float(p_under)))
            # A fair total implies a coin flip either way, so the disagreement is
            # probit(p), the same shape as the spread leg. The team field is the
            # game itself: a total is not a side, and using one of the two team
            # names here would let a parlay pair the over with that team's
            # moneyline as though they were independent.
            legs.append({"game": f"{r['away']}@{r['home']}",
                         "team": f"{r['away']}@{r['home']} total",
                         "desc": f"{ou} {float(tl):g} ({r['away']} at {r['home']})",
                         "p": p, "dec": SPREAD_JUICE,
                         "wild": square3_gap(p, 0.5) > SQUARE3_SIGMAS})
    parlays = []
    for k in (2, 3):
        for combo in itertools.combinations(legs, k):
            # One leg per game, and one per team. The game guard alone would
            # let a team that plays twice in the window appear on both sides of
            # a parlay, and those legs are not independent: the same roster,
            # form and injuries drive both, so multiplying their probabilities
            # would overstate the parlay's chance of landing.
            if len({c["game"] for c in combo}) < k:
                continue
            if len({c["team"] for c in combo}) < k:
                continue
            p = float(np.prod([c["p"] for c in combo]))
            dec = float(np.prod([c["dec"] for c in combo]))
            wild = any(c["wild"] for c in combo)
            if wild or dec > 11:
                band = "moon"
            elif dec <= 2.2:
                band = "safe"
            elif dec <= 4.0:
                band = "balanced"
            else:
                band = "long"
            parlays.append({"legs": ", ".join(c["desc"] for c in combo),
                            "n": k, "p": p, "payout_dec": dec,
                            "ev": p * dec - 1, "band": band})
    out = []
    for band, key in (("safe", lambda x: x["p"]),
                      ("balanced", lambda x: x["ev"]),
                      ("long", lambda x: x["ev"]),
                      ("moon", lambda x: x["ev"])):
        rows = sorted([x for x in parlays if x["band"] == band],
                      key=key, reverse=True)[:top_n]
        out.extend(rows)
    return out


CSS = """
:root{--bg:#0a0e0b;--panel2:#131a15;--white:#f2f5f2;
  --green:#2ee06f;--yellow:#f5c542;--red:#e5533d;
  --dim:#f2f5f2;--line:#20291f}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--white);
  font:15px/1.5 "Segoe UI",system-ui,sans-serif;margin:0}
.wrap{max-width:920px;margin:0 auto;padding:18px 14px 40px}
h1{font-family:"Arial Narrow","Segoe UI",sans-serif;font-size:1.7rem;
  text-transform:uppercase;letter-spacing:.05em}
h1 span{color:var(--green)}
.sub{color:var(--dim);font-size:.85rem;margin:2px 0 10px}
nav{position:sticky;top:0;z-index:10;background:var(--bg);
  border-bottom:1px solid var(--line);display:flex;gap:6px;padding:8px 14px;
  overflow-x:auto;justify-content:center}
nav button{background:none;border:1px solid var(--white);color:var(--white);
  padding:7px 16px;border-radius:99px;font:600 .85rem "Segoe UI",sans-serif;
  cursor:pointer;white-space:nowrap}
nav button.on{background:var(--green);border-color:var(--green);color:#08120b}
.panel{display:none}.panel.on{display:block}
.grid{display:grid;gap:10px;align-items:start}
@media(min-width:700px){.grid{grid-template-columns:1fr 1fr}}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:11px 13px}
.match{display:flex;justify-content:space-between;align-items:baseline}
.teams{font-family:"Arial Narrow",sans-serif;font-size:1.05rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.02em}
.date{color:var(--dim);font-size:.75rem}
/* Model line and market line: same shape, values right aligned so the two
   numbers sit directly above one another and compare at a glance. */
/* Inline, not a two-column grid. The number belongs against the words that
   name it; pushing it to the far edge made the reader's eye travel for it. The
   space that frees up pays for a larger figure. */
.lrow{padding:2px 0;line-height:1.4}
.llab{color:var(--dim);font-size:.78rem;text-transform:uppercase;
  letter-spacing:.03em}
.lval{font-size:1.14rem;font-weight:700;color:var(--green);margin-left:3px;
  font-variant-numeric:tabular-nums}
.lval.vval{color:var(--white)}
.lnote{color:var(--dim);font-size:.72rem;margin-left:3px}
/* The model's outright call, always shown, value or not. */
/* The outright call's probability sits beside the team, readable at a glance
   but a step below it, so the pick itself stays the loudest thing in the row. */
.lpct{color:var(--white);font-size:.92rem;font-weight:600;margin-left:3px}
.lpct b{color:var(--green)}
.reason{background:none;border:1px solid var(--line);border-radius:8px;
  padding:5px 9px;margin:9px 0 0;overflow-x:auto}
.reason summary{font-size:.75rem;color:var(--green);font-weight:600}
.rplain{font-size:.8rem;line-height:1.5;margin:7px 0 6px;color:var(--white)}
.rplain b{color:var(--green)}
.rlead{color:var(--dim);font-size:.68rem;margin:6px 0 5px;line-height:1.35}
.rtab{width:100%;border-collapse:collapse;font-size:.7rem;margin:0}
.rtab td{padding:3px 4px;border-bottom:1px solid #17201850;vertical-align:top}
.rpull{white-space:nowrap;font-weight:700;font-variant-numeric:tabular-nums}
.rhome{color:var(--green)}
.raway{color:var(--white)}
.rnil{color:var(--dim);font-weight:400}
.rname{white-space:nowrap;color:var(--white)}
.rval{display:block;color:var(--dim);font-weight:400;
  font-variant-numeric:tabular-nums}
.rdesc{color:var(--dim);line-height:1.3}
/* The value recommendation is one of the four bold rows, so it takes .lval's
   size and weight; only the "nothing to bet" and "(small edge)" notes step down. */
.pnone{font-size:.85rem;font-weight:600;color:var(--dim)}
.psmall{font-size:.7rem;font-weight:600;color:var(--dim)}
.bars{display:grid;gap:4px;margin:4px 0 2px}
.brow{display:grid;grid-template-columns:96px 1fr 44px;align-items:center;
  gap:8px;font-size:.7rem}
.blab{color:var(--dim);text-transform:uppercase;letter-spacing:.04em}
.btrack{height:10px;background:#1a231c;border-radius:5px;overflow:hidden}
.bfill{height:100%;border-radius:5px}
.bfill.model{background:var(--green)}
.bfill.mkt{background:var(--white)}
.bval{text-align:right;font-variant-numeric:tabular-nums}
.gap{font-size:.72rem;margin-top:5px;color:var(--dim)}
.gap b{font-variant-numeric:tabular-nums}
.sq3{color:var(--yellow)}
.sprow{margin-top:3px}
.date{white-space:nowrap}
.verdict{display:inline-block;margin-top:7px;padding:2px 10px;border-radius:99px;
  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.v-high{background:var(--green);color:#08120b}
.v-caut{background:var(--yellow);color:#141005}
.v-avoid{background:none;border:1px solid var(--red);color:var(--red)}
.v-none{background:none;border:1px solid var(--dim);color:var(--dim)}
.bandbar{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.bandbtn{background:none;border:1px solid var(--white);color:var(--white);
  padding:5px 14px;border-radius:99px;font:600 .8rem "Segoe UI",sans-serif;cursor:pointer}
.bandbtn.on{background:var(--green);border-color:var(--green);color:#08120b}
.bandbtn .cnt{opacity:.6;font-weight:400;margin-left:5px;
  font-variant-numeric:tabular-nums}
.bandbtn.on .cnt{opacity:.75}
.bandbtn:disabled{opacity:.35;cursor:default}
.nomatch{color:var(--dim);font-size:.85rem;margin:10px 0}
h2{font-family:"Arial Narrow",sans-serif;font-size:1.15rem;text-transform:uppercase;
  color:var(--green);margin:18px 0 8px}
h3{font-family:"Arial Narrow",sans-serif;font-size:.98rem;text-transform:uppercase;
  color:var(--white);margin:18px 0 4px;letter-spacing:.04em}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin:8px 0}
.xscroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.xscroll table{min-width:640px}
th{color:var(--green);text-align:left;font-weight:600;padding:6px 7px;
  border-bottom:1px solid var(--line);text-transform:uppercase;font-size:.72rem;
  letter-spacing:.04em}
td{padding:4px 7px;border-bottom:1px solid #17201850;
  font-variant-numeric:tabular-nums}
.hit{color:var(--green)}.miss{color:var(--red)}
.ev-hi{color:var(--green);font-weight:700}.ev-md{color:var(--yellow)}
.ev-lo{color:var(--red)}
details{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:10px 14px;margin:8px 0}
summary{cursor:pointer;font-weight:600;color:var(--green);font-size:.9rem}
.b101 p{margin:10px 0;font-size:.92rem}
.b101 b{color:var(--green)}
.b101 table{max-width:640px}
footer{color:var(--dim);font-size:.78rem;margin-top:26px}
"""

TABS_JS = """
function band(b){
  document.querySelectorAll('.prow').forEach(r=>{
    r.style.display=(b==='all'||r.classList.contains('band-'+b))?'':'none';});
  document.querySelectorAll('#parlay-bands .bandbtn').forEach(
    x=>x.classList.remove('on'));
  document.getElementById('band-'+b).classList.add('on');
}
function tier(t){
  var shown=0;
  document.querySelectorAll('.gcard').forEach(c=>{
    var hit=(t==='all'||c.classList.contains('tier-'+t));
    c.style.display=hit?'':'none';
    if(hit){shown++;}});
  document.querySelectorAll('#week-tiers .bandbtn').forEach(
    x=>x.classList.remove('on'));
  document.getElementById('tier-'+t).classList.add('on');
  var e=document.getElementById('week-empty');
  if(e){e.style.display=shown?'none':'';}
}
function tab(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  document.getElementById('b-'+id).classList.add('on');
  window.scrollTo(0,0);
}
"""

PRICES_JS = """
/* Fast layer. The page ships with the model's probabilities baked in and the
   market prices as of the last full build. This refetches only the market side
   from prices.json and recombines the two in the browser, so a page built on
   Wednesday still shows Sunday's prices. Everything here is a no-op if the
   fetch fails: the prices baked in at build time simply stand. */
var TIERS = ['high', 'small', 'none', 'nopr'];
var TIER_CLS = {high: 'v-high', small: 'v-caut', none: 'v-avoid', nopr: 'v-none'};
function feeOf(p){ return 0.07 * p * (1 - p); }
/* Square-3 threshold, in units of the model's own predictive sd. Must match
   SQUARE3_SIGMAS in website.py: the card would otherwise caution at build time
   and stop cautioning the moment prices refreshed. */
var SQUARE3_SIGMAS = 0.40;
/* Inverse normal CDF (Acklam's rational approximation). Needed because the
   disagreement is measured in sigmas, and for a Gaussian predictive margin that
   is the difference of probits. Accurate to ~1e-9, far beyond what a caution
   threshold needs. */
function probit(p){
  if (p <= 0) return -38; if (p >= 1) return 38;
  var a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
           1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00],
      b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
           6.680131188771972e+01, -1.328068155288572e+01],
      c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
           -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00],
      d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
           3.754408661907416e+00];
  var lo = 0.02425, q, r;
  if (p < lo){
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  if (p > 1 - lo){
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  q = p - 0.5; r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q /
         (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}
/* Mirrors square3_text() in website.py word for word. */
function square3Text(z, sigma){
  return 'Model and market are about <b>' + Math.round(z * sigma) +
    '</b> points apart on this game, against a typical miss of &plusmn;' +
    Math.round(sigma) + '. A gap that size usually means the model has not seen ' +
    'some news, so check injuries and inactives before acting on the pick above.';
}
function tierOfVerdict(v){
  if (v.indexOf('HIGH VALUE') === 0) return 'high';
  if (v.indexOf('CAUTIOUS') === 0) return 'small';
  if (v.indexOf('NO VALUE') === 0) return 'none';
  return 'nopr';
}
/* Same three cases as game_card's Python prose, kept in step deliberately: if
   these ever diverge the page would explain a verdict it is not showing. */
function recountTiers(){
  var counts = {high: 0, small: 0, none: 0, nopr: 0}, total = 0;
  document.querySelectorAll('.gcard').forEach(function(c){
    total++;
    TIERS.forEach(function(t){ if (c.classList.contains('tier-' + t)) counts[t]++; });
  });
  var all = document.getElementById('tier-all');
  if (all && all.querySelector('.cnt')) all.querySelector('.cnt').textContent = total;
  TIERS.forEach(function(t){
    var b = document.getElementById('tier-' + t);
    if (!b) return;
    if (b.querySelector('.cnt')) b.querySelector('.cnt').textContent = counts[t];
    b.disabled = counts[t] === 0;
    /* Never strand the reader on a filter that just emptied. */
    if (counts[t] === 0 && b.classList.contains('on')) tier('all');
  });
}
function applyCard(card, events){
  var trip = (card.dataset.keys || '').split(',');
  var ev = null, ac = null, hc = null;
  for (var i = 0; i < trip.length && !ev; i++){
    var parts = trip[i].split(':');
    if (parts.length === 3 && events[parts[0]]){
      ev = events[parts[0]]; ac = parts[1]; hc = parts[2];
    }
  }
  if (!ev) return false;
  var home = card.dataset.home, away = card.dataset.away;
  var pm = parseFloat(card.dataset.p);
  var hp = ev[hc], ap = ev[ac];
  var hAsk = hp && hp.ask != null ? hp.ask : null;
  var aAsk = ap && ap.ask != null ? ap.ask : null;
  if (hAsk === null) return false;

  var eHome = pm - hAsk - feeOf(hAsk);
  var eAway = aAsk === null ? -1 : (1 - pm) - aAsk - feeOf(aAsk);
  var best = Math.max(eHome, eAway);
  var side = eHome >= eAway ? home : away;
  var verdict, t;
  if (best > 0.04){ verdict = 'HIGH VALUE &mdash; ' + side; t = 'high'; }
  else if (best > 0){ verdict = 'CAUTIOUS &mdash; small edge on ' + side; t = 'small'; }
  else { verdict = 'NO VALUE at current price'; t = 'none'; }

  /* Mirrors game_card: both bars follow the model's favourite, which is the
     team every sentence on the card is about. Fixing them to the home side made
     a card argue for one team while showing two numbers for the other. */
  var focus = pm >= 0.5 ? home : away;
  var pFocus = pm >= 0.5 ? pm : 1 - pm;
  var mFocus = focus === home ? hAsk : aAsk;
  var mbar = card.querySelector('.brow:not(.mktrow) .bfill.model');
  var mval = card.querySelector('.brow:not(.mktrow) .bval');
  if (mbar) mbar.style.width = (pFocus * 100).toFixed(1) + '%';
  if (mval) mval.innerHTML = Math.round(pFocus * 100) + '%';

  var bar = card.querySelector('.mktrow .bfill.mkt');
  var val = card.querySelector('.mktrow .bval');
  if (mFocus != null){
    if (bar) bar.style.width = (mFocus * 100).toFixed(1) + '%';
    if (val) val.innerHTML = Math.round(mFocus * 100) + '&cent;';
  }

  var gap = card.querySelector('.gaptxt');
  if (gap && mFocus != null){
    var g = (pFocus - mFocus) * 100;
    gap.innerHTML = 'Both bars: chance <b>' + focus + '</b> wins. Disagreement: <b>' +
      (g >= 0 ? '+' : '') + g.toFixed(0) + '</b> points of probability ' +
      (g > 0 ? 'toward ' : 'against ') + focus;
  }
  var badge = card.querySelector('.verdict');
  if (badge){ badge.className = 'verdict ' + TIER_CLS[t]; badge.innerHTML = verdict; }
  /* The picks row is the loudest thing on the card, so it has to move with the
     price. The spread pick is model versus the Vegas line and never changes. */
  var plist = card.querySelector('.plist');
  if (plist){
    var picks = [];
    if (t === 'high') picks.push(side + ' ML');
    else if (t === 'small') picks.push(side +
      ' ML <span class="psmall">(small edge)</span>');
    if (card.dataset.sp) picks.push(card.dataset.sp);
    plist.innerHTML = picks.length ? picks.join(' &middot; ')
      : '<span class="pnone">None at current prices</span>';
  }
  /* The moneyline gap moves with the price, so the caution has to move with it
     too. The spread gap is model versus the Vegas line and rides in data-zsp. */
  var sq = card.querySelector('.sq3');
  if (sq){
    var vModel = side === home ? pm : 1 - pm;
    var vMkt = side === home ? hAsk : aAsk;
    var zMl = (vMkt == null || !picks.length) ? 0 : probit(vModel) - probit(vMkt);
    var zMax = Math.max(zMl, parseFloat(card.dataset.zsp || '0'));
    if (zMax > SQUARE3_SIGMAS){
      sq.innerHTML = square3Text(zMax, parseFloat(card.dataset.sigma));
      sq.hidden = false;
    } else {
      sq.hidden = true;
    }
  }
  TIERS.forEach(function(x){ card.classList.remove('tier-' + x); });
  card.classList.add('tier-' + t);
  return true;
}
function applyPrices(){
  if (typeof PRICES_URL !== 'string' || !PRICES_URL) return;
  fetch(PRICES_URL, {cache: 'no-store'}).then(function(r){
    if (!r.ok) throw new Error('http ' + r.status);
    return r.json();
  }).then(function(d){
    var events = (d && d.events) || {};
    var n = 0;
    document.querySelectorAll('.gcard').forEach(function(c){
      if (applyCard(c, events)) n++;
    });
    recountTiers();
    var el = document.getElementById('price-age');
    /* No game matched the feed: the page is still showing build-time prices,
       so it must not claim to be live. */
    if (el && d.ts && n > 0){
      var age = (Date.now() - Date.parse(d.ts)) / 60000;
      /* A clock time, not "5 min ago". A reader checking a price wants to know
         which moment it belongs to, and a relative figure quietly goes stale in
         a tab left open. The date is added once it is not today. */
      var at = new Date(Date.parse(d.ts));
      var sameDay = at.toDateString() === new Date().toDateString();
      var when = (sameDay ? '' : at.toLocaleDateString([], {weekday: 'short',
                    month: 'short', day: 'numeric'}) + ', ')
                 + at.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit',
                    timeZoneName: 'short'});
      /* The feed is republished every 10 minutes by the logging machine, with
         a GitHub Actions job as a backstop. Anything past 20 minutes means the
         primary publisher is down, so the page stops calling itself live rather
         than dressing up an old number: a reader deciding on a price deserves
         to know it is not the current one. 45 minutes is a stall. */
      var warn = age > 45
        ? ' <span style="color:var(--yellow)">(the price feed has stalled; ' +
          'treat these as indicative and check the market yourself)</span>'
        : '';
      var lead = age < 20 ? 'Market prices are live, updated <b>'
                          : 'Market prices are not current, last updated <b>';
      el.innerHTML = lead + when + '</b> for ' + n +
        ' of ' + document.querySelectorAll('.gcard').length +
        ' games. Model probabilities are from the last full rebuild.' + warn;
    }
  }).catch(function(){ /* keep the prices baked in at build time */ });
}
/* Kickoffs are published as real instants, in UTC. Render them on the reader's
   own clock: the build machine's timezone is nobody else's business, and "4:25
   PM ET" is a small sum to do in your head on the way to a decision. */
function localiseKickoffs(){
  document.querySelectorAll('.gcard .date[data-kick]').forEach(function(el){
    var iso = el.dataset.kick;
    if (!iso) return;
    var d = new Date(iso);
    if (isNaN(d.getTime())) return;          /* keep the server's ET text */
    el.textContent =
      d.toLocaleDateString([], {weekday: 'short', month: 'short', day: 'numeric'}) +
      ', ' + d.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit',
                                       timeZoneName: 'short'});
  });
}
/* Same treatment for the header stamps: published as instants, rendered on the
   reader's clock, and left exactly as the server wrote them if anything fails. */
function localiseStamps(){
  document.querySelectorAll('.stamp[data-stamp]').forEach(function(el){
    var d = new Date(el.dataset.stamp);
    if (isNaN(d.getTime())) return;
    el.textContent =
      d.toLocaleDateString([], {month: 'short', day: 'numeric', year: 'numeric'}) +
      ' at ' + d.toLocaleTimeString([], {hour: 'numeric', minute: '2-digit',
                                         timeZoneName: 'short'});
  });
}
function bootWeek(){ localiseKickoffs(); localiseStamps(); applyPrices(); }
if (document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', bootWeek);
} else { bootWeek(); }
/* Refresh while the tab sits open, and again when the reader returns to it. The
   CDN caps real freshness at 5 minutes, so polling faster would be wasted. */
setInterval(applyPrices, 300000);
document.addEventListener('visibilitychange', function(){
  if (!document.hidden) applyPrices();
});
"""

MLB_JS = r"""
/* MLB tab. Read-only. Every number here is GooseLine's model, fetched at page
   load from their repo; nothing in this file computes a baseball prediction and
   nothing on the NFL side depends on it. If their feed is unreachable the tab
   says so and the rest of the page is unaffected. */
function mlbCsv(text){
  var lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  var head = lines[0].split(',');
  var rows = [];
  for (var i = 1; i < lines.length; i++){
    /* The note column is free text and may be quoted with embedded commas. */
    var cells = [], cur = '', q = false, ln = lines[i];
    for (var j = 0; j < ln.length; j++){
      var ch = ln[j];
      if (ch === '"'){ if (q && ln[j+1] === '"'){ cur += '"'; j++; } else q = !q; }
      else if (ch === ',' && !q){ cells.push(cur); cur = ''; }
      else cur += ch;
    }
    cells.push(cur);
    var o = {};
    for (var k = 0; k < head.length; k++) o[head[k]] = cells[k];
    rows.push(o);
  }
  return rows;
}
function mlbNum(v){ var f = parseFloat(v); return isNaN(f) ? null : f; }
function mlbBadge(v){
  if (!v) return '<span class="verdict v-none">No price</span>';
  if (v.indexOf('HIGH VALUE') === 0) return '<span class="verdict v-high">' + v + '</span>';
  if (v.indexOf('CAUTIOUS') === 0) return '<span class="verdict v-caut">' + v + '</span>';
  if (v.indexOf('NO VALUE') === 0) return '<span class="verdict v-avoid">' + v + '</span>';
  return '<span class="verdict v-none">' + v + '</span>';
}
function mlbCard(r){
  var p = mlbNum(r.p_home), mk = mlbNum(r.mkt_home), mu = mlbNum(r.mu);
  var sg = mlbNum(r.sigma), edge = mlbNum(r.edge);
  if (p === null) return '';
  var call = mu === null ? '' :
    (mu >= 0 ? 'GooseLine model: <b>' + r.home + ' by ' + Math.abs(mu).toFixed(1) + '</b>'
             : 'GooseLine model: <b>' + r.away + ' by ' + Math.abs(mu).toFixed(1) + '</b>') +
    (sg === null ? '' : ' &plusmn;' + sg.toFixed(1) + ' runs');
  var bars =
    '<div class="brow"><span class="blab">GooseLine model</span>' +
    '<div class="btrack"><div class="bfill model" style="width:' + (p*100).toFixed(1) + '%"></div></div>' +
    '<span class="bval">' + Math.round(p*100) + '%</span></div>' +
    '<div class="brow"><span class="blab">Market price</span><div class="btrack">' +
    (mk === null ? '' : '<div class="bfill mkt" style="width:' + (mk*100).toFixed(1) + '%"></div>') +
    '</div><span class="bval">' + (mk === null ? '&mdash;' : Math.round(mk*100) + '&cent;') + '</span></div>';
  var sp = (r.away_sp || r.home_sp)
    ? '<div class="gap">Starters: ' + (r.away_sp || 'TBD') + ' at ' + (r.home_sp || 'TBD') +
      (r.sp_unknown === 'True' ? ' <span style="color:var(--yellow)">(a starter was unannounced, so the range is widened)</span>' : '') + '</div>'
    : '';
  var eg = (edge === null || mk === null) ? '' :
    '<div class="gap">Edge after fees: <b>' + (edge >= 0 ? '+' : '') + (edge*100).toFixed(1) + '%</b></div>';
  var verdict = (r.verdict && r.verdict !== 'pass') ? r.verdict
              : (mk === null ? '' : 'NO VALUE at current price');
  return '<div class="card"><div class="match"><span class="teams">' + r.away + ' @ ' +
    r.home + '</span><span class="date">' + r.date + '</span></div>' +
    '<div class="call">' + call + '</div><div class="bars">' + bars + '</div>' +
    sp + eg + mlbBadge(verdict) + '</div>';
}
function loadMlb(){
  var host = document.getElementById('mlb-slate');
  var meta = document.getElementById('mlb-meta');
  if (!host || typeof MLB_FEED_URL !== 'string' || !MLB_FEED_URL) return;
  fetch(MLB_FEED_URL, {cache: 'no-store'}).then(function(r){
    if (!r.ok) throw new Error('http ' + r.status);
    return r.text();
  }).then(function(t){
    var rows = mlbCsv(t);
    if (!rows.length) throw new Error('empty feed');
    /* Show only the newest slate, and only games not yet settled. */
    var latest = rows.reduce(function(m, r){ return r.date > m ? r.date : m; }, '');
    var slate = rows.filter(function(r){ return r.date === latest; });
    var open = slate.filter(function(r){ return !r.result; });
    var show = open.length ? open : slate;
    var cards = show.map(mlbCard).filter(Boolean).join('');
    host.innerHTML = cards || '<p class="sub">No MLB games on the latest slate.</p>';
    var settled = rows.filter(function(r){ return r.result; }).length;
    if (meta){
      meta.innerHTML = 'Slate of <b>' + latest + '</b>, ' + show.length + ' games' +
        (open.length ? '' : ' (all settled)') + '. Feed carries ' + settled +
        ' settled games since ' + rows[0].date + '.';
    }
  }).catch(function(e){
    host.innerHTML = '<p class="sub">The GooseLine MLB feed is unreachable right now (' +
      e.message + '). This tab is the only thing affected; the NFL model on this ' +
      'site does not depend on it.</p>';
    if (meta) meta.innerHTML = '';
  });
}
if (document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', loadMlb);
} else { loadMlb(); }
"""

B101 = """
<div class="b101">
<p><b>The one idea behind everything here.</b> A prediction is not a number, it is a
range of belief. This model never says "the Chiefs will win by 2." It says "our best
guess is Chiefs by 2, and here is exactly how sure we are." Bayesian modeling is the
math of keeping honest track of that sureness: start with a reasonable belief, let
each game's evidence pull it, and never claim more certainty than the evidence paid
for.</p>
<p><b>Why every prediction says plus-or-minus 13.</b> That 13 is measured, not
assumed. Take every NFL game since 2010, compare the final margin to the best
pre-game prediction anyone can make, and the typical miss is about 13 points. Vegas,
with all its money and information, also misses by about 13. Football is decided by
fumbles, tipped balls, and kickers, and no film study predicts those. The skill is
not shrinking the 13; it is knowing your 13 honestly while the market prices as if
it were smaller or larger.</p>
<p><b>Even the model's weights are probabilities.</b> A normal model learns one
number for how much each factor matters, say "quarterback continuity is worth 2.4
points," and treats it as gospel. This model refuses to commit: every learned weight
is a bell curve, a most likely value plus honest error bars, because 16 seasons of
noisy football cannot pin any of these numbers down exactly. To make a prediction it
samples a full set of plausible weights, forming one plausible analyst, and repeats
until it has a room of them. Where the room agrees, the data genuinely supports the
call. Where the room scatters, the model is telling you it does not understand this
game, and no bet should survive that.</p>
<p><b>The Kalman filter: the model's memory.</b> Every team carries a rating, points
better or worse than average on a neutral field, updated after every game by a
Kalman filter, the same math that navigated Apollo to the moon. The filter's genius
is knowing how far to move: a 20-point blowout by a team it already trusts barely
moves the rating; the same blowout by a mystery team moves it a lot. Between
seasons every rating shrinks 30% toward average and its uncertainty balloons,
because offseasons erase certainty. Home field advantage is learned, not assumed,
and converged to almost exactly 2 points.</p>
<p><b>How it all intertwines.</b> The pipeline is a relay. The Kalman filter watches
scores and maintains ratings with error bars. Those ratings, their uncertainty, and
the stat sheet below become the inputs to the room of five neural networks, which
learn the patterns and produce a margin and a per-game uncertainty. A calibration
step then nudges that uncertainty against held-out seasons so the stated confidence
matches reality. Finally the finished probability is compared to the market's price,
minus fees, and only survivors of that comparison and a human news check appear as
value on the This Week tab.</p>
<p><b>The stat sheet.</b> What the model is actually fed, for every game, always
computed only from games played before it:</p>
<table>
<tr><th>Input</th><th>What it is</th></tr>
<tr><td>Kalman rating gap</td><td>Team strength difference, points on a neutral field</td></tr>
<tr><td>Rating uncertainty</td><td>How well the filter currently knows both teams</td></tr>
<tr><td>Scoring form</td><td>Recent point differential, recent games weighted more</td></tr>
<tr><td>EPA, passing and rushing, offense and defense</td><td>Expected Points Added per play: how much each play helped, given down, distance, and field position. Credits efficiency, not raw yards</td></tr>
<tr><td>CPOE</td><td>Completion Percentage Over Expected: does the QB complete throws harder than they look</td></tr>
<tr><td>QB continuity</td><td>Share of the last 16 games started by this week's listed starter. A backup shows up as a 0</td></tr>
<tr><td>Rest difference</td><td>Days off, home minus away</td></tr>
<tr><td>Division game</td><td>Rivals play closer games than ratings suggest</td></tr>
<tr><td>Roof</td><td>Indoor or outdoor stadium</td></tr>
</table>
<p>Notably absent: yards per carry, yards after catch, and other yardage stats.
EPA already contains what they measure, with the context they lack, so adding them
would add noise, not knowledge. Also deliberately absent: betting lines. A model
fed the market's answer can only agree with the market; this one has to form its
own opinion so the two can genuinely disagree.</p>
<p><b>From margin to money.</b> A predicted margin plus its uncertainty gives the
chance of any outcome: the chance the margin beats zero is the moneyline, the
chance it beats the spread is the cover probability. A market price is also a
probability, since 65 cents means 65%. Value exists only when the model's number
and the price disagree by more than the fees, in a game where the room agrees.
Most weeks that is a short list. That is the design working, not failing.</p>
<p><b>The one part of this site that is not Bayesian.</b> Over/unders are priced
by a different model, on purpose. The margin question is "who wins and how sure are
we", where the whole value of the answer is the honesty of the uncertainty, and
that is what the Bayesian machinery above is for. A total is a different animal:
the thing being predicted is a count, its spread barely changes from game to game,
and what matters is the shape of the whole distribution rather than a belief about
weights. So the engines were made to compete on one measure, CRPS, which scores an
entire predicted distribution against the single number that actually happened, and
the winner shipped. The room of neural networks lost. Boosted trees lost, and lost
by more the more capacity they were given. A large pretrained tabular transformer,
a model that has never seen a football game and does its learning inside a single
forward pass over the data you hand it, finished a hair ahead on one season and
inside the noise on the next. What won was a plain weighted least squares fit on
pace, points per drive, both offences, both defences and the weather. Two lessons
worth keeping: more model is not more accuracy, and a method earns its place by
measurement, not by being the most interesting one in the room.</p>
<p><b>What this model honestly cannot see.</b> Injuries announced this week,
coaches resting starters, weather. Every green badge gets a human news check before
anything happens. When the market disagrees with the model, the market is usually
right; this system exists to find the exceptions and to know the difference.</p>
</div>"""



def ats_label(x):
    if pd.isna(x.spread_line):
        return "&mdash;"
    if x.ats_pick == x.home_team:
        line = f"-{abs(x.spread_line):g}" if x.spread_line > 0 else f"+{abs(x.spread_line):g}"
    else:
        line = f"-{abs(x.spread_line):g}" if x.spread_line < 0 else f"+{abs(x.spread_line):g}"
    res = ("HIT" if x.ats_correct == 1 else "miss" if x.ats_correct == 0 else "push")
    return f"{x.ats_pick} {line} &middot; {res}"


DEFAULT_MLB_FEED = ("https://raw.githubusercontent.com/jdev-02/"
                    "gooseline-model-hq/main/data/mlb/narrative/log.csv")
DEFAULT_PRICES_URL = ("https://raw.githubusercontent.com/HowlsCastle97/"
                      "nfl-model-hq/prices/prices.json")


def kalshi_lookup(away, home):
    """Alias candidates as 'eventkey:awaycode:homecode' triples for the page JS.

    Mirrors rundown.match_event: the browser tries each candidate in turn and
    uses the first Kalshi event that exists, so LA/LAR and JAX/JAC resolve the
    same way client-side as they do at build time.
    """
    out = []
    for a in rd.KALSHI_ALIASES.get(away, [away]):
        for h in rd.KALSHI_ALIASES.get(home, [home]):
            out.append(f"{a}{h}:{a}:{h}")
    return ",".join(out)


def verdict_tier(v):
    """Bucket a verdict string into the class used by the This Week filter."""
    if v.startswith("HIGH VALUE"):
        return "high"
    if v.startswith("CAUTIOUS"):
        return "small"
    if v.startswith("NO VALUE"):
        return "none"
    return "nopr"


TIER_BUTTONS = (("high", "High value"), ("small", "Small edge"),
                ("none", "No value"), ("nopr", "No price"))


def verdict_badge(v):
    if v.startswith("HIGH VALUE"):
        return f'<span class="verdict v-high">{v}</span>'
    if v.startswith("CAUTIOUS"):
        return f'<span class="verdict v-caut">{v}</span>'
    if v.startswith("NO VALUE"):
        return f'<span class="verdict v-avoid">{v}</span>'
    return '<span class="verdict v-none">No price yet</span>'


def square3_text(z, sigma):
    """Caution copy for a square-3 sized disagreement, in points a bettor reads."""
    if z <= SQUARE3_SIGMAS:
        return ""
    return (f'Model and market are about <b>{z * sigma:.0f}</b> points apart on '
            f'this game, against a typical miss of &plusmn;{sigma:.0f}. A gap '
            f'that size usually means the model has not seen some news, so check '
            f'injuries and inactives before acting on the pick above.')


# Display name and plain-English meaning for every input, keyed by the column
# name in rd.V3_COLS. Same wording as the stat sheet in Bayesian 101, so a
# reader who learns the term in one place recognises it in the other.
FEATURE_INFO = {
    "kalman_diff": ("Kalman rating gap",
                    "Team strength difference, points on a neutral field"),
    "kalman_var": ("Rating uncertainty",
                   "How well the filter currently knows both teams"),
    "pdiff_ewma_diff": ("Scoring form",
                        "Recent point differential, recent games weighted more"),
    "off_pass_diff": ("Passing offence, EPA",
                      "Expected Points Added per dropback: how much each play "
                      "helped, given down, distance and field position"),
    "off_rush_diff": ("Rushing offence, EPA",
                      "Expected Points Added per carry. Credits efficiency, "
                      "not raw yards"),
    "def_pass_diff": ("Pass defence, EPA",
                      "Expected Points Added given up per opponent dropback"),
    "def_rush_diff": ("Rush defence, EPA",
                      "Expected Points Added given up per opponent carry"),
    "cpoe_diff": ("CPOE",
                  "Completion Percentage Over Expected: does the QB complete "
                  "throws harder than they look"),
    "rest_diff": ("Rest difference", "Days off, home minus away"),
    "div_game": ("Division game",
                 "Rivals play closer games than ratings suggest"),
    "qb_fam_diff": ("QB continuity",
                    "Share of the last 16 games started by this week's listed "
                    "starter. A backup shows up as a 0"),
    "indoor": ("Roof", "Indoors removes weather and tends to raise scoring"),
}


def feature_value(col, v):
    """The raw input in the unit it is actually measured in."""
    if col == "div_game":
        return "yes" if v >= 0.5 else "no"
    if col == "indoor":
        return "indoor" if v >= 0.5 else "outdoor"
    if col == "rest_diff":
        return f"{v:+.0f} days"
    if col == "kalman_var":
        return f"{v:.0f}"
    return f"{v:+.2f}"


def season_context(df, season, stats_path="team_game_stats.csv",
                   qb_path="qb_game_stats.csv"):
    """Plain season-to-date facts for the reasoning prose, per team and per passer.

    None of this is a model input and the panel says so. It exists because "the
    better side" is a claim with the evidence taken out, and a reader who can see
    12 points a game against 34, or six interceptions against two, can judge the
    model's read instead of taking it on faith.

    Turnovers in particular are shown and not modelled. An interception is already
    inside the EPA numbers the model does use, charged at what the play cost, so a
    separate turnover input would count it twice. Quoting the count is honest;
    feeding it in would not be.

    Falls back to the previous season for a team with nothing played yet, flagged
    so the prose can say which year it is describing.
    """
    played = df[(df["season"] == season) & df["result"].notna()]
    use, past = season, False
    if played.empty:
        use, past = season - 1, True
        played = df[(df["season"] == use) & df["result"].notna()]

    teams = {}
    for t in sorted(set(played["home_team"]) | set(played["away_team"])):
        rows = played[(played["home_team"] == t) | (played["away_team"] == t)]
        at_home = rows["home_team"] == t
        margin = np.where(at_home, rows["result"], -rows["result"])
        pf = np.where(at_home, rows["home_score"], rows["away_score"])
        pa = np.where(at_home, rows["away_score"], rows["home_score"])
        teams[t] = {"games": len(rows), "w": int((margin > 0).sum()),
                    "l": int((margin < 0).sum()), "t": int((margin == 0).sum()),
                    "pf": float(np.mean(pf)), "pa": float(np.mean(pa)),
                    "margin": float(np.mean(margin)), "season": use, "past": past}

    try:
        st = pd.read_csv(stats_path)
        st = st[st["game_id"].str.startswith(f"{use}_")]
        if "giveaways" in st.columns:
            agg = st.groupby("team")[["giveaways", "takeaways"]].sum()
            for t, r in agg.iterrows():
                if t in teams:
                    teams[t]["giveaways"] = int(r["giveaways"])
                    teams[t]["takeaways"] = int(r["takeaways"])
    except Exception:
        pass

    qbs = {}
    try:
        q = pd.read_csv(qb_path)
        q = q[q["game_id"].str.startswith(f"{use}_")]
        if "interceptions" in q.columns:
            for pid, r in q.groupby("passer_id").agg(
                    dropbacks=("dropbacks", "sum"), qb_epa=("qb_epa", "sum"),
                    ints=("interceptions", "sum"),
                    name=("passer", "last")).iterrows():
                if r["dropbacks"] >= 20:
                    qbs[pid] = {"dropbacks": int(r["dropbacks"]),
                                "epa_per": float(r["qb_epa"] / r["dropbacks"]),
                                "ints": int(r["ints"]), "name": str(r["name"]),
                                "season": use}
    except Exception:
        pass
    return teams, qbs


def plain_summary(vals, con, home, away, mu, hqb, aqb, played=0,
                  fam_h=None, fam_a=None, lev=None, ctx=None, qbx=None):
    """The same arithmetic as the table, told the way you would tell a friend.

    Two rules keep it honest.

    Tense follows the evidence. In Week 1 every input is last season's, so the
    prose says so and stays in the past; it moves to the present once these two
    have played enough games this season for the EWMA to be describing now.
    Saying "have been getting more out of each carry" about a season that has
    not started is a small lie that costs a lot of trust.

    Nothing is claimed that the feature does not measure. qb_fam_diff is the
    share of a team's last 16 starts belonging to this week's listed starter, so
    a low number means the model has seen little of him lately, which can simply
    mean he was hurt. It does NOT mean a backup is playing, and an earlier
    version of this text said exactly that about Joe Burrow, who started 7 of
    Cincinnati's last 16 after an injury and is very much their starter.

    Every claim carries its number. This used to say things like "X were simply
    the better side", which is an assertion with the evidence deleted: a reader
    could not tell whether the model was looking at a two point edge or a ten
    point one, nor check it against what they had watched. Now each clause quotes
    the figure it rests on, per team rather than as a gap, and names the passer
    where there is a passer to name.

    lev holds each side's own EPA levels, ctx the season-to-date record and
    turnovers, qbx the listed starters' own numbers. All three are optional: with
    none of them the prose degrades to the shape it had before rather than
    failing, which is what the card tests without a full frame rely on.
    """
    g = dict(zip(rd.V3_COLS, con))
    v = dict(zip(rd.V3_COLS, vals))
    qb = {home: hqb, away: aqb}
    past = played == 0
    era = ("last season" if past
           else "so far this season" if played < 4 else "this season")
    themes = {
        "class": g["kalman_diff"],
        "air": g["off_pass_diff"] + g["def_pass_diff"] + g["cpoe_diff"],
        "ground": g["off_rush_diff"] + g["def_rush_diff"],
        "form": g["pdiff_ewma_diff"],
        "qb": g["qb_fam_diff"],
        "spot": g["rest_diff"] + g["div_game"] + g["indoor"] + g["kalman_var"],
    }
    def who(x):
        return (home, away) if x > 0 else (away, home)

    lev = lev or {}
    ctx = ctx or {}
    qbx = qbx or {}
    # Tense follows the evidence, the same rule the older clauses kept by hand.
    # In week 1 every number here is last season's and the prose has to say so in
    # the verb as well as in the era, or it describes a season that has not
    # happened in the present tense.
    IS = "was" if past else "is"
    ARE = "were" if past else "are"
    HAS = "had" if past else "has"

    def side_of(team, stat):
        """One team's own level of an EPA stat, or None if the frame lacks it."""
        key = f"{stat}_{'home' if team == home else 'away'}"
        v = lev.get(key)
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    def pair(team, opp, stat, fmt="{:+.2f}", unit="", floor=0.03):
        """"+0.30 against -0.09" for the two sides of one stat, team first.

        Returns nothing when the two sides are within `floor` of each other. A
        sentence calling one defence "the leakier" off +0.01 against -0.01 is
        technically true and rhetorically false, and the reader cannot tell which
        without the numbers, which is the whole reason the numbers are here.
        """
        a, b = side_of(team, stat), side_of(opp, stat)
        if a is None or b is None or abs(a - b) < floor:
            return ""
        return f"{fmt.format(a)}{unit} against {fmt.format(b)}{unit}"

    # How much of the efficiency read is still last season. The EWMA half-life is
    # eight games, so after n games this much weight remains on what came before.
    carry = 0.5 ** (played / 8.0) if played else 1.0

    # Whether this card was given the per side levels at all. It matters because
    # an empty `pair` means two different things: no numbers to quote, in which
    # case the older vague wording is the best available, or numbers that are too
    # close to be worth a sentence, in which case the sentence should not exist.
    has_lev = any(side_of(home, st) is not None for st in EPA_STATS)

    def qb_gap():
        """This season's gap between the two listed starters, per dropback."""
        a = qbx.get(qbx.get("id_home"))
        b = qbx.get(qbx.get("id_away"))
        if not a or not b:
            return None
        return a["epa_per"] - b["epa_per"]

    def record(team):
        c = ctx.get(team)
        if not c or not c.get("games"):
            return ""
        wl = f"{c['w']}-{c['l']}" + (f"-{c['t']}" if c.get("t") else "")
        return (f"{wl}, scoring {c['pf']:.0f} a game and allowing {c['pa']:.0f}")

    def giveaway_clause(team, opp):
        a, b = ctx.get(team, {}), ctx.get(opp, {})
        if "giveaways" not in a or "giveaways" not in b:
            return ""
        # Symmetric and countable, and silent when the two are level: "2 to 2"
        # is not a fact worth a sentence.
        if a["giveaways"] == b["giveaways"]:
            return ""
        have = "had" if past else "have"
        return (f"{team} {have} {a['giveaways']} giveaways {era} to "
                f"{opp}'s {b['giveaways']}")

    def qb_clause(team):
        """The listed starter's own production, named, or an empty string."""
        pid = qbx.get("id_home" if team == home else "id_away")
        rec = qbx.get(pid) if pid else None
        if not rec:
            return ""
        ints = (f", {rec['ints']} interception" + ("s" if rec["ints"] != 1 else "")
                if rec.get("ints") is not None else "")
        # A quarterback sitting a thousandth below zero is at zero, and "-0.00"
        # reads as a direction the number does not support.
        per = rec["epa_per"]
        per = 0.0 if abs(per) < 0.005 else per
        return (f"{rec['name']} {IS} at {per:+.2f} expected points per "
                f"dropback over {rec['dropbacks']} of them{ints}")

    out = []
    for name, tot in sorted(themes.items(), key=lambda kv: -abs(kv[1])):
        if abs(tot) < 0.3 or len(out) >= 3:
            continue
        t, opp = who(tot)
        # Plural decided from the string actually rendered, not a second
        # rounding of the float: 0.95 displays as 0.9 while round(0.95, 1) is
        # 1.0, which is how "about 0.9 point" once reached the page.
        shown = f"{abs(tot):.1f}"
        pts = f"about {shown} {'point' if shown == '1.0' else 'points'}"
        if name == "class":
            gap = abs(v["kalman_diff"])
            recs = [x for x in (record(t), record(opp)) if x]
            if len(recs) == 2:
                body = f"{t} {ARE} {recs[0]}; {opp} {ARE} {recs[1]}"
            else:
                body = (f"the ratings separate them by {gap:.1f} points on a "
                        f"neutral field")
            give = giveaway_clause(t, opp)
            tail = f". {give[0].upper()}{give[1:]}." if give else "."
            # Not `carry`: that name already holds how much of the read is last
            # season's, and shadowing it turned a number into a verb.
            carry_v = "carried" if past else "carry"
            # A rating gap that rounds to nothing is not "0.0 points of it", it is
            # two teams the filter cannot separate, and the sentence has to say
            # the second thing.
            held = (f"{gap:.1f} points of it on a neutral field and {pts} once "
                    f"the filter's uncertainty is taken into account"
                    if gap >= 0.05 else
                    f"worth {pts} here, almost all of it the filter's uncertainty "
                    f"rather than the ratings, which have them level")
            out.append(f"<b>{t}</b> {carry_v} the better team rating {era}, "
                       f"{held}: {body}{tail}")
        elif name == "air":
            bits = []
            if (v["off_pass_diff"] > 0) == (t == home) and v["off_pass_diff"]:
                num = pair(t, opp, "off_epa_pass")
                if num:
                    bits.append(f"{t}'s passing offence {IS} at {num} expected "
                                f"points per dropback {era}")
                elif not has_lev:
                    bits.append(f"{t} {ARE} the more efficient passing team {era}")
            if (v["def_pass_diff"] > 0) == (opp == home) and v["def_pass_diff"]:
                num = pair(opp, t, "def_epa_pass")
                if num:
                    bits.append(f"the {opp} pass defence {IS} the leakier, "
                                f"conceding {num} per throw")
                elif not has_lev:
                    bits.append(f"the {opp} pass defence {IS} the leakier of the two")
            if (v["cpoe_diff"] > 0) == (t == home) and v["cpoe_diff"]:
                num = pair(t, opp, "cpoe", "{:+.1f}", "%", floor=1.0)
                completed = "completed" if past else "completing"
                if num:
                    bits.append(f"{t} {completed} {num} above expectation")
                elif not has_lev:
                    bits.append(f"{t} {completed} more than expected")
            q_t, q_o = qb_clause(t), qb_clause(opp)
            if q_t and q_o:
                bits.append(f"{q_t}, while {q_o}")
            elif q_t:
                bits.append(q_t)
            body = "; ".join(bits) if bits else f"the passing matchup tilts {t}"
            # The honest awkward case: the model leans one way on passing while
            # this season's quarterback play points the other. That is not a
            # contradiction to hide, it is the eight game half-life showing, and
            # the reader is owed the number.
            gap = qb_gap()
            clash = ""
            if gap is not None and not past and played:
                leans_home = (t == home)
                if (gap > 0.05) != leans_home and abs(gap) > 0.05:
                    clash = (" That cuts against the lean: on this season's "
                             "play alone the quarterback gap runs the other way, "
                             "and the efficiency numbers above are still mostly "
                             "last season's.")
            out.append(f"Through the air it is worth {pts} to <b>{t}</b>: "
                       f"{body}.{clash}")
        elif name == "ground":
            bits = []
            if (v["off_rush_diff"] > 0) == (t == home) and v["off_rush_diff"]:
                num = pair(t, opp, "off_epa_rush")
                if num:
                    bits.append(f"{t} {ARE} at {num} expected points per carry {era}")
                elif not has_lev:
                    bits.append("they got more out of each carry" if past else
                                "they have been getting more out of each carry")
            if (v["def_rush_diff"] > 0) == (opp == home) and v["def_rush_diff"]:
                num = pair(opp, t, "def_epa_rush")
                conc = "conceded" if past else "concedes"
                gave = "was" if past else "has been"
                if num:
                    bits.append(f"the {opp} run defence {conc} {num}")
                elif not has_lev:
                    bits.append(f"the {opp} run defence {gave} giving it up")
            body = "; ".join(bits) if bits else f"the run matchup tilts {t}"
            out.append(f"On the ground it is worth {pts} to <b>{t}</b>: {body}.")
        elif name == "form":
            a, b = ctx.get(t, {}), ctx.get(opp, {})
            if a.get("games") and b.get("games"):
                fav = "favoured" if past else "favours"
                out.append(f"Recent scoring {fav} <b>{t}</b> by {pts}: "
                           f"{a['margin']:+.1f} points a game {era} against "
                           f"{opp}'s {b['margin']:+.1f}.")
            elif past:
                out.append(f"<b>{t}</b> outscored people down the stretch last "
                           f"season while {opp} did not, {pts} of it.")
            else:
                out.append(f"<b>{t}</b> have been outscoring people lately while "
                           f"{opp} have not, {pts} of it.")
        elif name == "qb":
            # Continuity, stated as continuity. Never as a benching.
            nt = None if fam_h is None else round(
                (fam_h if t == home else fam_a) * 16)
            no = None if fam_h is None else round(
                (fam_a if t == home else fam_h) * 16)
            tq, oq = qb.get(t), qb.get(opp)
            if nt is not None and tq and oq:
                own = qb_clause(t)
                extra = f" For what it is worth, {own}." if own else ""
                out.append(f"The model has seen more of <b>{t}</b>'s quarterback, "
                           f"{pts}: {tq} started {nt} of their last 16 games while "
                           f"{oq} started {no} of {opp}'s, so it has a firmer read "
                           f"on one than the other.{extra}")
            else:
                out.append(f"The model has seen more of <b>{t}</b>'s listed "
                           f"starter than {opp}'s in recent games, {pts}.")
        else:
            why = []
            if abs(g["rest_diff"]) > 0.1:
                why.append("the rest edge")
            if abs(g["div_game"]) > 0.1:
                why.append("a division game")
            if abs(g["indoor"]) > 0.1:
                why.append("the roof")
            if abs(g["kalman_var"]) > 0.1:
                why.append("how little the model still knows about these two")
            if len(why) > 1:
                why = [", ".join(why[:-1]) + " and " + why[-1]]
            out.append(f"Situationally it leans <b>{t}</b> by {pts}: "
                       f"{why[0] if why else 'the spot'}.")
    lead = ("Nothing has been played yet this season, so this is all last "
            "year's evidence. " if past else
            (f"Only {played} games have been played this season, so about "
             f"{carry*100:.0f}% of the model's efficiency read is still last "
             f"season's. " if 0 < played < 6 else ""))
    if not out:
        # Still says what it is working from. A reader in week 1 deserves the
        # basis even when the answer is "these two look level".
        return (f'<p class="rplain">{lead}Nothing here moves the needle much. '
                f'The model has these two close to level and the line reflects '
                f'that.</p>')
    s = half_point(mu)
    side = home if s >= 0 else away
    tail = (f" Add it up and the model wants <b>{side} -{abs(s):g}</b>."
            if s else " Add it up and the model has it a pick'em.")
    return f'<p class="rplain">{lead}{" ".join(out)}{tail}</p>'


def reasoning_panel(cols, values, contribs, home, away, mu, hqb="", aqb="",
                    lev=None, ctx=None, qbx=None, totals=None,
                    played=0, fam_h=None, fam_a=None):
    """Per-game breakdown of what is moving the prediction, biggest first.

    Sorted by size rather than listed in column order, because the point is
    immediacy: the reader should see the driver of this game in the first row
    without reading the rest.
    """
    rows = sorted(zip(cols, values, contribs), key=lambda t: -abs(t[2]))
    body = []
    for col, val, c in rows:
        label, desc = FEATURE_INFO.get(col, (col, ""))
        if abs(c) < 0.05:
            pull = '<span class="rnil">no effect</span>'
        else:
            team = home if c > 0 else away
            cls = "rhome" if c > 0 else "raway"
            pull = f'<span class="{cls}">{abs(c):.1f} to {team}</span>'
        body.append(f'<tr><td class="rpull">{pull}</td>'
                    f'<td class="rname">{label}<span class="rval">'
                    f'{feature_value(col, val)}</span></td>'
                    f'<td class="rdesc">{desc}</td></tr>')
    # Same rounding as the line above it. Quoting mu to a different precision
    # here is how "by 5" once ended up printed above "-5.5" on the same card.
    s_line = half_point(mu)
    side = home if s_line >= 0 else away
    return (
        '<details class="reason"><summary>Reasoning</summary>'
        + plain_summary(values, contribs, home, away, mu, hqb, aqb,
                        played, fam_h, fam_a, lev, ctx, qbx) +
        '<p class="rlead">And the same thing as arithmetic, biggest first. Each '
        'number '
        'is how much the prediction would move if that one input were neutral '
        'instead of what it is, measured on the same ensemble that produced the '
        f'prediction above. They will not add up to the {abs(s_line):g} points '
        f'on {side}: the model is a network, not a sum, so each input is '
        'measured on its own.</p>'
        f'<table class="rtab">{"".join(body)}</table>{totals or ""}</details>')


def game_card(r):
    mu, sigma, pm = r["mu"], r["sigma"], r["p_home"]
    fav0 = r["home"] if pm >= 0.5 else r["away"]
    fav_p = pm if pm >= 0.5 else 1 - pm
    # The model's line and the market's line, same shape and stacked, so the
    # comparison is a glance rather than a sentence. These were two separately
    # worded lines that between them said the number twice and never showed the
    # thing it wanted comparing against.
    sl = r.get("spread_line")
    has_sl = sl is not None and not pd.isna(sl)
    lines = (
        f'<div class="lrow"><span class="llab">Bayesian prediction:</span> '
        f'<span class="lval">{fav_line(mu, r["home"], r["away"])}</span> '
        f'<span class="lnote">&plusmn;{sigma:.0f} &middot; fair ML '
        f'{american(fav_p)}</span></div>'
        f'<div class="lrow"><span class="llab">Vegas market:</span> '
        f'<span class="lval vval">'
        f'{fav_line(sl, r["home"], r["away"]) if has_sl else "&mdash;"}</span>'
        f'</div>')
    v0 = str(r.get("verdict", ""))
    ml_pick = ""
    vside = ""
    if "&mdash;" in v0 and (v0.startswith("HIGH VALUE") or v0.startswith("CAUTIOUS")):
        vside = v0.split("&mdash;")[-1].strip().replace("small edge on ", "")
        ml_pick = f'{vside} ML' + ('' if v0.startswith("HIGH VALUE")
                                   else ' <span class="psmall">(small edge)</span>')
    # The spread pick is computed once here because both the picks row and the
    # spread detail row below need it. It is model versus the Vegas line only,
    # so live Kalshi prices never change it; the page JS leaves it alone.
    sp_side = sp_ln = None
    sp_p = 0.0
    if has_sl:
        p_ch = norm.cdf((mu - sl) / sigma)
        sp_side, sp_p = ((r["home"], p_ch) if p_ch >= 0.5
                         else (r["away"], 1 - p_ch))
        sp_ln = (f"-{abs(sl):g}" if (sp_side == r["home"]) == (sl > 0)
                 else f"+{abs(sl):g}")
    sp_pick = f"{sp_side} {sp_ln}" if sp_side is not None and sp_p >= 0.58 else ""
    # Square-3 measured against whichever markets the picks actually face: the
    # Kalshi price for the moneyline, the Vegas line for the spread. Two markets,
    # so two gaps, and the louder one drives the caution.
    mkt_h = r.get("mkt_home")
    has_mkt = mkt_h is not None and not pd.isna(mkt_h)
    z_ml = 0.0
    if ml_pick and has_mkt:
        v_model = pm if vside == r["home"] else 1 - pm
        v_mkt = float(mkt_h) if vside == r["home"] else 1 - float(mkt_h)
        z_ml = square3_gap(v_model, v_mkt)
    z_sp = square3_gap(sp_p, 0.5) if sp_pick else 0.0
    z_max = max(z_ml, z_sp)
    sq3_row = (f'<div class="gap sq3"{"" if z_max > SQUARE3_SIGMAS else " hidden"}>'
               f'{square3_text(z_max, sigma)}</div>')
    picks = [x for x in (ml_pick, sp_pick) if x]
    # The model's outright call and the value call are different questions, and
    # the card used to answer the first only inside a sentence about the second:
    # "the model still makes LAC the winner, but the market charges too much".
    # Each now has its own row in the same bold block as the lines, so a reader
    # sees who the model thinks wins even on a game with nothing to bet, and
    # sees separately what, if anything, is worth buying.
    if abs(fav_p - 0.5) < 0.005:
        ml_row = ('<div class="lrow"><span class="llab">ML pick:</span> '
                  '<span class="lval">too close to call</span></div>')
    else:
        ml_row = (f'<div class="lrow"><span class="llab">ML pick:</span> '
                  f'<span class="lval">{fav0}</span> '
                  f'<span class="lpct">predicted to win <b>{fav_p*100:.0f}%</b> '
                  f'of the time</span></div>')
    if picks:
        value_html = ' &middot; '.join(picks)
    elif has_mkt:
        value_html = '<span class="pnone">None at current prices</span>'
    else:
        value_html = '<span class="pnone">No market price yet</span>'
    value_row = ('<div class="lrow"><span class="llab">Value recommendation:</span> '
                 f'<span class="lval plist">{value_html}</span></div>')
    # Both bars follow the team the rest of the card is about, which is the
    # model's favourite. They used to be fixed to the home side, so a card whose
    # every sentence discussed the away team showed two home-team numbers and
    # the reader had to invert them: the model reading 37% for CAR directly
    # under a sentence saying the model likes CHI at 63%.
    focus = fav0
    pf = pm if focus == r["home"] else 1 - pm
    model_bar = (f'<div class="brow"><span class="blab">Bayesian model</span>'
                 f'<div class="btrack"><div class="bfill model" '
                 f'style="width:{pf*100:.1f}%"></div></div>'
                 f'<span class="bval">{pf*100:.0f}%</span></div>')
    mkt_focus = (r.get("mkt_home") if focus == r["home"] else r.get("mkt_away"))
    if mkt_focus is not None and not pd.isna(mkt_focus):
        mk = float(mkt_focus)
        gap = (pf - mk) * 100
        mkt_bar = (f'<div class="brow mktrow"><span class="blab">Market price</span>'
                   f'<div class="btrack"><div class="bfill mkt" '
                   f'style="width:{mk*100:.1f}%"></div></div>'
                   f'<span class="bval">{mk*100:.0f}&cent;</span></div>')
        gaptxt = (f'<div class="gap gaptxt">Both bars: chance <b>{focus}</b> wins. '
                  f'Disagreement: <b>{gap:+.0f}</b> points of probability '
                  f'{"toward" if gap > 0 else "against"} {focus}</div>')
    else:
        mkt_bar = (f'<div class="brow mktrow"><span class="blab">Market price</span>'
                   f'<div class="btrack"><div class="bfill mkt" style="width:0%">'
                   f'</div></div><span class="bval">&mdash;</span></div>')
        gaptxt = '<div class="gap gaptxt">Market has not opened this game yet</div>'
    v = str(r["verdict"])
    spread_row = ""
    if sp_side is not None:
        side, p, line = sp_side, sp_p, sp_ln
        if p >= 0.58:
            tag = '<span class="hit">value at a book\'s -110</span>'
        elif p >= 0.545:
            tag = '<span style="color:var(--yellow)">slight lean at -110</span>'
        else:
            tag = 'no edge at -110'
        # The Vegas number is now on its own line above, so it is not repeated.
        # Sits directly under the two lines it refers to, so it needs no
        # preamble naming the market it is talking about.
        spread_row = (f'<div class="gap sprow">Model covers '
                      f'<b>{side} {line}</b> {p*100:.0f}% of the time &middot; '
                      f'fair price {american(p)} &middot; {tag}</div>')
    # Order: what the model thinks the game looks like, then what it would take,
    # then what is worth buying. The two totals rows join those same groups rather
    # than trailing after the value line, so the predictions read together and the
    # value row keeps the last word.
    ou_pred, ou_pick = ou_rows(r)
    return (f'<div class="card gcard tier-{verdict_tier(v)}" '
            f'data-keys="{kalshi_lookup(r["away"], r["home"])}" '
            f'data-away="{r["away"]}" data-home="{r["home"]}" '
            f'data-sp="{sp_pick}" data-zsp="{z_sp:.4f}" '
            f'data-sigma="{sigma:.3f}" data-p="{pm:.6f}">'
            f'<div class="match"><span class="teams">{r["away"]} @ '
            f'{r["home"]}</span><span class="date" '
            f'data-kick="{r.get("kick_iso", "")}">'
            f'{r.get("kick_txt") or r["date"]}</span></div>'
            f'{lines}{ou_pred}{ml_row}{ou_pick}{value_row}{spread_row}'
            f'<div class="bars">{model_bar}{mkt_bar}</div>{gaptxt}'
            f'{sq3_row}'
            f'{reasoning_panel(rd.V3_COLS, r.get("x", []), r.get("contrib", []), r["home"], r["away"], mu, r.get("hqb", ""), r.get("aqb", ""),
            r.get("lev"), r.get("ctx"), r.get("qbx"), totals_reason(r),
            r.get("played", 0), r.get("fam_h"), r.get("fam_a")) if len(r.get("contrib", [])) else ""}'
            f'{verdict_badge(v)}</div>')
def build_site(out_path="site.html", games_path="games.csv",
               stats_path="team_game_stats.csv", db_path="kalshi_prices.db",
               horizon_days=None, edge_threshold=0.04,
               prices_url=DEFAULT_PRICES_URL, mlb_feed=DEFAULT_MLB_FEED):
    df = rd.build_frame(games_path, stats_path)
    # Three different facts, and the header used to show only the weakest of them.
    # When the page was built is not when the data was pulled, and neither is the
    # same as how far the results actually run. A reader looking at a stale card
    # needs the last two, so all three are printed and the date now carries a time.
    built = pd.Timestamp.now(tz="UTC")
    try:
        pulled = pd.Timestamp(os.path.getmtime(games_path), unit="s", tz="UTC")
    except OSError:
        pulled = built
    _done_all = df[df["result"].notna()]
    through = ""
    if len(_done_all):
        _last = _done_all.sort_values("gameday").iloc[-1]
        through = (f"{int(_last['season'])} week {int(_last['week'])}, "
                   f"last game {_last['gameday'].date()}")
    hist, by_season, calib = history_tables(df)
    rec_week, rec_season = rec_records(hist, df)
    tot_hist = totals_history(df)

    today = pd.Timestamp.today().normalize()
    future = df[df["result"].isna() & (df["gameday"] >= today)]
    week_note = ""
    if horizon_days is None:
        upcoming, week_note = select_week(future)
    else:
        upcoming = future[future["gameday"] <= today + pd.Timedelta(days=horizon_days)]
        if len(upcoming) == 0 and len(future):
            first = future["gameday"].min()
            upcoming = future[future["gameday"] <= first + pd.Timedelta(days=6)]
            week_note = (f'<p class="sub">No games in the next {horizon_days} days; '
                         f'showing the next scheduled week '
                         f'({first.date()} onward).</p>')
    # Kickoff order, not file order. nflverse gametime is US Eastern, so it is
    # localised there and then carried to the browser as a real instant, which
    # lets each reader see the time on their own clock rather than mine.
    upcoming = upcoming.copy()
    times = (upcoming["gametime"].fillna("13:00")
             if "gametime" in upcoming.columns else "13:00")
    stamps = upcoming["gameday"].dt.strftime("%Y-%m-%d") + " " + times
    upcoming["kick_et"] = pd.to_datetime(stamps, errors="coerce").dt.tz_localize(
        ZoneInfo("America/New_York"), nonexistent="shift_forward", ambiguous=True)
    upcoming = upcoming.sort_values(["kick_et", "home_team"], kind="stable")
    week_rows, parlays = [], []
    price_age = ""
    if "week_note" not in dir():
        week_note = ""
    if len(upcoming):
        lin, ens = rd.fit_models(df, int(upcoming["season"].max()))
        Xu = upcoming[V3].values
        mu, sigma = ens.predict_dist(Xu)
        p_home = norm.cdf(mu / sigma)
        # Attribution for the Reasoning panel, against the same ensemble that
        # produced mu above rather than a linear stand-in for it.
        # How much of this season these two have actually played. Week 1 is
        # zero, and the summary has to say so rather than describing last year
        # in the present tense.
        cur = int(upcoming["season"].max())
        done_now = df[(df["season"] == cur) & df["result"].notna()]
        played_by = pd.concat([done_now["home_team"],
                               done_now["away_team"]]).value_counts().to_dict()
        _train = df[df["result"].notna()][V3].values
        contrib = rd.feature_contributions(ens, Xu, rd.neutral_row(_train))
        # The totals model is a separate engine with its own features, fitted on
        # the same completed games. Cheap enough to refit here every build.
        team_ctx, qb_ctx = season_context(df, cur, stats_path)
        lev_cols = [f"{st}_{side}" for st in EPA_STATS + PACE_STATS
                    for side in ("home", "away")]
        lev_cols = [c for c in lev_cols if c in upcoming.columns]
        tmodel = tot.fit_totals(df, cur)
        Xt = upcoming[tot.TOTAL_COLS].values
        tot_mu, tot_sigma = tmodel.predict_dist(Xt)
        # Exact, because the totals engine is a linear fit: base plus these terms
        # is the published number, with nothing left over.
        tot_c, tot_base, tot_mean = tot.contributions(tmodel, Xt)
        prices = rd.latest_prices(db_path)
        price_age = ""
        try:
            import sqlite3 as _sq
            _con = _sq.connect(db_path)
            _ts = _con.execute("SELECT MAX(ts_utc) FROM snapshots").fetchone()[0]
            _con.close()
            if _ts:
                _age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(_ts)
                hrs = _age.total_seconds() / 3600
                stale = (' <span style="color:var(--yellow)">(getting old; run the '
                         'logger for fresh prices)</span>' if hrs > 24 else "")
                price_age = (f'<p class="sub" id="price-age">Market prices last logged: '
                             f'{pd.Timestamp(_ts).strftime("%b %d, %H:%M UTC")}, '
                             f'about {hrs:.0f}h ago{stale}. "No price yet" means no '
                             f'price existed in that snapshot; the market may have '
                             f'opened since.</p>')
        except Exception:
            pass
        for j, row in enumerate(upcoming.itertuples(index=False)):
            _k = getattr(row, "kick_et", None)
            rec = {"date": row.gameday.date(),
                   "kick_iso": "" if pd.isna(_k) else _k.tz_convert("UTC")
                   .strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "kick_txt": (str(row.gameday.date()) if pd.isna(_k)
                                else _k.strftime("%a %d %b, %I:%M %p ET")
                                .replace(" 0", " ")),
                   "played": min(played_by.get(row.home_team, 0),
                                 played_by.get(row.away_team, 0)),
                   "lev": {c: getattr(row, c, None) for c in lev_cols},
                   "ctx": {t: team_ctx.get(t, {})
                           for t in (row.home_team, row.away_team)},
                   "qbx": {**qb_ctx,
                           "id_home": getattr(row, "home_qb_id", None),
                           "id_away": getattr(row, "away_qb_id", None)},
                   "tot_mu": float(tot_mu[j]), "tot_sigma": float(tot_sigma[j]),
                   "tot_x": Xt[j], "tot_c": tot_c[j], "tot_base": tot_base,
                   "tot_mean": tot_mean,
                   "total_line": getattr(row, "total_line", None),
                   "fam_h": getattr(row, "qb_fam_home", None),
                   "fam_a": getattr(row, "qb_fam_away", None),
                   "hqb": getattr(row, "home_qb_name", "") or "",
                   "aqb": getattr(row, "away_qb_name", "") or "",
                   "away": row.away_team,
                   "home": row.home_team, "mu": mu[j], "sigma": sigma[j],
                   "p_home": p_home[j], "mkt_home": None, "mkt_away": None,
                   "spread_line": getattr(row, "spread_line", None),
                   "x": Xu[j], "contrib": contrib[j],
                   "verdict": "no price"}
            ev = rd.match_event(prices, row.away_team, row.home_team)
            if ev:
                hp = ev.get(row.home_team)
                if hp and hp.get("ask") is not None:
                    rec["mkt_home"] = hp["ask"]
                    _ap = ev.get(row.away_team)
                    if _ap and _ap.get("ask") is not None:
                        rec["mkt_away"] = _ap["ask"]
                    e = p_home[j] - hp["ask"] - rd.kalshi_fee(hp["ask"])
                    ap = ev.get(row.away_team)
                    ea = ((1 - p_home[j]) - ap["ask"] - rd.kalshi_fee(ap["ask"])
                          if ap and ap.get("ask") is not None else -1)
                    best = max(e, ea)
                    side = row.home_team if e >= ea else row.away_team
                    if best > edge_threshold:
                        rec["verdict"] = f"HIGH VALUE &mdash; {side}"
                    elif best > 0:
                        rec["verdict"] = f"CAUTIOUS &mdash; small edge on {side}"
                    else:
                        rec["verdict"] = "NO VALUE at current price"
            week_rows.append(rec)
        parlays = build_parlays(week_rows)

    cards = "".join(game_card(r) for r in week_rows) or \
        '<p class="sub">No games in the upcoming window.</p>'

    if not price_age:
        price_age = ('<p class="sub" id="price-age">Market prices load from the '
                     'live feed, republished every 10 minutes. If this line '
                     'still says this after the page has loaded, the feed did '
                     'not answer and the prices shown are from the last '
                     'rebuild.</p>')

    tier_bar = ""
    if week_rows:
        counts = Counter(verdict_tier(str(r["verdict"])) for r in week_rows)
        btns = ['<button id="tier-all" class="bandbtn on" '
                "onclick=\"tier('all')\">All"
                f'<span class="cnt">{len(week_rows)}</span></button>']
        for key, label in TIER_BUTTONS:
            n = counts.get(key, 0)
            dis = "" if n else " disabled"
            btns.append(f'<button id="tier-{key}" class="bandbtn"{dis} '
                        f"onclick=\"tier('{key}')\">{label}"
                        f'<span class="cnt">{n}</span></button>')
        tier_bar = (f'<div class="bandbar" id="week-tiers">{"".join(btns)}</div>'
                    '<p class="nomatch" id="week-empty" style="display:none">'
                    'No games in that tier this week.</p>')


    live_seasons = {int(s) for s, r in by_season.iterrows() if r.live}
    srows = "".join(
        f'<tr><td>{s}{" (live)" if int(s) in live_seasons else ""}</td>'
        f'<td>{int(r.games)}</td><td>{r.winner_pct:.1f}%</td>'
        f'<td>{r.ats_pct:.1f}%</td><td>{r.avg_miss:.1f}</td></tr>'
        for s, r in by_season.iterrows())
    crows = "".join(
        f'<tr><td>{idx}</td><td>{int(r.games)}</td><td>{r.model_said:.0f}%</td>'
        f'<td>{r.home_actually_won:.0f}%</td></tr>'
        for idx, r in calib.iterrows())

    # The weekly record covers the season in progress, since that is the one a
    # reader is following. With nothing live, the newest finished season stands in.
    rrows_season = rec_rows(rec_season, "season")
    # Why a winning record still loses money, in the tab's own numbers. Every
    # judgement in this paragraph is computed, including the ones a writer would
    # normally supply: whether the model overstated itself, and which way the
    # market's price sits against the result. A sentence that says "honest to
    # within half a point" is a hostage to the next fifty games.
    calib_line = ""
    if len(rec_season):
        a = rec_season.iloc[-1]
        if not pd.isna(a.all_hit) and not pd.isna(a.all_imp):
            claim = a.all_said - a.all_hit
            if abs(claim) < 1.0:
                honesty = "which is its own forecast, near enough"
            elif claim > 0:
                honesty = f"so it overstated itself by {claim:.1f} points"
            else:
                honesty = f"so it understated itself by {-claim:.1f} points"
            edge = a.all_hit - a.all_imp
            if edge > 0:
                verdict = (f'That is {edge:.1f} points better than the price asked '
                           f'for, which is what a real edge looks like: the number '
                           f'to watch is whether it survives the next season.')
            else:
                verdict = (f'That gap is the whole game: the market charged for '
                           f'{a.all_imp:.1f}% and delivered {a.all_hit:.1f}%, so '
                           f'picking winners {a.all_hit:.0f}% of the time is not '
                           f'the same as making money. The model has to be right '
                           f'about a price, not just about a game.')
            calib_line = (
                f'<p class="sub"><b>Why the record and the return disagree.</b> '
                f'Across every season here the model said its outright picks would '
                f'win <b>{a.all_said:.1f}%</b> of the time. They won '
                f'<b>{a.all_hit:.1f}%</b>, {honesty}. The prices those picks were '
                f'bought at implied <b>{a.all_imp:.1f}%</b>. {verdict}</p>')
    week_season = max(live_seasons) if live_seasons else (
        int(rec_week["season"].max()) if len(rec_week) else None)
    rec_week_view = (rec_week[rec_week["season"] == week_season]
                     if week_season is not None else rec_week)
    rrows_week = rec_rows(rec_week_view, "week", lambda w: f"Week {int(w)}")
    week_rec_title = (f"{week_season} week by week" if week_season is not None
                      else "Week by week")

    # Quoted in the This Week copy. Taken from the table rather than typed, because
    # a hardcoded "best season" is a sentence that goes quietly wrong in October.
    tot_best_txt = "no season graded yet"
    if len(tot_hist):
        _b = tot_hist.loc[tot_hist["ou_pct"].idxmax()]
        tot_best_txt = f"best season {_b.ou_pct:.1f}% in {int(_b.season)}"
    trows = "".join(
        f'<tr><td>{int(r.season)}{" (live)" if int(r.season) in live_seasons else ""}</td>'
        f'<td>{int(r.games)}</td><td>{r.mean_total:.1f}</td>'
        f'<td>{r.mean_actual:.1f}</td><td>{r.avg_miss:.1f}</td>'
        f'<td>{r.ou_pct:.1f}% <span class="psmall">on {int(r.graded)}</span></td></tr>'
        for r in tot_hist.itertuples(index=False)) or \
        '<tr><td colspan="6">Nothing graded yet.</td></tr>'

    prow_html = []
    for p in parlays:
        cls = "ev-hi" if p["ev"] > 0.04 else "ev-md" if p["ev"] > 0 else "ev-lo"
        prow_html.append(
            f'<tr class="prow band-{p["band"]}"><td>{p["legs"]}</td>'
            f'<td>{p["p"]*100:.0f}%</td>'
            f'<td>{100/p["payout_dec"]:.0f}%</td>'
            f'<td>{american(1/p["payout_dec"])}</td>'
            f'<td class="{cls}">{"+$" + format(p["ev"]*10, ".2f") if p["ev"] >= 0 else "-$" + format(abs(p["ev"])*10, ".2f")}</td></tr>')
    prows = "".join(prow_html) or \
        '<tr><td colspan="5">Needs upcoming games and prices.</td></tr>'

    season_blocks = []
    for season, g in hist.groupby("season"):
        rows = "".join(
            f'<tr><td>W{int(x.week)}</td><td>{x.away_team} @ {x.home_team}</td>'
            f'<td>{fav_line(x.mu, x.home_team, x.away_team)}</td>'
            f'<td>{fav_line(x.spread_line, x.home_team, x.away_team) if not pd.isna(x.spread_line) else "&mdash;"}</td>'
            f'<td>{x.p_home*100:.0f}%</td>'
            f'<td>{"+" if x.y > 0 else ""}{x.y:.0f} ({x.home_team} {x.home_score:.0f} - {x.away_team} {x.away_score:.0f})</td>'
            f'<td class="{"hit" if x.su_correct else "miss"}">'
            f'{x.su_pick} &middot; {"HIT" if x.su_correct else "miss"}</td>'
            f'<td class="{"hit" if x.ats_correct == 1 else "miss" if x.ats_correct == 0 else ""}">'
            f'{ats_label(x)}</td></tr>'
            for x in g.itertuples(index=False))
        season_blocks.append(
            f'<details><summary>{season} season'
            f'{" (live, updates weekly)" if int(season) in live_seasons else ""}'
            f' &mdash; every game ({len(g)})</summary><table><tr><th>Wk</th><th>Game</th>'
            f'<th>Bayesian Model line</th><th>Vegas line</th>'
            f'<th>Model: home team wins</th><th>Final margin (home score first)</th>'
            f'<th>Bayesian Model winner pick</th><th>Bayesian Model spread pick</th></tr>{rows}</table></details>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Model HQ</title><style>{CSS}</style>
<script>{TABS_JS}</script>
<script>var PRICES_URL={prices_url!r};var MLB_FEED_URL={mlb_feed!r};</script>
<script>{PRICES_JS}</script>
<script>{MLB_JS}</script></head><body>
<nav><button id="b-week" class="on" onclick="tab('week')">This Week</button>
<button id="b-parlays" onclick="tab('parlays')">Parlay Lab</button>
<button id="b-record" onclick="tab('record')">Track Record</button>
<button id="b-b101" onclick="tab('b101')">Bayesian 101</button>
<button id="b-mlb" onclick="tab('mlb')">MLB</button></nav>
<div class="wrap">
<h1>NFL <span>Model</span> HQ</h1>
<p class="sub">A Bayesian margin model &middot; built
<span class="stamp" data-stamp="{built.strftime('%Y-%m-%dT%H:%M:%SZ')}">{stamp_text(built)}</span>
&middot; schedule and results pulled
<span class="stamp" data-stamp="{pulled.strftime('%Y-%m-%dT%H:%M:%SZ')}">{stamp_text(pulled)}</span>{(', complete through ' + through) if through else ''}</p>
{price_age}

<div id="week" class="panel on">
<h2>This Week</h2>
<p class="sub">Green bar: the Bayesian Model's chance the team named under the
bars wins, which is whichever side the model favours. White bar: what the market
charges for that same outcome, so the two are always directly comparable. Badges: green means real value
after fees, yellow means an edge too small to trust, red means the price is fair
or worse. Each card also grades the Vegas spread: the model's chance of covering
each side, and whether that beats the 52.4% needed to profit at a standard -110.
Every green light still gets a human news check first.</p>
<p class="sub">New: the over/under line on each card, from a separate totals model.
It is graded by the same words as the spread row, but it has not earned the same
trust: across every season back to {TRACK_FIRST_SEASON} it has never beaten the
52.4% break-even against the closing total, {tot_best_txt}. Treat a green tag there
as the model's strongest lean on a total, not as an edge. The Track Record tab
keeps its scorecard.</p>
{week_note}{tier_bar}<div class="grid">{cards}</div>
</div>

<div id="parlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub">Now including over/unders, priced at a standard -110 like the
spread legs and flagged by the same missing news rule. Combinations of moneylines (at logged Kalshi prices, fees
included) and spreads (at the standard -110). Pick your risk appetite:
<b>Safe</b> caps the payout near +120 and ranks by hit chance; these are legs
where the model and the market mostly agree, so expect them to land often but
carry little or no edge; you are paying the fees for the fun, not beating anyone.
<b>Balanced</b> (+120 to +300) and <b>Longshot</b> (+300 to +1000) rank by the
model's expected profit, which is where real disagreements live. <b>Moonshot</b>
is the lottery-ticket tier: payouts above +1000, including legs where the model
disagrees with the market by an amount too large to fully trust, since a gap
that big often means the model is missing news rather than the market giving
money away. Moonshot numbers are the model at its least reliable; bet them for
the sweat, not the math. Read each row as: the model says this combo hits X%,
the market's prices imply Y%, and the last column is the average result of a $10 bet, blending
wins and losses at the model's probability: it is not what a winning ticket pays
(the Payout column is), it is what the bet earns or costs on average if you made
it many times. Positive means the model thinks you are being paid to take the
bet; negative means you are paying for the entertainment. Legs
are independent games only, and parlays multiply the house's cut as well as the
thrill.</p>
<div class="bandbar" id="parlay-bands">
<button id="band-safe" class="bandbtn" onclick="band('safe')">Safe &le;+120</button>
<button id="band-balanced" class="bandbtn" onclick="band('balanced')">Balanced</button>
<button id="band-long" class="bandbtn" onclick="band('long')">Longshot</button>
<button id="band-moon" class="bandbtn" onclick="band('moon')">Moonshot +1000</button>
<button id="band-all" class="bandbtn on" onclick="band('all')">All</button>
</div>
<table><tr><th>Legs</th><th>Model chance</th><th>Market implied chance</th>
<th>Payout</th><th>Avg profit per $10 bet, win or lose</th></tr>
{prows}</table>
</div>

<div id="record" class="panel">
<h2>Track Record</h2>
<p class="sub">Everything on this tab is stated from the home team's perspective:
a positive margin means the home team won by that much, and every probability is
the home team's chance of winning. Every prediction below was made by the Bayesian
Model before it had seen the game: each week it trains only on games already
played, exactly as it runs live. It is the same deep ensemble, with the same
uncertainty calibration, that makes this week's picks, so this tab grades the
predictions you are actually reading. Seasons before 2021 are excluded because the model's settings were chosen
using that era. The current season is the only fully honest test, since every
earlier season existed while the model was being built. Two separate report cards: Both scorecards below belong to the Bayesian
Model, never to Vegas: "winner pick" is the model picking the game outright, and
"spread pick" is the model's chosen side against the Vegas closing line (the pick
is spelled out in each row, for example "LAC +3"). 52.4% against the spread is
break-even at standard juice.</p>
<table><tr><th>Season</th><th>Games</th><th>Bayesian Model picks winner</th>
<th>Model vs the spread</th><th>Avg miss (pts)</th></tr>{srows}</table>
<p class="sub">Honesty check. Everything in this table is from the home team's
point of view. Take all the games where the Bayesian Model gave the home team a
certain range of winning chances, then check how often the home team really won.
A single game cannot test a probability, since it either happens or it
does not; only a pile of games can. So games are grouped by what the model
predicted. Reading the first row: take every game where the model gave the home
team somewhere in the 0-35% range; those predictions averaged 28%, and home teams
in that pile actually won 34% of the time. The test is always the last two columns
against each other: when they roughly match in every row, the model's stated
confidence is honest.</p>
<table><tr><th>Games grouped by prediction</th><th>Games</th>
<th>The group's predictions averaged</th><th>Home teams in the group actually won</th></tr>
{crows}</table>

<h3>Recommendation record</h3>
<p class="sub">The two tables above grade predictions. These two grade
<b>recommendations</b>: what would have happened to a flat one unit bet on every
pick this site would have published, week by week. ROI is profit divided by the
amount staked, so +5% means five cents back on every dollar risked and -5% means
five cents gone.<br><br>
<b>Every ML pick</b> is the model's outright call, the side it makes the favourite,
bought at that side's closing price whatever the price was. That is the "ML pick"
row on every card, and most of them are not value picks: the model often likes a
team the market likes more. It is here because those picks are published and can
be acted on, so they should be graded.<br><br>
<b>ML value picks only</b> is the narrower subset where the model's win probability
beat what the market charged, which is what earns the green badge. Same games, same
prices, a filter on top.<br><br>
The price used for both is the closing Vegas moneyline, because Kalshi prices have
only been logged here since 2026 and a record starting this September would say
nothing at all. Two honest consequences: the Vegas price includes the book's cut,
which makes this a slightly harder test than a Kalshi ask, and Kalshi's 7% fee on
winnings is not deducted here. Read it as the record of the model's rules, not a
transcript of a Kalshi account. An outright tie voids the bet.<br><br>
<b>Every spread pick</b> is the model's side against the Vegas closing line in
every game that had one. It is the same thing the accuracy table above reports as a
percentage, shown here as money.<br><br>
<b>Spread picks the card printed</b> is the subset where the model gave its side a
58% chance or better, which is exactly when a spread pick appears on a card. Both
are priced at standard -110, where a win returns 0.909 per unit risked and 52.4% is
break even, and pushes are voided rather than counted as half a win.<br><br>
So two of these four columns are filtered and two are not, which is the comparison
worth reading: a filter earns its place only by beating the unfiltered version next
to it.<br><br>
Both counts include picks the card flagged with the missing news caution, since
the caution is a warning and not a veto. And the warning that matters most: a
week holds a handful of bets, so a weekly ROI of plus or minus 40% is what noise
looks like at this sample size, not a hot or cold streak. The All seasons row is
the only line with enough bets to mean much, and even that one is thin.</p>
<div class="xscroll"><table><tr><th>Season</th><th>Games</th>
<th>Every ML pick</th><th>ROI</th><th>ML value picks only</th><th>ROI</th>
<th>Every spread pick</th><th>ROI</th><th>Spread picks the card printed</th>
<th>ROI</th></tr>{rrows_season}</table></div>
{calib_line}
<h3>{week_rec_title}</h3>
<div class="xscroll"><table><tr><th>Week</th><th>Games</th>
<th>Every ML pick</th><th>ROI</th><th>ML value picks only</th><th>ROI</th>
<th>Every spread pick</th><th>ROI</th><th>Spread picks the card printed</th>
<th>ROI</th></tr>{rrows_week}</table></div>

<h3>Totals model scorecard</h3>
<p class="sub">This one is not the Bayesian Model and does not belong in the
columns above. Over/unders are priced by a separate, simpler engine, picked by
walk-forward score against a deep ensemble, boosted trees and a large pretrained
tabular transformer; the plain one won, so the plain one ships. Same honesty rules:
every number below was predicted before the game, and a season is graded only on
games with a posted closing total, pushes dropped the way a book drops them.<br><br>
Read the last column first. 52.4% is break even at a standard -110, and the totals
model has never reached it. That is the point of showing the scorecard next to the
over/under row on the cards: the model has a number for every game, and no
demonstrated edge on any of them.</p>
<table><tr><th>Season</th><th>Games</th><th>Model's average total</th>
<th>Actual average</th><th>Avg miss (pts)</th>
<th>Beat the closing total</th></tr>{trows}</table>
{"".join(season_blocks)}
</div>

<div id="mlb" class="panel">
<h2>MLB &middot; from GooseLine</h2>
<p class="sub">This tab is not my model. It is <a href="https://jdev-02.github.io/gooseline-model-hq/"
style="color:var(--green)">GooseLine Solutions' MLB model</a>, loaded live from
their public data when this page opens. I do not build, tune, or vouch for the
baseball numbers; they are here so both sports sit in one place. One difference
from the NFL tab: these bars are always the <b>home</b> team's chance, because
this feed is read as published rather than rebuilt here. Green is that model's
chance the home team wins, white is what the market charges. Baseball is much closer to a coin flip than
football, so edges are smaller and rarer, and the same rule applies: a green
badge is a starting point for a news check, not a bet.</p>
<p class="sub" id="mlb-meta"></p>
<div class="grid" id="mlb-slate"><p class="sub">Loading the GooseLine feed&hellip;</p></div>
<p class="sub" style="margin-top:14px">Source:
<a href="https://github.com/jdev-02/gooseline-model-hq" style="color:var(--green)">jdev-02/gooseline-model-hq</a>.
Their model, their data, their call; this page only displays it.</p>
</div>

<div id="b101" class="panel">
<h2>Bayesian 101</h2>{B101}
</div>

<footer>One model, honestly uncertain. Nothing here is financial advice.</footer>
</div></body></html>"""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"site written to {out_path} ({len(html)//1024} KB)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site.html")
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    ap.add_argument("--db", default="kalshi_prices.db")
    ap.add_argument("--days", type=int, default=None,
                    help="show a rolling window of N days instead of the "
                         "next unplayed NFL week (the default)")
    ap.add_argument("--mlb-feed", default=DEFAULT_MLB_FEED,
                    help="CSV feed for the read-only MLB tab; empty to disable")
    ap.add_argument("--prices-url", default=DEFAULT_PRICES_URL,
                    help="URL the published page polls for live Kalshi prices; "
                         "pass an empty string to disable the fast layer")
    args = ap.parse_args()
    build_site(args.out, args.games, args.stats, args.db, args.days,
               prices_url=args.prices_url, mlb_feed=args.mlb_feed)
