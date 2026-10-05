"""Grade one week's published picks, game by game, the way a book would settle them.

Every number here is walk-forward: the models are refit on games strictly before
the week in question, so this reproduces what the cards actually said at the time
rather than what the models would say now knowing the answers.

Three ledgers, each shown twice, unfiltered and filtered, because a filter only
earns its place by beating the thing it filters:

  moneyline   the model's outright pick at that side's closing Vegas price, and
              the subset where the model's number beat that price (the green badge)
  spread      the model's side against the closing line at -110, and the subset
              the card prints, where the model gives it 58% or better
  over/under  the totals model against the closing total at -110, same thresholds

Prices are the closing Vegas numbers out of games.csv, not Kalshi asks, for the
same reason the Track Record tab uses them: Kalshi history only starts in 2026.
The Vegas price carries the book's cut and Kalshi's 7% win fee is absent, so read
this as the record of the rules, not as a Kalshi statement.

    python weekly_report.py                 the latest week with results
    python weekly_report.py --week 4
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

import rundown as rd
import totals as tot
from models import prob_total_over
from walkforward import walk_forward

JUICE = 1.909          # -110: a win returns 0.909 on top of the stake
PICK_P = 0.58          # the card's own threshold for printing a spread or O/U pick


def american_dec(ml):
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml > 0, 1 + ml / 100.0, 1 + 100.0 / np.abs(ml))


def settle(won, push, dec):
    """Profit per unit staked. Pushes and voids stake nothing and return nothing."""
    if push:
        return 0.0, 0
    return (dec - 1.0 if won else -1.0), 1


def grade(season, week, games_path="games.csv", stats_path="team_game_stats.csv"):
    df = rd.build_frame(games_path, stats_path)
    done = df[df["result"].notna()]
    if week is None:
        wk = done[done["season"] == season]["week"]
        if wk.empty:
            print(f"no completed {season} games")
            return None
        week = int(wk.max())

    margin = walk_forward(done, rd.V3_COLS, season, half_life_seasons=rd.DECAY_HL,
                          model_factory=rd.DeployedModel)
    totals = walk_forward(df[df["y_total"].notna()], tot.TOTAL_COLS, season,
                          half_life_seasons=tot.TOTAL_DECAY_HL,
                          model_factory=tot.DeployedTotals, target="y_total",
                          keep_cols=("total_line",))
    m = margin[margin["week"] == week].copy()
    t = totals[totals["week"] == week][["game_id", "mu", "sigma", "total_line"]]
    t = t.rename(columns={"mu": "t_mu", "sigma": "t_sigma"})
    m = m.merge(t, on="game_id", how="left").merge(
        df[["game_id", "spread_line", "home_moneyline", "away_moneyline",
            "home_score", "away_score"]], on="game_id", how="left")
    if m.empty:
        print(f"nothing graded for {season} week {week}")
        return None

    books = {k: {"w": 0, "l": 0, "dead": 0, "profit": 0.0, "staked": 0}
             for k in ("ml", "ml_value", "sp", "sp_printed", "ou", "ou_printed")}
    lines = []

    for r in m.itertuples(index=False):
        p_home = float(norm.cdf(r.mu / r.sigma))
        fav_home = p_home >= 0.5
        side = r.home_team if fav_home else r.away_team
        dec_h, dec_a = american_dec(r.home_moneyline), american_dec(r.away_moneyline)
        dec = float(dec_h if fav_home else dec_a)
        won = (r.y > 0) if fav_home else (r.y < 0)
        tie = r.y == 0
        prof, st = settle(won, tie or np.isnan(dec), dec)
        for key in ("ml",) + (("ml_value",) if
                              (p_home if fav_home else 1 - p_home) > 1 / dec else ()):
            b = books[key]
            b["profit"] += prof
            b["staked"] += st
            if not st:
                b["dead"] += 1
            elif won:
                b["w"] += 1
            else:
                b["l"] += 1

        sp_txt = ou_txt = "no line"
        if not pd.isna(r.spread_line):
            p_cov_home = float(norm.cdf((r.mu - r.spread_line) / r.sigma))
            sp_home = p_cov_home >= 0.5
            sp_p = p_cov_home if sp_home else 1 - p_cov_home
            sp_side = r.home_team if sp_home else r.away_team
            ln = (-abs(r.spread_line) if (sp_home == (r.spread_line > 0))
                  else abs(r.spread_line))
            push = r.y == r.spread_line
            cov = (r.y > r.spread_line) if sp_home else (r.y < r.spread_line)
            prof, st = settle(cov, push, JUICE)
            for key in ("sp",) + (("sp_printed",) if sp_p >= PICK_P else ()):
                b = books[key]
                b["profit"] += prof
                b["staked"] += st
                if not st:
                    b["dead"] += 1
                elif cov:
                    b["w"] += 1
                else:
                    b["l"] += 1
            mark = "PUSH" if push else ("HIT " if cov else "miss")
            sp_txt = (f"{sp_side} {ln:+g} {mark} ({sp_p*100:.0f}%"
                      f"{', printed' if sp_p >= PICK_P else ''})")

        if not pd.isna(getattr(r, "t_mu", np.nan)) and not pd.isna(r.total_line):
            p_over, p_under, _ = prob_total_over(r.t_mu, r.t_sigma, float(r.total_line))
            over = p_over >= p_under
            ou_p = float(max(p_over, p_under))
            actual = r.home_score + r.away_score
            push = actual == r.total_line
            hit = (actual > r.total_line) if over else (actual < r.total_line)
            prof, st = settle(hit, push, JUICE)
            for key in ("ou",) + (("ou_printed",) if ou_p >= PICK_P else ()):
                b = books[key]
                b["profit"] += prof
                b["staked"] += st
                if not st:
                    b["dead"] += 1
                elif hit:
                    b["w"] += 1
                else:
                    b["l"] += 1
            mark = "PUSH" if push else ("HIT " if hit else "miss")
            ou_txt = (f"{'Over' if over else 'Under'} {float(r.total_line):g} {mark} "
                      f"(model {r.t_mu:.1f}, actual {actual:.0f}, {ou_p*100:.0f}%"
                      f"{', printed' if ou_p >= PICK_P else ''})")

        lines.append({
            "game": f"{r.away_team} @ {r.home_team}",
            "final": f"{r.home_team} {r.home_score:.0f}-{r.away_score:.0f}",
            "model": f"{'home' if r.mu > 0 else 'away'} by {abs(r.mu):.1f}",
            # American odds, not a percentage: the spread and O/U columns show the
            # model's confidence in brackets, and printing a payout in the same
            # position invites reading a 12% price as a 12% belief.
            "ml": f"{side} {'HIT ' if won else 'PUSH' if tie else 'miss'} "
                  f"[{'+' if dec >= 2 else ''}"
                  f"{(100*(dec-1)) if dec >= 2 else (-100/(dec-1)):.0f}, "
                  f"model {(p_home if fav_home else 1 - p_home)*100:.0f}%]",
            "spread": sp_txt, "ou": ou_txt})

    print(f"\n=== {season} week {week}: {len(m)} games graded "
          f"(walk-forward, refit on games before the week)\n")
    w = pd.DataFrame(lines)
    print(w.to_string(index=False))

    print(f"\n{'ledger':<28}{'record':>14}{'ROI':>9}")
    label = {"ml": "Every ML pick", "ml_value": "  of those, value picks",
             "sp": "Every spread pick", "sp_printed": "  of those, the card printed",
             "ou": "Every O/U pick", "ou_printed": "  of those, the card printed"}
    out = {}
    for key, lab in label.items():
        b = books[key]
        if not b["staked"]:
            print(f"{lab:<28}{'none':>14}{'':>9}")
            continue
        roi = 100 * b["profit"] / b["staked"]
        dead = f" +{b['dead']}v" if b["dead"] else ""
        print(f"{lab:<28}{f'{b[chr(119)]}-{b[chr(108)]}{dead}':>14}{roi:>+8.1f}%")
        out[key] = {"w": b["w"], "l": b["l"], "roi": roi}
    print("\n52.4% is break even at -110. One week is a handful of bets: a swing "
          "of plus or minus 40% here is noise, not a trend.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    args = ap.parse_args()
    season = args.season
    if season is None:
        g = pd.read_csv(args.games, usecols=["season", "result"], low_memory=False)
        season = int(g[g["result"].notna()]["season"].max())
    grade(season, args.week, args.games, args.stats)


if __name__ == "__main__":
    main()
