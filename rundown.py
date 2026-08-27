import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import os
import re
import sqlite3
import numpy as np
import pandas as pd

from features import load_games, load_team_game_stats, build_features, FEATURE_COLS
from kalman import TeamKalman
from models import LinearGaussianModel
from walkforward import season_decay_weights
from ensemble import DeepEnsemble

KALMAN_PARAMS = {"obs_var": 150.0, "weekly_q": 0.8, "season_inflate": 8.0,
                 "season_revert": 0.7}
V3_COLS = ["kalman_diff", "kalman_var"] + FEATURE_COLS + ["qb_fam_diff", "indoor"]
RECAL_SCALE = 1.039
LIN_LAM = 100.0
DECAY_HL = 2.0
TICKER_RE = re.compile(r"^KXNFLGAME-(\d{2}[A-Z]{3}\d{2})([A-Z]+)$")


def norm_cdf(z):
    from scipy.stats import norm
    return norm.cdf(z)


def kalshi_fee(p):
    return 0.07 * p * (1 - p)


def build_frame(games_path, stats_path):
    games = load_games(games_path, first_season=2010, keep_unplayed=True)
    stats = load_team_game_stats(stats_path)
    df = build_features(games, stats, form_half_life_games=8)
    df = TeamKalman(**KALMAN_PARAMS).run(df)
    return df


def fit_models(df, asof_season):
    train = df[df["result"].notna()]
    sw = season_decay_weights(train["season"].values, asof_season, DECAY_HL)
    X, y = train[V3_COLS].values, train["y"].values
    lin = LinearGaussianModel(lam=LIN_LAM).fit(X, y, sample_weight=sw)
    ens = DeepEnsemble(n_members=5, hidden=16, weight_decay=1e-2,
                       epochs=200, seed=0).fit(X, y, sample_weight=sw)
    return lin, ens


def latest_prices(db_path):
    if not db_path or not os.path.exists(db_path):
        return {}
    con = sqlite3.connect(db_path)
    rows = con.execute("""
        SELECT ticker, event_ticker, yes_bid, yes_ask, last_price
        FROM snapshots WHERE series_ticker='KXNFLGAME'
        AND ts_utc = (SELECT MAX(ts_utc) FROM snapshots s2
                      WHERE s2.ticker = snapshots.ticker)""").fetchall()
    con.close()
    prices = {}
    for ticker, event, bid, ask, last in rows:
        team = ticker.rsplit("-", 1)[-1]
        m = TICKER_RE.match(event or "")
        if not m:
            continue
        prices.setdefault(m.group(2), {})[team] = {
            "bid": bid, "ask": ask, "last": last}
    return prices


KALSHI_ALIASES = {"LA": ["LAR", "LA"], "JAX": ["JAX", "JAC"],
                  "WAS": ["WAS", "WSH"], "LV": ["LV", "LVR"]}


def match_event(prices, away, home):
    for a in KALSHI_ALIASES.get(away, [away]):
        for h in KALSHI_ALIASES.get(home, [home]):
            ev = prices.get(f"{a}{h}")
            if ev:
                if a != away or h != home:
                    ev = {(away if k == a else home if k == h else k): v
                          for k, v in ev.items()}
                return ev
    return None


def render_html(table, trained_through, out_path="rundown.html"):
    css = """
    :root{--ink:#12190f;--slate:#1c2a1e;--chalk:#ece9df;--turf:#3a8a54;
      --amber:#e0a32e;--dim:#8fa08f;}
    *{box-sizing:border-box;margin:0}
    body{background:var(--ink);color:var(--chalk);
      font:16px/1.5 "Segoe UI",system-ui,sans-serif;padding:24px 16px;max-width:760px;margin:0 auto}
    h1{font-family:"Arial Narrow","Segoe UI",sans-serif;font-weight:700;
      font-size:2rem;letter-spacing:.04em;text-transform:uppercase}
    .sub{color:var(--dim);margin:4px 0 24px;font-size:.9rem}
    details{background:var(--slate);border-radius:10px;padding:14px 18px;margin-bottom:24px}
    summary{cursor:pointer;font-weight:600;color:var(--amber)}
    details p{margin:10px 0 0;font-size:.92rem;color:var(--chalk)}
    .card{background:var(--slate);border-radius:12px;padding:16px 18px;margin-bottom:14px}
    .match{display:flex;justify-content:space-between;align-items:baseline}
    .teams{font-family:"Arial Narrow",sans-serif;font-size:1.35rem;font-weight:700;
      letter-spacing:.03em;text-transform:uppercase}
    .date{color:var(--dim);font-size:.85rem}
    .call{margin:8px 0 12px;font-size:.95rem}
    .call b{font-variant-numeric:tabular-nums}
    .strip{position:relative;height:34px;background:
      repeating-linear-gradient(90deg,#24382a 0 10%,#203225 10% 20%);
      border-radius:6px;overflow:hidden}
    .gapfill{position:absolute;top:0;bottom:0;background:rgba(224,163,46,.35)}
    .mark{position:absolute;top:0;bottom:0;width:3px}
    .mark.model{background:var(--turf)}
    .mark.market{background:var(--amber)}
    .legend{display:flex;gap:18px;margin-top:6px;font-size:.8rem;color:var(--dim)}
    .dotm,.dotk{display:inline-block;width:10px;height:10px;border-radius:2px;
      margin-right:5px;vertical-align:-1px}
    .dotm{background:var(--turf)}.dotk{background:var(--amber)}
    .verdict{display:inline-block;margin-top:10px;padding:3px 12px;border-radius:99px;
      font-size:.82rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
    .v-pass{background:#2a3a2e;color:var(--dim)}
    .v-cand{background:var(--amber);color:var(--ink)}
    .v-none{background:transparent;border:1px solid var(--dim);color:var(--dim)}
    footer{color:var(--dim);font-size:.8rem;margin-top:28px}
    """
    explainer = """
    <details><summary>How to read this</summary>
    <p><b>The call.</b> The model predicts the final scoring margin for every game,
    for example "Chiefs by 3". No model can predict football precisely: the honest
    give-or-take on any NFL prediction is about two touchdowns, and the number after
    the &plusmn; says exactly how much for that game.</p>
    <p><b>The field strip.</b> The green marker is the model's chance that the home
    team wins. The gold marker is what the market currently charges for that outcome
    (a 65&cent; price means the market calls it 65%). The shaded zone between them is
    the disagreement. A bet only makes sense when that zone is wide, in the right
    direction, and survives the trading fee.</p>
    <p><b>Verdicts.</b> PASS means the price is fair; most games are passes and that
    is normal. CANDIDATE means the model sees value after fees, pending a news check:
    the model cannot see this week's injuries or resting starters, so a human looks
    before anything happens. NO PRICE means the market has not opened yet.</p>
    </details>"""

    cards = []
    for r in table.itertuples(index=False):
        pm = r.p_home
        call = (f"Model: <b>{r.home}</b> by <b>{abs(r.mu):.0f}</b>" if r.mu >= 0
                else f"Model: <b>{r.away}</b> by <b>{abs(r.mu):.0f}</b>")
        call += f" &plusmn;{r.sigma:.0f} &nbsp;&middot;&nbsp; home win chance <b>{pm*100:.0f}%</b>"
        if r.mkt_home is not None and not (isinstance(r.mkt_home, float) and np.isnan(r.mkt_home)):
            mk = float(r.mkt_home)
            lo, hi = sorted([pm * 100, mk * 100])
            strip = (f'<div class="strip">'
                     f'<div class="gapfill" style="left:{lo:.1f}%;width:{hi-lo:.1f}%"></div>'
                     f'<div class="mark model" style="left:{pm*100:.1f}%"></div>'
                     f'<div class="mark market" style="left:{mk*100:.1f}%"></div></div>'
                     f'<div class="legend"><span><span class="dotm"></span>model '
                     f'{pm*100:.0f}%</span><span><span class="dotk"></span>market '
                     f'{mk*100:.0f}&cent;</span></div>')
        else:
            strip = ('<div class="strip"></div>'
                     '<div class="legend"><span>market not open yet</span></div>')
        v = str(r.verdict)
        if v.startswith("CANDIDATE"):
            badge = f'<span class="verdict v-cand">{v}</span>'
        elif v == "pass":
            badge = '<span class="verdict v-pass">Pass &mdash; price is fair</span>'
        else:
            badge = '<span class="verdict v-none">No price yet</span>'
        cards.append(
            f'<div class="card"><div class="match"><span class="teams">{r.away} @ '
            f'{r.home}</span><span class="date">{r.date}</span></div>'
            f'<div class="call">{call}</div>{strip}{badge}</div>')

    html = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>NFL Model Rundown</title><style>{css}</style></head><body>'
            f'<h1>NFL Model Rundown</h1>'
            f'<p class="sub">Bayesian margin model &middot; trained through '
            f'{trained_through} &middot; generated {pd.Timestamp.today().date()}</p>'
            f'{explainer}{"".join(cards)}'
            f'<footer>Green: model probability. Gold: market price. Shaded: the '
            f'disagreement. Nothing here is financial advice; it is one model, '
            f'honestly uncertain.</footer></body></html>')
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


def rundown(games_path="games.csv", stats_path="team_game_stats.csv",
            db_path="kalshi_prices.db", horizon_days=8, edge_threshold=0.04,
            html_out=None):
    df = build_frame(games_path, stats_path)
    today = pd.Timestamp.today().normalize()
    upcoming = df[df["result"].isna()
                  & (df["gameday"] >= today)
                  & (df["gameday"] <= today + pd.Timedelta(days=horizon_days))]
    if len(upcoming) == 0:
        print("No games in the horizon window.")
        return None

    asof_season = int(upcoming["season"].max())
    lin, ens = fit_models(df, asof_season)

    Xu = upcoming[V3_COLS].values
    lin_mu = lin.predict_mu(Xu)
    mu, ale, epi = ens.predict_split(Xu)
    sigma = RECAL_SCALE * np.sqrt(ale + epi)
    p_home = norm_cdf(mu / sigma)

    prices = latest_prices(db_path)
    out = []
    for j, row in enumerate(upcoming.itertuples(index=False)):
        rec = {"date": row.gameday.date(), "away": row.away_team,
               "home": row.home_team, "lin_mu": round(lin_mu[j], 1),
               "mu": round(mu[j], 1), "sigma": round(sigma[j], 1),
               "epi_sig": round(np.sqrt(epi[j]), 2),
               "p_home": round(p_home[j], 3),
               "mkt_home": None, "edge": None, "verdict": "no price"}
        ev = match_event(prices, row.away_team, row.home_team)
        if ev:
            side = None
            hp = ev.get(row.home_team)
            ap = ev.get(row.away_team)
            if hp and hp.get("ask") is not None:
                e_home = p_home[j] - hp["ask"] - kalshi_fee(hp["ask"])
                rec["mkt_home"] = hp["ask"]
                if e_home > edge_threshold:
                    side = (row.home_team, e_home)
            if ap and ap.get("ask") is not None:
                e_away = (1 - p_home[j]) - ap["ask"] - kalshi_fee(ap["ask"])
                if e_away > edge_threshold and (side is None or e_away > side[1]):
                    side = (row.away_team, e_away)
            if side:
                rec["edge"] = round(side[1], 3)
                rec["verdict"] = f"CANDIDATE {side[0]}"
            else:
                rec["edge"] = round(p_home[j] - hp["ask"] - kalshi_fee(hp["ask"]), 3) \
                    if hp and hp.get("ask") is not None else None
                rec["verdict"] = "pass"
        out.append(rec)

    table = pd.DataFrame(out).sort_values(["date", "home"])
    print(f"\n=== Rundown: next {horizon_days} days, trained through "
          f"{int(df[df['result'].notna()]['season'].max())} week "
          f"{int(df[df['result'].notna()].iloc[-1]['week'])} ===")
    print(table.to_string(index=False))
    print("\nCANDIDATE means model edge exceeds threshold after estimated fees. "
          "Always apply the news check (injuries, rest decisions) before acting; "
          "the model cannot see them.")
    if html_out:
        done = df[df["result"].notna()]
        meta = f"{int(done['season'].max())} week {int(done.iloc[-1]['week'])}"
        path = render_html(table, meta, html_out)
        print(f"HTML report written to {path}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="games.csv")
    ap.add_argument("--stats", default="team_game_stats.csv")
    ap.add_argument("--db", default="kalshi_prices.db")
    ap.add_argument("--days", type=int, default=8)
    ap.add_argument("--edge", type=float, default=0.04)
    ap.add_argument("--html", nargs="?", const="rundown.html", default=None,
                    help="also write an HTML report (default rundown.html; "
                         "use docs/index.html for GitHub Pages)")
    args = ap.parse_args()
    rundown(args.games, args.stats, args.db, args.days, args.edge, args.html)
