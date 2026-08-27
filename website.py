import argparse
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm

import rundown as rd
from features import FEATURE_COLS
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


def fav_line(mu, home, away):
    s = round(mu * 2) / 2
    if s == 0:
        return "pick'em"
    team = home if s > 0 else away
    return f"{team} -{abs(s):g}"


def history_tables(df, seasons=(2021, 2022, 2023, 2024, 2025)):
    frames = []
    for season in seasons:
        p = walk_forward(df, V3, season, lam=rd.LIN_LAM,
                         half_life_seasons=rd.DECAY_HL)
        p["season"] = season
        frames.append(p)
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

    edges = [0, .35, .45, .55, .65, 1.0]
    labels = ["0-35%", "35-45%", "45-55%", "55-65%", "65-100%"]
    bins = pd.cut(hist["p_home"], edges, labels=labels)
    calib = hist.groupby(bins, observed=True).apply(lambda g: pd.Series({
        "games": len(g),
        "model_said": 100 * g["p_home"].mean(),
        "home_actually_won": 100 * (g["y"] > 0).mean(),
    }), include_groups=False).round(1)
    return hist, by_season, calib


def build_parlays(upcoming_rows, top_n=10):
    legs = []
    for r in upcoming_rows:
        mu, sigma = r["mu"], r["sigma"]
        p_home = norm.cdf(mu / sigma)
        if r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home")):
            side, p = (r["home"], p_home) if p_home >= 0.5 else (r["away"], 1 - p_home)
            price = r["mkt_home"] if side == r["home"] else 1 - r["mkt_home"]
            if 0.02 < price < 0.98:
                legs.append({"game": f"{r['away']}@{r['home']}", "desc": f"{side} ML",
                             "p": p, "dec": 1 / (price + rd.kalshi_fee(price)),
                             "wild": (p - price) > 0.15})
        sl = r.get("spread_line")
        if sl is not None and not pd.isna(sl):
            p_cover_home = norm.cdf((mu - sl) / sigma)
            side, p = ((r["home"], p_cover_home) if p_cover_home >= 0.5
                       else (r["away"], 1 - p_cover_home))
            line = f"-{abs(sl):g}" if (side == r["home"]) == (sl > 0) else f"+{abs(sl):g}"
            legs.append({"game": f"{r['away']}@{r['home']}", "desc": f"{side} {line}",
                         "p": p, "dec": SPREAD_JUICE, "wild": False})
    parlays = []
    for k in (2, 3):
        for combo in itertools.combinations(legs, k):
            if len({c["game"] for c in combo}) < k:
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
.grid{display:grid;gap:10px}
@media(min-width:700px){.grid{grid-template-columns:1fr 1fr}}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:11px 13px}
.match{display:flex;justify-content:space-between;align-items:baseline}
.teams{font-family:"Arial Narrow",sans-serif;font-size:1.05rem;font-weight:700;
  text-transform:uppercase;letter-spacing:.02em}
.date{color:var(--dim);font-size:.75rem}
.call{font-size:.83rem;margin:5px 0 2px}
.gline{color:var(--green);font-size:.8rem;margin-bottom:8px}
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
.verdict{display:inline-block;margin-top:7px;padding:2px 10px;border-radius:99px;
  font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em}
.v-high{background:var(--green);color:#08120b}
.v-caut{background:var(--yellow);color:#141005}
.v-avoid{background:none;border:1px solid var(--red);color:var(--red)}
.v-none{background:none;border:1px solid var(--dim);color:var(--dim)}
.bandbar{display:flex;gap:6px;margin:10px 0}
.bandbtn{background:none;border:1px solid var(--white);color:var(--white);
  padding:5px 14px;border-radius:99px;font:600 .8rem "Segoe UI",sans-serif;cursor:pointer}
.bandbtn.on{background:var(--green);border-color:var(--green);color:#08120b}
h2{font-family:"Arial Narrow",sans-serif;font-size:1.15rem;text-transform:uppercase;
  color:var(--green);margin:18px 0 8px}
table{width:100%;border-collapse:collapse;font-size:.82rem;margin:8px 0}
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
  document.querySelectorAll('.bandbtn').forEach(x=>x.classList.remove('on'));
  document.getElementById('band-'+b).classList.add('on');
}
function tab(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.querySelectorAll('nav button').forEach(b=>b.classList.remove('on'));
  document.getElementById(id).classList.add('on');
  document.getElementById('b-'+id).classList.add('on');
  window.scrollTo(0,0);
}
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


def verdict_badge(v):
    if v.startswith("HIGH VALUE"):
        return f'<span class="verdict v-high">{v}</span>'
    if v.startswith("CAUTIOUS"):
        return f'<span class="verdict v-caut">{v}</span>'
    if v.startswith("NO VALUE"):
        return f'<span class="verdict v-avoid">{v}</span>'
    return '<span class="verdict v-none">No price yet</span>'


def game_card(r):
    mu, sigma, pm = r["mu"], r["sigma"], r["p_home"]
    call = (f"Bayesian Model: <b>{r['home']} by {abs(mu):.0f}</b>" if mu >= 0
            else f"Bayesian Model: <b>{r['away']} by {abs(mu):.0f}</b>")
    call += f" &plusmn;{sigma:.0f}"
    fav0 = r["home"] if pm >= 0.5 else r["away"]
    fav_p = pm if pm >= 0.5 else 1 - pm
    gline = (f'Gambler terms: <b>{fav_line(mu, r["home"], r["away"])}</b> &middot; '
             f'{fav0} ML <b>{american(fav_p)}</b>')
    v0 = str(r.get("verdict", ""))
    if "&mdash;" in v0 and (v0.startswith("HIGH VALUE") or v0.startswith("CAUTIOUS")):
        vside = v0.split("&mdash;")[-1].strip().replace("small edge on ", "")
        if vside != fav0:
            gline += (f' &middot; or for value: <b>{vside} ML '
                      f'{american(1 - fav_p)}</b> (why below)')
    model_bar = (f'<div class="brow"><span class="blab">Bayesian model</span>'
                 f'<div class="btrack"><div class="bfill model" '
                 f'style="width:{pm*100:.1f}%"></div></div>'
                 f'<span class="bval">{pm*100:.0f}%</span></div>')
    if r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home")):
        mk = float(r["mkt_home"])
        gap = (pm - mk) * 100
        mkt_bar = (f'<div class="brow"><span class="blab">Market price</span>'
                   f'<div class="btrack"><div class="bfill mkt" '
                   f'style="width:{mk*100:.1f}%"></div></div>'
                   f'<span class="bval">{mk*100:.0f}&cent;</span></div>')
        gaptxt = (f'<div class="gap">Both bars: chance the home team wins. '
                  f'Disagreement: <b>{gap:+.0f}</b> points of probability '
                  f'{"toward" if gap > 0 else "against"} {r["home"]}</div>')
    else:
        mkt_bar = (f'<div class="brow"><span class="blab">Market price</span>'
                   f'<div class="btrack"></div><span class="bval">&mdash;</span></div>')
        gaptxt = '<div class="gap">Market has not opened this game yet</div>'
    v = str(r["verdict"])
    note = ""
    fav = r["home"] if pm >= 0.5 else r["away"]
    has_price = r.get("mkt_home") is not None and not pd.isna(r.get("mkt_home"))
    if has_price:
        mkt_fav = r["home"] if float(r["mkt_home"]) >= 0.5 else r["away"]
        if "&mdash;" in v and (v.startswith("HIGH VALUE") or v.startswith("CAUTIOUS")):
            side = v.split("&mdash;")[-1].strip().replace("small edge on ", "")
            soft = "" if v.startswith("HIGH VALUE") else                 " The edge is small, so treat this one lightly."
            if side != fav:
                note = (f'For value: the model still expects <b>{fav}</b> to win, '
                        f'but the market charges too much for {fav}. The value '
                        f'play is <b>{side}</b>: buying the underpriced side, '
                        f'not picking the winner.' + soft)
            elif side == mkt_fav:
                note = (f'For value: the model and the market agree <b>{side}</b> '
                        f'is the likely winner, but the model is more confident '
                        f'than the price implies. The value play is <b>{side}</b>.'
                        + soft)
            else:
                note = (f'For value: the model calls an upset. It makes '
                        f'<b>{side}</b> the favorite while the market does not, '
                        f'so {side} comes cheap if the model is right. The value '
                        f'play is <b>{side}</b>.' + soft)
        elif v.startswith("NO VALUE"):
            if fav == mkt_fav:
                note = (f'The model and the market see this game the same way: '
                        f'<b>{fav}</b> likely wins, and the price already says so. '
                        f'Fair price, no bet.')
            else:
                note = (f'The model leans <b>{fav}</b> while the market leans '
                        f'{mkt_fav}, but not by enough to beat the price after '
                        f'fees. No bet.')
        if note:
            note = f'<div class="gap">{note}</div>'
    spread_row = ""
    sl = r.get("spread_line")
    if sl is not None and not pd.isna(sl):
        p_ch = norm.cdf((mu - sl) / sigma)
        side, p = ((r["home"], p_ch) if p_ch >= 0.5 else (r["away"], 1 - p_ch))
        line = (f"-{abs(sl):g}" if (side == r["home"]) == (sl > 0)
                else f"+{abs(sl):g}")
        if p >= 0.58:
            tag = '<span class="hit">value at a book\'s -110</span>'
        elif p >= 0.545:
            tag = '<span style="color:var(--yellow)">slight lean at -110</span>'
        else:
            tag = 'no edge at -110'
        spread_row = (f'<div class="gap">Spread (Vegas: '
                      f'{fav_line(sl, r["home"], r["away"])}): model covers '
                      f'<b>{side} {line}</b> {p*100:.0f}% of the time &middot; '
                      f'fair price {american(p)} &middot; {tag}</div>')
    return (f'<div class="card"><div class="match"><span class="teams">{r["away"]} @ '
            f'{r["home"]}</span><span class="date">{r["date"]}</span></div>'
            f'<div class="call">{call}</div><div class="gline">{gline}</div>'
            f'<div class="bars">{model_bar}{mkt_bar}</div>{gaptxt}{spread_row}{note}'
            f'{verdict_badge(v)}</div>')
def build_site(out_path="site.html", games_path="games.csv",
               stats_path="team_game_stats.csv", db_path="kalshi_prices.db",
               horizon_days=8, edge_threshold=0.04):
    df = rd.build_frame(games_path, stats_path)
    hist, by_season, calib = history_tables(df)

    today = pd.Timestamp.today().normalize()
    future = df[df["result"].isna() & (df["gameday"] >= today)]
    upcoming = future[future["gameday"] <= today + pd.Timedelta(days=horizon_days)]
    week_note = ""
    if len(upcoming) == 0 and len(future):
        first = future["gameday"].min()
        upcoming = future[future["gameday"] <= first + pd.Timedelta(days=6)]
        week_note = (f'<p class="sub">No games in the next {horizon_days} days; '
                     f'showing the next scheduled week '
                     f'({first.date()} onward).</p>')
    week_rows, parlays = [], []
    price_age = ""
    if "week_note" not in dir():
        week_note = ""
    if len(upcoming):
        lin, ens = rd.fit_models(df, int(upcoming["season"].max()))
        Xu = upcoming[V3].values
        mu, ale, epi = ens.predict_split(Xu)
        sigma = rd.RECAL_SCALE * np.sqrt(ale + epi)
        p_home = norm.cdf(mu / sigma)
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
                price_age = (f'<p class="sub">Market prices last logged: '
                             f'{pd.Timestamp(_ts).strftime("%b %d, %H:%M UTC")}, '
                             f'about {hrs:.0f}h ago{stale}. "No price yet" means no '
                             f'price existed in that snapshot; the market may have '
                             f'opened since.</p>')
        except Exception:
            pass
        for j, row in enumerate(upcoming.itertuples(index=False)):
            rec = {"date": row.gameday.date(), "away": row.away_team,
                   "home": row.home_team, "mu": mu[j], "sigma": sigma[j],
                   "p_home": p_home[j], "mkt_home": None,
                   "spread_line": getattr(row, "spread_line", None),
                   "verdict": "no price"}
            ev = rd.match_event(prices, row.away_team, row.home_team)
            if ev:
                hp = ev.get(row.home_team)
                if hp and hp.get("ask") is not None:
                    rec["mkt_home"] = hp["ask"]
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


    srows = "".join(
        f'<tr><td>{s}</td><td>{int(r.games)}</td><td>{r.winner_pct:.1f}%</td>'
        f'<td>{r.ats_pct:.1f}%</td><td>{r.avg_miss:.1f}</td></tr>'
        for s, r in by_season.iterrows())
    crows = "".join(
        f'<tr><td>{idx}</td><td>{int(r.games)}</td><td>{r.model_said:.0f}%</td>'
        f'<td>{r.home_actually_won:.0f}%</td></tr>'
        for idx, r in calib.iterrows())

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
            f'<details><summary>{season} season &mdash; every game '
            f'({len(g)})</summary><table><tr><th>Wk</th><th>Game</th>'
            f'<th>Bayesian Model line</th><th>Vegas line</th>'
            f'<th>Model: home team wins</th><th>Final margin (home score first)</th>'
            f'<th>Bayesian Model winner pick</th><th>Bayesian Model spread pick</th></tr>{rows}</table></details>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Model HQ</title><style>{CSS}</style>
<script>{TABS_JS}</script></head><body>
<nav><button id="b-week" class="on" onclick="tab('week')">This Week</button>
<button id="b-parlays" onclick="tab('parlays')">Parlay Lab</button>
<button id="b-record" onclick="tab('record')">Track Record</button>
<button id="b-b101" onclick="tab('b101')">Bayesian 101</button></nav>
<div class="wrap">
<h1>NFL <span>Model</span> HQ</h1>
<p class="sub">A Bayesian margin model &middot; generated {today.date()}</p>

<div id="week" class="panel on">
<h2>This Week</h2>
<p class="sub">Green bar: the Bayesian Model's chance the home team wins. White
bar: what the market charges for that outcome. Badges: green means real value
after fees, yellow means an edge too small to trust, red means the price is fair
or worse. Each card also grades the Vegas spread: the model's chance of covering
each side, and whether that beats the 52.4% needed to profit at a standard -110.
Every green light still gets a human news check first.</p>
{week_note}{price_age}<div class="grid">{cards}</div>
</div>

<div id="parlays" class="panel">
<h2>Parlay Lab</h2>
<p class="sub">Combinations of moneylines (at logged Kalshi prices, fees
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
<div class="bandbar">
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
played, exactly as it runs live. Seasons before 2021 are excluded because the model's settings were chosen
using that era. Two separate report cards: Both scorecards below belong to the Bayesian
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
{"".join(season_blocks)}
</div>

<div id="b101" class="panel">
<h2>Bayesian 101</h2>{B101}
</div>

<footer>One model, honestly uncertain. Nothing here is financial advice.</footer>
</div></body></html>"""
    import os
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
    ap.add_argument("--days", type=int, default=8)
    args = ap.parse_args()
    build_site(args.out, args.games, args.stats, args.db, args.days)
