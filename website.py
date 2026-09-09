import argparse
import itertools
from collections import Counter
from zoneinfo import ZoneInfo
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
            legs.append({"game": f"{r['away']}@{r['home']}", "desc": f"{side} {line}",
                         "p": p, "dec": SPREAD_JUICE,
                         "wild": square3_gap(p, 0.5) > SQUARE3_SIGMAS})
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
.lval{font-size:1.14rem;font-weight:700;color:var(--green);margin-left:7px;
  font-variant-numeric:tabular-nums}
.lval.vval{color:var(--white)}
.lnote{color:var(--dim);font-size:.72rem;margin-left:7px}
/* The model's outright call, always shown, value or not. */
.overall{margin:8px 0 0;padding:7px 9px;border-radius:7px;font-size:.78rem;
  line-height:1.42;background:#3fb95018;border-left:3px solid var(--green)}
.overall b{color:var(--green)}
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
.picks{margin:0 0 9px}
.picks .plab{display:block;font-size:.68rem;font-weight:600;color:var(--dim);
  text-transform:uppercase;letter-spacing:.05em;margin-bottom:1px}
/* Level with .lval on purpose: the model's line and the recommended play are
   the two numbers on the card, and neither should outrank the other. */
.picks .plist{font-size:1.14rem;font-weight:700;color:var(--green);
  line-height:1.3}
.picks .pnone{font-size:.85rem;font-weight:600;color:var(--dim)}
.picks .psmall{font-size:.7rem;font-weight:600;color:var(--dim)}
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
/* Mirrors overall_text() in website.py case for case. It leads with the
   model's outright call every time, including when nothing is worth betting,
   because that call is the thing a reader most often wants and it is worth
   tracking whether or not it ever carried an edge. */
function overallText(fav, favP, t, side, mktFav){
  var call = 'Model makes <b>' + fav + '</b> the winner <b>' +
             Math.round(favP * 100) + '%</b> of the time';
  if (t === 'none'){
    return fav === mktFav
      ? (call + ', and the market prices it about the same. Fair price, ' +
         'nothing to bet.')
      : (call + ', against a market that leans ' + mktFav + ', but not by ' +
         'enough to beat the price after fees. Nothing to bet.');
  }
  var soft = t === 'high' ? '' : ' The edge is small, so treat it lightly.';
  if (side !== fav){
    return (call + ', but the market charges too much for ' + fav + ' to be ' +
            'worth backing. The value is in <b>' + side + '</b> instead: buying ' +
            'the underpriced side, not picking the winner.' + soft);
  }
  if (side === mktFav){
    return (call + ', and the market agrees on the winner but is not as ' +
            'confident, so <b>' + side + '</b> is underpriced.' + soft);
  }
  return (call + ', which the market does not: it makes ' + mktFav +
          ' the favourite, so <b>' + side + '</b> comes cheap if the model is ' +
          'right.' + soft);
}
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
  var fav = pm >= 0.5 ? home : away;
  var favP = pm >= 0.5 ? pm : 1 - pm;
  var mktFav = hAsk >= 0.5 ? home : away;
  var note = card.querySelector('.overall');
  if (note){ note.innerHTML = overallText(fav, favP, t, side, mktFav); }
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
      : '<span class="pnone">No recommended play</span>';
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
function bootWeek(){ localiseKickoffs(); applyPrices(); }
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


def plain_summary(vals, con, home, away, mu, hqb, aqb, played=0,
                  fam_h=None, fam_a=None):
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
            were = "were" if past else "have been"
            out.append(f"<b>{t}</b> {were} simply the better side {era}, worth "
                       f"{pts} here before anything else about the matchup.")
        elif name == "air":
            bits = []
            if (v["off_pass_diff"] > 0) == (t == home) and v["off_pass_diff"]:
                mover = qb.get(t)
                verb = "were" if past else "have been"
                bits.append((f"{mover} and the {t} pass game {verb} the more "
                             f"efficient of the two per dropback") if mover else
                            f"they {verb} the more efficient passing team")
            if (v["def_pass_diff"] > 0) == (opp == home) and v["def_pass_diff"]:
                verb = "was" if past else "has been"
                bits.append(f"the {opp} pass defence {verb} the leakier one, "
                            f"giving up more per throw")
            if (v["cpoe_diff"] > 0) == (t == home) and v["cpoe_diff"]:
                mover = qb.get(t)
                verb = "completed" if past else "has been completing"
                bits.append((f"{mover} {verb} throws he had no business "
                             f"completing") if mover else
                            f"they {'completed' if past else 'have been completing'}"
                            f" more than expected")
            body = ", and ".join(bits) if bits else f"the passing matchup tilts {t}"
            out.append(f"Through the air it is worth {pts} to <b>{t}</b>: {body}.")
        elif name == "ground":
            bits = []
            if (v["off_rush_diff"] > 0) == (t == home) and v["off_rush_diff"]:
                bits.append("they got more out of each carry" if past else
                            "they have been getting more out of each carry")
            if (v["def_rush_diff"] > 0) == (opp == home) and v["def_rush_diff"]:
                verb = "was" if past else "has been"
                bits.append(f"the {opp} run defence {verb} giving it up")
            body = ", and ".join(bits) if bits else f"the run matchup tilts {t}"
            out.append(f"On the ground it is worth {pts} to <b>{t}</b>: {body}.")
        elif name == "form":
            if past:
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
                out.append(f"The model has seen more of <b>{t}</b>'s quarterback, "
                           f"{pts}: {tq} started {nt} of their last 16 games while "
                           f"{oq} started {no} of {opp}'s, so it has a firmer read "
                           f"on one than the other.")
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
            "year's evidence. " if past else "")
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
                        played, fam_h, fam_a) +
        '<p class="rlead">And the same thing as arithmetic, biggest first. Each '
        'number '
        'is how much the prediction would move if that one input were neutral '
        'instead of what it is, measured on the same ensemble that produced the '
        f'prediction above. They will not add up to the {abs(s_line):g} points '
        f'on {side}: the model is a network, not a sum, so each input is '
        'measured on its own.</p>'
        f'<table class="rtab">{"".join(body)}</table></details>')


def overall_text(fav, fav_p, tier, side, mkt_fav, has_price):
    """One sentence: who the model likes outright, then whether that is buyable.

    Always present, including when there is nothing to bet. A reader who wants
    the model's outright opinion should not have to infer it from the absence of
    a recommendation, and that opinion is worth tracking whether or not it ever
    carried an edge.
    """
    call = f'Model makes <b>{fav}</b> the winner <b>{fav_p*100:.0f}%</b> of the time'
    if not has_price:
        return f'{call}. No market price yet, so there is nothing to compare it to.'
    if tier == "none":
        if fav == mkt_fav:
            return (f'{call}, and the market prices it about the same. '
                    f'Fair price, nothing to bet.')
        return (f'{call}, against a market that leans {mkt_fav}, but not by '
                f'enough to beat the price after fees. Nothing to bet.')
    soft = "" if tier == "high" else " The edge is small, so treat it lightly."
    if side != fav:
        return (f'{call}, but the market charges too much for {fav} to be worth '
                f'backing. The value is in <b>{side}</b> instead: buying the '
                f'underpriced side, not picking the winner.' + soft)
    if side == mkt_fav:
        return (f'{call}, and the market agrees on the winner but is not as '
                f'confident, so <b>{side}</b> is underpriced.' + soft)
    return (f'{call}, which the market does not: it makes {mkt_fav} the '
            f'favourite, so <b>{side}</b> comes cheap if the model is right.'
            + soft)


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
        f'<div class="lrow"><span class="llab">Model predicts the line '
        f'should be:</span><span class="lval">'
        f'{fav_line(mu, r["home"], r["away"])}</span>'
        f'<span class="lnote">&plusmn;{sigma:.0f} &middot; fair ML '
        f'{american(fav_p)}</span></div>'
        f'<div class="lrow"><span class="llab">Vegas market line:</span>'
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
    picks_row = ('<div class="picks"><span class="plab">Recommended picks</span>'
                 '<span class="plist">'
                 + (' &middot; '.join(picks) if picks
                    else '<span class="pnone">No recommended play</span>')
                 + '</span></div>')
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
    fav = r["home"] if pm >= 0.5 else r["away"]
    has_price = has_mkt
    tier = verdict_tier(v)
    side_v = fav
    if "&mdash;" in v and (v.startswith("HIGH VALUE") or v.startswith("CAUTIOUS")):
        side_v = v.split("&mdash;")[-1].strip().replace("small edge on ", "")
    mkt_fav = (r["home"] if has_price and float(mkt_h) >= 0.5 else r["away"]) \
        if has_price else ""
    # Always rendered, including when there is nothing to bet: the model's
    # outright call is worth showing on its own, and worth tracking whether or
    # not it ever carried an edge.
    note = ('<div class="overall">'
            + overall_text(fav, fav_p, tier, side_v, mkt_fav, has_price)
            + '</div>')
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
    return (f'<div class="card gcard tier-{verdict_tier(v)}" '
            f'data-keys="{kalshi_lookup(r["away"], r["home"])}" '
            f'data-away="{r["away"]}" data-home="{r["home"]}" '
            f'data-sp="{sp_pick}" data-zsp="{z_sp:.4f}" '
            f'data-sigma="{sigma:.3f}" data-p="{pm:.6f}">'
            f'<div class="match"><span class="teams">{r["away"]} @ '
            f'{r["home"]}</span><span class="date" '
            f'data-kick="{r.get("kick_iso", "")}">'
            f'{r.get("kick_txt") or r["date"]}</span></div>'
            f'{lines}{spread_row}{picks_row}'
            f'<div class="bars">{model_bar}{mkt_bar}</div>{gaptxt}'
            f'{note}{sq3_row}'
            f'{reasoning_panel(rd.V3_COLS, r.get("x", []), r.get("contrib", []), r["home"], r["away"], mu, r.get("hqb", ""), r.get("aqb", ""),
            r.get("played", 0), r.get("fam_h"), r.get("fam_a")) if len(r.get("contrib", [])) else ""}'
            f'{verdict_badge(v)}</div>')
def build_site(out_path="site.html", games_path="games.csv",
               stats_path="team_game_stats.csv", db_path="kalshi_prices.db",
               horizon_days=8, edge_threshold=0.04,
               prices_url=DEFAULT_PRICES_URL, mlb_feed=DEFAULT_MLB_FEED):
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
        mu, ale, epi = ens.predict_split(Xu)
        sigma = rd.RECAL_SCALE * np.sqrt(ale + epi)
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
<p class="sub">A Bayesian margin model &middot; generated {today.date()}</p>
{price_age}

<div id="week" class="panel on">
<h2>This Week</h2>
<p class="sub">Green bar: the Bayesian Model's chance the home team wins. White
bar: what the market charges for that outcome. Badges: green means real value
after fees, yellow means an edge too small to trust, red means the price is fair
or worse. Each card also grades the Vegas spread: the model's chance of covering
each side, and whether that beats the 52.4% needed to profit at a standard -110.
Every green light still gets a human news check first.</p>
{week_note}{tier_bar}<div class="grid">{cards}</div>
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

<div id="mlb" class="panel">
<h2>MLB &middot; from GooseLine</h2>
<p class="sub">This tab is not my model. It is <a href="https://jdev-02.github.io/gooseline-model-hq/"
style="color:var(--green)">GooseLine Solutions' MLB model</a>, loaded live from
their public data when this page opens. I do not build, tune, or vouch for the
baseball numbers; they are here so both sports sit in one place. Read the bars
the same way as the NFL tab: green is the model's chance the home team wins,
white is what the market charges. Baseball is much closer to a coin flip than
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
    ap.add_argument("--mlb-feed", default=DEFAULT_MLB_FEED,
                    help="CSV feed for the read-only MLB tab; empty to disable")
    ap.add_argument("--prices-url", default=DEFAULT_PRICES_URL,
                    help="URL the published page polls for live Kalshi prices; "
                         "pass an empty string to disable the fast layer")
    args = ap.parse_args()
    build_site(args.out, args.games, args.stats, args.db, args.days,
               prices_url=args.prices_url, mlb_feed=args.mlb_feed)
