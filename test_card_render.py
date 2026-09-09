"""The rebuilt card: merged line rows, always-on outright call, Reasoning."""
import os, re, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import website as W
import rundown as rd

fail = []
def ok(c, m):
    if not c: fail.append(m)

COLS = rd.V3_COLS
X = np.array([6.2, 40.0, 3.1, .05, -.02, .01, .03, 1.2, 3.0, 1.0, .31, 0.0])
C = np.array([7.1, -0.3, 1.2, .4, -.2, .1, .02, .5, .6, -1.4, -0.9, 0.0])

def card(**kw):
    r = dict(home="LAC", away="ARI", date="2026-09-13", mu=5.3, sigma=16.0,
             p_home=0.635479, spread_line=10.0, mkt_home=0.82,
             verdict="HIGH VALUE &mdash; ARI", x=X, contrib=C)
    r.update(kw)
    return W.game_card(r)

def sect(h, cls):
    m = re.search(r'<div class="%s"[^>]*>(.*?)</div>' % cls, h, re.S)
    return m.group(1) if m else None

h = card()

# --- the two line rows, and the old duplicated prose gone ---
rows = re.findall(r'<span class="llab">(.*?)</span><span class="lval[^"]*">(.*?)</span>', h)
print("line rows:", rows)
ok(len(rows) == 2, f"expected 2 line rows, got {len(rows)}")
ok(rows[0][0] == "Model predicts the line should be:" and rows[0][1] == "LAC -5.5",
   f"model line wrong: {rows[0]}")
ok(rows[1][0] == "Vegas market line:" and rows[1][1] == "LAC -10",
   f"vegas line wrong: {rows[1]}")
ok("Bayesian Model: <b>" not in h, "old 'Bayesian Model:' headline still present")
ok("Model says the line should be" not in h, "old duplicate line still present")
ok("Gambler terms" not in h, "ancient label still present")
ok("Spread (Vegas:" not in h, "spread row still repeats the Vegas number")
ok("&plusmn;16" in h, "sigma dropped from the card")

# --- the outright call is always there ---
def overall(x):
    m = re.search(r'<div class="overall">(.*?)</div>', x, re.S)
    return re.sub(r"<[^>]+>", "", m.group(1)) if m else None

print("value/other side :", overall(h))
ok("Model makes LAC the winner 64% of the time" in overall(h), "outright call wrong")
ok("value is in ARI" in overall(h), "value side not named")

hn = card(verdict="NO VALUE at current price")
print("no value         :", overall(hn))
ok("Model makes LAC the winner 64%" in overall(hn),
   "outright call must survive a no-value verdict")
ok("othing to bet" in overall(hn), "no-value wording missing")

hp = card(verdict="no price", mkt_home=None)
print("no price         :", overall(hp))
ok("No market price yet" in overall(hp), "no-price wording missing")

hs = card(verdict="HIGH VALUE &mdash; LAC", mkt_home=0.40)
print("model+market same:", overall(hs))
ok("comes cheap" in overall(hs) or "underpriced" in overall(hs),
   "agree/upset wording missing")

hc = card(verdict="CAUTIOUS &mdash; small edge on ARI")
ok("treat it lightly" in overall(hc), "cautious softener missing")

# --- Reasoning ---
ok("<summary>Reasoning</summary>" in h, "no Reasoning dropdown")
pulls = re.findall(r'<td class="rpull">(.*?)</td>', h, re.S)
names = re.findall(r'<td class="rname">(.*?)<span', h, re.S)
print("reasoning order:", list(zip([re.sub(r'<[^>]+>', '', p) for p in pulls[:4]], names[:4])))
ok(len(names) == len(COLS), f"expected {len(COLS)} feature rows, got {len(names)}")
ok(names[0] == "Kalman rating gap", f"biggest driver not first: {names[0]}")
ok("7.1 to LAC" in pulls[0], f"top contribution wrong: {pulls[0]}")
# Sorted by absolute size, so the 1.4 against the home team outranks the 1.2 for it.
ok(names[1] == "Division game", f"second row should be the 1.4: {names[1]}")
ok("1.4 to ARI" in pulls[1], "negative contribution should point at the away team")
ok('<span class="rnil">no effect</span>' in h, "a zero contribution should read as none")
ok("Completion Percentage Over Expected" in h, "CPOE description missing")
ok("A backup shows up as a 0" in h, "QB continuity description missing")
ok("+3 days" in h, "rest difference should be in days")
ok(">yes<" in h, "division game should read yes")
ok("outdoor" in h, "roof should read indoor/outdoor")
ok("will not add up" in h, "must disclose that parts do not sum")
# The lead must quote the same rounded margin as the line above it. Quoting it
# to a different precision is how "by 5" once sat above "-5.5" on one card.
lead = re.search(r'class="rlead">(.*?)</p>', h, re.S).group(1)
print("lead margin:", re.search(r"add up to the ([\d.]+) points on (\w+)", lead).groups())
ok("add up to the 5.5 points on LAC" in lead,
   f"reasoning lead disagrees with the quoted line: {lead[-90:]}")

# --- the plain-English summary ---
plain = re.search(r'<p class="rplain">(.*?)</p>', h, re.S).group(1)
print("plain summary:", re.sub(r"<[^>]+>", "", plain)[:120])
ok("Kalman" not in plain and "EPA" not in plain and "_diff" not in plain,
   "plain summary leaks jargon at the reader")
ok("Add it up and the model wants" in plain, "summary does not land on the line")

# Plural must follow the number actually rendered. 0.95 displays as 0.9 while
# round(0.95, 1) is 1.0, which is how "about 0.9 point" once reached the page.
zero = np.zeros(len(COLS))
ki = COLS.index("kalman_diff")
for val, want in ((1.0, "1.0 point"), (1.04, "1.0 point"), (0.95, "0.9 points"),
                  (1.3, "1.3 points"), (2.0, "2.0 points")):
    c = zero.copy()
    c[ki] = val
    txt = W.plain_summary(zero, c, "LAC", "ARI", val, "", "")
    got = re.search(r"about [0-9.]+ points?", txt).group(0)
    ok(got == "about " + want, f"contribution {val} reads '{got}', want 'about {want}'")


# --- tense follows the evidence ---
# In Week 1 every input is last season's. Describing it in the present tense is
# a small lie that costs trust, so the copy has to say what it is working from.
gz = np.zeros(len(COLS))
gz[COLS.index("off_rush_diff")] = 0.7
vz = np.zeros(len(COLS))
vz[COLS.index("off_rush_diff")] = 0.05   # the raw value the clause describes
wk1 = W.plain_summary(vz, gz, "CIN", "TB", -0.2, "Joe Burrow", "Baker Mayfield",
                      played=0, fam_h=7/16, fam_a=1.0)
mid = W.plain_summary(vz, gz, "CIN", "TB", -0.2, "Joe Burrow", "Baker Mayfield",
                      played=8, fam_h=7/16, fam_a=1.0)
ok("Nothing has been played yet this season" in wk1,
   "week 1 summary does not say it is working from last season")
ok("they got more out of each carry" in wk1,
   f"week 1 should be past tense: {re.sub(r'<[^>]+>', '', wk1)[:150]}")
ok("Nothing has been played yet" not in mid, "mid-season still claims nothing played")
ok("have been getting more out of each carry" in mid, "mid-season should be present")

# --- the QB clause must describe continuity, never a benching ---
# qb_fam_diff is the share of a team's last 16 starts belonging to this week's
# listed starter. Burrow's is 7/16 because he was injured; he is still the
# starter. An earlier version rendered that as "CIN do not have their usual man
# under centre", which was simply false.
gq = np.zeros(len(COLS))
gq[COLS.index("qb_fam_diff")] = -0.4
qtxt = W.plain_summary(vz, gq, "CIN", "TB", -0.2, "Joe Burrow", "Baker Mayfield",
                       played=0, fam_h=7/16, fam_a=1.0)
flat = re.sub(r"<[^>]+>", "", qtxt)
print("qb clause:", flat[flat.find("The model has seen"):][:130])
ok("usual man under centre" not in flat, "summary still implies a benching")
ok("do not" not in flat, "summary still asserts a team lacks its starter")
ok("Joe Burrow started 7 of" in flat and "Baker Mayfield started 16 of" in flat,
   f"QB clause should quote both start counts: {flat[:160]}")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else "PASS: rebuilt card"))

sys.exit(1 if fail else 0)
