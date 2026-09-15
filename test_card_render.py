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

# --- the four bold rows, in the order a reader asked for them ---
def block(x):
    """label -> plain text of the value, for each bold row."""
    out = {}
    for lab, rest in re.findall(r'<div class="lrow"><span class="llab">(.*?)</span>(.*?)</div>', x, re.S):
        out[lab] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", rest)).strip()
    return out

b = block(h)
print("bold rows:", b)
ok(list(b) == ["Bayesian prediction:", "Vegas market:", "ML pick:", "Value recommendation:"],
   f"rows missing or out of order: {list(b)}")
ok(b["Bayesian prediction:"].startswith("LAC -5.5"), f"model line wrong: {b}")
ok(b["Vegas market:"] == "LAC -10", f"vegas line wrong: {b}")
ok("&plusmn;16" in h, "sigma dropped from the card")
for gone in ("Bayesian Model: <b>", "Model says the line should be", "Gambler terms",
             "Spread (Vegas:", "Model predicts the line should be", 'class="overall"',
             "Recommended picks"):
    ok(gone not in h, f"old copy still present: {gone}")

# --- the ML pick is the outright call, independent of any value verdict ---
# It used to live only inside a sentence about value ("the model still makes LAC
# the winner, but..."), so on a card with value on the other side a reader had
# to dig for who the model actually thinks wins.
ok(b["ML pick:"] == "LAC predicted to win 64% of the time", f"ML pick wrong: {b['ML pick:']}")
ok(b["Value recommendation:"] == "ARI ML · ARI +10".replace("·", "&middot;")
   or b["Value recommendation:"].replace("&middot;", "·") == "ARI ML · ARI +10",
   f"value row wrong: {b['Value recommendation:']}")

bn = block(card(verdict="NO VALUE at current price", spread_line=5.0))
ok(bn["ML pick:"] == "LAC predicted to win 64% of the time",
   "the ML pick must survive a no-value verdict")
ok(bn["Value recommendation:"] == "None at current prices", f"no-value row wrong: {bn}")

bp = block(card(verdict="no price", mkt_home=None, spread_line=5.0))
ok(bp["Value recommendation:"] == "No market price yet", f"no-price row wrong: {bp}")

bc = block(card(verdict="CAUTIOUS &mdash; small edge on ARI"))
ok("(small edge)" in bc["Value recommendation:"], "cautious value lost its qualifier")

# Away favourite: the pick names the away team and quotes its own probability.
ba = block(card(mu=-4.5, p_home=0.37, spread_line=-3.0, verdict="NO VALUE at current price"))
ok(ba["ML pick:"] == "ARI predicted to win 63% of the time", f"away favourite wrong: {ba}")

bt = block(card(mu=0.1, p_home=0.502, verdict="NO VALUE at current price"))
ok(bt["ML pick:"] == "too close to call", f"coin flip should not name a side: {bt}")

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
