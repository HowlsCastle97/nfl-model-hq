"""The totals plumbing: scoring rules, push mass, and the target hook.

Nothing here trains a network. These are the pieces where a quiet sign error or
an off by one on a whole number strike would make a wrong price look plausible,
so each is checked against a number worked out by hand or by a second method.
"""
import numpy as np
import pandas as pd

from models import (crps_ensemble, crps_gaussian, gaussian_nll,
                    prob_total_over, prob_total_over_pmf)
from walkforward import evaluate, walk_forward

fail = []


def ok(cond, msg):
    if not cond:
        fail.append(msg)


# CRPS of a Gaussian against itself: the known value for y = mu is
# sigma * (2 * phi(0) - 1/sqrt(pi)) = sigma * (1/sqrt(pi)) * (sqrt(2/pi) ... )
# rather than trust an algebraic rearrangement, check the closed form against the
# sample based version, which is derived independently.
rng = np.random.default_rng(0)
mu, sigma = np.array([45.0, 38.0, 52.0]), np.array([13.0, 10.0, 16.0])
y = np.array([44.0, 51.0, 30.0])
closed = crps_gaussian(y, mu, sigma)
samples = mu[:, None] + sigma[:, None] * rng.standard_normal((3, 40000))
sampled = crps_ensemble(y, samples)
ok(np.allclose(closed, sampled, atol=0.06),
   f"closed form CRPS {closed.round(3)} disagrees with samples {sampled.round(3)}")
print(f"CRPS closed form {closed.round(3)} vs 40k samples {sampled.round(3)}")

# CRPS is minimised by the truth and grows as the forecast moves away from it.
near = crps_gaussian([45.0], [45.0], [13.0])[0]
far = crps_gaussian([45.0], [55.0], [13.0])[0]
ok(far > near, "CRPS did not punish a forecast ten points off")

# A sharper forecast that is right scores better; a sharper one that is wrong
# scores worse. Both directions, because a score that only rewards sharpness
# would happily ship an overconfident model.
ok(crps_gaussian([45.0], [45.0], [8.0])[0] < near, "sharp and right scored worse")
ok(crps_gaussian([65.0], [45.0], [8.0])[0]
   > crps_gaussian([65.0], [45.0], [13.0])[0], "sharp and wrong scored better")

# Push mass. A whole number strike can land exactly on the total; a half point
# strike cannot. The three parts must always sum to one.
po, pu, pp = prob_total_over(45.0, 13.0, 45.0)
ok(pp > 0, "a whole number strike was given no push mass")
ok(abs(po + pu + pp - 1) < 1e-9, "over, under and push do not sum to one")
ok(abs(po - pu) < 1e-9, f"a strike at the mean should be symmetric: {po} vs {pu}")
ho, hu, hp = prob_total_over(45.0, 13.0, 45.5)
ok(hp == 0, f"a half point strike cannot push, got {hp}")
ok(abs(ho + hu - 1) < 1e-9, "half point over and under do not sum to one")
# The naive tail probability hands the push to the under. That was the bug this
# correction exists to avoid, so the corrected over must be the larger number.
from scipy.stats import norm
naive = 1 - norm.cdf((45.0 - 45.0) / 13.0)
ok(po < naive, "the correction did not take the push out of the over")
print(f"strike 45 on mu 45: over {po:.4f} under {pu:.4f} push {pp:.4f} "
      f"(naive over {naive:.4f})")

# The discrete version takes a push literally.
support = np.arange(0, 101)
pmf = np.zeros(101)
pmf[[43, 44, 45, 46, 47]] = [0.1, 0.2, 0.3, 0.2, 0.2]
do, du, dp = prob_total_over_pmf(support, pmf, 45)
ok(abs(dp - 0.3) < 1e-12, f"discrete push mass {dp} should be the mass on 45")
ok(abs(do - 0.4) < 1e-12 and abs(du - 0.3) < 1e-12,
   f"discrete split wrong: {do}, {du}")

# The target hook. Two walk-forwards over the same frame, one on each target,
# must come back scoring the column they were asked for and nothing else.
n = 600
rs = np.random.default_rng(1)
frame = pd.DataFrame({
    "game_id": [f"g{i}" for i in range(n)],
    "season": np.repeat([2021, 2022, 2023], n // 3),
    "week": np.tile(np.arange(1, 21), n // 20),
    "home_team": "AAA", "away_team": "BBB",
    "x1": rs.standard_normal(n), "x2": rs.standard_normal(n),
})
frame["y"] = 2 * frame["x1"] + rs.standard_normal(n)
frame["y_total"] = 45 + 5 * frame["x2"] + rs.standard_normal(n)
frame["total_line"] = 45.0

m = walk_forward(frame, ["x1", "x2"], 2023, lam=1.0)
t = walk_forward(frame, ["x1", "x2"], 2023, lam=1.0, target="y_total",
                 keep_cols=("total_line",))
ok(abs(m["y"].mean()) < 1.0, f"margin target came back centred on {m['y'].mean():.2f}")
ok(40 < t["y"].mean() < 50, f"totals target came back centred on {t['y'].mean():.2f}")
ok("total_line" in t.columns, "keep_cols did not carry the market total through")
ok("total_line" not in m.columns, "the margin run invented a total_line column")
ok(len(m) == len(t), "the two targets graded different numbers of games")
ok(t["mu"].mean() > 40, f"totals mu {t['mu'].mean():.1f} is not on the totals scale")
print(f"target hook: margin mu {m['mu'].mean():+.2f}, totals mu {t['mu'].mean():.1f}")

# A NaN target is dropped rather than predicted. Half the final season is wiped.
holed = frame.copy()
holed.loc[holed.index[-40:], "y_total"] = np.nan
h = walk_forward(holed, ["x1", "x2"], 2023, lam=1.0, target="y_total")
ok(len(h) == len(t) - 40, f"unplayed rows leaked in: {len(h)} vs {len(t) - 40}")

# evaluate reports CRPS alongside NLL, and both agree with a direct computation.
e = evaluate(t)
ok(abs(e["crps"] - crps_gaussian(t["y"], t["mu"], t["sigma"]).mean()) < 1e-12,
   "evaluate's CRPS does not match the function it claims to call")
ok(abs(e["nll"] - gaussian_nll(t["y"].values, t["mu"].values,
                               t["sigma"].values).mean()) < 1e-12,
   "evaluate's NLL drifted")
print(f"evaluate: CRPS {e['crps']:.3f} NLL {e['nll']:.3f} RMSE {e['rmse']:.3f}")

# --- The quantile path. A quantile engine is scored without assuming a shape, so
# the way to test those functions is to feed them a shape whose answers are known:
# the exact quantiles of a Gaussian. Everything should come back agreeing with the
# closed form, within the error of a 23 point grid.
from models import (crps_from_quantiles, moments_from_quantiles,
                    nll_from_quantiles, prob_total_over_quantiles)

TAUS = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
gm, gs = np.array([45.0, 38.0, 52.0]), np.array([13.0, 10.0, 16.0])
gy = np.array([44.0, 51.0, 30.0])
GQ = gm[:, None] + gs[:, None] * norm.ppf(TAUS)[None, :]

cq = crps_from_quantiles(gy, GQ, TAUS)
cg = crps_gaussian(gy, gm, gs)
ok(np.allclose(cq, cg, rtol=0.03),
   f"quantile CRPS {cq.round(3)} disagrees with the closed form {cg.round(3)}")
print(f"quantile CRPS {cq.round(3)} vs Gaussian closed form {cg.round(3)}")

mq, sq = moments_from_quantiles(GQ, TAUS)
ok(np.allclose(mq, gm, atol=0.2), f"recovered means {mq.round(2)} not {gm}")
# Truncation at 1% and 99% loses a little of the tail, so sd comes back low. The
# test pins the direction and the size, because a surprise here would silently
# make every published sigma too small.
ok(np.all(sq < gs) and np.all(sq > 0.9 * gs),
   f"recovered sds {sq.round(2)} should sit just under {gs}")
print(f"recovered sd {sq.round(2)} against true {gs} (truncation, as expected)")

po_q, pu_q, pp_q = prob_total_over_quantiles(GQ, TAUS, np.array([45.0, 38.5, 52.0]))
po_g, pu_g, pp_g = prob_total_over(gm, gs, np.array([45.0, 38.5, 52.0]))
ok(np.allclose(po_q, po_g, atol=0.02),
   f"quantile over probabilities {po_q.round(3)} vs Gaussian {po_g.round(3)}")
ok(abs(pp_q[1]) < 1e-9, "a half point strike pushed in the quantile version")
ok(pp_q[0] > 0 and pp_q[2] > 0, "whole number strikes got no push mass")
print(f"P(over) quantile {po_q.round(3)} vs Gaussian {po_g.round(3)}")

# A result outside the grid must cost a lot and stay finite. An infinite NLL would
# take a whole season's average with it.
far = nll_from_quantiles(np.array([120.0]), GQ[:1], TAUS)
mid = nll_from_quantiles(np.array([45.0]), GQ[:1], TAUS)
ok(np.isfinite(far).all(), "a total outside the quantile grid gave infinite NLL")
ok(far[0] > mid[0] + 2, f"an absurd total scored {far[0]:.2f} against {mid[0]:.2f}")
print(f"NLL at the mean {mid[0]:.2f}, at 120 points {far[0]:.2f}, both finite")

# Sorting is not optional: separate quantile fits can cross, and a crossed row
# would make the CDF non monotone and the density negative.
crossed = GQ.copy()
crossed[:, [5, 6]] = crossed[:, [6, 5]]
ok(np.allclose(crps_from_quantiles(gy, np.sort(crossed, axis=1), TAUS), cq),
   "sorting a crossed row changed its CRPS")
mS, sS = moments_from_quantiles(crossed, TAUS)
ok(np.allclose(sS, sq), "moments_from_quantiles did not sort its input")
print("crossed quantiles are sorted before use")

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail
              else "PASS: totals scoring, push mass and the target hook"))
raise SystemExit(1 if fail else 0)
