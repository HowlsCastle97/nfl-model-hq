import numpy as np
from scipy.stats import norm


class LinearGaussianModel:
    """Linear model with Gaussian predictive distribution over home margin.

    Fit by closed-form weighted ridge. lam = 0 gives the MLE solution;
    lam > 0 is MAP with a mean-zero Gaussian prior, where lam = sigma^2 / sigma_prior^2.
    The intercept is not penalized. sigma is fit by weighted MLE on train residuals.
    """

    def __init__(self, lam=0.0):
        self.lam = lam
        self.intercept_ = None
        self.theta_ = None
        self.sigma_ = None
        self.x_mean_ = None
        self.x_std_ = None

    def _standardize(self, X):
        return (X - self.x_mean_) / self.x_std_

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        Z = self._standardize(X)

        wsum = w.sum()
        zbar = (w[:, None] * Z).sum(axis=0) / wsum
        ybar = (w * y).sum() / wsum
        Zc = Z - zbar
        yc = y - ybar

        A = (Zc * w[:, None]).T @ Zc + self.lam * np.eye(d)
        b = (Zc * w[:, None]).T @ yc
        self.theta_ = np.linalg.solve(A, b)
        self.intercept_ = ybar - zbar @ self.theta_

        r = y - self.predict_mu(X)
        self.sigma_ = float(np.sqrt((w * r**2).sum() / wsum))
        return self

    def predict_mu(self, X):
        Z = self._standardize(np.asarray(X, dtype=float))
        return self.intercept_ + Z @ self.theta_

    def predict_dist(self, X):
        mu = self.predict_mu(X)
        return mu, np.full_like(mu, self.sigma_)


def gaussian_nll(y, mu, sigma):
    return 0.5 * np.log(2 * np.pi * sigma**2) + (y - mu) ** 2 / (2 * sigma**2)


def crps_gaussian(y, mu, sigma):
    """Continuous ranked probability score, closed form for a Gaussian.

    In points, and lower is better. It scores the whole predictive distribution
    against the single number that happened, and unlike NLL it does not blow up
    on one badly placed outlier, which is why it is the yardstick for the totals
    engines: a 70 point game should cost a model something, not everything.
    """
    y, mu, sigma = np.asarray(y, float), np.asarray(mu, float), np.asarray(sigma, float)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_ensemble(y, samples):
    """CRPS from samples, for a predictive distribution with no closed form.

    Energy form: E|Y - y| minus half E|Y - Y'|, both over the sample set.
    samples is (n_obs, n_samples). Exact for the empirical distribution, so a
    simulator and a Gaussian can be compared on one scale.
    """
    y = np.asarray(y, float)[:, None]
    S = np.sort(np.asarray(samples, float), axis=1)
    m = S.shape[1]
    term1 = np.abs(S - y).mean(axis=1)
    # The pairwise sum over a sorted row collapses to a single weighted sum.
    w = (2 * np.arange(m) + 1 - m)
    term2 = (S * w).sum(axis=1) / (m * m)
    return term1 - term2


def crps_from_quantiles(y, Q, taus):
    """CRPS of a predictive distribution given only as a set of quantiles.

    CRPS is twice the integral of the pinball loss over the probability level, so
    a model that outputs quantiles can be scored on exactly the same axis as a
    Gaussian one with no distributional assumption bolted on afterwards. The
    integral is a trapezoid over the tau grid, which is why the grid should reach
    into both tails: everything outside it is invisible to this estimate.

    Q is (n_obs, n_taus), taus the matching probability levels, ascending.
    """
    y = np.asarray(y, float)[:, None]
    Q = np.asarray(Q, float)
    taus = np.asarray(taus, float)
    d = y - Q
    pinball = np.where(d >= 0, taus * d, (taus - 1) * d)
    return 2.0 * np.trapz(pinball, taus, axis=1)


def quantile_density(Q, taus, tail_scale=1.0):
    """Piecewise linear density implied by a quantile function, with exponential tails.

    Between two quantiles the distribution is treated as uniform, which is the
    only thing the quantiles actually assert. Beyond the outermost ones the mass
    that is left over decays exponentially at the rate of the nearest interval,
    so a value outside the grid gets a finite density instead of a zero that
    would make NLL infinite.

    Returns a function of y, vectorised over the same rows as Q.
    """
    Q = np.sort(np.asarray(Q, float), axis=1)
    taus = np.asarray(taus, float)
    widths = np.diff(Q, axis=1)
    widths = np.maximum(widths, 1e-6)
    dens = np.diff(taus)[None, :] / widths
    lo_rate = dens[:, 0] / max(taus[0], 1e-6)
    hi_rate = dens[:, -1] / max(1 - taus[-1], 1e-6)

    def pdf(y):
        y = np.asarray(y, float)
        out = np.empty(len(y))
        for i, v in enumerate(y):
            row = Q[i]
            if v < row[0]:
                out[i] = taus[0] * lo_rate[i] * np.exp(
                    -lo_rate[i] * (row[0] - v) / max(tail_scale, 1e-6))
            elif v > row[-1]:
                out[i] = (1 - taus[-1]) * hi_rate[i] * np.exp(
                    -hi_rate[i] * (v - row[-1]) / max(tail_scale, 1e-6))
            else:
                j = min(np.searchsorted(row, v, side="right") - 1, len(row) - 2)
                out[i] = dens[i, j]
        return np.maximum(out, 1e-12)

    return pdf


def nll_from_quantiles(y, Q, taus):
    """Negative log predictive density under the piecewise linear quantile fit."""
    return -np.log(quantile_density(Q, taus)(y))


def moments_from_quantiles(Q, taus):
    """Mean and sd of the distribution a quantile grid describes.

    The quantile function integrated over tau, by trapezoid. Truncated at the
    outermost tau, so the sd is a slight underestimate; that is stated rather than
    corrected, since correcting it would mean inventing tails.
    """
    Q = np.sort(np.asarray(Q, float), axis=1)
    taus = np.asarray(taus, float)
    span = taus[-1] - taus[0]
    mu = np.trapz(Q, taus, axis=1) / span
    var = np.trapz((Q - mu[:, None]) ** 2, taus, axis=1) / span
    return mu, np.sqrt(var)


def prob_total_over_quantiles(Q, taus, strike, push_half_point=0.5):
    """Over, under and push probability read straight off a quantile function.

    The CDF at a point is found by inverting the grid linearly. A whole number
    strike gets the same half point of push mass the Gaussian version uses, so the
    two engines price a push the same way and only their shapes differ.
    """
    Q = np.sort(np.asarray(Q, float), axis=1)
    taus = np.asarray(taus, float)
    strike = np.atleast_1d(np.asarray(strike, float))
    whole = np.isclose(strike, np.round(strike))
    hi = np.where(whole, strike + push_half_point, strike)
    lo = np.where(whole, strike - push_half_point, strike)

    def cdf(v):
        out = np.empty(len(Q))
        for i in range(len(Q)):
            out[i] = np.interp(v[i], Q[i], taus, left=0.0, right=1.0)
        return out

    p_under = cdf(lo)
    p_over = 1.0 - cdf(hi)
    return p_over, p_under, np.clip(1.0 - p_over - p_under, 0.0, 1.0)


def prob_total_over(mu, sigma, strike, push_half_point=0.5):
    """P(total > strike) under a Gaussian, with the push handled.

    A whole number total can land exactly on the strike, which is a push and
    neither a win nor a loss. A continuous distribution gives that outcome zero
    mass, so a naive tail probability quietly prices a push as a loss for the
    over. The continuity correction spreads the integer's mass over the interval
    around it and returns the three parts explicitly.

    Returns (p_over, p_under, p_push). A half point strike cannot push, so its
    push mass is zero and the other two sum to one.
    """
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    strike = np.asarray(strike, float)
    whole = np.isclose(strike, np.round(strike))
    hi = np.where(whole, strike + push_half_point, strike)
    lo = np.where(whole, strike - push_half_point, strike)
    p_over = 1.0 - norm.cdf((hi - mu) / sigma)
    p_under = norm.cdf((lo - mu) / sigma)
    return p_over, p_under, np.clip(1.0 - p_over - p_under, 0.0, 1.0)


def prob_total_over_pmf(support, pmf, strike):
    """The same three way split for a discrete predictive distribution.

    support is the grid of totals the distribution is defined on, pmf the mass on
    each point, either one row shared or one row per game. Mass sitting exactly
    on a whole number strike is the push, taken literally instead of approximated.
    """
    support = np.asarray(support, float)
    P = np.atleast_2d(np.asarray(pmf, float))
    strike = np.atleast_1d(np.asarray(strike, float))[:, None]
    over = support[None, :] > strike
    under = support[None, :] < strike
    push = ~(over | under)
    return ((P * over).sum(axis=1), (P * under).sum(axis=1), (P * push).sum(axis=1))


def prob_margin_over(mu, sigma, strike):
    return 1.0 - norm.cdf((strike - mu) / sigma)
