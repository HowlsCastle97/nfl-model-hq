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


def prob_margin_over(mu, sigma, strike):
    return 1.0 - norm.cdf((strike - mu) / sigma)
