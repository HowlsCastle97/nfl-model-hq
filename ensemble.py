import numpy as np
import torch
import torch.nn as nn


class HeteroMLP(nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, 2)
        nn.init.constant_(self.head.bias, 0.0)
        with torch.no_grad():
            self.head.bias[1] = 0.5

    def forward(self, x):
        out = self.head(self.body(x))
        mu = out[:, 0]
        sigma = nn.functional.softplus(out[:, 1]) + 1e-3
        return mu, sigma


class DeepEnsemble:
    """Ensemble of heteroscedastic MLPs trained on weighted Gaussian NLL.

    Matches the walk-forward harness interface: fit(X, y, sample_weight) and
    predict_dist(X). predict_split(X) additionally returns the aleatoric and
    epistemic components of the predictive variance.
    """

    def __init__(self, n_members=5, hidden=32, weight_decay=1e-3,
                 epochs=300, lr=1e-2, seed=0, beta=0.5):
        self.n_members = n_members
        self.hidden = hidden
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.beta = beta
        self.members_ = []

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)

        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        wsum = w.sum()
        self.y_mean_ = (w * y).sum() / wsum
        self.y_std_ = float(np.sqrt((w * (y - self.y_mean_) ** 2).sum() / wsum))

        Xt = torch.tensor((X - self.x_mean_) / self.x_std_, dtype=torch.float32)
        yt = torch.tensor((y - self.y_mean_) / self.y_std_, dtype=torch.float32)
        wt = torch.tensor(w / w.mean(), dtype=torch.float32)

        self.members_ = []
        for j in range(self.n_members):
            torch.manual_seed(self.seed * 1000 + j)
            model = HeteroMLP(d, self.hidden)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
            model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                mu, sigma = model(Xt)
                per = torch.log(sigma) + 0.5 * ((yt - mu) / sigma) ** 2
                if self.beta > 0:
                    per = per * (sigma.detach() ** (2 * self.beta))
                nll = (wt * per).mean()
                nll.backward()
                opt.step()
                sched.step()
            model.eval()
            self.members_.append(model)
        return self

    def _member_predictions(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = torch.tensor((X - self.x_mean_) / self.x_std_, dtype=torch.float32)
        mus, sigmas = [], []
        with torch.no_grad():
            for model in self.members_:
                mu, sigma = model(Xt)
                mus.append(self.y_mean_ + self.y_std_ * mu.numpy())
                sigmas.append(self.y_std_ * sigma.numpy())
        return np.stack(mus), np.stack(sigmas)

    def predict_split(self, X):
        mus, sigmas = self._member_predictions(X)
        mu = mus.mean(axis=0)
        aleatoric = (sigmas ** 2).mean(axis=0)
        epistemic = ((mus - mu) ** 2).mean(axis=0)
        return mu, aleatoric, epistemic

    def predict_dist(self, X):
        mu, aleatoric, epistemic = self.predict_split(X)
        return mu, np.sqrt(aleatoric + epistemic)
