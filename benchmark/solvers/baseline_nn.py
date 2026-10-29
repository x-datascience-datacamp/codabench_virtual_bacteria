"""Neural-network baseline — the getting-started notebook, wired to the metric.

Builds the design matrix from the three data sources
(``benchmark_utils.features``), trains a small MLP to regress ``log1p(counts)``
(stable) on at most ``max_rows`` training observations, then estimates the
single NB dispersion ``sigma`` that minimizes the competition metric on them.
Prediction returns ``(mu, sigma)``.
Depends only on the base stack + torch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.features import FeatureBuilder
from benchmark_utils.nb_loss import nb_nll


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class CountNet(nn.Module):
    """MLP mapping features to a ``log1p(count)`` prediction.

    The output is anchored at the training mean log-count so it starts near
    the constant floor and training stays stable.
    """

    def __init__(self, n_features, hidden, log_mu0=0.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        )
        self.head = nn.Linear(hidden // 2, 1)
        self.register_buffer("log_mu0", torch.as_tensor(log_mu0,
                                                        dtype=torch.float32))

    def forward(self, x):
        return (self.log_mu0 + self.head(self.body(x)).squeeze(1)).clamp(0.0,
                                                                         14.0)


class NBModel:
    """Fit the design + net + dispersion; predict ``(mu, sigma)``."""

    def __init__(self, meta, n_epochs, lr, hidden, max_rows, device):
        self.fb = FeatureBuilder(meta)
        self.n_epochs = n_epochs
        self.max_rows = max_rows
        self.lr = lr
        self.hidden = hidden
        self.device = device
        self.net = None
        self.sigma = 1.0

    def _mu(self, X):
        self.net.eval()
        with torch.no_grad():
            pred = self.net(torch.as_tensor(X, device=self.device))
        # A count far above a near-zero mean costs ~ y * log(1 / mu).
        return np.maximum(np.expm1(pred.cpu().numpy()), 0.5)

    def fit(self, train_df, samples_df, genomes_dir):
        if len(train_df) > self.max_rows:
            train_df = train_df.sample(self.max_rows, random_state=0)
        X = self.fb.fit(train_df, samples_df, genomes_dir).transform(
            train_df, samples_df)
        target = "readcount" if "readcount" in train_df else "counts"
        y = train_df[target].to_numpy(dtype=np.float32)
        Xt = torch.as_tensor(X, device=self.device)
        yt = torch.as_tensor(np.log1p(y), device=self.device)

        torch.manual_seed(0)  # deterministic weight init / shuffling
        log_mu0 = float(np.mean(np.log1p(y)))
        self.net = CountNet(X.shape[1], self.hidden, log_mu0).to(self.device)
        # Weight decay keeps the net from fingerprinting the training genes
        # through their length and promoter composition.
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=1e-2)
        # At most ~200 steps per epoch.
        n = len(y)
        bs = max(512, n // 200)
        self.net.train()
        for _ in range(self.n_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = F.smooth_l1_loss(self.net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()

        # Calibrate a single dispersion by minimizing the metric on train.
        mu = np.clip(self._mu(X), 1e-6, None)
        res = minimize_scalar(
            lambda log_r: nb_nll(y, mu, np.exp(log_r)),
            bounds=(-4.0, 8.0), method="bounded")
        self.sigma = float(np.exp(res.x))
        return self

    def predict(self, test_df, samples_df, genomes_dir):
        n = len(test_df)
        if self.net is None:  # inference-only run without training: floor
            return np.column_stack([np.ones(n), np.ones(n)])
        # Chunks keep the dense design matrix small on the large test sets.
        mu = np.concatenate([
            self._mu(self.fb.transform(test_df.iloc[i:i + 200_000],
                                       samples_df))
            for i in range(0, n, 200_000)])
        return np.column_stack([np.clip(mu, 0.0, None),
                                np.full(n, self.sigma)])


class Solver(GeneCompetSolver):

    name = "BaselineNN"

    requirements = ["scikit-learn", "pip::torch", "pip::biopython"]

    parameters = {"n_epochs": [150], "lr": [1e-2], "hidden": [64],
                  "max_rows": [1_000_000]}
    test_parameters = {"n_epochs": [5], "lr": [1e-2], "hidden": [16],
                       "max_rows": [1_000_000]}

    def get_model(self, meta):
        return NBModel(meta, self.n_epochs, self.lr, self.hidden,
                       self.max_rows, _device())

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
