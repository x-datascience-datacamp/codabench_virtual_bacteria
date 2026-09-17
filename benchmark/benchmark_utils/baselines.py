"""Reference models used by the built-in solvers and by ``benchopt test``.

Each model follows the competition contract: a ``predict(test_df, samples_df,
genomes_dir)`` returning an ``(N, 2)`` array of ``(mu, sigma)``.
"""

import numpy as np


class ConstantModel:
    """Predict a constant ``(mu, sigma)`` for every observation.

    ``mu`` is set to the mean training count (a sane floor); ``sigma`` is a
    fixed dispersion. Serves as the leaderboard floor and as the dummy result
    for ``benchopt test``.
    """

    def __init__(self, mu=1.0, sigma=1.0):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def fit(self, train_df, samples_df, genomes_dir):
        if "counts" in train_df.columns:
            self.mu = float(max(train_df["counts"].mean(), 1e-6))
        return self

    def predict(self, test_df, samples_df, genomes_dir):
        n = len(test_df)
        return np.column_stack([
            np.full(n, self.mu), np.full(n, self.sigma),
        ])
