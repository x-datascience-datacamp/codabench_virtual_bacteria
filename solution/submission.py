"""Sample submission — the starting point you copy and improve.

A submission is a benchopt solver: a ``submission.py`` defining
``class Solver(GeneCompetSolver)``. This baseline builds features from the
three data sources (``benchmark_utils.features.FeatureBuilder``), fits a Ridge
regression on ``log1p(counts)``, and calibrates a single Negative-Binomial
dispersion by minimizing the competition metric on the training set. It
returns ``(mu, sigma)`` per observation.

Run it locally exactly as the platform does::

    python tools/setup_data.py --demo         # or your own data
    BENCHOPT_DATA_HOME=phases/main/input_data/data \
        benchopt run benchmark -d Bacteria -s solution/submission.py

Ideas to improve it: richer promoter representations (k-mers, learned sequence
encoders), more experimental metadata, a neural network (see
``benchmark/solvers/baseline_nn.py``), or gene x condition interactions.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.features import FeatureBuilder
from benchmark_utils.nb_loss import nb_nll


class SubmissionModel:

    def __init__(self, numeric, categorical):
        self.fb = FeatureBuilder(numeric, categorical)
        # A plain sklearn estimator: Ridge on log1p(counts) via the target
        # transformer, so predictions come back on the count scale.
        self.reg = TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1)
        self.sigma = 1.0

    def fit(self, train_df, samples_df, genomes_dir):
        X = self.fb.fit(train_df, samples_df, genomes_dir).transform(
            train_df, samples_df, genomes_dir)
        y = train_df["counts"].to_numpy(dtype=float)
        self.reg.fit(X, y)
        mu = np.clip(self.reg.predict(X), 1e-6, None)
        res = minimize_scalar(
            lambda log_r: nb_nll(y, mu, np.exp(log_r)),
            bounds=(-4.0, 8.0), method="bounded")
        self.sigma = float(np.exp(res.x))
        return self

    def predict(self, test_df, samples_df, genomes_dir):
        X = self.fb.transform(test_df, samples_df, genomes_dir)
        mu = np.clip(self.reg.predict(X), 0.0, None)
        return np.column_stack([mu, np.full(len(test_df), self.sigma)])


class Solver(GeneCompetSolver):

    name = "submission"

    def get_model(self, meta):
        return SubmissionModel(
            meta["numeric_features"], meta["categorical_features"])

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
