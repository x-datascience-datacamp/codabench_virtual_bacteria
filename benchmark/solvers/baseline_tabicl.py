"""Baseline: TabICL, a pretrained tabular foundation model, in context.

TabICL predicts from a context of labelled rows, without training. Its inputs
are the gene and condition effects of the additive baseline, a flag for each
telling whether it is known, and the two normalization offsets; its target is
the normalized log count. The effects come from the training rows outside the
context, and some context rows hide them, so the model also sees the regions
with a new gene or a new run. A single NB dispersion is then calibrated on
held-out rows.

The checkpoint is read from ``tabicl-regressor-v2-20260212.ckpt`` next to
this file when present (as in a platform submission), else downloaded from
the Hugging Face Hub.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll

CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
CHUNK = 100_000
GENE_KEYS = ["genome_id", "feature_id"]
COND_KEYS = ["genome_id", "run_accession"]


def _numeric(df, col):
    if col not in df.columns:
        return np.zeros(len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)


def _offset(df):
    return _numeric(df, "log_length_ratio") + _numeric(df, "log_size_factor")


class TabICLNBModel:

    def __init__(self, n_context=20_000, n_estimators=2, mask_rate=0.25,
                 shrinkage=1.0, model_path=None, seed=42):
        self.n_context = n_context
        self.n_estimators = n_estimators
        self.mask_rate = mask_rate
        self.shrinkage = shrinkage
        self.model_path = model_path
        self.seed = seed

    def _effects(self, df, keys, values):
        stats = pd.DataFrame({"v": values}).groupby(
            [df[k].to_numpy() for k in keys])["v"].agg(["sum", "count"])
        return stats["sum"] / (stats["count"] + self.shrinkage)

    def _lookup(self, effects, df, keys):
        index = pd.MultiIndex.from_arrays([df[k].to_numpy() for k in keys])
        return effects.reindex(index).to_numpy(float)

    def _features(self, df, mask=None):
        alpha = self._lookup(self.alpha_, df, GENE_KEYS)
        beta = self._lookup(self.beta_, df, COND_KEYS)
        if mask is not None:
            alpha[mask[:, 0]] = np.nan
            beta[mask[:, 1]] = np.nan
        return np.column_stack([
            np.nan_to_num(alpha), np.nan_to_num(beta),
            ~np.isnan(alpha), ~np.isnan(beta),
            _numeric(df, "log_length_ratio"), _numeric(df, "log_size_factor"),
        ]).astype(np.float32)

    def fit(self, train_df, samples_df=None, genomes_dir=None):
        from tabicl import TabICLRegressor

        target = "readcount" if "readcount" in train_df.columns else "counts"
        y = train_df[target].to_numpy(float)
        log_y = np.log1p(np.clip(y, 0.0, None)) - _offset(train_df)

        # Context and calibration rows stay out of the effects they read.
        rng = np.random.default_rng(self.seed)
        n_ctx = min(self.n_context, len(y) // 3)
        held = rng.choice(len(y), 2 * n_ctx, replace=False)
        ctx, cal = held[:n_ctx], held[n_ctx:]
        rest = np.ones(len(y), bool)
        rest[held] = False

        rest_df = train_df.iloc[np.flatnonzero(rest)]
        self.mu_0_ = float(log_y[rest].mean())
        centered = log_y[rest] - self.mu_0_
        self.alpha_ = self._effects(rest_df, GENE_KEYS, centered)
        alpha = np.nan_to_num(self._lookup(self.alpha_, rest_df, GENE_KEYS))
        self.beta_ = self._effects(rest_df, COND_KEYS, centered - alpha)

        masks = [rng.random((len(rows), 2)) < self.mask_rate
                 for rows in (ctx, cal)]
        X_ctx = self._features(train_df.iloc[ctx], masks[0])
        self.regressor_ = TabICLRegressor(
            n_estimators=self.n_estimators, model_path=self.model_path,
            allow_auto_download=self.model_path is None,
            random_state=self.seed)
        self.regressor_.fit(X_ctx, log_y[ctx] - self.mu_0_)

        cal_df = train_df.iloc[cal]
        mu_cal = self._mu(self._features(cal_df, masks[1]), cal_df)
        res = minimize_scalar(lambda log_r: nb_nll(y[cal], mu_cal,
                                                   np.exp(log_r)),
                              bounds=(-4.0, 8.0), method="bounded")
        self.sigma_ = float(np.exp(res.x))
        return self

    def _mu(self, X, df):
        # Chunks bound the GPU memory of the test rows.
        log_pred = np.concatenate([
            self.regressor_.predict(X[i:i + CHUNK])
            for i in range(0, len(X), CHUNK)]) + self.mu_0_ + _offset(df)
        return np.maximum(np.expm1(np.clip(log_pred, -15.0, 15.0)), 0.5)

    def predict(self, test_df, samples_df=None, genomes_dir=None):
        mu = self._mu(self._features(test_df), test_df)
        return np.column_stack([mu, np.full(len(mu), self.sigma_)])


class Solver(GeneCompetSolver):

    name = "BaselineTabICL"

    requirements = ["pip::tabicl"]

    def get_model(self, meta):
        path = meta["submission_dir"] / CHECKPOINT
        return TabICLNBModel(model_path=path if path.exists() else None)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
