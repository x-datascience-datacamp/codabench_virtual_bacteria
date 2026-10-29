"""Baseline Level 2: Gradient Boosting with NB dispersion grid search.

Inputs restricted to the 3 unique IDs and 2 normalization offsets:
- Categoricals : genome_id, feature_id, run_accession
- Numerics     : log_length_ratio, log_size_factor

Procedure:
1. Train a Gradient Boosting Regressor on train fold predicting
   log1p(readcount).
2. Evaluate Negative-Binomial NLL across candidate dispersion values
   r in (0.5, 1.0, 2.0, 5.0, 10.0).
3. Select optimal dispersion r* minimizing validation NB-NLL.
4. Output (mu, r*) predictions.

If CatBoost is installed, it is used with native categorical handling.
Otherwise, falls back to scikit-learn's HistGradientBoostingRegressor.
"""

import numpy as np
import pandas as pd

from benchmark_utils.base_solver import GeneCompetSolver
from benchmark_utils.nb_loss import nb_nll


def _extract_tabular_features(df: pd.DataFrame):
    """Extract standard categorical IDs and numeric offsets."""
    cat_cols = ["genome_id", "feature_id", "run_accession"]
    num_cols = ["log_length_ratio", "log_size_factor"]

    X = pd.DataFrame(index=df.index)
    for c in cat_cols:
        X[c] = df[c].astype(str) if c in df.columns else "missing"

    for c in num_cols:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0

    return X, cat_cols, num_cols


class CatBoostNBModel:

    def __init__(
        self,
        iterations: int = 150,
        lr: float = 0.1,
        candidates=(0.5, 1.0, 2.0, 5.0, 10.0),
    ):
        self.iterations = iterations
        self.lr = lr
        self.candidates = list(candidates)
        self.best_model = None
        self.best_r = float(self.candidates[0])
        self.is_catboost = False

    def fit(self, train_df: pd.DataFrame, samples_df=None, genomes_dir=None):
        target_col = (
            "readcount" if "readcount" in train_df.columns else "counts"
        )
        y = train_df[target_col].to_numpy(dtype=float)
        X, cat_cols, num_cols = _extract_tabular_features(train_df)

        n = len(y)
        # 15% validation split for dispersion hyperparameter selection
        rng = np.random.default_rng(42)
        perm = rng.permutation(n)
        val_size = max(int(0.15 * n), 10)
        val_idx = perm[:val_size]
        tr_idx = perm[val_size:]

        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        yt_tr = np.log1p(np.clip(y_tr, 0.0, None))

        # Try CatBoost first
        try:
            from catboost import CatBoostRegressor

            cb = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.lr,
                cat_features=cat_cols,
                loss_function="RMSE",
                verbose=0,
                random_seed=42,
                allow_writing_files=False,
            )
            cb.fit(X_tr, yt_tr)
            pred_val = cb.predict(X_val)
            mu_val = np.maximum(np.expm1(pred_val), 0.5)

            best_loss = float("inf")
            self.best_model = cb
            self.best_r = float(self.candidates[0])

            for r_cand in self.candidates:
                loss = nb_nll(y_val, mu_val, r_cand)
                if loss < best_loss:
                    best_loss = loss
                    self.best_r = float(r_cand)

            self.is_catboost = True
            return self

        except ImportError:
            pass

        # Fallback to sklearn HistGradientBoostingRegressor
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder

        oe = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        X_cat_tr = oe.fit_transform(X_tr[cat_cols])
        X_tr_enc = np.column_stack([X_cat_tr, X_tr[num_cols].to_numpy()])

        X_cat_val = oe.transform(X_val[cat_cols])
        X_val_enc = np.column_stack([X_cat_val, X_val[num_cols].to_numpy()])

        # HistGradientBoosting bins at most 255 categories: larger id columns
        # go in as ordinal codes.
        categorical_mask = [len(c) <= 255 for c in oe.categories_] + [
            False, False]
        hgb = HistGradientBoostingRegressor(
            max_iter=self.iterations,
            learning_rate=self.lr,
            categorical_features=categorical_mask,
            random_state=42,
        )
        hgb.fit(X_tr_enc, yt_tr)
        pred_val = hgb.predict(X_val_enc)
        mu_val = np.maximum(np.expm1(pred_val), 0.5)

        best_loss = float("inf")
        self.best_model = hgb
        self.best_r = float(self.candidates[0])

        for r_cand in self.candidates:
            loss = nb_nll(y_val, mu_val, r_cand)
            if loss < best_loss:
                best_loss = loss
                self.best_r = float(r_cand)

        self.encoder_ = oe
        self.is_catboost = False
        return self

    def predict(
        self, test_df: pd.DataFrame, samples_df=None, genomes_dir=None
    ):
        X, cat_cols, num_cols = _extract_tabular_features(test_df)
        n = len(X)
        if self.best_model is None:
            return np.column_stack([np.ones(n), np.ones(n)])

        if self.is_catboost:
            pred_log = self.best_model.predict(X)
        else:
            X_cat = self.encoder_.transform(X[cat_cols])
            X_enc = np.column_stack([X_cat, X[num_cols].to_numpy()])
            pred_log = self.best_model.predict(X_enc)

        mu = np.maximum(np.expm1(np.clip(pred_log, -15.0, 15.0)), 0.5)
        return np.column_stack([mu, np.full(n, self.best_r)])


class Solver(GeneCompetSolver):

    name = "BaselineCatBoost"

    requirements = ["pip::catboost", "scikit-learn"]

    parameters = {"iterations": [150], "lr": [0.1]}

    def get_model(self, meta):
        return CatBoostNBModel(iterations=self.iterations, lr=self.lr)

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
