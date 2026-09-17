"""Negative-Binomial scoring for the gene-expression competition.

A submission predicts, per (gene x sample) observation, two numbers:

- ``mu``    : the expected RNA-seq count (mean of the Negative Binomial), >= 0;
- ``sigma`` : the NB dispersion / size parameter ``r``, > 0. The count variance
              is ``mu + mu**2 / sigma``; large sigma -> Poisson-like.

The ranking metric is the mean NB negative log-likelihood (lower is better).
MAE and R^2 on ``log1p(counts)`` are reported alongside for interpretability.
"""

import numpy as np
from scipy.special import gammaln

_EPS = 1e-8


def nb_nll(counts, mu, sigma):
    """Mean Negative-Binomial negative log-likelihood (lower is better)."""
    y = np.asarray(counts, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), _EPS, None)
    r = np.clip(np.asarray(sigma, dtype=np.float64), _EPS, None)
    # log-likelihood of NB(mean=mu, size=r):
    #   lgamma(y+r) - lgamma(r) - lgamma(y+1)
    #   + r*log(r/(r+mu)) + y*log(mu/(r+mu))
    log_denom = np.log(r + mu)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + r * (np.log(r) - log_denom) + y * (np.log(mu) - log_denom))
    return float(-np.mean(ll))


def log_mae(counts, mu):
    """Mean absolute error on ``log1p(counts)``."""
    y = np.asarray(counts, dtype=np.float64)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 0.0, None)
    return float(np.mean(np.abs(np.log1p(y) - np.log1p(mu))))


def log_r2(counts, mu):
    """R^2 on ``log1p(counts)`` (0 when the target has no variance)."""
    y = np.log1p(np.asarray(counts, dtype=np.float64))
    p = np.log1p(np.clip(np.asarray(mu, dtype=np.float64), 0.0, None))
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def check_predictions(preds, n):
    """Validate a submission's ``(n, 2)`` prediction array; raise on defects.

    Returns ``(mu, sigma)`` on success. Fails loudly on a wrong shape, NaN/inf,
    a negative mean, or a non-positive dispersion — a bad number must crash the
    run, not sit on the leaderboard.
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.shape != (n, 2):
        raise ValueError(
            f"predictions must have shape ({n}, 2), got {preds.shape}"
        )
    if not np.isfinite(preds).all():
        raise ValueError("predictions contain NaN or inf")
    mu, sigma = preds[:, 0], preds[:, 1]
    if (mu < 0).any():
        raise ValueError("mu (expected count) must be >= 0")
    if (sigma <= 0).any():
        raise ValueError("sigma (NB dispersion) must be > 0")
    return mu, sigma
