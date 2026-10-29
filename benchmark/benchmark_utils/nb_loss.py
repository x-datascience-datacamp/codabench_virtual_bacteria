"""Negative-Binomial scoring for the gene-expression competition.

A submission predicts, per (gene x sample) observation, two numbers:

- ``mu``    : the expected RNA-seq count (mean of the Negative Binomial), >= 0;
- ``sigma`` : the NB dispersion / size parameter ``r``, > 0. The count variance
              is ``mu + mu**2 / sigma``; large sigma -> Poisson-like.

The ranking metric is the mean NB negative log-likelihood (lower is better),
each observation's loss capped at ``NLL_CAP`` so that one catastrophic
prediction (e.g. ``mu`` near 0 for a highly expressed gene) cannot dominate a
region. MAE and R^2 on ``log1p(counts)`` are reported alongside for
interpretability.
"""

import numpy as np
from scipy.special import gammaln
from sklearn.metrics import mean_absolute_error, r2_score

_EPS = 1e-8
# Typical observations cost about 7 nats, 99% of them less than 50.
NLL_CAP = 100.0


def nb_nll(counts, mu, sigma):
    """Mean Negative-Binomial negative log-likelihood (lower is better), each
    observation's loss capped at ``NLL_CAP``.

    Continuous Gamma relaxation that supports both integer and float counts:
        log_pmf = gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r*log(r) + y*log(mu) - (y + r)*log(r + mu)
    """
    y = np.clip(np.asarray(counts, dtype=np.float64), 0.0, None)
    mu = np.clip(np.asarray(mu, dtype=np.float64), _EPS, None)
    r = np.clip(np.asarray(sigma, dtype=np.float64), _EPS, None)

    y_log_mu = np.where(y > 0, y * np.log(mu), 0.0)
    log_pmf = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1.0)
        + r * np.log(r)
        + y_log_mu
        - (y + r) * np.log(r + mu)
    )
    return float(np.mean(np.minimum(-log_pmf, NLL_CAP)))


def log_mae(counts, mu):
    """Mean absolute error on ``log1p(counts)``."""
    y = np.log1p(np.asarray(counts, dtype=np.float64))
    p = np.log1p(np.clip(np.asarray(mu, dtype=np.float64), 0.0, None))
    return float(mean_absolute_error(y, p))


def log_r2(counts, mu):
    """R^2 on ``log1p(counts)``."""
    y = np.log1p(np.asarray(counts, dtype=np.float64))
    p = np.log1p(np.clip(np.asarray(mu, dtype=np.float64), 0.0, None))
    return float(r2_score(y, p))


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
