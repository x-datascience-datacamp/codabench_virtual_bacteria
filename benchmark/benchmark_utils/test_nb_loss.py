"""Fast checks for the Negative-Binomial scoring (run with ``pytest``)."""

import numpy as np
import pytest

from benchmark_utils.nb_loss import (
    nb_nll, log_mae, log_r2, check_predictions,
)


def test_nb_nll_matches_scipy():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 50, 200)
    mu = rng.uniform(1, 40, 200)
    r = rng.uniform(0.5, 5, 200)
    from scipy.stats import nbinom
    # scipy's nbinom(n=r, p=r/(r+mu)) has mean mu.
    expected = -np.mean(nbinom.logpmf(y, r, r / (r + mu)))
    assert nb_nll(y, mu, r) == pytest.approx(expected, rel=1e-10)


def test_perfect_mu_gives_zero_error():
    y = np.array([0, 3, 10, 100])
    assert log_mae(y, y) == pytest.approx(0.0)
    assert log_r2(y, y) == pytest.approx(1.0)


def test_check_predictions_rejects_bad_output():
    with pytest.raises(ValueError):
        check_predictions(np.zeros((3, 2)), 4)          # wrong length
    with pytest.raises(ValueError):
        check_predictions(np.array([[-1.0, 1.0]]), 1)   # negative mu
    with pytest.raises(ValueError):
        check_predictions(np.array([[1.0, 0.0]]), 1)    # sigma <= 0
    with pytest.raises(ValueError):
        check_predictions(np.array([[np.nan, 1.0]]), 1)  # non-finite
    mu, sigma = check_predictions(np.array([[2.0, 3.0]]), 1)
    assert mu[0] == 2.0 and sigma[0] == 3.0
