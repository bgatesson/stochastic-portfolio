"""
Tests for spo.models modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.models import calibrate_gbm, simulate_gbm_paths
from spo.models.simulate import simulated_moments


@pytest.fixture
def synthetic_returns():
    """3-asset panel with known parameters; we'll try to recover them."""
    rng = np.random.default_rng(0)
    n = 5000
    true_sigma = np.array([0.01, 0.015, 0.012])
    true_corr = np.array([
        [1.0, 0.4, 0.2],
        [0.4, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ])
    true_cov = np.outer(true_sigma, true_sigma) * true_corr
    true_mu_log = np.array([0.0003, 0.0005, 0.0004])  # E[log-return]
    log_r = rng.multivariate_normal(true_mu_log, true_cov, size=n)
    idx = pd.bdate_range("2010-01-01", periods=n)
    df = pd.DataFrame(log_r, index=idx, columns=["A", "B", "C"])
    return df, true_sigma, true_corr, true_mu_log


def test_calibration_recovers_sigma(synthetic_returns):
    df, true_sigma, _, _ = synthetic_returns
    params = calibrate_gbm(df, cov_method="sample")
    np.testing.assert_allclose(params.sigma.values, true_sigma, rtol=0.05)


def test_calibration_recovers_correlation(synthetic_returns):
    df, _, true_corr, _ = synthetic_returns
    params = calibrate_gbm(df, cov_method="sample")
    np.testing.assert_allclose(params.corr.values, true_corr, atol=0.03)


def test_drift_includes_ito_correction(synthetic_returns):
    df, true_sigma, _, true_mu_log = synthetic_returns
    params = calibrate_gbm(df, cov_method="sample")
    # μ should equal mean(log_r) + 0.5 σ²
    expected_mu = true_mu_log + 0.5 * true_sigma ** 2
    np.testing.assert_allclose(params.mu.values, expected_mu, atol=5e-4)


def test_simulation_shapes():
    params = _toy_params()
    sim = simulate_gbm_paths(params, n_paths=100, n_steps=21, seed=0)
    assert sim["prices"].shape == (100, 22, 3)         # +1 for t=0
    assert sim["log_returns"].shape == (100, 21, 3)
    assert sim["terminal"].shape == (100, 3)


def test_simulation_starts_at_s0():
    params = _toy_params()
    sim = simulate_gbm_paths(params, n_paths=50, n_steps=10, s0=np.array([100., 50., 25.]), seed=1)
    assert np.allclose(sim["prices"][:, 0, :], [100., 50., 25.])


def test_simulated_moments_match_calibration(synthetic_returns):
    """Large MC draws should recover the calibration moments to within MC noise."""
    df, _, _, _ = synthetic_returns
    params = calibrate_gbm(df, cov_method="sample")
    sim = simulate_gbm_paths(params, n_paths=20_000, n_steps=1, seed=42)
    mu_emp = sim["log_returns"].mean(axis=(0, 1))
    cov_emp = np.cov(sim["log_returns"].reshape(-1, 3), rowvar=False)
    # Per-step log-return mean ≈ μ - 0.5 σ² ≈ original sample mean
    expected_mean = params.mu.values - 0.5 * params.sigma.values ** 2
    np.testing.assert_allclose(mu_emp, expected_mean, atol=5e-4)
    np.testing.assert_allclose(cov_emp, params.cov.values, atol=1e-4)


def test_horizon_moments_scale_correctly(synthetic_returns):
    df, _, _, _ = synthetic_returns
    params = calibrate_gbm(df, cov_method="sample")
    sim = simulate_gbm_paths(params, n_paths=10_000, n_steps=20, seed=7)
    mu_h, cov_h = simulated_moments(sim, params.tickers)
    expected_mean = (params.mu.values - 0.5 * params.sigma.values ** 2) * 20
    np.testing.assert_allclose(mu_h.values, expected_mean, atol=2e-3)
    np.testing.assert_allclose(cov_h.values, params.cov.values * 20, atol=2e-3)


def _toy_params():
    """Tiny 3-asset GBMParams for shape tests."""
    tickers = ["A", "B", "C"]
    sigma = pd.Series([0.01, 0.012, 0.015], index=tickers)
    mu = pd.Series([0.0005, 0.0006, 0.0007], index=tickers)
    corr = pd.DataFrame(np.eye(3) + 0.2 * (np.ones((3, 3)) - np.eye(3)),
                        index=tickers, columns=tickers)
    cov = corr * np.outer(sigma, sigma)
    from spo.models.gbm import GBMParams
    return GBMParams(mu=mu, sigma=sigma, cov=cov, corr=corr, tickers=tickers, n_obs=1000)