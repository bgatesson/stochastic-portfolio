"""
Tests for Heston calibration and simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.models import (
    calibrate_gbm,
    calibrate_heston_multi,
    calibrate_heston_single,
    simulate_gbm_paths,
    simulate_heston_paths,
)


@pytest.fixture
def fat_tailed_returns():
    """Synthetic returns with deliberate fat tails (mixture of normals)."""
    rng = np.random.default_rng(42)
    n = 2000
    # 95% calm + 5% high-vol days should give fat tails
    calm = rng.normal(0, 0.008, size=int(n * 0.95))
    crisis = rng.normal(0, 0.04, size=n - len(calm))
    r = np.concatenate([calm, crisis])
    rng.shuffle(r)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(r, index=idx, name="FAT")


def test_calibration_runs(fat_tailed_returns):
    p = calibrate_heston_single(fat_tailed_returns, ticker="FAT")
    assert p.kappa > 0
    assert p.theta > 0
    assert p.sigma > 0
    assert -1.0 < p.rho < 1.0


def test_calibration_picks_up_fat_tails(fat_tailed_returns):
    """
    Calibration should produce nontrivial vol-of-vol on fat-tailed data.
    """
    p = calibrate_heston_single(fat_tailed_returns, ticker="FAT")
    # sigma should be non-negligible relative to theta
    assert p.sigma / np.sqrt(p.theta) > 0.1


def test_heston_simulated_kurtosis_exceeds_gaussian(fat_tailed_returns):
    """
    Simulated Heston returns should have kurtosis > 3.
    """
    # Single-asset sim via the multi-asset interface
    df = fat_tailed_returns.to_frame()
    multi = calibrate_heston_multi(df)
    sim = simulate_heston_paths(multi, n_paths=2000, n_steps=252, seed=0)
    # Flatten across paths and time
    sim_returns = sim["log_returns"][:, :, 0].flatten()
    kurt = ((sim_returns - sim_returns.mean()) ** 4).mean() / sim_returns.var() ** 2
    assert kurt > 3.2, f"Heston sim kurtosis {kurt:.2f} not > 3"

def test_heston_fatter_tails_than_gbm(fat_tailed_returns):
    """
    Heston-simulated returns should have higher kurtosis than GBM-simulated returns.
    """
    df = fat_tailed_returns.to_frame()

    gbm = calibrate_gbm(df, cov_method="sample")
    sim_gbm = simulate_gbm_paths(gbm, n_paths=2000, n_steps=252, seed=1)
    gbm_returns = sim_gbm["log_returns"][:, :, 0].flatten()
    gbm_kurt = ((gbm_returns - gbm_returns.mean()) ** 4).mean() / gbm_returns.var() ** 2

    heston_multi = calibrate_heston_multi(df)
    sim_h = simulate_heston_paths(heston_multi, n_paths=2000, n_steps=252, seed=1)
    h_returns = sim_h["log_returns"][:, :, 0].flatten()
    h_kurt = ((h_returns - h_returns.mean()) ** 4).mean() / h_returns.var() ** 2

    assert h_kurt > gbm_kurt, (
        f"Heston kurtosis {h_kurt:.2f} ≤ GBM kurtosis {gbm_kurt:.2f}"
    )


def test_heston_simulation_shapes(fat_tailed_returns):
    df = pd.concat([fat_tailed_returns.rename("A"),
                    fat_tailed_returns.rename("B")], axis=1)
    multi = calibrate_heston_multi(df)
    sim = simulate_heston_paths(multi, n_paths=100, n_steps=21, seed=0)
    assert sim["prices"].shape == (100, 22, 2)
    assert sim["log_returns"].shape == (100, 21, 2)


def test_multi_asset_calibration_preserves_correlation():
    """
    Calibrated cross-asset correlation should match empirical.
    """
    rng = np.random.default_rng(7)
    n = 1500
    rho_true = 0.6
    L = np.array([[1.0, 0.0], [rho_true, np.sqrt(1 - rho_true ** 2)]])
    z = rng.standard_normal((n, 2)) * 0.01
    r = z @ L.T
    df = pd.DataFrame(r, columns=["A", "B"], index=pd.bdate_range("2018-01-01", periods=n))
    multi = calibrate_heston_multi(df)
    np.testing.assert_allclose(multi.return_corr.loc["A", "B"], rho_true, atol=0.05)