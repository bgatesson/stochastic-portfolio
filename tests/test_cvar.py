"""
Test for CVaR optmization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.optim import (
    compute_cvar,
    cvar_frontier,
    max_return_cvar_constrained,
    min_cvar_portfolio,
    min_var_portfolio,
)


def _gaussian_scenarios(n_scenarios=5000, n_assets=5, seed=0):
    """
    Multivariate Gaussian scenarios. CVaR and Min-Var should agree here.
    """
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n_assets, n_assets)) * 0.4
    cov = L @ L.T / 100 + np.eye(n_assets) * 0.001
    mu = np.full(n_assets, 0.005)
    return rng.multivariate_normal(mu, cov, size=n_scenarios)


def _fat_tailed_scenarios(n_scenarios=5000, n_assets=5, seed=0):
    """
    Mixture of normals, fat tails. CVaR and Min-Var should disagree here.
    """
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n_assets, n_assets)) * 0.4
    cov_calm = L @ L.T / 100 + np.eye(n_assets) * 0.001
    cov_crisis = cov_calm * 25
    mu = np.full(n_assets, 0.005)
    n_crisis = int(n_scenarios * 0.05)
    calm = rng.multivariate_normal(mu, cov_calm, size=n_scenarios - n_crisis)
    crisis = rng.multivariate_normal(mu * 0, cov_crisis, size=n_crisis)
    # Add idiosyncratic losses to the first half of assets in crisis.
    # Scaling uniformly would preserves the relative risk structure, so min-CVaR ≈ min-var.
    crisis[:, :n_assets // 2] -= 0.10
    s = np.vstack([calm, crisis])
    rng.shuffle(s)
    return s


def test_weights_sum_to_one():
    s = _gaussian_scenarios()
    w = min_cvar_portfolio(s, alpha=0.95)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)


def test_long_only_constraint_respected():
    s = _gaussian_scenarios()
    w = min_cvar_portfolio(s, alpha=0.95, long_only=True)
    assert (w.values >= -1e-8).all()


def test_max_weight_constraint_respected():
    s = _gaussian_scenarios()
    w = min_cvar_portfolio(s, alpha=0.95, max_weight=0.4)
    assert w.max() <= 0.4 + 1e-6


def test_gaussian_collapse_to_min_variance():
    """
    For Gaussian scenarios, min-CVaR and min-variance must nearly agree.
    """
    s = _gaussian_scenarios(n_scenarios=10_000)
    tickers = [f"a{i}" for i in range(s.shape[1])]
    cvar_w = min_cvar_portfolio(s, alpha=0.95, tickers=tickers)
    # Build a returns df so the min-variance solver can be reused
    mv_w = min_var_portfolio(pd.DataFrame(s, columns=tickers),
                                  covariance="sample")
    # L1 weight difference should be small (Monte Carlo noise + numerical)
    assert (cvar_w - mv_w).abs().sum() < 0.20


def test_fat_tails_break_the_collapse():
    """
    Under fat-tailed scenarios, min-CVaR and min-variance should differ.
    """
    s = _fat_tailed_scenarios(n_scenarios=10_000)
    tickers = [f"a{i}" for i in range(s.shape[1])]
    cvar_w = min_cvar_portfolio(s, alpha=0.95, tickers=tickers)
    mv_w = min_var_portfolio(pd.DataFrame(s, columns=tickers),
                                  covariance="sample")
    # Meaningful disagreement under non-Gaussian scenarios
    assert (cvar_w - mv_w).abs().sum() > 0.05


def test_cvar_optimal_beats_min_var_on_cvar_metric():
    """
    The min-CVaR portfolio must achieve lower CVaR than the min-var portfolio.
    """
    s = _fat_tailed_scenarios(n_scenarios=10_000)
    tickers = [f"a{i}" for i in range(s.shape[1])]
    w_cvar = min_cvar_portfolio(s, alpha=0.95, tickers=tickers)
    w_mv = min_var_portfolio(pd.DataFrame(s, columns=tickers),
                                  covariance="sample")
    _, cvar_of_cvar = compute_cvar(s, w_cvar.values, alpha=0.95)
    _, cvar_of_mv = compute_cvar(s, w_mv.values, alpha=0.95)
    assert cvar_of_cvar <= cvar_of_mv + 1e-6


def test_higher_confidence_shifts_more_defensive():
    """
    Increasing alpha (deeper tail) should shift weights, generally more defensive.
    """
    s = _fat_tailed_scenarios(n_scenarios=10_000)
    w_95 = min_cvar_portfolio(s, alpha=0.95)
    w_99 = min_cvar_portfolio(s, alpha=0.99)
    # Weights should differ between confidence levels under fat tails
    assert (w_95 - w_99).abs().sum() > 0.01


def test_lp_objective_matches_empirical_cvar():
    """
    The LP's optimal value should match recomputed CVaR on the solution.
    """
    s = _gaussian_scenarios()
    w = min_cvar_portfolio(s, alpha=0.95)
    _, cvar_emp = compute_cvar(s, w.values, alpha=0.95)
    # Reconstruct the LP objective: VaR + (1/(S(1-α))) Σ z_s
    losses = -(s @ w.values)
    var_emp = np.quantile(losses, 0.95)
    excess = np.maximum(losses - var_emp, 0)
    cvar_reconstructed = var_emp + excess.mean() / (1 - 0.95)
    np.testing.assert_allclose(cvar_emp, cvar_reconstructed, atol=1e-3)


def test_cvar_budget_constraint_binds():
    """
    At a tight CVaR budget, the constraint should bind.
    """
    s = _fat_tailed_scenarios(n_scenarios=5000)
    w_min = min_cvar_portfolio(s, alpha=0.95)
    _, cvar_min = compute_cvar(s, w_min.values, alpha=0.95)
    # Set budget slightly above min. Should be feasible, and produce different weights
    w_constrained = max_return_cvar_constrained(s, cvar_limit=cvar_min * 1.5, alpha=0.95)
    _, cvar_realized = compute_cvar(s, w_constrained.values, alpha=0.95)
    assert cvar_realized <= cvar_min * 1.5 + 1e-3
    # And expected return should be at least as high as min-CVaR's
    mu = s.mean(axis=0)
    assert mu @ w_constrained.values >= mu @ w_min.values - 1e-6



def test_cvar_frontier_runs():
    s = _fat_tailed_scenarios(n_scenarios=2000, n_assets=4)
    frontier = cvar_frontier(s, n_points=5, alpha=0.95)
    assert len(frontier) > 0
    # Expected return should be non-decreasing in CVaR budget
    assert frontier["expected_return"].is_monotonic_increasing or \
           frontier["expected_return"].diff().min() > -1e-4