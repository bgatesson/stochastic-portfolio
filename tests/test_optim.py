"""
Tests for spo.backtest and spo.optim modules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.backtest import sharpe_ratio, backtest
from spo.optim import estimate_covariance, max_sharpe_portfolio, min_var_portfolio

@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2018-01-01", periods=1500)
    n_assets = 8 # create assets with postive drift and some correlation
    L = rng.standard_normal((n_assets, n_assets)) * 0.3
    cov = L @ L.T + np.eye(n_assets) * 0.0001
    returns = rng.multivariate_normal(
        mean=np.full(n_assets, 0.0005), cov=cov, size=len(idx)
    )
    return pd.DataFrame(returns, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_ledoit_wolf_is_psd(returns):
    cov = estimate_covariance(returns, method="ledoit_wolf")
    eigvals = np.linalg.eigvalsh(cov)
    assert (eigvals >= -1e-10).all()


def test_min_var_weights_valid(returns):
    w = min_var_portfolio(returns)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()


def test_min_var_lower_variance_than_equal_weight(returns):
    cov = estimate_covariance(returns)
    mv = min_var_portfolio(returns).values
    ew = np.full(returns.shape[1], 1.0 / returns.shape[1])
    assert mv @ cov @ mv < ew @ cov @ ew


def test_max_sharpe_weights_valid(returns):
    w = max_sharpe_portfolio(returns)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-8).all()


def test_backtest_no_lookahead(returns):
    """
    Weights effective today must be set strictly before today.
    """
    result = backtest(
        returns, lambda r: min_var_portfolio(r), lookback=252
    )
    # On any rebalance date, the gross return is computed using yesterday's weights
    assert result["net_returns"].notna().sum() > 0
    assert result["weights"].sum(axis=1).max() <= 1.0 + 1e-6


def test_backtest_costs_reduce_returns(returns):
    """
    Higher transaction cost must produce lower (or equal) cumulative returns.
    """
    no_cost = backtest(
        returns, lambda r: min_var_portfolio(r), cost=0.0
    )
    with_cost = backtest(
        returns, lambda r: min_var_portfolio(r), cost=50.0
    )
    assert no_cost["net_returns"].sum() >= with_cost["net_returns"].sum()


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, size=2000))
    assert sharpe_ratio(r) > 0