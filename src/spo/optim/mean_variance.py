"""
Mean-variance optimization with Ledoit-Wolf covariance shrinkage.
"""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

def estimate_covariance(returns: pd.DataFrame, method: str="ledoit_wolf") -> np.ndarray:
    """
    Estimate the covariance matrix of a returns table.

    Available methods:
        - "sample"
        - "ledoit_wolf"
    """
    X = returns.values
    if method == "sample":
        return np.cov(X, rowvar=False)
    if method == "ledoit_wolf":
        return LedoitWolf().fit(X).covariance_
    raise ValueError(f"Unknown method: {method}")


def min_var_portfolio(returns: pd.DataFrame, covariance: str="ledoit_wolf", long_only: bool=True, max_weight: float | None=None) -> pd.Series:
    """
    Minimum-variance portfolio optimization.
    Solves:
        min w' Σ w
        s.t. w' 1 = 1, w >= 0 (if long positions only), w <= max_weight (if applicable)
    """
    returns = returns.dropna(axis=1)
    cov  = estimate_covariance(returns, method=covariance)
    n = cov.shape[0]
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints=constraints)
    prob.solve()
    if w.value is None:
        raise RuntimeError(f"Min-variance optimization failed: {prob.status}")
    return pd.Series(w.value, index=returns.columns)


def max_sharpe_portfolio(returns: pd.DataFrame, rf: float=0.0, covariance: str="ledoit_wolf", long_only: bool=True) -> pd.Series:
    """
    Maximum-Sharpe portfolio via long position only Markowitz transform.
    Solves:
        min x' Σ x
        s.t. (μ - rf)' x = 1, x >= 0

    Falls back to min-variance if no asset has positive expected excess return.
    """
    returns = returns.dropna(axis=1)
    cov = estimate_covariance(returns, method=covariance)
    mu = returns.mean().values
    excess = mu - rf

    if excess.max() <= 0:
        logger.warning("No positive excess returns. Using min-variance instead.")
        return min_var_portfolio(returns, covariance=covariance, long_only=long_only)
    
    n = cov.shape[0]
    x = cp.Variable(n)
    constraints = [excess @ x == 1]
    if long_only:
        constraints.append(x >= 0)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x, cp.psd_wrap(cov))), constraints=constraints)
    prob.solve()
    if x.value is None:
        raise RuntimeError(f"Max-Sharpe optimization failed: {prob.status}")
    weights = x.value / x.value.sum()
    return pd.Series(weights, index=returns.columns)


def efficient_frontier(returns: pd.DataFrame, n_points: int=30, covariance: str="ledoit_wolf", long_only: bool=True) -> pd.DataFrame:
    """
    Trace an efficient frontier from the target returns.
    """
    returns = returns.dropna(axis=1)
    cov = estimate_covariance(returns, method=covariance)
    mu = returns.mean().values
    n = cov.shape[0]

    mv_w = min_var_portfolio(returns, covariance, long_only).values
    low = float(mu @ mv_w)
    high = float(mu.max())
    targets = np.linspace(low, high, n_points)

    rows = []
    for t in targets:
        w = cp.Variable(n)
        constraints = [cp.sum(w) == 1, mu @ w >= t]
        if long_only:
            constraints.append(w >= 0)
        prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), constraints)
        prob.solve()
        if w.value is None:
            continue
        vol = float(np.sqrt(w.value @ cov @ w.value))
        rows.append({"target_return": t, "vol": vol, "expected_return": float(mu @ w.value)})
    return pd.DataFrame(rows)