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

def _hierarchical_clean_correlation(
    corr: np.ndarray,
    n_clusters: int | None = None,
) -> np.ndarray:
    """
    Block-average correlation cleaning via hierarchical clustering.
    Replaces within-block and cross-block correlations with their averages.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = corr.shape[0]
    if n_clusters is None:
        n_clusters = max(2, int(np.sqrt(n)))

    # Correlation -> distance
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, 4.0))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Block averaging: within-block avg for each cluster, cross-block avg for each pair
    cleaned = np.zeros_like(corr)
    for a in range(1, n_clusters + 1):
        for b in range(1, n_clusters + 1):
            mask_a = labels == a
            mask_b = labels == b
            block = corr[np.ix_(mask_a, mask_b)]
            if a == b:
                # Within block: average off-diagonal, set diagonal to 1
                off_diag = block[~np.eye(block.shape[0], dtype=bool)]
                avg = off_diag.mean() if off_diag.size else 0.0
                filled = np.full(block.shape, avg)
                np.fill_diagonal(filled, 1.0)
                cleaned[np.ix_(mask_a, mask_b)] = filled
            else:
                cleaned[np.ix_(mask_a, mask_b)] = block.mean()

    # Ensure exact symmetry and unit diagonal
    cleaned = 0.5 * (cleaned + cleaned.T)
    np.fill_diagonal(cleaned, 1.0)
    return cleaned

def estimate_covariance(returns: pd.DataFrame, method: str="ledoit_wolf") -> np.ndarray:
    """
    Estimate the covariance matrix of a returns table.

    Available methods:
        - "sample"
        - "ledoit_wolf"
        - "hierarchical_clean"
    """
    X = returns.values
    if method == "sample":
        return np.atleast_2d(np.cov(X, rowvar=False))
    if method == "ledoit_wolf":
        return LedoitWolf().fit(X).covariance_
    if method == "hierarchical_clean":
        cov_sample = np.cov(X, rowvar=False)
        std = np.sqrt(np.diag(cov_sample))
        corr = cov_sample / np.outer(std, std)
        corr_clean = _hierarchical_clean_correlation(corr)
        return np.outer(std, std) * corr_clean
    raise ValueError(f"Unknown method: {method}")


def min_var_portfolio(returns: pd.DataFrame | None=None, covariance: str="ledoit_wolf", long_only: bool=True, max_weight: float | None=None, cov: pd.DataFrame | None=None) -> pd.Series:
    """
    Minimum-variance portfolio optimization.
    Solves:
        min w' Σ w
        s.t. w' 1 = 1, w >= 0 (if long positions only), w <= max_weight (if applicable)

    Pass a pre-computed `cov` DataFrame to skip estimation (e.g. from simulation).
    """
    if cov is not None:
        tickers = list(cov.columns)
        cov_arr = np.asarray(cov)
    else:
        if returns is None:
            raise ValueError("Either returns or cov must be provided.")
        returns = returns.dropna(axis=1)
        cov_arr = estimate_covariance(returns, method=covariance)
        tickers = list(returns.columns)

    n = len(tickers)
    w = cp.Variable(n)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov_arr))), constraints=constraints)
    prob.solve()
    if w.value is None:
        raise RuntimeError(f"Min-variance optimization failed: {prob.status}")
    return pd.Series(w.value, index=tickers)


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