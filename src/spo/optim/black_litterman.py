"""
Black-Litterman posterior expected returns + resulting Markowitz portfolio.

Reference: Black & Litterman (1992). We follow the standard He-Litterman
formulation (Goldman Sachs Fixed Income Research, 1999).
"""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

from spo.optim.mean_variance import estimate_covariance

logger = logging.getLogger(__name__)


def implied_equilibrium_returns(market_weights: pd.Series, cov: np.ndarray, risk_aversion: float=2.5) -> pd.Series:
    """
    Reverse-optimize the equilibrium (CAPM-implied) expected returns.

    Formula:
        π = δ Σ w_mkt
    
    δ is set a 2.5 (implies equity-risk-premium of 4-5%)
    """
    pi = risk_aversion * cov @ market_weights.values
    return pd.Series(pi, index=market_weights.index)


def black_litterman_posterior(
        market_weights: pd.Series,
        cov: np.ndarray,
        risk_aversion: float=2.5,
        tau: float = 0.05,
        views: pd.DataFrame | None = None,
        view_returns: pd.Series | None = None,
        view_confidences: pd.Series | None = None,
) -> tuple[pd.Series, np.ndarray]:
    """
    Compute Black-Litterman posterior expected returns and posterior covariance.

    Parameters:
    market_weights : pd.Series
        Prior "market" weights.
    cov : np.ndarray
        Prior covariance.
    risk_aversion : float
        δ in π = δ Σ w. Typical value 2.5.
    tau : float
        Uncertainty in the prior. Small (0.01-0.10). Higher tau = less trust
        in equilibrium.
    views : pd.DataFrame, optional
        (K, N) matrix P: each row is a view over N assets. E.g. row
        (+1 in col A, -1 in col B) is "A outperforms B."
    view_returns : pd.Series, optional
        (K,) vector Q: expected return of each view.
    view_confidences : pd.Series, optional
        (K,) vector of variances Ω_kk. Higher = less confidence.
    """
    n = len(market_weights)
    tickers = list(market_weights.index)
    pi = implied_equilibrium_returns(market_weights, cov, risk_aversion).values
    tau_cov_inv = np.linalg.inv(tau * cov)

    if views is None or view_returns is None:
        return pd.Series(pi, index=tickers), cov
    
    P = views.values
    Q = view_returns.values 
    if view_confidences is None:
        omega = np.diag(np.diag(P @ (tau * cov) @ P.T))
    else:
        omega = np.diag(view_confidences.values)
    omega_inv = np.linalg.inv(omega)

    # Posterior mean
    A = tau_cov_inv + P.T @ omega_inv @ P
    b = tau_cov_inv @ pi + P.T @ omega_inv @ Q
    mu_bl = np.linalg.solve(A, b)
