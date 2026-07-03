"""
Geometric Brownian Motion. Uses log returns.

Formula:
    dS_i = μ_i * S_i dt + σ_i * S_i dW_i
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spo.optim.mean_variance import estimate_covariance


class GBMParams:
    """
    GBM parameters. All quantities are per-period (dt=1).
    """
    def __init__(
            self,
            mu: pd.Series,
            sigma: pd.Series,
            cov: pd.DataFrame,
            corr: pd.DataFrame,
            tickers: list[str],
            n_obs: int
        ):
        self.mu = mu
        self.sigma = sigma
        self.cov = cov
        self.corr = corr
        self.tickers = tickers
        self.n_obs = n_obs

    def annualize(self, period: int=252) -> "GBMParams":
        """
        Create a copy with annualized parameters.
        """
        return GBMParams(
            mu=self.mu * period,
            sigma=self.sigma * np.sqrt(period),
            cov=self.cov * period,
            corr=self.corr,
            tickers=self.tickers,
            n_obs=self.n_obs,
        )


def calibrate_gbm(log_returns: pd.DataFrame, cov_method: str="ledoit_wolf") -> GBMParams:
    """
    Fit a multivariate GBM to log_returns data.
    """
    log_returns = log_returns.dropna(axis=1)

    cov = estimate_covariance(log_returns, method=cov_method)
    sigma2 = np.diag(cov)
    sigma = np.sqrt(sigma2)
    mean_r = log_returns.mean().values
    mu = mean_r + 0.5 * sigma2  # Itô correction: E[log S_t/S_0] = (μ - 0.5σ²)t

    inv_sd = 1.0 / sigma
    corr = cov * np.outer(inv_sd, inv_sd)
    np.fill_diagonal(corr, 1.0)

    tickers = list(log_returns.columns)
    return GBMParams(
        mu=pd.Series(mu, index=tickers),
        sigma=pd.Series(sigma, index=tickers),
        cov=pd.DataFrame(cov, index=tickers, columns=tickers),
        corr=pd.DataFrame(corr, index=tickers, columns=tickers),
        tickers=tickers,
        n_obs=len(log_returns),
    )