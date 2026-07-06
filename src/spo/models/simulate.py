"""
Monte Carlo simulation script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from spo.models.gbm import GBMParams
from spo.models.heston import HestonParams, MultiAssetHestonParams

import logging

logger = logging.getLogger(__name__)


def _psd_cholesky(corr: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """
    Cholesky factor of the nearest PSD correlation matrix.

    Sample correlation matrices can have slightly negative eigenvalues due to
    floating-point error or pairwise-NaN construction.
    We clip eigenvalues to epsilon, rescale the diagonal back to 1, then Cholesky.
    """
    eigvals, eigvecs = np.linalg.eigh(corr)
    n_negative = int((eigvals < 0).sum())
    if n_negative > 0:
        logger.warning("Clipped %d/%d negative eigenvalues (min was %.2e)", n_negative, len(eigvals), eigvals.min())
    eigvals = np.maximum(eigvals, epsilon)
    psd = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.maximum(np.diag(psd), epsilon))
    psd = psd / np.outer(d, d)
    np.fill_diagonal(psd, 1.0)
    return np.linalg.cholesky(psd)


def simulate_gbm_paths(
        params: GBMParams,
        n_paths: int=10000,
        n_steps: int=21,
        dt: float=1.0,
        s0: np.ndarray | None=None,
        seed: int | None=None
) -> dict[str, np.ndarray]:
    """
    Simulate GBM paths.
    Formula assuming log returns and multivariate setting:
        log S_t = log S_t-1 + (μ - 0.5 * σ^2) * dt + σ * L * z * √dt

        With:
            L - The Cholesky decomposition factor of the correlation matrix
            z ~ N(0, I) (Assumptions from dW)
    """
    rng = np.random.default_rng(seed)
    n_assets = len(params.tickers)
    if s0 is None:
        s0 = np.ones(n_assets)
    s0 = np.asarray(s0, dtype=float)

    # (μ - 0.5 * σ^2) * dt
    drift = (params.mu.values - 0.5 * params.sigma.values ** 2) * dt
    # σ * √dt
    vol_step = params.sigma.values * np.sqrt(dt)
    # L: cholesky decomposition factor of correlation matrix
    L = _psd_cholesky(params.corr.values)
 

    z = rng.standard_normal(size=(n_paths, n_steps, n_assets))
    corr_z = z @ L.T
    log_returns = drift + vol_step * corr_z

    log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
    log_prices = np.concatenate(
        [np.broadcast_to(np.log(s0), (n_paths, 1, n_assets)), log_prices],
        axis=1,
    )
    prices = np.exp(log_prices)

    return {
        "prices": prices,
        "log_returns": log_returns,
        "terminal": prices[:, -1, :]
    }


def simulated_moments(simulation: dict[str, np.ndarray], tickers: list[str]) -> tuple[pd.Series, pd.DataFrame]:
    """
    Mean and Covariance of total log returns.
    """
    log_returns = simulation["log_returns"]
    total_log_returns = log_returns.sum(axis=1)
    mu = pd.Series(total_log_returns.mean(axis=0), index=tickers)
    cov = pd.DataFrame(np.atleast_2d(np.cov(total_log_returns, rowvar=False)), index=tickers, columns=tickers)
    return mu, cov



def _simulate_heston_single(
        params: HestonParams,
        n_paths: int,
        n_steps: int,
        dt: float,
        z_s: np.ndarray,
        z_v: np.ndarray
) -> np.ndarray:
    """
    Simulate one asset's log returns under Heston model.
    """
    kappa, theta, sigma, rho = params.kappa, params.theta, params.sigma, params.rho
    # Correlate price Brownian and variance Brownian
    z_v_corr = rho * z_s + np.sqrt(1.0 - rho ** 2) * z_v

    v = np.full(n_paths, params.v0)
    log_returns = np.empty((n_paths, n_steps))

    for t in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        # Log-return increment with Ito correction
        log_returns[:, t] = (params.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * z_s[:, t]
        # clamp variance v to [0, inf) so it never goes negative between steps
        v = np.maximum(
            v + kappa * (theta - v_pos) * dt + sigma * np.sqrt(v_pos * dt) * z_v_corr[:, t],
            0.0,
        )
    
    return log_returns


def simulate_heston_paths(
        params: MultiAssetHestonParams,
        n_paths: int=10000,
        n_steps: int=21,
        dt: float=1.0,
        s0: np.ndarray | None=None,
        seed: int | None=None
) -> dict[str, np.ndarray]:
    """
    Simulate multivariate Heston paths.
    """
    rng = np.random.default_rng(seed)
    n_assets = params.n_assets
    if s0 is None:
        s0 = np.ones(n_assets)
    s0 = np.asarray(s0, dtype=float)

    # L: cholesky decomposition factor of price-noise correlation matrix
    L = _psd_cholesky(params.return_corr.values)

    # Generate price noises with cross-asset correlation
    z_s_raw = rng.standard_normal(size=(n_paths, n_steps, n_assets))   # i.i.d.
    z_s = z_s_raw @ L.T # correlate across assets
    # Variance noises: independent across assets, will be correlated to z_s
    # within each asset by the per-asset rho inside _simulate_heston_single.
    z_v = rng.standard_normal(size=(n_paths, n_steps, n_assets))

    log_returns = np.empty((n_paths, n_steps, n_assets))
    for i, ticker in enumerate(params.tickers):
        log_returns[:, :, i] = _simulate_heston_single(
            params.assets[ticker], n_paths, n_steps, dt,
            z_s=z_s[:, :, i], z_v=z_v[:, :, i],
        )

    log_prices = np.log(s0) + np.cumsum(log_returns, axis=1)
    log_prices = np.concatenate(
        [np.broadcast_to(np.log(s0), (n_paths, 1, n_assets)), log_prices],
        axis=1,
    )
    prices = np.exp(log_prices)
    return {
        "prices": prices,
        "log_returns": log_returns,
        "terminal": prices[:, -1, :],
    }
