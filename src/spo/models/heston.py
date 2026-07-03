"""
Heston model: per-asset and multi-asset (independent Heston processes correlated through price Brownians) calibration.
Formula:
    dS_t = μ * S_t * dt + √V_t * S_t * dW_1t
    dV_t = κ(θ - V_t) * dt + σ * √V_t * dW_2t
    dW_1t * dW_2t = ρdt

    Where: 
        μ is the long term drift,
        κ is mean-reversion speed, 
        θ the long term variance,
        σ the volatility of the volatility,
        ρ the correlation between returns and variance changes
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


class HestonParams:
    """
    Heston parameters for a single asset (per-period, dt=1)
    """
    def __init__(
            self,
            mu: float,
            kappa: float,
            theta: float,
            sigma: float,
            rho: float,
            v0: float,
            ticker: str
        ):
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.ticker = ticker

    
    @property
    def feller_check(self) -> bool:
        """
        Feller condition: 2κθ >= σ^2. Variance has positive probability of hitting 0 if violated.
        """
        return 2 * self.kappa * self.theta >= self.sigma ** 2
    
    def annualize(self, periods: int=252) -> "HestonParams":
        """
        Scale to annual units. 
        """
        return HestonParams(
            mu = self.mu * periods,
            kappa = self.kappa * periods,
            theta = self.theta * periods,
            sigma = self.sigma * np.sqrt(periods),
            rho = self.rho,
            v0 = self.v0 * periods,
            ticker = self.ticker
        )


class MultiAssetHestonParams:
    """
    Joint multi-asset Heston calibration.
    """
    def __init__(
            self,
            assets: dict[str, HestonParams],
            return_corr: pd.DataFrame,
            tickers: list[str]
        ):
        self.assets = assets
        self.return_corr = return_corr
        self.tickers = tickers
    

    @property
    def n_assets(self) -> int:
        return len(self.tickers)


# Single asset Generalized Method of Moments (GMM)
# See "Estimating stochastic volatility diffusion using conditional moments of integrated volatility" Bollerslev & Zhou 2002

def _empirical_moments(r: np.ndarray) -> dict[str, float]:
    """
    Sample moments used to calibrate the Heston model.
    """
    r = np.asarray(r, dtype=float)
    mean = r.mean()
    r_centered = r - mean # remove sample mean
    var = r_centered.var(ddof=1)
    kurt = pd.Series(r_centered).kurt()
    ac1 = pd.Series(r_centered ** 2).autocorr(lag=1)
    corr = np.corrcoef(r_centered[:-1], np.abs(r_centered)[1:])[0, 1]
    return {
        "mean": mean,
        "var": var,
        "kurt": kurt,
        "ac1": ac1,
        "corr": corr
    }


def _theoretical_moments(kappa: float, theta: float, sigma: float, rho: float) -> dict[str, float]:
    """
    Closed-form Heston model moments for log returns.
    Following Bollerslev-Zhou (2002).
    """
    var = theta
    kurt = 6.0 * sigma ** 2 /  max(4.0 * kappa * theta, 1e-10)
    ac1 = np.exp(-kappa)
    corr = rho * sigma / np.sqrt(theta) * 0.5
    return {
        "var": var,
        "kurt": kurt,
        "ac1": ac1,
        "corr": corr
    }


def calibrate_heston_single(log_returns: pd.Series, ticker: str) -> HestonParams:
    """
    Calibrate Heston to a single asset's log return series via indirect inference.

    Priors fixed from literature (not identifiable from daily returns alone):
      rho = -0.4: leverage effect consensus (Christie 1982, Black 1976)
      kappa = 15/252: half-life ~17 days; kappa*dt = 0.06 << 1 (Euler stable)

    theta = Var(r) directly from the variance moment.

    sigma is found by Brent's method on a simulation (1000 paths x 252 steps),
    targeting the simulated excess kurtosis to match the empirical value. This is
    indirect inference: it calibrates to what the model actually produces under
    full-truncation Euler, not to what the first-order closed-form predicts.

    Empirical kurtosis is capped at 8 to prevent outlier days (e.g. ADM's 69)
    from driving sigma to unrealistic values. Falls back to the closed-form
    approximation if root-finding fails.
    """
    from spo.models.simulate import _simulate_heston_single

    r = log_returns.dropna().values
    if len(r) < 252:
        raise ValueError(f"Need >= 252 obs to calibrate Heston. Got {len(r)}.")
    empirical = _empirical_moments(r)

    rho = -0.4
    kappa = 15.0 / 252
    theta = float(np.clip(empirical["var"], 1e-8, 1.0))
    target_kurt = float(np.clip(empirical["kurt"], 0.5, 8.0))

    def sim_excess_kurt(sigma: float) -> float:
        params = HestonParams(mu=0.0, kappa=kappa, theta=theta, sigma=sigma,
                              rho=rho, v0=theta, ticker=ticker)
        rng = np.random.default_rng(0)
        z_s = rng.standard_normal((1000, 252))
        z_v = rng.standard_normal((1000, 252))
        log_r = _simulate_heston_single(params, 1000, 252, 1.0, z_s, z_v)
        flat = log_r.flatten()
        centered = flat - flat.mean()
        return float((centered ** 4).mean() / centered.var() ** 2 - 3.0)

    sigma_fallback = float(np.clip(
        np.sqrt(max(target_kurt, 0.0) * (2.0 / 3.0) * kappa * theta), 1e-8, 1.0
    ))
    try:
        low_k = sim_excess_kurt(1e-4)
        high_k = sim_excess_kurt(0.15)  # sigma_daily=0.15 → sigma_ann≈2.4, safe upper bound
        if target_kurt <= low_k:
            sigma = 1e-4
        elif target_kurt >= high_k:
            sigma = 0.15
        else:
            sigma = brentq(
                lambda s: sim_excess_kurt(s) - target_kurt,
                1e-4, 0.15, xtol=5e-3, maxiter=20,
            )
    except (ValueError, RuntimeError) as e:
        logger.warning("Indirect inference failed for %s: %s — using closed-form.", ticker, e)
        sigma = sigma_fallback

    return HestonParams(
        mu=empirical["mean"] + 0.5 * theta,
        kappa=kappa, theta=theta, sigma=sigma, rho=rho,
        v0=theta, ticker=ticker,
    )


def calibrate_heston_multi(log_returns: pd.DataFrame) -> MultiAssetHestonParams:
    """
    Calibrate per-asset Heston and use the residual correlation matrix.
    """
    assets = {}
    for ticker in log_returns.columns:
        assets[ticker] = calibrate_heston_single(log_returns[ticker], ticker=ticker)
        logger.info(
            "Heston %s: κ=%.4f θ=%.5f σ=%.4f ρ=%+.3f  Feller=%s",
            ticker, assets[ticker].kappa, assets[ticker].theta,
            assets[ticker].sigma, assets[ticker].rho,
            "NOT VIOLATED" if assets[ticker].feller_check else "VIOLATED"
        )
    
    return_corr = log_returns.dropna(how="any").corr()
    return MultiAssetHestonParams(
        assets=assets,
        return_corr=return_corr,
        tickers=list(log_returns.columns)
    )