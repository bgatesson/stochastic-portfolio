"""
Performance metrics for backtesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm


def sharpe_ratio(returns: pd.Series, rf: float=0.0, period: int=252) -> float:
    excess = returns - rf / period
    return float(excess.mean() / excess.std() * np.sqrt(period))


def sortino_ratio(returns: pd.Series, rf: float=0.0, period: int=252) -> float:
    excess = returns - rf / period
    downside = excess[excess < 0].std()
    return float(excess.mean() / downside * np.sqrt(period))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns).cumprod()
    return float((cum / cum.cummax() - 1).min())


def annualized_returns(returns: pd.Series, period: int=252) -> float:
    n = len(returns)
    return float((1 + returns).prod() ** (period / n) - 1)


def annualized_vol(returns: pd.Series, period: int=252) -> float:
    return float(returns.std() * np.sqrt(period))


def calmar_ratio(returns: pd.Series, period: int=252) -> float:
    mdd = max_drawdown(returns)
    return float(annualized_returns(returns, period) / abs(mdd))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    sr_benchmark: float = 0.0,
    period: int = 252,
) -> float:
    """
    Probability that the true Sharpe exceeds sr_benchmark (Bailey & López de Prado 2012).

    Uses the non-normality-adjusted standard error of the SR estimator, so it
    penalises fat tails and negative skew — both common in CVaR-optimised portfolios.
    Returns a probability in [0, 1].
    """
    n = len(returns)
    sr_per = float(returns.mean() / returns.std())          # per-period, no annualisation
    sr_bench_per = sr_benchmark / np.sqrt(period)
    skew = float(returns.skew())
    excess_kurt = float(returns.kurtosis())                 # pandas: Fisher (excess) kurtosis

    # Opdyke (2007) variance of the SR estimator under non-normality
    var_sr = (1 + sr_per ** 2 / 2 - skew * sr_per + excess_kurt * sr_per ** 2 / 4) / (n - 1)
    if var_sr <= 0:
        return np.nan
    return float(_norm.cdf((sr_per - sr_bench_per) / np.sqrt(var_sr)))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    period: int = 252,
) -> float:
    """
    DSR: PSR where the benchmark is the expected maximum SR under n_trials independent tests.

    Corrects for selection bias (multiple strategies tested), non-normality, and
    finite-sample noise. Pass n_trials = number of distinct strategy configurations
    evaluated (e.g. 3 for the three strategies in notebook 5).
    Returns a probability in [0, 1].
    """
    n = len(returns)
    sr_per = float(returns.mean() / returns.std())
    skew = float(returns.skew())
    excess_kurt = float(returns.kurtosis())

    var_sr = (1 + sr_per ** 2 / 2 - skew * sr_per + excess_kurt * sr_per ** 2 / 4) / (n - 1)
    if var_sr <= 0:
        return np.nan
    se = np.sqrt(var_sr)

    if n_trials <= 1:
        sr_star = 0.0
    else:
        # Expected maximum of n_trials IID N(0,1) variables (Lau 1980 approximation)
        _gamma = 0.5772156649  # Euler-Mascheroni constant
        e_max_z = ((1 - _gamma) * _norm.ppf(1 - 1 / n_trials) +
                   _gamma * _norm.ppf(1 - 1 / (n_trials * np.e)))
        sr_star = se * e_max_z   # scale to SR units

    return float(_norm.cdf((sr_per - sr_star) / se))


def summary(returns: pd.Series, periods: int = 252) -> pd.Series:
    return pd.Series(
        {
            "Ann. Returns": annualized_returns(returns, periods),
            "Ann. Vol": annualized_vol(returns, periods),
            "Sharpe": sharpe_ratio(returns, 0.0, periods),
            "PSR (SR*=0)": probabilistic_sharpe_ratio(returns, 0.0, periods),
            "Sortino": sortino_ratio(returns, 0.0, periods),
            "Max DD": max_drawdown(returns),
            "Calmar": calmar_ratio(returns, periods),
        }
    )