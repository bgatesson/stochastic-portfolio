"""
Performance metrics for backtesting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def summary(returns: pd.Series, periods: int = 252) -> pd.Series:
    return pd.Series(
        {
            "Ann. Returns": annualized_returns(returns, periods),
            "Ann. Vol": annualized_vol(returns, periods),
            "Sharpe": sharpe_ratio(returns, 0.0, periods),
            "Sortino": sortino_ratio(returns, 0.0, periods),
            "Max DD": max_drawdown(returns),
            "Calmar": calmar_ratio(returns, periods),
        }
    )