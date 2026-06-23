"""
Value at Risk (VaR) and Conditional VaR (CVaR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

def hist_var(returns: pd.Series, alpha: float=0.05) -> float:
    """
    Historical VaR. Alpha is the quantile of the return distribution (e.g. 0.05: minimum losses on the 5th-percentile).
    """
    return float(returns.quantile(alpha))

def hist_cvar(returns: pd.Series, alpha: float=0.05) -> float:
    """
    Historical CVaR: mean return given return <= VaR.
    """
    var = hist_var(returns, alpha)
    return float(returns[returns <= var].mean())

def param_var(returns: pd.Series, alpha: float=0.05) -> float:
    """
    Gaussian VaR (Assume normal distribution of returns). Can be used as baseline but severely underestimate risk. 
    """
    mu, sigma = returns.mean(), returns.std()
    return float(mu + sigma * stats.norm.ppf(alpha))