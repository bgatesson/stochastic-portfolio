"""
CVaR portfolio optmization via the Rockafellar-Uryasev linear program.
Parameters:
    alpha: confidence level (e.g. alpha=0.95 means we care about the 5% worst)
    scenarios: (n_scenarios, n_assets) horizon of simpel returns
    losses: L_s = -r_s * w, with sum(w) = 1
"""
from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def scenarios_from_sim(simulation: dict[str, np.ndarray], return_type: str="simple") -> np.ndarray:
    """
    Aggregate path log-returns into horizon scenarios.
    """
    horizon_log = simulation["log_returns"].sum(axis=1)
    if return_type == "log":
        return horizon_log
    if return_type == "simple":
        return np.expm1(horizon_log)
    raise ValueError(f"Unknown return_type: {return_type}")


def compute_cvar(scenarios: np.ndarray, weights: np.ndarray | pd.Series, alpha: float=0.95) -> tuple[float, float]:
    """
    Empirical (VaR, CVaR) of a portfolio over a set of scenarios.

    Returns (VaR_alpha, CVaR_alpha), both positive numbers for losses.
    A CVaR of 0.04 means the average loss in the worst (1-alpha)·100% of
    scenarios is 4% of portfolio value over the horizon.
    """
    w = np.asarray(weights)
    losses = -(scenarios @ w)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    cvar = float(tail.mean())
    return var, cvar


def _build_cvar_lp(scenarios: np.ndarray, alpha: float, long_only: bool, max_weight: float | None) -> tuple[cp.Variable, cp.Variable, cp.Variable, cp.Expression, list]:
    """
    Construct the shared LP variables, CVaR expression, and constraints.
    """
    S, N = scenarios.shape
    w = cp.Variable(N, name="weights")
    v = cp.Variable(name="VaR")
    z = cp.Variable(S, name="excess_losses")

    losses = -scenarios @ w
    cvar_lp = v + (1.0 / (S * (1.0 - alpha))) * cp.sum(z)

    constraints = [
        z >= losses - v,
        z >= 0,
        cp.sum(w) == 1
    ]
    if long_only:
        constraints.append(w >= 0)
    if max_weight is not None:
        constraints.append(w <= max_weight)
    
    return w, v, z, cvar_lp, constraints


def min_cvar_portfolio(scenarios: np.ndarray, alpha: float=0.95, long_only: bool=True, max_weight: float | None=None, tickers: list[str] | None=None) -> pd.Series:
    """
    Minimize CVaR over a set of horizon return scenarios.
    """
    if scenarios.ndim != 2:
        raise ValueError(f"scenarios must be 2D, got shape {scenarios.shape}")
    
    w, _, _, cvar_lp, constraints = _build_cvar_lp(scenarios, alpha, long_only, max_weight)
    prob = cp.Problem(cp.Minimize(cvar_lp), constraints)
    prob.solve()

    if w.value is None:
        raise RuntimeError(f"CVaR LP failed: status={prob.status}")
    
    return pd.Series(w.value, index=tickers)


def max_return_cvar_constrained(scenarios: np.ndarray, cvar_limit: float, alpha: float=0.95, long_only: bool=True, max_weight: float | None=None, tickers: list[str] | None=None) -> pd.Series:
    """
    Maximize expected return subject to a CVaR budget.
    """
    w, _, _, cvar_lp, constraints = _build_cvar_lp(scenarios, alpha, long_only, max_weight)
    constraints.append(cvar_lp <= cvar_limit)

    mu = scenarios.mean(axis=0)
    prob = cp.Problem(cp.Maximize(mu @ w), constraints)
    prob.solve()

    if w.value is None:
        raise RuntimeError(
            f"CVaR-constrained LP infeasible at budget={cvar_limit}: status={prob.status}"
        )
    return pd.Series(w.value, index=tickers)


def cvar_frontier(scenarios: np.ndarray, n_points: int=30, alpha: float=0.95, long_only: bool=True, tickers: list[str] | None=None) -> pd.DataFrame:
    """
    Trace the CVaR efficient frontier using CVaR budgets.
    """
    w_min = min_cvar_portfolio(scenarios, alpha=alpha, long_only=long_only, tickers=tickers)
    _, cvar_min = compute_cvar(scenarios, w_min, alpha)

    mu = scenarios.mean(axis=0)
    w_max = np.zeros(scenarios.shape[1])
    w_max[np.argmax(mu)] = 1.0
    _, cvar_max = compute_cvar(scenarios, w_max, alpha)

    budgets = np.linspace(cvar_min, cvar_max, n_points)
    rows = []
    for b in budgets:
        try:
            w = max_return_cvar_constrained(
                scenarios, cvar_limit=b, alpha=alpha,
                long_only=long_only, tickers=tickers,
            )
        except RuntimeError:
            continue
        var_r, cvar_r = compute_cvar(scenarios, w, alpha)
        row = {
            "cvar_budget": b,
            "expected_return": float(mu @ w.values),
            "realized_cvar": cvar_r,
            "realized_var": var_r
        }
        row.update(w.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)