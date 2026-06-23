"""
Backtesting engine.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
import pandas as pd

logger = logging.getLogger(__name__)

StrategyFn = Callable[[pd.DataFrame], pd.Series]


def backtest(returns: pd.DataFrame, strategy_fn: StrategyFn, rebalance_freq: str="BME", lookback: int=504, cost: float=10.0, min_period: int | None=None) -> dict[str, pd.Series | pd.DataFrame]:
    """
    Walk forward backtesting with portfolio rebalancing at each period.

    At each rebalance date, the strategy uses the lookback period to compute the new weights. 
    The weights are then applied to the returns from next day onward.
    """
    if min_period is None:
        min_period = lookback // 2

    schedule = pd.date_range(returns.index.min(), returns.index.max(), freq=rebalance_freq)
    rebalance_dates = schedule.intersection(returns.index)

    weights = pd.DataFrame(index=rebalance_dates, columns=returns.columns, dtype=float)

    for date in rebalance_dates:
        lb = returns.loc[:date].iloc[-lookback:]
        if len(lb) < min_period:
            continue
        
        try:
            w = strategy_fn(lb)
        except Exception as e:
            logger.warning("Strategy failed at %s: %s", date.date(), e)
            continue
        weights.loc[date] = w.reindex(returns.columns).fillna(0.0)
    
    curr_weights = weights.reindex(returns.index).ffill().fillna(0.0)

    gross_returns = (curr_weights.shift(1) * returns).sum(axis=1)

    turnover = curr_weights.diff().abs().sum(axis=1)
    costs = turnover * cost / 1e4 # cost is in bps 
    net_returns = gross_returns - costs

    return {
        "weights": curr_weights,
        "gross_returns": gross_returns,
        "net_returns": net_returns,
        "turnover": turnover,
        "rebalance_dates": rebalance_dates
    }

