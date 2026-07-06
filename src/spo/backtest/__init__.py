"""
Backtest engine + performance metrics.
"""
from spo.backtest.engine import backtest
from spo.backtest.metrics import( 
    annualized_returns, 
    annualized_vol, 
    calmar_ratio, 
    deflated_sharpe_ratio, 
    max_drawdown, 
    probabilistic_sharpe_ratio, 
    sharpe_ratio, 
    sortino_ratio, 
    summary
)
from spo.backtest.strategies import REGISTRY, StrategyFn, get_strategy

__all__ = [
    "annualized_returns",
    "annualized_vol",
    "calmar_ratio",
    "deflated_sharpe_ratio",
    "max_drawdown",
    "probabilistic_sharpe_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "summary",
    "backtest",
    "REGISTRY",
    "StrategyFn",
    "get_strategy"
]