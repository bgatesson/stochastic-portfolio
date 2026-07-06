"""
Data loading and cleaning
"""
from spo.data.loaders import fetch_market_caps, fetch_prices, load_universe
from spo.data.cleaning import clean_and_align_prices, log_returns, simple_returns, load_return_anomalies, apply_return_anomalies

__all__ = [
    "fetch_market_caps",
    "fetch_prices",
    "load_universe",
    "clean_and_align_prices",
    "log_returns",
    "simple_returns",
    "load_return_anomalies",
    "apply_return_anomalies",
]