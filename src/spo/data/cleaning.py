"""
Cleaning, aligning, and return calculations for the data.

Methodology:
- prices are aligned to a single business day index
- forward-fill is only up to 3 business days by default
- tickers with too much missing data are dropped
- returns are calculated as log returns by default
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

def load_return_anomalies(config_path: str | Path="config/return_anomalies.yaml") -> list[dict]:
    """
    Load return anomalies from YAML config.
    Returns a list of dicts with keys: ticker, date, reason, fix (optional), or empty list if config file is missing or empty.
    """
    path = Path(config_path)
    if not path.exists():
        logger.info(f"Return anomalies config file not found at {config_path}. Proceeding without anomalies.")
        return []
    with open(path, 'r') as f:
        config = yaml.safe_load(f) or {}
    anomalies = config.get('anomalies', [])
    logger.info(f"Loaded {len(anomalies)} return anomalies from config.")
    return anomalies

def apply_return_anomalies(returns: pd.DataFrame, anomalies: list[dict]) -> pd.DataFrame:
    """
    Nullify returns for specified anomalies based on the config.
    """
    if not anomalies:
        return returns
    out = returns.copy()
    applied = 0
    for anomaly in anomalies:
        ticker = anomaly['ticker']
        date = pd.to_datetime(anomaly['date'])
        fix = anomaly.get('fix', np.nan) # default to np.nan if not specified
        if ticker not in out.columns:
            logger.warning(f"Ticker {ticker} from anomalies config not found in returns DataFrame. Skipping.")
            continue
        if date not in out.index:
            logger.warning(f"Date {date.date()} from anomalies config not found in returns DataFrame index. Skipping.")
            continue
        out.loc[date, ticker] = fix
        applied += 1
    logger.info(f"Applied {applied}/{len(anomalies)} fixes")
    return out

def clean_and_align_prices(
    prices: pd.DataFrame,
    max_ffill_days: int = 3,
    max_missing_pct: float = 0.05,
    min_coverage_days: int = 252,
) -> pd.DataFrame:
    """
    Clean and align price data
    """
    if prices.empty:
        raise ValueError("Input price DataFrame is empty")
    
    # align to business day index
    bday_index = pd.bdate_range(start=prices.index.min(), end=prices.index.max())
    aligned_prices = prices.reindex(bday_index)

    # forward-fill missing values up to max_ffill_days
    aligned_prices = aligned_prices.ffill(limit=max_ffill_days)

    # drop tickers with too much missing data
    missing_pct = aligned_prices.isna().mean()
    coverage_days = aligned_prices.notna().sum()
    keep_tickers = (missing_pct <= max_missing_pct) & (coverage_days >= min_coverage_days)

    dropped_tickers = aligned_prices.columns[~keep_tickers].tolist()
    if dropped_tickers:
        logger.warning(f"Dropping tickers with too much missing data: {dropped_tickers}")
    cleaned_prices = aligned_prices.loc[:, keep_tickers]

    cleaned_prices = cleaned_prices.dropna(how='any')  # drop rows with any remaining NaNs

    logger.info(f"Cleaned and aligned prices. Final shape: {cleaned_prices.shape}")
    return cleaned_prices

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from price data
    """
    return np.log(prices / prices.shift(1)).dropna(how='all')

def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simple returns from price data
    """
    return prices.pct_change().dropna(how='all')