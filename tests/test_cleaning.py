"""
Tests for spo.data.cleaning module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spo.data.cleaning import clean_and_align_prices, log_returns, simple_returns


def _make_prices(n_days: int=500, n_tickers: int=5, seed: int=0) -> pd.DataFrame:
    """Synthetic price data for testing."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    returns = rng.normal(loc=0.0005, scale=0.01, size=(n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    return pd.DataFrame(prices, index=idx, columns=[f"Ticker{i}" for i in range(n_tickers)])

def test_clean_and_align_prices_all_valid():
    """Test clean_and_align_prices with all valid data."""
    prices = _make_prices()
    cleaned = clean_and_align_prices(prices)
    assert cleaned.shape == prices.shape
    assert not cleaned.isna().any().any()

def test_clean_and_align_prices_sparse_tickers():
    """Test clean_and_align_prices with some tickers having sparse data."""
    prices = _make_prices()
    prices.iloc[:480, 0] = np.nan  # Introduce NaNs in Ticker0
    cleaned = clean_and_align_prices(prices)
    assert "Ticker0" not in cleaned.columns
    assert cleaned.shape[1] == prices.shape[1] - 1

def test_clean_and_align_prices_ffill_short_gaps():
    """Test clean_and_align_prices with short gaps that can be forward-filled."""
    prices = _make_prices()
    prices.iloc[100, 0] = np.nan  # Short gap in Ticker0
    cleaned = clean_and_align_prices(prices)
    assert "Ticker0" in cleaned.columns
    assert not cleaned["Ticker0"].isna().any()

def test_log_returns():
    """Test log_returns function."""
    prices = _make_prices()
    log_ret = log_returns(prices)
    assert log_ret.shape == (prices.shape[0] - 1, prices.shape[1])
    assert np.isfinite(log_ret.values).all()

def test_log_vs_simple_returns():
    """Test that log_returns and simple_returns give similar results for small returns."""
    prices = _make_prices()
    log_ret = log_returns(prices)
    simple_ret = simple_returns(prices)
    # For small returns, log(1 + r) ~ r
    assert np.allclose(log_ret.values, simple_ret.values, atol=1e-3) # doesn't work for atol=1e-4 due to the scale of returns

def test_clean_and_align_prices_empty_dataframe():
    """Test clean_and_align_prices with an empty DataFrame."""
    with pytest.raises(ValueError):
        clean_and_align_prices(pd.DataFrame())