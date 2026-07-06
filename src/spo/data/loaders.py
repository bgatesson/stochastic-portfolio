"""
Price and universe loaders
"""
from __future__ import annotations
import logging
from pathlib import Path
import pandas as pd
import yaml
from typing import Literal

logger = logging.getLogger(__name__)

SOURCE = Literal["yfinance"] # can add more sources in the future

def load_universe(name: str, config_path: str | Path="config/universes.yaml") -> dict:
    """
    Load a universe from a YAML configuration file
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Universe '{name}' not found in config. Available universes: {available}")
    return config[name]

def _fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Fetch adjusted close prices data from yfinance
    """
    import yfinance as yf

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize data to have a single level of columns with tickers as column names
    if isinstance(data.columns, pd.MultiIndex):
        close = pd.DataFrame(
            {ticker: data[ticker]["Close"] for ticker in tickers if ticker in data.columns.levels[0]}
        )
    else:
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close

_MCAP_CACHE = Path("data/processed/market_caps_cache.json")

def fetch_market_caps(
    tickers: list[str],
    cache_path: str | Path | None = _MCAP_CACHE,
    batch_size: int = 50,
    min_coverage: float = 0.80,
) -> pd.Series:
    """Fetch current market caps from Yahoo Finance and return normalised weights.

    Results are cached to *cache_path* so subsequent calls are instant.
    Delete the cache file to force a refresh.

    Limitation: these are the current market caps, not the
    historical weights that existed during the backtest window.
    """
    import time
    import yfinance as yf

    # Cache read 
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            cached = pd.read_json(cache_file, typ="series")
            valid = cached.reindex(tickers).dropna()
            if len(valid) >= min_coverage * len(tickers):
                logger.info("Using cached market caps (%d/%d tickers)", len(valid), len(tickers))
                return valid / valid.sum()
            logger.info("Cache coverage %.0f%% < %.0f%% threshold — re-fetching",
                        100 * len(valid) / len(tickers), 100 * min_coverage)

    # Fetch 
    caps: dict[str, float] = {}
    batches = [tickers[i: i + batch_size] for i in range(0, len(tickers), batch_size)]
    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(2.0)
        bundle = yf.Tickers(" ".join(batch))
        for t in batch:
            for attempt in range(3):
                try:
                    mc = bundle.tickers[t].fast_info.market_cap
                    if mc and mc > 0:
                        caps[t] = float(mc)
                    break
                except Exception as e:
                    msg = str(e)
                    if "Too Many Requests" in msg and attempt < 2:
                        time.sleep(10 * (attempt + 1))
                    else:
                        logger.warning("Market cap fetch failed for %s: %s", t, e)
                        break

    if not caps:
        logger.warning("No market caps retrieved — falling back to equal weight")
        return pd.Series(1.0 / len(tickers), index=tickers)

    s = pd.Series(caps)

    # Cache write 
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        s.to_json(cache_file)
        logger.info("Market caps cached to %s", cache_file)

    return s / s.sum()


def fetch_prices(tickers: list[str], start: str, end: str, source: SOURCE="yfinance") -> pd.DataFrame:
    """
    Fetch adjusted close prices data for a list of tickers between start and end dates
    """
    if source == "yfinance":
        prices = _fetch_yfinance(tickers, start, end)
    else:
        raise ValueError(f"Unsupported data source: {source}")
    
    return prices.sort_index()