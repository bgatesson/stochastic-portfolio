"""
Fetch prices, clean the data, compute the returns, and save the data to a parquet file.

Usage:
    python -m scripts.fetch_data --universe <universe_name> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from spo.data import fetch_prices, clean_and_align_prices, load_universe, log_returns, simple_returns, load_return_anomalies, apply_return_anomalies

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("fetch_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and clean a price dataset.")
    parser.add_argument("--universe", default="sp500")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    parser.add_argument("--source", default="yfinance", choices=["yfinance"])
    parser.add_argument("--config", default="config/universes.yaml")
    args = parser.parse_args()

    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load the universe of assets
    universe = load_universe(args.universe, args.config)
    tickers = universe["tickers"]
    logger.info(f"Loaded universe '{universe['name']}' with {len(tickers)} tickers")

    # Fetch the price data
    all_tickers = list(tickers) + [universe["benchmark"]]
    logger.info(f"Fetching price data for {len(all_tickers)} tickers from {args.source}")
    raw_prices = fetch_prices(all_tickers, args.start_date, args.end_date, source=args.source)

    n_empty = raw_prices.isna().all().sum()
    if n_empty:
        logger.warning(f"{n_empty} tickers have no price data and will be dropped")
    
    raw_prices.to_parquet(raw_dir / f"{args.universe}_raw_prices.parquet")

    # Split benchmark from the rest of the assets and clean the data
    benchmark = (raw_prices[[universe["benchmark"]]].dropna() if universe["benchmark"] in raw_prices.columns else pd.DataFrame())
    prices = raw_prices.drop(columns=[universe["benchmark"]], errors="ignore")
    clean_prices = clean_and_align_prices(prices)

    # Compute returns
    log_ret = log_returns(clean_prices)
    simple_ret = simple_returns(clean_prices)

    # Fix found return anomalies if needed
    anomalies = load_return_anomalies()
    log_ret = apply_return_anomalies(log_ret, anomalies)
    simple_ret = apply_return_anomalies(simple_ret, anomalies)

    # Save the cleaned data and returns
    clean_prices.to_parquet(processed_dir / f"{args.universe}_cleaned_prices.parquet")
    log_ret.to_parquet(processed_dir / f"{args.universe}_log_returns.parquet")
    simple_ret.to_parquet(processed_dir / f"{args.universe}_simple_returns.parquet")
    if not benchmark.empty:
        benchmark.to_parquet(processed_dir / f"{args.universe}_benchmark.parquet")
    
    # Summary statistics
    annualized_vol = log_ret.std() * (252 ** 0.5)
    print("Universe:", universe["name"])
    print("Date range:", log_ret.index.min().date(), "to", log_ret.index.max().date())
    print("Trading days:", len(log_ret))
    print(f"Tickers with data: {log_ret.shape[1]} / {len(tickers)}")
    print(f"Annualized volatility: {annualized_vol.mean():.2%} (mean), ({annualized_vol.min():.2%} min, {annualized_vol.max():.2%} max)")


if __name__ == "__main__":
    main()