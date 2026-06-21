"""
Populate a universe's ticker list from an internet source.

Usage:
    python -m scripts.populate_tickers --universe <universe_name>

Currently supports the following universes:
    - 'sp500': The S&P 500 index
    - 'cac40': The CAC 40 index
    - 'all': All tickers from the above universes
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import requests
import yaml
from lxml import html as lxml_html

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("populate_tickers")


def _get_wikipedia_index(url: str, ticker_column: tuple[str, ...], expected_size: tuple[int, int], replace_dot: bool = False) -> list[str]:
    """
    Fetches a list of tickers from a Wikipedia page containing an index.

    Args:
        url (str): The URL of the Wikipedia page.
        ticker_column (tuple[str, ...]): The column(s) in the table that contains the tickers. 
        expected_size (tuple[int, int]): The expected size range of the table (min, max) to filter out irrelevant tables.
        replace_dot (bool): Whether to replace dots in tickers with dashes.

    Returns:
        list[str]: A list of tickers.
    """
    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    response = requests.get(url, headers={"User-Agent": ua}, timeout=15)
    response.raise_for_status()
    doc = lxml_html.fromstring(response.content)
    for table_el in doc.findall(".//table"):
        first_row = table_el.find(".//tr")
        if first_row is None:
            continue
        col_names = [th.text_content().strip() for th in first_row.findall("th")]
        matched_idx = next((i for i, h in enumerate(col_names) if h in ticker_column), None)
        if matched_idx is None:
            continue
        data_rows = [row for row in table_el.findall(".//tr") if row.findall("td")]
        if not (expected_size[0] <= len(data_rows) <= expected_size[1]):
            continue
        tickers = []
        for row in data_rows:
            cells = row.findall("td")
            if len(cells) > matched_idx:
                ticker = cells[matched_idx].text_content().strip()
                if replace_dot:
                    ticker = ticker.replace(".", "-")
                tickers.append(ticker)
        logger.info(f"Found {len(tickers)} tickers in table with column '{col_names[matched_idx]}'")
        return sorted(tickers)
    raise RuntimeError(f"No suitable table found at {url} with expected size range {expected_size[0]}-{expected_size[1]} and ticker column(s) {ticker_column}")


def fetch_sp500() -> list[str]:
    """Fetches the list of S&P 500 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return _get_wikipedia_index(url, ticker_column=('Symbol',), expected_size=(400, 600), replace_dot=True)


def fetch_cac40() -> list[str]:
    """Fetches the list of CAC 40 tickers from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/CAC_40"
    return _get_wikipedia_index(url, ticker_column=('Ticker',), expected_size=(30, 50))


FETCHERS = {
    'sp500': fetch_sp500,
    'cac40': fetch_cac40,
}


def update_universe(universe_key: str, config: dict) -> None:
    """Updates the ticker list for a given universe in the configuration."""
    if universe_key not in config:
        raise ValueError(f"Universe '{universe_key}' not found in configuration.")
    if universe_key not in FETCHERS:
        raise ValueError(f"No fetcher defined for universe '{universe_key}'. Available fetchers: {list(FETCHERS)}")
    tickers = FETCHERS[universe_key]()
    config[universe_key]['tickers'] = tickers
    logger.info(f"Updated '{universe_key}' universe with {len(tickers)} tickers.")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--universe', required=True, help=f"Universe key or 'all'. Available options: {list(FETCHERS)}")
    parser.add_argument('--config', default='config/universes.yaml')
    args = parser.parse_args()

    config_path = Path(args.config)
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    targets = list(FETCHERS) if args.universe == 'all' else [args.universe]
    for u in targets:
        update_universe(u, config)
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    logger.info(f"Saved updated configuration to {config_path}")


if __name__ == "__main__":
    main()