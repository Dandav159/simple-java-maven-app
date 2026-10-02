"""Download daily adjusted closes for current S&P 500 members (+ SPY, ^VIX).

Writes data/sp500_prices.csv and data/sp500_members.csv (ticker, sector,
date added). Only current members are available, so the universe is
survivorship-biased; analyze_factors.py mitigates part of it by admitting
each stock only from its index-inclusion date.
"""
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from fetch_data import DATA_DIR, HEADERS, fetch

WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def members():
    html = requests.get(WIKI, headers=HEADERS, timeout=30).text
    t = pd.read_html(io.StringIO(html))[0]
    return pd.DataFrame({
        "ticker": t["Symbol"].str.replace(".", "-", regex=False),
        "sector": t["GICS Sector"],
        "date_added": pd.to_datetime(t["Date added"], errors="coerce"),
    })


def main():
    DATA_DIR.mkdir(exist_ok=True)
    m = members()
    m.to_csv(DATA_DIR / "sp500_members.csv", index=False)
    tickers = list(m.ticker) + ["SPY", "^VIX"]
    with ThreadPoolExecutor(max_workers=6) as ex:
        series = [s for s in ex.map(fetch, tickers) if s is not None]
    prices = pd.concat(series, axis=1, sort=True)
    prices.to_csv(DATA_DIR / "sp500_prices.csv")
    print(f"saved {prices.shape}; missing: {sorted(set(tickers) - set(prices.columns))}")


if __name__ == "__main__":
    main()
