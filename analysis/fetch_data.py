"""Download daily adjusted close prices from Yahoo Finance's chart API.

Writes data/prices.csv (one column per ticker, indexed by date).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from universe import ALL_TICKERS, MARKET_TICKERS

DATA_DIR = Path(__file__).parent / "data"
URL = "https://query1.finance.yahoo.com/v8/finance/chart/{t}"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch(ticker, start="2014-01-01", end="2026-10-01"):
    params = {
        "period1": int(pd.Timestamp(start).timestamp()),
        "period2": int(pd.Timestamp(end).timestamp()),
        "interval": "1d",
        "events": "div,splits",
    }
    for attempt in range(4):
        try:
            r = requests.get(URL.format(t=ticker), params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = pd.to_datetime(res["timestamp"], unit="s").normalize()
            adj = res["indicators"]["adjclose"][0]["adjclose"]
            return pd.Series(adj, index=ts, name=ticker).groupby(level=0).last()
        except Exception as e:  # network hiccup / rate limit
            print(f"  {ticker}: attempt {attempt + 1} failed ({e})", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    return None


def main():
    DATA_DIR.mkdir(exist_ok=True)
    series = []
    for t in ALL_TICKERS + MARKET_TICKERS:
        s = fetch(t)
        if s is not None:
            series.append(s)
            print(f"{t}: {len(s)} rows {s.index.min().date()} -> {s.index.max().date()}")
        time.sleep(0.3)
    prices = pd.concat(series, axis=1, sort=True)
    prices.to_csv(DATA_DIR / "prices.csv")
    print(f"saved {prices.shape} to {DATA_DIR / 'prices.csv'}")


if __name__ == "__main__":
    main()
