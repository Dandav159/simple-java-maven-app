"""Download intraday bars from Yahoo Finance (regular session only).

  data/intraday_60m.parquet  - hourly bars, ~730 days (Yahoo's limit)
  data/intraday_5m.parquet   - 5-minute bars, ~60 days (Yahoo's limit)

Bars are NOT split-adjusted; analysis code drops days whose overnight gap
looks like a split. Timestamps are converted to New York time.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from fetch_data import DATA_DIR, HEADERS, URL

ETFS = ["SPY", "QQQ", "IWM"]


def fetch(ticker, interval, rng):
    params = {"interval": interval, "range": rng, "includePrePost": "false"}
    for attempt in range(4):
        try:
            r = requests.get(URL.format(t=ticker), params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")},
                              index=pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York"))
            df = df.dropna(subset=["open", "close"])
            df["ticker"] = ticker
            return df
        except Exception as e:
            print(f"  {ticker} {interval}: attempt {attempt + 1} failed ({e})", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    return None


def main():
    tickers = ETFS + pd.read_csv(DATA_DIR / "sp500_members.csv").ticker.tolist()
    for interval, rng, name in (("60m", "730d", "intraday_60m"), ("5m", "60d", "intraday_5m")):
        with ThreadPoolExecutor(max_workers=6) as ex:
            frames = [f for f in ex.map(lambda t: fetch(t, interval, rng), tickers) if f is not None]
        df = pd.concat(frames).rename_axis("time").reset_index()
        df.to_parquet(DATA_DIR / f"{name}.parquet", index=False)
        print(f"{name}: {len(df):,} bars, {df.ticker.nunique()} tickers, {df.time.min()} -> {df.time.max()}")


if __name__ == "__main__":
    main()
