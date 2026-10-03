"""Download outside ("macro") data and adjusted open/close prices.

Writes:
  data/macro_close.csv  - daily closes of rates, commodities, currencies, credit
                          and foreign stock indices
  data/macro_open.csv   - their opens
  data/sp500_open.csv   - split/dividend-adjusted opens for the S&P 500 members
                          (adjusted open = open * adjclose / close)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from fetch_data import DATA_DIR, HEADERS, URL

MACRO = {
    "^TNX": "10y Treasury yield", "^FVX": "5y Treasury yield", "^IRX": "3m T-bill yield",
    "CL=F": "crude oil", "GC=F": "gold", "HG=F": "copper", "NG=F": "natural gas",
    "DX-Y.NYB": "US dollar index", "HYG": "high-yield credit", "LQD": "investment-grade credit",
    "TLT": "long Treasuries", "EEM": "emerging markets", "^VIX": "VIX",
    "^N225": "Japan (Nikkei)", "^HSI": "Hong Kong (Hang Seng)", "^AXJO": "Australia (ASX 200)",
    "^KS11": "Korea (KOSPI)", "^GDAXI": "Germany (DAX)", "^FTSE": "UK (FTSE 100)",
}


def fetch_ohlc(ticker, start="2014-01-01", end="2026-10-01"):
    params = {"period1": int(pd.Timestamp(start).timestamp()), "period2": int(pd.Timestamp(end).timestamp()),
              "interval": "1d", "events": "div,splits"}
    for attempt in range(4):
        try:
            r = requests.get(URL.format(t=ticker), params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            # Stamp each bar with the exchange's LOCAL date (UTC can be the previous day,
            # e.g. Sydney opens at 23:00 UTC in its summer).
            offset = res["meta"].get("gmtoffset", 0)
            idx = pd.to_datetime([t + offset for t in res["timestamp"]], unit="s").normalize()
            q = res["indicators"]["quote"][0]
            close = pd.Series(q["close"], index=idx, dtype=float)
            adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
            adj = pd.Series(adj, index=idx, dtype=float) if adj else close
            df = pd.DataFrame({"open": pd.Series(q["open"], index=idx, dtype=float) * adj / close, "close": adj})
            return ticker, df.groupby(level=0).last()
        except Exception as e:
            print(f"  {ticker}: attempt {attempt + 1} failed ({e})", file=sys.stderr)
            time.sleep(2 ** (attempt + 1))
    return ticker, None


def download(tickers):
    with ThreadPoolExecutor(max_workers=6) as ex:
        got = {t: df for t, df in ex.map(fetch_ohlc, tickers) if df is not None}
    opens = pd.concat({t: df["open"] for t, df in got.items()}, axis=1, sort=True)
    closes = pd.concat({t: df["close"] for t, df in got.items()}, axis=1, sort=True)
    return opens, closes


def main():
    o, c = download(list(MACRO))
    o.to_csv(DATA_DIR / "macro_open.csv")
    c.to_csv(DATA_DIR / "macro_close.csv")
    print(f"macro: {c.shape}, missing {sorted(set(MACRO) - set(c.columns))}")
    members = pd.read_csv(DATA_DIR / "sp500_members.csv").ticker.tolist()
    o, _ = download(members + ["SPY"])
    o.to_csv(DATA_DIR / "sp500_open.csv")
    print(f"opens: {o.shape}")


if __name__ == "__main__":
    main()
