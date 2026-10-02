"""Candidate universe. Final stable/volatile buckets are assigned from realized
volatility in the *training* window only (see analyze.py), so these lists are
just the candidate pool, not the classification."""

# Historically defensive names: staples, utilities, healthcare, telecom.
STABLE_CANDIDATES = [
    "KO", "PEP", "PG", "CL", "KMB", "GIS", "JNJ", "MRK", "ABT", "MCD",
    "WMT", "COST", "DUK", "SO", "AEP", "ED", "XEL", "VZ", "T", "BRK-B",
    "ATO", "WM", "RSG", "HSY", "MDLZ",
]

# Historically high-beta / high-volatility names that existed for the full window.
VOLATILE_CANDIDATES = [
    "TSLA", "NVDA", "AMD", "MU", "NFLX", "SHOP", "XYZ", "ROKU", "ENPH", "FSLR",
    "MSTR", "RIOT", "MARA", "PLUG", "TWLO", "ETSY", "SEDG", "SMCI", "CVNA",
    "W", "AAL", "CCL", "OXY", "APA", "FCX",
]

ALL_TICKERS = STABLE_CANDIDATES + VOLATILE_CANDIDATES

# Market references: S&P 500 ETF, VIX index, long Treasuries, gold.
MARKET_TICKERS = ["SPY", "^VIX", "TLT", "GLD"]
