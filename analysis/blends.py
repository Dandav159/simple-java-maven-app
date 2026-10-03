"""Shared PCA/ICA fit: rotate stock returns into blends using TRAIN data only."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from common import TRAIN

HERE = Path(__file__).parent
M = 20                 # dimensions kept from PCA; all methods rotate within these
CLIP = 4.0             # cap standardized moves at 4 sd when FITTING, so a few crash days can't dominate


def load_returns():
    """Daily close-to-close returns for S&P 500 members with the full history."""
    prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
    prices = prices.loc[prices["SPY"].notna()].drop(columns=["^VIX", "SPY"])
    rets = prices.pct_change(fill_method=None).iloc[1:]
    return rets.loc[:, rets.notna().all()]


def fit(rets):
    """Return train mean, train sd, top-M PCA directions and ICA directions (stocks x M)."""
    tr = rets.loc[TRAIN[0]:TRAIN[1]]
    mu, sd = tr.mean(), tr.std()
    Ztr = ((tr - mu) / sd).clip(-CLIP, CLIP).values
    _, _, vt = np.linalg.svd(Ztr, full_matrices=False)
    P = vt[:M].T
    ica = FastICA(n_components=M, whiten="unit-variance", random_state=0, max_iter=2000)
    ica.fit(Ztr @ P)
    return mu, sd, P, P @ ica.components_.T
