"""Use a compressor as a live forecaster and trade on it.

The adaptive context model from compression.py is turned into a next-day
predictor: the probability it would assign to "up" when encoding tomorrow's
symbol is its forecast. It is trained on 2014-2020, then walks forward through
2021-2026 one day at a time, predicting every stock before seeing that day and
learning from the day afterwards (exactly how a compressor updates).

Each day is one symbol combining: stock direction, stock move size (above or
below its typical move), and market (SPY) direction -> 8 symbols. The model
looks back k days of these symbols, pooled across all stocks.

Each day it buys the 10% of stocks with the highest predicted P(up) and shorts
the 10% with the lowest, equal-weighted, paying costs on turnover.

Outputs: output/predict_compression_results.json, output/predict_compression.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common import BLUE, GRAY, ORANGE, OUT, TEST, style, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ORDERS = (1, 2, 3)
DECILE = 0.10
COST_BPS = 5              # one way, per unit of turnover
ALPHA = 0.5               # KT prior count, as in compression.py
GREEN = "#1baf7a"

prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns="^VIX")
rets = prices.pct_change(fill_method=None)
rets = rets.where(rets.abs() < 1.0).iloc[1:]
spy = rets.pop("SPY")

# ---------------------------------------------------------------- symbols (all info known at that day's close)
typical = rets.abs().rolling(63, min_periods=21).median().shift(1)    # past-only "normal" move size
up = (rets > 0).astype(float).where(rets.notna())
big = (rets.abs() > typical).astype(float).where(rets.notna() & typical.notna())
mkt_up = (spy > 0).astype(int)
sym = up * 4 + big * 2 + mkt_up.values[:, None]                      # 0..7, NaN if missing

dates = rets.index
test_start = dates.searchsorted(pd.Timestamp(TEST[0]))
S = sym.values                                                        # days x stocks
U = up.values
R = rets.values


def context_ids(k):
    """Context id for predicting day t from symbols t-k..t-1 (-1 if any missing)."""
    ctx = np.zeros_like(S)
    for j in range(1, k + 1):
        prev = np.vstack([np.full((j, S.shape[1]), np.nan), S[:-j]])
        ctx = ctx * 8 + prev
    return np.where(np.isnan(ctx), -1, ctx).astype(np.int64)


results = {}
daily_pnl = {}
for k in ORDERS:
    ctx = context_ids(k)
    n_ctx = 8 ** k
    ups = np.zeros(n_ctx)
    tot = np.zeros(n_ctx)
    # learn from the training years
    m = (ctx[:test_start] >= 0) & ~np.isnan(U[:test_start])
    np.add.at(ups, ctx[:test_start][m], U[:test_start][m])
    np.add.at(tot, ctx[:test_start][m], 1)

    hits = n_pred = 0
    pnl, gross, prev_long, prev_short = [], [], set(), set()
    for t in range(test_start, len(dates)):
        c, u, r = ctx[t], U[t], R[t]
        ok = (c >= 0) & ~np.isnan(u)
        p_up = (ups[c[ok]] + ALPHA) / (tot[c[ok]] + 2 * ALPHA)        # forecast BEFORE seeing day t
        idx = np.flatnonzero(ok)
        hits += int(((p_up > 0.5) == (u[ok] == 1)).sum())
        n_pred += len(idx)
        n = max(1, int(len(idx) * DECILE))
        order = np.argsort(p_up + 1e-9 * np.random.default_rng(t).random(len(p_up)))  # random tie-break
        longs, shorts = set(idx[order[-n:]]), set(idx[order[:n]])
        turn = (1 - len(longs & prev_long) / n) + (1 - len(shorts & prev_short) / n)
        g = np.nanmean(r[list(longs)]) - np.nanmean(r[list(shorts)])
        gross.append(g)
        pnl.append(g - 2 * turn * COST_BPS / 1e4)
        prev_long, prev_short = longs, shorts
        np.add.at(ups, c[ok], u[ok])                                  # then learn from day t
        np.add.at(tot, c[ok], 1)
    pnl = pd.Series(pnl, index=dates[test_start:])
    gross = pd.Series(gross, index=dates[test_start:])
    daily_pnl[k] = pnl
    results[f"lookback_{k}d"] = {
        "accuracy": hits / n_pred,
        "predictions": n_pred,
        "long_short_before_costs": summarize(gross),
        "long_short_after_costs": summarize(pnl),
    }

best = max(ORDERS, key=lambda k: results[f"lookback_{k}d"]["accuracy"])
results["best_lookback"] = best
results["up_day_rate_test"] = float(np.nanmean(U[test_start:]))
results["always_up_accuracy"] = results["up_day_rate_test"]

OUT.mkdir(exist_ok=True)
with open(OUT / "predict_compression_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, ax = plt.subplots(figsize=(9, 3.8))
for k, c in zip(ORDERS, (BLUE, ORANGE, GREEN)):
    ax.plot(daily_pnl[k].index, (1 + daily_pnl[k]).cumprod(), color=c, lw=1.5, label=f"looks back {k} day{'s' * (k > 1)}")
ax.axhline(1, color=GRAY, lw=0.8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Compressor as forecaster: long top 10% / short bottom 10%, growth of $1 after costs")
fig.tight_layout(); fig.savefig(OUT / "predict_compression.png", dpi=150); plt.close(fig)

print(f"test days {len(dates) - test_start}, up-day rate {results['up_day_rate_test']:.2%} "
      f"(accuracy of always guessing 'up')")
for k in ORDERS:
    r = results[f"lookback_{k}d"]
    g, s = r["long_short_before_costs"], r["long_short_after_costs"]
    print(f"look back {k}: accuracy {r['accuracy']:.2%} on {r['predictions']:,} predictions | "
          f"before costs {g['ann_return']:+.1%}/yr (Sharpe {g['sharpe']:.2f}) | "
          f"after costs {s['ann_return']:+.1%}/yr (Sharpe {s['sharpe']:.2f}, max DD {s['max_drawdown']:.0%})")
