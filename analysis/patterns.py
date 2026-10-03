"""Do classic chart patterns (floors, ceilings, breakouts...) beat random entries?

Every pattern is traded the same way: stop-loss 1 unit away, target 2 units away
(1 unit = 2x the stock's 20-day daily volatility), exit at whichever is hit
first on a daily close, or after 60 trading days. One open trade per stock per
pattern at a time. Each pattern's results are compared with RANDOM entries in
the SAME direction (stocks drift up, so buys are judged against random buys).

Patterns use closing prices only (the data has no intraday highs/lows):
  floor_bounce      BUY   close within 0.5 unit of the 60-day low, which was set
                          10+ days ago (an established floor), and today is up
  floor_break       SELL  close falls below that established 60-day low
  ceiling_reject    SELL  close within 0.5 unit of the 60-day high, set 10+ days
                          ago (an established ceiling), and today is down
  ceiling_breakout  BUY   close rises above that established 60-day high
  trend_pullback    BUY   above the 200-day average, but down 1+ unit over 5 days
  golden_cross      BUY   50-day average crosses above the 200-day
  death_cross       SELL  50-day average crosses below the 200-day

Outputs: output/patterns_results.json, output/patterns.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common import BLUE, GRAY, ORANGE, OUT, TEST, TRAIN, style

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
TARGET, STOP = 2.0, 1.0
MAX_DAYS = 60
LOOK = 60                  # window for floors and ceilings
MIN_AGE = 10               # the floor/ceiling must have been set at least this long ago
COST_BPS = 10              # round trip
N_RANDOM = 4               # random entries per stock-day sampled = this x pattern count
rng = np.random.default_rng(0)

prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns=["^VIX", "SPY"])
dates = prices.index
logp = np.log(prices)
vol = logp.diff().rolling(20).std()
unit = 2 * vol

# ---------------------------------------------------------------- pattern signals (all use data up to today's close)
prior = logp.shift(1)
low_prev = prior.rolling(LOOK - 1).min()                   # lowest close of the prior window
high_prev = prior.rolling(LOOK - 1).max()
low_age = prior.rolling(LOOK - 1).apply(lambda x: len(x) - 1 - np.nanargmin(x), raw=True)
high_age = prior.rolling(LOOK - 1).apply(lambda x: len(x) - 1 - np.nanargmax(x), raw=True)
up_day, down_day = logp.diff() > 0, logp.diff() < 0
ma50, ma200 = logp.rolling(50).mean(), logp.rolling(200).mean()

signals = {
    "floor_bounce": (1, (logp >= low_prev) & (logp - low_prev <= 0.5 * unit) & (low_age >= MIN_AGE) & up_day),
    "floor_break": (-1, (logp < low_prev) & (prior >= low_prev) & (low_age >= MIN_AGE)),
    "ceiling_reject": (-1, (logp <= high_prev) & (high_prev - logp <= 0.5 * unit) & (high_age >= MIN_AGE) & down_day),
    "ceiling_breakout": (1, (logp > high_prev) & (prior <= high_prev) & (high_age >= MIN_AGE)),
    "trend_pullback": (1, (logp > ma200) & (logp - logp.shift(5) <= -unit)),
    "golden_cross": (1, (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))),
    "death_cross": (-1, (ma50 < ma200) & (ma50.shift(1) >= ma200.shift(1))),
}

L, U = logp.values, unit.values


def trade(t, s, side):
    """Outcome of one trade entered at the close of day t, in risk units after costs."""
    u = U[t, s]
    path = side * (L[t + 1:t + MAX_DAYS + 1, s] - L[t, s]) / u
    path = path[~np.isnan(path)]
    if len(path) == 0:
        return None
    hit_t = np.flatnonzero(path >= TARGET)
    hit_s = np.flatnonzero(path <= -STOP)
    i_t = hit_t[0] if len(hit_t) else len(path)
    i_s = hit_s[0] if len(hit_s) else len(path)
    if i_t < i_s:
        kind, exit_i = "win", i_t
    elif i_s < i_t:
        kind, exit_i = "loss", i_s
    else:
        kind, exit_i = "timeout", len(path) - 1
    return kind, path[exit_i] - COST_BPS / 1e4 / u, exit_i + 1


def valid(t, s):
    return t + 1 < len(dates) and U[t, s] > 0 and not np.isnan(L[t, s])


def run_pattern(mask, side):
    rows = []
    M = mask.values
    for s in range(M.shape[1]):
        busy_until = -1
        for t in np.flatnonzero(M[:, s]):
            if t <= busy_until or not valid(t, s):
                continue
            out = trade(t, s, side)
            if out:
                rows.append((dates[t], out[0], out[1]))
                busy_until = t + out[2]
    return pd.DataFrame(rows, columns=["date", "kind", "units"])


def run_random(side, n):
    rows = []
    while len(rows) < n:
        t, s = rng.integers(200, len(dates) - 2), rng.integers(L.shape[1])
        if valid(t, s):
            out = trade(t, s, side)
            if out:
                rows.append((dates[t], out[0], out[1]))
    return pd.DataFrame(rows, columns=["date", "kind", "units"])


def stats(df):
    """Win rate, average result, and a t-stat on MONTHLY averages (trades in the same
    month move together, so per-trade t-stats would overstate confidence)."""
    if len(df) == 0:
        return None
    monthly = df.groupby(df.date.dt.to_period("M")).units.mean()
    return {
        "trades": len(df),
        "win_rate": float((df.kind == "win").mean()),
        "loss_rate": float((df.kind == "loss").mean()),
        "avg_units": float(df.units.mean()),
        "t_monthly": float(monthly.mean() / monthly.std() * np.sqrt(len(monthly))) if len(monthly) > 2 else float("nan"),
    }


random_trades = {side: run_random(side, 80_000) for side in (1, -1)}
results = {}
for name, (side, mask) in signals.items():
    df = run_pattern(mask, side)
    rnd = random_trades[side]
    results[name] = {"direction": "buy" if side == 1 else "sell"}
    for period, (a, b) in (("train", TRAIN), ("test", TEST), ("all", ("2014", "2026"))):
        p = stats(df[(df.date >= a) & (df.date <= b)])
        r = stats(rnd[(rnd.date >= a) & (rnd.date <= b)])
        merged = pd.concat([df.assign(g=1), rnd.assign(g=0)])
        merged = merged[(merged.date >= a) & (merged.date <= b)]
        # edge vs random, t-stat on the monthly difference of averages
        mp = merged.groupby([merged.date.dt.to_period("M"), "g"]).units.mean().unstack()
        d = (mp[1] - mp[0]).dropna()
        results[name][period] = {
            "pattern": p, "random_same_direction": r,
            "edge_units": float(d.mean()),          # average monthly gap vs random, matches its t-stat
            "edge_win_rate": p["win_rate"] - r["win_rate"],
            "edge_t_monthly": float(d.mean() / d.std() * np.sqrt(len(d))),
        }

OUT.mkdir(exist_ok=True)
with open(OUT / "patterns_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

# ---------------------------------------------------------------- chart: win rate vs random, train and test
fig, ax = plt.subplots(figsize=(9, 4.2))
names = list(signals)
y = np.arange(len(names))[::-1]
for i, n in enumerate(names):
    for period, c, off in (("train", BLUE, 0.15), ("test", ORANGE, -0.15)):
        e = results[n][period]["edge_win_rate"] * 100
        ax.barh(y[i] + off, e, 0.28, color=c, label=f"{period} {'2014-20' if period == 'train' else '2021-26'}" if i == 0 else None)
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_yticks(y, [f"{n.replace('_', ' ')} ({results[n]['direction']})" for n in names], fontsize=8)
ax.set_xlabel("win rate minus random entries in the same direction (percentage points)", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax, "Do chart patterns win more often than random trades? (2:1 target, 1-unit stop)")
fig.tight_layout(); fig.savefig(OUT / "patterns.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- console
for side in (1, -1):
    r = stats(random_trades[side])
    print(f"random {'buys ' if side == 1 else 'sells'}: win {r['win_rate']:.1%}, avg {r['avg_units']:+.3f} units/trade")
print(f"\n{'pattern':18s} {'dir':4s} {'trades':>7s} | {'win% train':>10s} {'vs rand':>7s} | {'win% test':>9s} {'vs rand':>7s} "
      f"| {'units/trade test':>16s} {'edge':>7s} {'t':>5s}")
for n, v in results.items():
    tr, te = v["train"], v["test"]
    print(f"{n:18s} {v['direction']:4s} {v['all']['pattern']['trades']:7d} | {tr['pattern']['win_rate']:10.1%} "
          f"{tr['edge_win_rate'] * 100:+6.1f}p | {te['pattern']['win_rate']:9.1%} {te['edge_win_rate'] * 100:+6.1f}p | "
          f"{te['pattern']['avg_units']:+16.3f} {te['edge_units']:+7.3f} {te['edge_t_monthly']:5.1f}")
