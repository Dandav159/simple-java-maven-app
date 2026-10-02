"""Economic structure as a prior: cross-sectional stock selection on S&P 500 members.

Instead of letting a model search for patterns, the signals and their directions
come from published research, decided before looking at this data:

  factor        sign  economic story                                   source
  momentum_12_1  +    investors under-react to news                    Jegadeesh & Titman (1993)
  reversal_1m    -    short-term liquidity shocks revert               Jegadeesh (1990), Lehmann (1990)
  low_vol        +    leverage-constrained investors overpay for       Frazzini & Pedersen (2014),
                      risky stocks                                     Ang et al. (2006)
  near_52w_high  +    anchoring to the 52-week high delays            George & Hwang (2004)
                      reaction to good news

Three ways to set factor weights are compared:
  prior    - literature signs, equal weights, nothing fit to data (1/N, DeMiguel et al. 2009)
  data     - Fama-MacBeth slopes estimated on TRAIN only (signs and sizes from data)
  bayes    - prior premium shrunk toward the TRAIN estimate by precision weighting

Plus a time-series prior: volatility clusters, so scaling SPY exposure by inverse
recent volatility should improve risk-adjusted return (Moreira & Muir 2017).

Outputs: output/factors_results.json and output/factors_*.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common import ANN, BLUE, GRAY, ORANGE, OUT, TEST, TRAIN, style, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
COST_BPS = 10            # one-way, per unit of turnover (monthly rebalance)
QUANTILE = 0.2           # top / bottom quintile
GREEN = "#1baf7a"

# Prior for each factor's monthly return per 1 cross-sectional z-score.
# Published long-short spreads are ~0.6-1%/month across ~3 z-units (~0.3%/z);
# McLean & Pontiff (2016) find ~half of that survives publication -> 0.15%/z.
# Prior sd equals the mean, so "no effect" sits one sd away.
PRIOR_SIGN = {"momentum_12_1": 1, "reversal_1m": -1, "low_vol": 1, "near_52w_high": 1}
PRIOR_MEAN, PRIOR_SD = 0.0015, 0.0015
FACTORS = list(PRIOR_SIGN)

# ---------------------------------------------------------------- data
prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()]
prices.pop("^VIX")
spy = prices.pop("SPY")
members = pd.read_csv(HERE / "data" / "sp500_members.csv", parse_dates=["date_added"]).set_index("ticker")
sector = members["sector"]

daily = prices.pct_change(fill_method=None)
daily = daily.where(daily.abs() < 1.0)
month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index

# ---------------------------------------------------------------- signals at each month end
def zscore_by_sector(x):
    """Rank-based z-score within sector: removes sector bets, robust to outliers."""
    df = pd.DataFrame({"x": x, "sector": sector.reindex(x.index)}).dropna()
    r = df.groupby("sector")["x"].rank(pct=True)
    z = (r - r.groupby(df["sector"]).transform("mean")) / r.groupby(df["sector"]).transform("std")
    return z


panel = []
for i, d in enumerate(month_ends[:-1]):
    loc = prices.index.get_loc(d)
    if loc < 253:
        continue
    nxt = month_ends[i + 1]
    eligible = members.index[(members.date_added.isna()) | (members.date_added <= d)]
    p = prices.loc[:d, prices.columns.intersection(eligible)]
    hist = p.iloc[-253:]
    ok = hist.notna().all()
    hist = hist.loc[:, ok]
    raw = pd.DataFrame({
        "momentum_12_1": hist.iloc[-22] / hist.iloc[0] - 1,
        "reversal_1m": hist.iloc[-1] / hist.iloc[-22] - 1,
        "low_vol": -daily.loc[hist.index[1]:d, hist.columns].std(),
        "near_52w_high": hist.iloc[-1] / hist.max(),
    })
    z = raw.apply(zscore_by_sector)
    fwd = prices.loc[nxt, z.index] / prices.loc[d, z.index] - 1
    z["fwd"] = fwd
    z["date"] = d
    panel.append(z.dropna())
panel = pd.concat(panel).rename_axis("ticker").reset_index()
dates = panel.date.unique()
is_train = lambda d: pd.Timestamp(TRAIN[0]) <= d <= pd.Timestamp(TRAIN[1])  # noqa: E731
train_panel = panel[panel.date.map(is_train)]

# ---------------------------------------------------------------- 1. per-factor evidence (rank IC)
def ic_stats(df):
    ic = df.groupby("date").apply(lambda g: g[FACTORS].corrwith(g.fwd, method="spearman"))
    return pd.DataFrame({
        "mean_ic": ic.mean(),
        "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
        "hit_rate": (ic > 0).mean(),
    })


test_panel = panel[panel.date >= pd.Timestamp(TEST[0])]
ic = {"train": ic_stats(train_panel), "test": ic_stats(test_panel), "full": ic_stats(panel)}

# ---------------------------------------------------------------- 2. factor weights
def fama_macbeth(df):
    slopes = []
    for _, g in df.groupby("date"):
        X = np.column_stack([np.ones(len(g)), g[FACTORS].values])
        b = np.linalg.lstsq(X, g.fwd.values, rcond=None)[0][1:]
        slopes.append(b)
    slopes = pd.DataFrame(slopes, columns=FACTORS)
    return slopes.mean(), slopes.std() / np.sqrt(len(slopes))


b_hat, se = fama_macbeth(train_panel)
prior_mu = pd.Series({f: PRIOR_SIGN[f] * PRIOR_MEAN for f in FACTORS})
post_prec = 1 / PRIOR_SD**2 + 1 / se**2
b_post = (prior_mu / PRIOR_SD**2 + b_hat / se**2) / post_prec
weights = {
    "prior": prior_mu / prior_mu.abs().sum(),
    "data": b_hat / b_hat.abs().sum(),
    "bayes": b_post / b_post.abs().sum(),
}

# ---------------------------------------------------------------- 3. portfolios
def backtest(w):
    """Monthly long-short quintile portfolio and long-only top-quintile vs universe."""
    ls, lo, prev_long, prev_short = {}, {}, set(), set()
    for d, g in panel.groupby("date"):
        score = g[FACTORS].values @ w.values
        q_hi, q_lo = np.quantile(score, 1 - QUANTILE), np.quantile(score, QUANTILE)
        longs, shorts = set(g.ticker[score >= q_hi]), set(g.ticker[score <= q_lo])
        r_long = g.fwd[score >= q_hi].mean()
        r_short = g.fwd[score <= q_lo].mean()
        turn_l = 1 - len(longs & prev_long) / len(longs)
        turn_s = 1 - len(shorts & prev_short) / len(shorts)
        cost = COST_BPS / 1e4
        ls[d] = r_long - r_short - 2 * cost * (turn_l + turn_s)
        lo[d] = r_long - g.fwd.mean() - 2 * cost * turn_l          # active return vs equal-weight universe
        prev_long, prev_short = longs, shorts
    return pd.Series(ls), pd.Series(lo)


def msummary(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {
        "ann_return": float(eq.iloc[-1] ** (12 / len(r)) - 1),
        "sharpe": float(r.mean() / r.std() * np.sqrt(12)),
        "t_stat": float(r.mean() / r.std() * np.sqrt(len(r))),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "months": len(r),
    }


def msplit(r):
    return r[[is_train(d) for d in r.index]], r[r.index >= pd.Timestamp(TEST[0])]


port = {}
series = {}
for name, w in list(weights.items()) + [(f"single:{f}", pd.Series({g: float(g == f) * PRIOR_SIGN[f] for g in FACTORS})) for f in FACTORS]:
    ls, lo = backtest(w)
    series[name] = ls
    port[name] = {
        "long_short": {"train": msummary(msplit(ls)[0]), "test": msummary(msplit(ls)[1])},
        "long_only_active": {"train": msummary(msplit(lo)[0]), "test": msummary(msplit(lo)[1])},
    }

# ---------------------------------------------------------------- 4. volatility-managed SPY
spy_r = spy.pct_change().dropna()
rv = spy_r.rolling(21).std() * np.sqrt(ANN)
target = float(rv.loc[TRAIN[0]:TRAIN[1]].median())
vm = {}
for cap in (1.0, 2.0):
    w = (target / rv).clip(upper=cap).shift(1)
    turnover = w.diff().abs()
    vm[f"cap_{cap:g}x"] = (w * spy_r - turnover * 2 / 1e4).dropna()
vm["spy"] = spy_r
vol_managed = {k: {"train": summarize(v.loc[TRAIN[0]:TRAIN[1]]), "test": summarize(v.loc[TEST[0]:TEST[1]])} for k, v in vm.items()}

# ---------------------------------------------------------------- output
results = {
    "universe": {"stocks": int(panel.ticker.nunique()), "months": len(dates),
                 "avg_stocks_per_month": float(panel.groupby("date").size().mean())},
    "rank_ic": {k: v.round(4).to_dict("index") for k, v in ic.items()},
    "fama_macbeth_train": {"slope": b_hat.to_dict(), "se": se.to_dict(), "t": (b_hat / se).to_dict()},
    "posterior_slope": b_post.to_dict(),
    "weights": {k: v.round(3).to_dict() for k, v in weights.items()},
    "portfolios": port,
    "vol_managed_spy": vol_managed,
    "vol_target": target,
}
OUT.mkdir(exist_ok=True)
with open(OUT / "factors_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(FACTORS))
signed = lambda s: [s[f] * PRIOR_SIGN[f] for f in FACTORS]  # noqa: E731
ax.bar(x - 0.2, signed(ic["train"].mean_ic), 0.38, color=BLUE, label="train 2015-20")
ax.bar(x + 0.2, signed(ic["test"].mean_ic), 0.38, color=ORANGE, label="test 2021-26")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xticks(x, [f.replace("_", " ") for f in FACTORS], fontsize=9)
ax.set_ylabel("mean monthly rank IC\n(+ = works in prior's direction)", fontsize=8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Does each factor work in the direction economics predicts?")
fig.tight_layout(); fig.savefig(OUT / "factors_ic.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
for name, c in (("prior", BLUE), ("data", ORANGE), ("bayes", GREEN)):
    r = msplit(series[name])[1]
    ax.plot(r.index, (1 + r).cumprod(), color=c, lw=1.6, label=f"{name} weights")
ax.axhline(1, color=GRAY, lw=0.8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Out-of-sample long-short composite, growth of $1 (2021-2026, after costs)")
fig.tight_layout(); fig.savefig(OUT / "factors_oos.png", dpi=150); plt.close(fig)

print(json.dumps({k: results[k] for k in ("universe", "rank_ic", "fama_macbeth_train", "weights")}, indent=1, default=str))
print(f"\n{'portfolio':28s} {'LS train SR':>11s} {'LS test SR':>10s} {'LS test t':>9s} {'LS test ann':>11s} {'LO active test SR':>17s} {'LO test ann':>11s}")
for k, v in port.items():
    a, b = v["long_short"], v["long_only_active"]
    print(f"{k:28s} {a['train']['sharpe']:11.2f} {a['test']['sharpe']:10.2f} {a['test']['t_stat']:9.2f} "
          f"{a['test']['ann_return']:11.1%} {b['test']['sharpe']:17.2f} {b['test']['ann_return']:11.1%}")
print("\nvol-managed SPY:")
for k, v in vol_managed.items():
    print(f"  {k:10s} train SR {v['train']['sharpe']:.2f}  test SR {v['test']['sharpe']:.2f}  "
          f"test ann {v['test']['ann_return']:.1%}  test MDD {v['test']['max_drawdown']:.1%}")
