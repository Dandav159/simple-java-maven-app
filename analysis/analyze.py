"""Stable-vs-volatile stock relationship study.

Every candidate "pattern" is discovered on the TRAIN window and then evaluated,
unchanged, on the TEST window. Anything that only works in-sample is overfit.

Outputs: output/results.json and output/*.png
"""
import json
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import coint, grangercausalitytests

from universe import STABLE_CANDIDATES, VOLATILE_CANDIDATES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "output"
TRAIN = ("2014-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-12-31")
N_BUCKET = 15            # names per bucket
COST_BPS = 5             # one-way cost per unit of turnover
ANN = 252
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
results = {}


# ---------------------------------------------------------------- helpers
def sharpe(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ANN)) if r.std() > 0 else float("nan")


def summarize(r):
    r = r.dropna()
    eq = (1 + r).cumprod()
    return {
        "ann_return": float((eq.iloc[-1]) ** (ANN / len(r)) - 1),
        "ann_vol": float(r.std() * np.sqrt(ANN)),
        "sharpe": sharpe(r),
        "max_drawdown": float((eq / eq.cummax() - 1).min()),
        "t_stat": float(r.mean() / r.std() * np.sqrt(len(r))),
        "days": len(r),
    }


def with_costs(position, asset_ret):
    """Daily P&L of holding `position` (decided at close t-1) in `asset_ret`."""
    pos = position.shift(1).fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return pos * asset_ret - turnover * COST_BPS / 1e4


def split(x):
    return x.loc[TRAIN[0]:TRAIN[1]], x.loc[TEST[0]:TEST[1]]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.6)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- data
prices = pd.read_csv(HERE / "data" / "prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()]          # trading days only
vix = prices.pop("^VIX").ffill()
rets = prices.pct_change(fill_method=None)
rets = rets.where(rets.abs() < 1.0)                  # drop bad ticks (>100% day)

# ---------------------------------------------------------------- 1. buckets
tr_rets, _ = split(rets)
pool = [t for t in STABLE_CANDIDATES + VOLATILE_CANDIDATES if tr_rets[t].count() >= 750]
train_vol = (tr_rets[pool].std() * np.sqrt(ANN)).sort_values()
stable = list(train_vol.index[:N_BUCKET])
volatile = list(train_vol.index[-N_BUCKET:])
results["buckets"] = {
    "stable": {t: round(float(train_vol[t]), 3) for t in stable},
    "volatile": {t: round(float(train_vol[t]), 3) for t in volatile},
}
S = rets[stable].mean(axis=1)
V = rets[volatile].mean(axis=1)
mkt = rets["SPY"]
baskets = pd.DataFrame({"stable": S, "volatile": V, "spy": mkt}).dropna()

# ---------------------------------------------------------------- 2. contemporaneous correlation
corr = rets[stable + volatile].corr()
ss = corr.loc[stable, stable].values[np.triu_indices(N_BUCKET, 1)].mean()
vv = corr.loc[volatile, volatile].values[np.triu_indices(N_BUCKET, 1)].mean()
sv = corr.loc[stable, volatile].values.mean()
roll = baskets["stable"].rolling(63).corr(baskets["volatile"])

v_al = vix.reindex(baskets.index)
regimes = {
    "calm (VIX<15)": v_al < 15,
    "normal (15-25)": (v_al >= 15) & (v_al < 25),
    "stressed (VIX>=25)": v_al >= 25,
}
spy_q = baskets["spy"].quantile([0.1, 0.9])
regime_corr = {k: float(baskets.loc[m, ["stable", "volatile"]].corr().iloc[0, 1]) for k, m in regimes.items()}
regime_corr["SPY worst 10% days"] = float(baskets.loc[baskets.spy <= spy_q[0.1], ["stable", "volatile"]].corr().iloc[0, 1])
regime_corr["SPY best 10% days"] = float(baskets.loc[baskets.spy >= spy_q[0.9], ["stable", "volatile"]].corr().iloc[0, 1])

beta = {
    k: float(np.cov(baskets[k], baskets.spy)[0, 1] / baskets.spy.var()) for k in ("stable", "volatile")
}
results["correlation"] = {
    "avg_pairwise_stable_stable": float(ss),
    "avg_pairwise_volatile_volatile": float(vv),
    "avg_pairwise_stable_volatile": float(sv),
    "basket_corr_full": float(baskets[["stable", "volatile"]].corr().iloc[0, 1]),
    "rolling_63d_min": float(roll.min()), "rolling_63d_max": float(roll.max()),
    "by_regime": regime_corr,
    "beta_to_spy": beta,
}

# ---------------------------------------------------------------- 3. lead-lag (basket level)
tr_b, te_b = split(baskets)
lags = range(-5, 6)
xcorr = {}
for name, df in (("train", tr_b), ("test", te_b)):
    # positive lag k: stable(t-k) vs volatile(t)  => stable leads
    xcorr[name] = {k: float(df["volatile"].corr(df["stable"].shift(k))) for k in lags}
granger = {}
for cause, effect in (("stable", "volatile"), ("volatile", "stable")):
    g = grangercausalitytests(tr_b[[effect, cause]], maxlag=5)
    granger[f"{cause}->{effect}"] = {int(lag): float(r[0]["ssr_ftest"][1]) for lag, r in g.items()}
# Rank correlation is robust to a handful of crash days dominating the Pearson number
rank_lag1 = {name: float(df["volatile"].corr(df["stable"].shift(1), method="spearman"))
             for name, df in (("train", tr_b), ("test", te_b))}
ex_covid = tr_b.drop(tr_b.loc["2020-02-15":"2020-05-15"].index)
results["lead_lag"] = {
    "cross_corr": xcorr,
    "lag1_spearman": rank_lag1,
    "lag1_pearson_train_ex_covid_crash": float(ex_covid["volatile"].corr(ex_covid["stable"].shift(1))),
    "granger_pvalues_train": granger,
}

# ---------------------------------------------------------------- 4. lead-lag (all 225 pairs), train -> test replication
rows = []
tr_r, te_r = split(rets)
for s, v in product(stable, volatile):
    for lead, lag_ in ((s, v), (v, s)):
        a, b = tr_r[lead].shift(1), tr_r[lag_]
        m = a.notna() & b.notna()
        r, p = stats.pearsonr(a[m], b[m])
        a2, b2 = te_r[lead].shift(1), te_r[lag_]
        m2 = a2.notna() & b2.notna()
        r2 = float(np.corrcoef(a2[m2], b2[m2])[0, 1])
        rows.append({"leader": lead, "follower": lag_, "train_r": r, "train_p": p, "test_r": r2})
pairs = pd.DataFrame(rows)
pairs["train_p_bh"] = multipletests(pairs.train_p, method="fdr_bh")[1]
sig = pairs[pairs.train_p_bh < 0.05]
same_sign = int((np.sign(sig.train_r) == np.sign(sig.test_r)).sum()) if len(sig) else 0
results["pair_lead_lag"] = {
    "tests": len(pairs),
    "raw_p_lt_0.05": int((pairs.train_p < 0.05).sum()),
    "significant_after_BH": len(sig),
    "of_which_same_sign_in_test": same_sign,
    "train_test_r_correlation": float(pairs.train_r.corr(pairs.test_r)),
    "top_train": sig.sort_values("train_p").head(10).round(4).to_dict("records"),
}

# ---------------------------------------------------------------- 5. strategies (parameters chosen on train only)
strategies = {}

# 5a. Lead-lag timing: trade volatile basket off yesterday's stable-basket move,
#     direction (follow vs fade) taken from the sign of the train lag-1 correlation
direction = np.sign(xcorr["train"][1])
pos = direction * np.sign(baskets["stable"])
strategies["leadlag_stable_signals_volatile"] = with_costs(pos, baskets["volatile"])

# 5b. Rotation: momentum vs reversal on the stable-minus-volatile spread, lookback picked on train
spread = baskets["stable"] - baskets["volatile"]
best = None
grid = {}
for lb in (5, 10, 21, 63, 126, 252):
    for sign_, label in ((1, "momentum"), (-1, "reversal")):
        p = sign_ * np.sign(spread.rolling(lb).sum())
        sr = sharpe(split(with_costs(p, spread))[0])
        grid[f"{label}_{lb}d"] = sr
        if best is None or sr > best[0]:
            best = (sr, lb, sign_, label)
_, lb, sign_, label = best
strategies[f"rotation_{label}_{lb}d (train-best)"] = with_costs(sign_ * np.sign(spread.rolling(lb).sum()), spread)
results["rotation_grid_train_sharpe"] = grid

# 5c. VIX regime switch: hold volatile basket when VIX < train median, else stable
vmed = split(vix)[0].median()
in_vol = (vix.reindex(baskets.index) < vmed).astype(float)
switch = with_costs(in_vol, baskets["volatile"]) + with_costs(1 - in_vol, baskets["stable"])
strategies[f"vix_switch (VIX<{vmed:.1f} -> volatile)"] = switch

# 5d. Beta-neutral low-minus-high ("betting against beta"), betas from trailing 252d
b_s = baskets.stable.rolling(252).cov(baskets.spy) / baskets.spy.rolling(252).var()
b_v = baskets.volatile.rolling(252).cov(baskets.spy) / baskets.spy.rolling(252).var()
w_s, w_v = 1 / b_s.clip(lower=0.33), 1 / b_v.clip(lower=0.33)   # cap leverage at 3x
bab = with_costs(w_s, baskets.stable) - with_costs(w_v, baskets.volatile)
strategies["beta_neutral_long_stable_short_volatile"] = bab

# 5e. Cointegration pairs trading: pairs found on train, traded on test
logp = np.log(prices)
tr_lp = logp.loc[TRAIN[0]:TRAIN[1]]
coint_rows = []
for s, v in product(stable, volatile):
    d = tr_lp[[s, v]].dropna()
    if len(d) < 750:
        continue
    p = coint(d[s], d[v])[1]
    coint_rows.append((s, v, p))
cdf = pd.DataFrame(coint_rows, columns=["stable", "volatile", "p"])
cdf["p_bh"] = multipletests(cdf.p, method="fdr_bh")[1]
cdf = cdf.sort_values("p")
pair_pnls = []
for _, row in cdf.head(5).iterrows():
    s, v = row.stable, row.volatile
    d = tr_lp[[s, v]].dropna()
    hedge = np.polyfit(d[v], d[s], 1)[0]
    sp = logp[s] - hedge * logp[v]
    mu, sd = sp.loc[TRAIN[0]:TRAIN[1]].mean(), sp.loc[TRAIN[0]:TRAIN[1]].std()
    z = (sp - mu) / sd
    p = pd.Series(np.nan, index=z.index)
    p[z > 2], p[z < -2], p[z.abs() < 0.5] = -1.0, 1.0, 0.0
    p = p.ffill().fillna(0).clip(-1, 1)
    leg = (rets[s] - hedge * rets[v]) / (1 + abs(hedge))
    pair_pnls.append(with_costs(p, leg))
strategies["coint_pairs_top5"] = pd.concat(pair_pnls, axis=1).mean(axis=1)
results["cointegration"] = {
    "pairs_tested": len(cdf),
    "raw_p_lt_0.05": int((cdf.p < 0.05).sum()),
    "significant_after_BH": int((cdf.p_bh < 0.05).sum()),
    "top5": cdf.head(5).round(4).to_dict("records"),
}

# Benchmarks
strategies["benchmark_SPY"] = baskets.spy
strategies["benchmark_stable_basket"] = baskets.stable
strategies["benchmark_volatile_basket"] = baskets.volatile

results["strategies"] = {
    k: {"train": summarize(split(v)[0]), "test": summarize(split(v)[1])} for k, v in strategies.items()
}

# ---------------------------------------------------------------- charts
OUT.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(roll.index, roll, color=BLUE, lw=1.5)
ax.axhline(results["correlation"]["basket_corr_full"], color=GRAY, lw=1, ls="--")
ax.set_ylim(-0.2, 1)
style(ax, "Rolling 63-day correlation: stable basket vs volatile basket")
fig.tight_layout(); fig.savefig(OUT / "rolling_correlation.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.6))
x = np.array(list(lags))
ax.bar(x - 0.18, [xcorr["train"][k] for k in lags], 0.34, color=BLUE, label="train 2014-20")
ax.bar(x + 0.18, [xcorr["test"][k] for k in lags], 0.34, color=ORANGE, label="test 2021-26")
n = len(te_b)
ax.axhspan(-2 / np.sqrt(n), 2 / np.sqrt(n), color="#e8e7e2", zorder=0)
ax.set_xticks(x)
ax.set_xlabel("lag k (days): corr(stable[t-k], volatile[t]); k>0 = stable leads", fontsize=8, color="#52514e")
ax.legend(frameon=False, fontsize=8)
style(ax, "Cross-correlation of basket returns (gray band = noise, ±2/√n)")
fig.tight_layout(); fig.savefig(OUT / "lead_lag.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.scatter(pairs.train_r, pairs.test_r, s=10, color=BLUE, alpha=0.5)
ax.scatter(sig.train_r, sig.test_r, s=18, color=ORANGE, label="BH-significant in train")
lim = max(pairs[["train_r", "test_r"]].abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=0.8, ls="--")
ax.axhline(0, color="#c3c2b7", lw=0.6); ax.axvline(0, color="#c3c2b7", lw=0.6)
ax.set_xlabel("train lag-1 correlation", fontsize=8); ax.set_ylabel("test lag-1 correlation", fontsize=8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Do pair lead-lag effects replicate?")
fig.tight_layout(); fig.savefig(OUT / "pair_replication.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.2))
show = [k for k in strategies if not k.startswith("benchmark")] + ["benchmark_SPY"]
pal = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
for k, c in zip(show, pal):
    r = split(strategies[k])[1].dropna()
    eq = (1 + r).cumprod()
    ax.plot(eq.index, eq, lw=1.5, color=c, label=k)
ax.set_yscale("log")
ax.legend(frameon=False, fontsize=7, loc="upper left")
style(ax, "Out-of-sample growth of $1 (2021-2026, after 5 bps costs)")
fig.tight_layout(); fig.savefig(OUT / "oos_equity.png", dpi=150); plt.close(fig)

with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
pairs.to_csv(OUT / "pair_lead_lag.csv", index=False)
cdf.to_csv(OUT / "cointegration.csv", index=False)

# ---------------------------------------------------------------- console summary
print(json.dumps({k: results[k] for k in ("buckets", "correlation", "lead_lag", "pair_lead_lag", "cointegration", "rotation_grid_train_sharpe")}, indent=1, default=str))
print(f"\n{'strategy':55s} {'train SR':>9s} {'test SR':>8s} {'test ann':>9s} {'test MDD':>9s} {'test t':>7s}")
for k, v in results["strategies"].items():
    print(f"{k:55s} {v['train']['sharpe']:9.2f} {v['test']['sharpe']:8.2f} "
          f"{v['test']['ann_return']:9.1%} {v['test']['max_drawdown']:9.1%} {v['test']['t_stat']:7.2f}")
