"""Market structure as a point on a Grassmannian.

The top-k principal directions of ~450 stocks' standardized daily returns span a
k-dimensional subspace: a point on the Grassmannian Gr(k, n). Two questions:

A. RISK  - Does the speed at which that subspace rotates (geodesic distance
           between consecutive quarters) predict next month's market volatility
           beyond what current volatility already predicts?
B. PROFIT - Project out the shared subspace and trade the stock-specific
           residuals for reversal (Avellaneda & Lee 2010 style stat-arb).
           Control: the same strategy on raw returns, with no geometry.

Train 2014-2020, test 2021-2026, as everywhere else.
Outputs: output/grassmann_results.json, output/grassmann_*.png
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
K = 5                  # dimension of the shared subspace
WIN = 63               # one quarter of trading days per subspace estimate
FIT = 252              # days used to estimate loadings for the residual strategy
HOLD = 5               # rebalance weekly
DECILE = 0.10
COST_BPS = 5

prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns="^VIX")
rets = prices.pct_change(fill_method=None)
rets = rets.where(rets.abs() < 1.0).iloc[1:]
spy = rets.pop("SPY")
dates = rets.index


def subspace(block, k=K):
    """Orthonormal basis (stocks x k) of the top-k principal directions."""
    z = (block - block.mean()) / block.std()
    _, _, vt = np.linalg.svd(z.values, full_matrices=False)
    return vt[:k].T


def grassmann_distance(u1, u2):
    """Geodesic distance on Gr(k, n): root-sum-square of the principal angles."""
    s = np.clip(np.linalg.svd(u1.T @ u2, compute_uv=False), -1, 1)
    return float(np.sqrt((np.arccos(s) ** 2).sum()))


# ---------------------------------------------------------------- A. rotation speed vs future volatility
rows = []
month_ends = rets.groupby(dates.to_period("M")).tail(1).index
for d in month_ends:
    i = dates.get_loc(d)
    if i < 2 * WIN or i + 21 >= len(dates):
        continue
    now, before = rets.iloc[i - WIN + 1:i + 1], rets.iloc[i - 2 * WIN + 1:i - WIN + 1]
    cols = now.columns[now.notna().all() & before.notna().all()]
    u_now, u_before = subspace(now[cols]), subspace(before[cols])
    z = (now[cols] - now[cols].mean()) / now[cols].std()
    eig = np.linalg.svd(z.values, compute_uv=False) ** 2
    rows.append({
        "date": d,
        "rotation": grassmann_distance(u_now, u_before),
        "absorption": eig[:K].sum() / eig.sum(),        # share of variance in the subspace (Kritzman et al. 2011)
        "vol_now": spy.iloc[i - 20:i + 1].std() * np.sqrt(ANN),
        "vol_next": spy.iloc[i + 1:i + 22].std() * np.sqrt(ANN),
        "ret_next": (1 + spy.iloc[i + 1:i + 22]).prod() - 1,
    })
geo = pd.DataFrame(rows).set_index("date")
geo["log_vol_now"], geo["log_vol_next"] = np.log(geo.vol_now), np.log(geo.vol_next)


def oos_r2(cols):
    """Fit log future vol on train months, report R^2 on test months."""
    tr, te = geo.loc[TRAIN[0]:TRAIN[1]], geo.loc[TEST[0]:TEST[1]]
    X = lambda df: np.column_stack([np.ones(len(df))] + [df[c] for c in cols])  # noqa: E731
    b = np.linalg.lstsq(X(tr), tr.log_vol_next, rcond=None)[0]
    pred = X(te) @ b
    return float(1 - ((te.log_vol_next - pred) ** 2).sum() / ((te.log_vol_next - te.log_vol_next.mean()) ** 2).sum()), b.tolist()


risk = {
    "months": len(geo),
    "corr_rotation_vs_next_vol": float(geo.rotation.corr(geo.log_vol_next)),
    "corr_rotation_vs_next_return": float(geo.rotation.corr(geo.ret_next)),
    "corr_rotation_vs_current_vol": float(geo.rotation.corr(geo.log_vol_now)),
}
for name, cols in {"current_vol_only": ["log_vol_now"],
                   "plus_rotation": ["log_vol_now", "rotation"],
                   "plus_absorption": ["log_vol_now", "absorption"],
                   "plus_both": ["log_vol_now", "rotation", "absorption"]}.items():
    r2, b = oos_r2(cols)
    risk[f"test_r2_{name}"] = r2
    risk[f"coef_{name}"] = b

# ---------------------------------------------------------------- B. residual reversal stat-arb
def backtest(use_geometry):
    pnl, gross_l, idx_out = [], [], []
    prev_l, prev_s = set(), set()
    for i in range(FIT, len(dates) - HOLD, HOLD):
        hist = rets.iloc[i - FIT:i]
        cols = hist.columns[hist.notna().all() & rets.iloc[i:i + HOLD].notna().all()]
        recent = hist[cols].iloc[-HOLD:]
        if use_geometry:
            mu, sd = hist[cols].mean(), hist[cols].std()
            u = subspace(hist[cols])
            z = ((recent - mu) / sd).values
            resid = z - (z @ u) @ u.T                     # remove the shared directions
            signal = -resid.sum(0)                        # bet residuals revert
        else:
            signal = -((1 + recent).prod() - 1).values    # plain 5-day reversal
        n = int(len(cols) * DECILE)
        order = np.argsort(signal)
        longs, shorts = set(cols[order[-n:]]), set(cols[order[:n]])
        fwd = (1 + rets.iloc[i:i + HOLD][cols]).prod() - 1
        g = fwd[list(longs)].mean() - fwd[list(shorts)].mean()
        turn = (1 - len(longs & prev_l) / n) + (1 - len(shorts & prev_s) / n)
        pnl.append(g - 2 * turn * COST_BPS / 1e4)
        gross_l.append(g)
        idx_out.append(dates[i])
        prev_l, prev_s = longs, shorts
    return pd.Series(pnl, index=idx_out), pd.Series(gross_l, index=idx_out)


def wsummary(r):
    eq = (1 + r).cumprod()
    per_year = ANN / HOLD
    return {"ann_return": float(eq.iloc[-1] ** (per_year / len(r)) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(per_year)),
            "t_stat": float(r.mean() / r.std() * np.sqrt(len(r))),
            "max_drawdown": float((eq / eq.cummax() - 1).min())}


stat_arb, curves = {}, {}
for name, geom in (("grassmann_residual_reversal", True), ("plain_reversal_control", False)):
    net, gross = backtest(geom)
    curves[name] = net
    stat_arb[name] = {
        period: {"before_costs": wsummary(gross.loc[a:b]), "after_costs": wsummary(net.loc[a:b])}
        for period, (a, b) in (("train", TRAIN), ("test", TEST))
    }
diff = curves["grassmann_residual_reversal"] - curves["plain_reversal_control"]
stat_arb["geometry_minus_control_test"] = wsummary(diff.loc[TEST[0]:TEST[1]])

# ---------------------------------------------------------------- output
results = {"risk": risk, "stat_arb": stat_arb, "spy_test": summarize(spy.loc[TEST[0]:TEST[1]])}
OUT.mkdir(exist_ok=True)
with open(OUT / "grassmann_results.json", "w") as f:
    json.dump(results, f, indent=2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax1.plot(geo.index, geo.rotation, color=BLUE, lw=1.5)
style(ax1, "How far the market's top-5 subspace rotated vs the prior quarter (radians)")
ax2.plot(geo.index, geo.vol_now, color=ORANGE, lw=1.5)
style(ax2, "SPY volatility, trailing month (annualized)")
fig.tight_layout(); fig.savefig(OUT / "grassmann_rotation.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
for name, c in (("grassmann_residual_reversal", BLUE), ("plain_reversal_control", ORANGE)):
    r = curves[name].loc[TEST[0]:TEST[1]]
    ax.plot(r.index, (1 + r).cumprod(), color=c, lw=1.5, label=name.replace("_", " "))
ax.axhline(1, color=GRAY, lw=0.8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Weekly reversal, growth of $1 after costs (2021-2026)")
fig.tight_layout(); fig.savefig(OUT / "grassmann_statarb.png", dpi=150); plt.close(fig)

print(json.dumps(risk, indent=1))
for name in ("grassmann_residual_reversal", "plain_reversal_control"):
    for period in ("train", "test"):
        b, a = stat_arb[name][period]["before_costs"], stat_arb[name][period]["after_costs"]
        print(f"{name:30s} {period}: before costs {b['ann_return']:+.1%} (SR {b['sharpe']:.2f}, t {b['t_stat']:.2f}) | "
              f"after {a['ann_return']:+.1%} (SR {a['sharpe']:.2f}, MDD {a['max_drawdown']:.0%})")
d = stat_arb["geometry_minus_control_test"]
print(f"geometry minus control (test): {d['ann_return']:+.1%}/yr, t {d['t_stat']:.2f}")
