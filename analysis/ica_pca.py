"""Is some *blend* of stocks predictable, even if single stocks are not?

Three ways to rotate the space of ~450 stock returns into portfolios, all fit
on TRAIN (2014-2020) only:

  PCA       - blends that explain the most variance
  ICA       - blends that are statistically independent (FastICA)
  Box-Tiao  - blends whose next-day return is most predictable from today's
              (canonical predictability, Box & Tiao 1977): maximize
              var(w' A x_{t-1}) / var(w' x_t) under a VAR(1) fit

To keep the fit from memorizing noise, every method works inside the top
M principal components. Each blend becomes a portfolio; it is traded on TEST
(2021-2026) by forecasting tomorrow's blend return from today's with the
train autocorrelation, going long or short the blend, after costs.

Outputs: output/ica_pca_results.json, output/ica_pca.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA

from common import BLUE, GRAY, ORANGE, OUT, TEST, TRAIN, style, summarize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
M = 20                 # dimensions kept from PCA; all methods rotate within these
TOP = 3                # blends per method chosen on train by predictability
COST_BPS = 5
CLIP = 4.0             # cap standardized moves at 4 sd when FITTING, so a few crash days can't dominate
GREEN = "#1baf7a"

prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns=["^VIX", "SPY"])
rets = prices.pct_change(fill_method=None).iloc[1:]
rets = rets.loc[:, rets.notna().all()]                       # stocks with the full history
tr, te = rets.loc[TRAIN[0]:TRAIN[1]], rets.loc[TEST[0]:TEST[1]]
mu, sd = tr.mean(), tr.std()
Z = (rets - mu) / sd                                         # standardized with train stats only
Ztr = Z.loc[TRAIN[0]:TRAIN[1]].clip(-CLIP, CLIP).values

# ---------------------------------------------------------------- the three rotations (stocks x components)
_, _, vt = np.linalg.svd(Ztr, full_matrices=False)
P = vt[:M].T                                                 # top-M principal directions
X = Ztr @ P                                                  # train data in PC coordinates

W = {"PCA": P}

ica = FastICA(n_components=M, whiten="unit-variance", random_state=0, max_iter=2000)
ica.fit(X)
W["ICA"] = P @ ica.components_.T                             # unmixing, mapped back to stocks

X0, X1 = X[:-1], X[1:]
A = np.linalg.lstsq(X0, X1, rcond=None)[0].T                 # VAR(1): x_t = A x_{t-1}
S = np.cov(X.T)
Sp = A @ np.cov(X0.T) @ A.T                                  # covariance of the predictable part
L = np.linalg.cholesky(S)
Li = np.linalg.inv(L)
evals, evecs = np.linalg.eigh(Li @ Sp @ Li.T)                # generalized eigenproblem Sp v = l S v
order = np.argsort(evals)[::-1]
W["Box-Tiao"] = P @ (Li.T @ evecs[:, order])
bt_train_r2 = evals[order]


def ar1(y):
    return float(np.corrcoef(y[:-1], y[1:])[0, 1])


# ---------------------------------------------------------------- evaluate every blend
rows, pnl_by_method, gross_by_method = [], {}, {}
for method, Wm in W.items():
    blends = Z.values @ Wm                                   # daily blend values (standardized space)
    blends = pd.DataFrame(blends, index=Z.index)
    for j in range(Wm.shape[1]):
        y_tr, y_te = blends[j].loc[TRAIN[0]:TRAIN[1]].values, blends[j].loc[TEST[0]:TEST[1]].values
        rows.append({"method": method, "component": j, "train_ar1": ar1(y_tr), "test_ar1": ar1(y_te)})
    df = pd.DataFrame([r for r in rows if r["method"] == method])
    picks = df.reindex(df.train_ar1.abs().sort_values(ascending=False).index).head(TOP)
    legs, gross_legs = [], []
    for _, p in picks.iterrows():
        j = int(p.component)
        w = Wm[:, j] / sd.values                             # weights on raw returns
        w = w / np.abs(w).sum()                              # $1 gross exposure
        port = pd.Series(rets.values @ w, index=rets.index)
        signal = np.sign(p.train_ar1) * np.sign(blends[j])   # forecast sign of tomorrow
        pos = signal.shift(1).fillna(0)
        cost = pos.diff().abs().fillna(0) * COST_BPS / 1e4   # a flip trades 2x the $1 gross book
        legs.append(pos * port - cost)
        gross_legs.append(pos * port)
    pnl_by_method[method] = pd.concat(legs, axis=1).mean(axis=1)
    gross_by_method[method] = pd.concat(gross_legs, axis=1).mean(axis=1)
comp = pd.DataFrame(rows)

# ---------------------------------------------------------------- summary
results = {
    "stocks": rets.shape[1],
    "dims": M,
    "box_tiao_train_r2_top5": bt_train_r2[:5].tolist(),
    "replication": {},
    "trading": {},
}
for method in W:
    c = comp[comp.method == method]
    n_test = len(te)
    band = 2 / np.sqrt(n_test)
    results["replication"][method] = {
        "corr_train_vs_test_ar1": float(c.train_ar1.corr(c.test_ar1)),
        "max_abs_train_ar1": float(c.train_ar1.abs().max()),
        "max_abs_test_ar1": float(c.test_ar1.abs().max()),
        "test_noise_band": band,
        "same_sign_top3": int((np.sign(c.reindex(c.train_ar1.abs().sort_values(ascending=False).index).head(TOP)[["train_ar1", "test_ar1"]]).prod(axis=1) > 0).sum()),
    }
    r, g = pnl_by_method[method], gross_by_method[method]
    results["trading"][method] = {
        "train": summarize(r.loc[TRAIN[0]:TRAIN[1]]), "test": summarize(r.loc[TEST[0]:TEST[1]]),
        "train_before_costs": summarize(g.loc[TRAIN[0]:TRAIN[1]]), "test_before_costs": summarize(g.loc[TEST[0]:TEST[1]]),
    }

OUT.mkdir(exist_ok=True)
with open(OUT / "ica_pca_results.json", "w") as f:
    json.dump(results, f, indent=2)
comp.to_csv(OUT / "ica_pca_components.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for method, c, mk in (("PCA", BLUE, "o"), ("ICA", ORANGE, "s"), ("Box-Tiao", GREEN, "^")):
    d = comp[comp.method == method]
    ax.scatter(d.train_ar1, d.test_ar1, color=c, marker=mk, s=30, label=method, edgecolor="white", linewidth=0.8)
lim = max(comp[["train_ar1", "test_ar1"]].abs().max()) * 1.1
ax.plot([-lim, lim], [-lim, lim], color=GRAY, lw=0.8, ls="--")
ax.axhspan(-2 / np.sqrt(len(te)), 2 / np.sqrt(len(te)), color="#e8e7e2", zorder=0)
ax.axhline(0, color="#c3c2b7", lw=0.6); ax.axvline(0, color="#c3c2b7", lw=0.6)
ax.set_xlabel("train: today -> tomorrow correlation", fontsize=8)
ax.set_ylabel("test: today -> tomorrow correlation", fontsize=8)
ax.legend(frameon=False, fontsize=8)
style(ax, "Does each blend's predictability carry over?\n(dashed = perfect carry-over, gray = noise)")
fig.tight_layout(); fig.savefig(OUT / "ica_pca.png", dpi=150); plt.close(fig)

print(json.dumps({k: results[k] for k in ("stocks", "dims", "box_tiao_train_r2_top5", "replication")}, indent=1))
for m, v in results["trading"].items():
    gt, gs = v["train_before_costs"], v["test_before_costs"]
    print(f"{m:9s} BEFORE costs train {gt['ann_return']:+.1%} (SR {gt['sharpe']:.2f}) test {gs['ann_return']:+.1%} "
          f"(SR {gs['sharpe']:.2f}, t {gs['t_stat']:.2f}) | AFTER costs test {v['test']['ann_return']:+.1%}")
