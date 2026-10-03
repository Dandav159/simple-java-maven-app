"""Find the market's hidden drivers with ICA, then predict them from OUTSIDE data.

Step 1 - ICA (fit on 2014-2020) splits ~450 stocks into 20 independent blends.
Step 2 - Name each blend by which outside series moves with it the same day
         (yields, oil, gold, copper, dollar, credit, foreign markets).
Step 3 - Try to PREDICT the blends with outside data that is known in advance:
  A  yesterday's outside moves      -> today's blend returns (close to close)
  B  Asia's close (Japan, Hong Kong, Australia, Korea finish before the US
     opens)                         -> today's blend returns from US open to close
  C  last month's outside moves     -> next month's blend returns
Each is a ridge regression fit on TRAIN and judged on TEST (2021-2026): forecast
accuracy, plus trading every blend long/short by the forecast's sign, after costs.
SPY gets the same treatment as a control: do the blends add anything over
predicting the market as a whole?

Outputs: output/ica_macro_results.json, output/ica_macro.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from blends import fit, load_returns
from common import BLUE, GRAY, ORANGE, OUT, TEST, TRAIN, style
from fetch_macro import MACRO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
COST_BPS = 5
ALPHAS = np.logspace(-1, 5, 25)
YIELDS = ["^TNX", "^FVX", "^IRX"]
ASIA = ["^N225", "^HSI", "^AXJO", "^KS11"]
GREEN = "#1baf7a"

# ---------------------------------------------------------------- data
rets = load_returns()
dates = rets.index
closes = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True).reindex(dates)
opens = pd.read_csv(HERE / "data" / "sp500_open.csv", index_col=0, parse_dates=True).reindex(dates)
intraday = (closes[rets.columns] / opens[rets.columns] - 1)          # open -> close, same day
intraday = intraday.where(intraday.abs() < 0.5)
spy_c2c = closes["SPY"].pct_change()
spy_o2c = closes["SPY"] / opens["SPY"] - 1

macro = pd.read_csv(HERE / "data" / "macro_close.csv", index_col=0, parse_dates=True).dropna(how="all")
chg = macro.pct_change(fill_method=None)
chg[YIELDS] = macro[YIELDS].diff()                                    # yields: change in percentage points
chg = chg.where(chg.abs() < 15 * chg.std())                         # drop broken ticks (e.g. oil at -$37)

# ---------------------------------------------------------------- step 1: ICA blends as $1-gross portfolios
mu, sd, _, W = fit(rets)
weights = W / sd.values[:, None]
weights = weights / np.abs(weights).sum(0)
names = [f"IC{j + 1}" for j in range(W.shape[1])]
blend_c2c = pd.DataFrame(rets.fillna(0).values @ weights, index=dates, columns=names)
blend_o2c = pd.DataFrame(intraday.fillna(0).values @ weights, index=dates, columns=names)
blend_c2c["SPY"], blend_o2c["SPY"] = spy_c2c, spy_o2c

# ---------------------------------------------------------------- step 2: name each blend
us_hours = [c for c in MACRO if c not in ASIA]
same_day = chg[us_hours].reindex(dates)
train_days = slice(TRAIN[0], TRAIN[1])
labels = {}
for c in names + ["SPY"]:
    corr = same_day.loc[train_days].corrwith(blend_c2c.loc[train_days, c])
    top = corr.abs().sort_values(ascending=False).head(2).index
    y = blend_c2c.loc[train_days, c].dropna()
    X = StandardScaler().fit_transform(same_day.loc[y.index].fillna(0))
    r2 = RidgeCV(alphas=ALPHAS).fit(X, y).score(X, y)                 # in-sample: a description, not a forecast
    labels[c] = {"top_links": {MACRO[t]: round(float(corr[t]), 2) for t in top}, "same_day_r2": round(float(r2), 3)}


# ---------------------------------------------------------------- step 3: predict
def evaluate(features, targets, cost_per_period, periods_per_year):
    """Fit ridge on train for each target, forecast test; return accuracy and P&L."""
    out = {}
    pnl_all, gross_all = [], []
    for c in targets.columns:
        df = pd.concat([features, targets[c].rename("y")], axis=1).dropna()
        tr, te = df.loc[TRAIN[0]:TRAIN[1]], df.loc[TEST[0]:TEST[1]]
        sc = StandardScaler().fit(tr.drop(columns="y"))
        model = RidgeCV(alphas=ALPHAS).fit(sc.transform(tr.drop(columns="y")), tr.y)
        f = pd.Series(model.predict(sc.transform(te.drop(columns="y"))), index=te.index)
        pos = np.sign(f)
        gross = pos * te.y
        net = gross - cost_per_period(pos)
        out[c] = {
            "oos_r2_vs_zero": float(1 - ((te.y - f) ** 2).sum() / (te.y ** 2).sum()),
            "forecast_corr": float(np.corrcoef(f, te.y)[0, 1]) if f.std() > 0 else 0.0,
            "hit_rate": float((np.sign(f) == np.sign(te.y)).mean()),
            "n_test": len(te),
        }
        if c != "SPY":
            pnl_all.append(net)
            gross_all.append(gross)
    net = pd.concat(pnl_all, axis=1).mean(axis=1)
    gross = pd.concat(gross_all, axis=1).mean(axis=1)
    blends_only = {k: v for k, v in out.items() if k != "SPY"}
    ann = lambda r: float(r.mean() * periods_per_year)  # noqa: E731
    sr = lambda r: float(r.mean() / r.std() * np.sqrt(periods_per_year))  # noqa: E731
    return {
        "per_target": out,
        "blends_avg_forecast_corr": float(np.mean([v["forecast_corr"] for v in blends_only.values()])),
        "blends_avg_oos_r2": float(np.mean([v["oos_r2_vs_zero"] for v in blends_only.values()])),
        "blends_beating_zero": int(sum(v["oos_r2_vs_zero"] > 0 for v in blends_only.values())),
        "noise_band_corr": float(2 / np.sqrt(np.mean([v["n_test"] for v in blends_only.values()]))),
        "trade_before_costs": {"ann_return": ann(gross), "sharpe": sr(gross),
                               "t_stat": float(gross.mean() / gross.std() * np.sqrt(len(gross)))},
        "trade_after_costs": {"ann_return": ann(net), "sharpe": sr(net)},
    }


flip_cost = lambda pos: pos.diff().abs().fillna(pos.abs()) * COST_BPS / 1e4  # noqa: E731
round_trip = lambda pos: pos.abs() * 2 * COST_BPS / 1e4                      # noqa: E731  in at open, out at close

# A: yesterday's outside moves -> today
feat_a = chg.reindex(dates).shift(1)
results_a = evaluate(feat_a, blend_c2c, flip_cost, 252)

# B: Asia's close today (before the US opens) -> US open-to-close today
asia_today = chg[ASIA].reindex(dates)
results_b = evaluate(asia_today, blend_o2c, round_trip, 252)
# B2: same, but also give the model the overnight gap (also known at the open): has the
# open already priced Asia in?
gap = (opens["SPY"] / closes["SPY"].shift(1) - 1).rename("spy_gap")
results_b2 = evaluate(pd.concat([asia_today, gap], axis=1), blend_o2c, round_trip, 252)

# C: last month's outside moves -> next month
month_end = dates.to_series().groupby(dates.to_period("M")).last()
m_macro = macro.reindex(dates).ffill().loc[month_end]
m_chg = m_macro.pct_change(fill_method=None)
m_chg[YIELDS] = m_macro[YIELDS].diff()
m_blend = (1 + blend_c2c).groupby(dates.to_period("M")).prod() - 1
m_blend.index = month_end.values
results_c = evaluate(m_chg.shift(0).iloc[:-1].set_axis(m_blend.index[1:]), m_blend.iloc[1:], flip_cost, 12)

results = {"labels": labels, "A_yesterday_macro": results_a, "B_asia_to_us_intraday": results_b,
           "B2_asia_plus_overnight_gap": results_b2, "C_last_month_macro": results_c}
OUT.mkdir(exist_ok=True)
with open(OUT / "ica_macro_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---------------------------------------------------------------- chart
fig, ax = plt.subplots(figsize=(9, 4))
tests = [("A: yesterday's\noutside data", results_a), ("B: Asia close ->\nUS open-to-close", results_b),
         ("B2: Asia + US\novernight gap", results_b2), ("C: last month's\noutside data", results_c)]
x = np.arange(len(tests))
for i, (lab, r) in enumerate(tests):
    vals = [v["forecast_corr"] for k, v in r["per_target"].items() if k != "SPY"]
    ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, color=BLUE, s=14, alpha=0.7,
               label="each ICA driver" if i == 0 else None)
    ax.scatter([i + 0.3], [r["per_target"]["SPY"]["forecast_corr"]], color=ORANGE, marker="D", s=36,
               label="SPY (control)" if i == 0 else None)
    band = r["noise_band_corr"]
    ax.fill_between([i - 0.25, i + 0.4], -band, band, color="#e8e7e2", zorder=0)
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xticks(x, [t[0] for t in tests], fontsize=8)
ax.set_ylabel("test-period correlation:\nforecast vs what happened", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper right")
style(ax, "Can outside data predict the market's hidden drivers? (gray = noise)")
fig.tight_layout(); fig.savefig(OUT / "ica_macro.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- console
print("What each driver moves with (train, same day):")
for c, v in labels.items():
    print(f"  {c:5s} R2 {v['same_day_r2']:.2f}  {v['top_links']}")
for name, r in [(t[0].replace(chr(10), ' '), t[1]) for t in tests]:
    b, a = r["trade_before_costs"], r["trade_after_costs"]
    print(f"\n{name}: avg forecast corr {r['blends_avg_forecast_corr']:+.3f} (noise ±{r['noise_band_corr']:.3f}), "
          f"{r['blends_beating_zero']}/20 beat a zero forecast, SPY corr {r['per_target']['SPY']['forecast_corr']:+.3f}")
    print(f"  trade: before costs {b['ann_return']:+.1%}/yr (SR {b['sharpe']:.2f}, t {b['t_stat']:.2f}) | "
          f"after costs {a['ann_return']:+.1%}/yr (SR {a['sharpe']:.2f})")
