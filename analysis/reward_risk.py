"""Does a good reward:risk ratio make money on its own?

Random entries (long or short, coin-flip direction) on S&P 500 stocks, 2014-2026.
Stop-loss 1 unit away, target R units away, where 1 unit = 2x the stock's recent
daily volatility. Exit at whichever is hit first (checked on daily closes), or
after 60 trading days. If reward:risk alone created profit, these random trades
would earn ~ +0.5 units per trade at 2:1.

Outputs: output/reward_risk_results.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "output"
RATIOS = (1, 2, 3, 5)
TRADES = 200_000
MAX_DAYS = 60
COST_BPS = 10            # round trip
rng = np.random.default_rng(0)

prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns=["^VIX", "SPY"])
P = prices.values
logp = np.log(P)
vol = pd.DataFrame(np.diff(logp, axis=0, prepend=np.nan)).rolling(20).std().values

results = {}
for R in RATIOS:
    wins = losses = timeouts = 0
    pnl_units = []
    n = 0
    while n < TRADES:
        t = rng.integers(21, len(P) - MAX_DAYS - 1)
        s = rng.integers(P.shape[1])
        if not vol[t, s] > 0 or np.isnan(logp[t:t + MAX_DAYS + 1, s]).any():
            continue
        unit = 2 * vol[t, s]                                  # risk unit in log-return terms
        side = rng.choice([-1, 1])
        path = side * (logp[t + 1:t + MAX_DAYS + 1, s] - logp[t, s]) / unit
        hit_target = np.argmax(path >= R) if (path >= R).any() else MAX_DAYS
        hit_stop = np.argmax(path <= -1) if (path <= -1).any() else MAX_DAYS
        if hit_target < hit_stop:
            wins += 1; out = path[hit_target]
        elif hit_stop < hit_target:
            losses += 1; out = path[hit_stop]
        else:
            timeouts += 1; out = path[-1]
        pnl_units.append(out - COST_BPS / 1e4 / unit)
        n += 1
    pnl = np.array(pnl_units)
    results[f"{R}:1"] = {
        "win_rate": wins / TRADES, "loss_rate": losses / TRADES, "timeout_rate": timeouts / TRADES,
        "theory_win_rate_random_walk": 1 / (1 + R),
        "avg_units_per_trade_after_costs": float(pnl.mean()),
        "t_stat": float(pnl.mean() / pnl.std() * np.sqrt(len(pnl))),
    }
    r = results[f"{R}:1"]
    print(f"{R}:1  win {r['win_rate']:.1%} (random-walk theory {1 / (1 + R):.1%}), "
          f"loss {r['loss_rate']:.1%}, timeout {r['timeout_rate']:.1%}  ->  "
          f"{r['avg_units_per_trade_after_costs']:+.3f} units/trade (t {r['t_stat']:.1f})")

with open(OUT / "reward_risk_results.json", "w") as f:
    json.dump(results, f, indent=2)
