"""Prediction = compression: is there structure in daily price moves?

Each stock's daily returns are turned into symbols, then compressed two ways:
  1. adaptive context model (an online predictor; its code length is exactly
     what an arithmetic coder would use to store the sequence)
  2. off-the-shelf compressors (lzma, bz2)
Both are compared with the same symbols in shuffled order. Shuffling keeps how
often each symbol appears but destroys any sequence pattern, so
"real < shuffled" means the order of moves carries predictable information.

Two questions are asked of the same data:
  direction  - up or down?          (what you'd need to profit from)
  size       - quiet or wild day?   (volatility)

Outputs: output/compression_results.json
"""
import bz2
import json
import lzma
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

HERE = Path(__file__).parent
OUT = HERE / "output"
N_BINS = 4                 # quartiles of the stock's own return history
ORDERS = (0, 1, 2, 3, 5)   # how many past days the predictor looks at
N_SHUFFLES = 5
rng = np.random.default_rng(0)


def symbolize(r, kind):
    r = r.dropna()
    if kind == "direction":
        return (r > 0).astype(np.uint8).values, 2
    # quartile of absolute move, ranked within the stock's own history
    return pd.qcut(r.abs().rank(method="first"), N_BINS, labels=False).astype(np.uint8).values, N_BINS


def context_counts(seq, k, alphabet):
    """Symbol counts per order-k context (contexts indexed as base-(alphabet+1) numbers)."""
    seq = seq.astype(np.int64)
    ctx = np.zeros(len(seq), dtype=np.int64)
    padded = np.concatenate([np.full(k, alphabet), seq])      # pad = "no history yet"
    for j in range(1, k + 1):
        ctx = ctx * (alphabet + 1) + padded[k - j:len(padded) - j]
    counts = np.zeros(((alphabet + 1) ** k, alphabet))
    np.add.at(counts, (ctx, seq), 1)
    return counts


def kt_bits(counts):
    """Bits an adaptive order-k model (KT estimator) needs to encode the sequence.

    Sequential KT coding has a closed form that depends only on the final
    symbol counts in each context, so no per-symbol loop is needed.
    """
    alphabet = counts.shape[1]
    counts = counts[counts.sum(1) > 0]
    nats = (gammaln(counts.sum(1) + alphabet / 2) - gammaln(alphabet / 2)).sum() \
        - (gammaln(counts + 0.5) - gammaln(0.5)).sum()
    return nats / np.log(2)


def context_bits(seq, k, alphabet):
    return kt_bits(context_counts(seq, k, alphabet))


def lib_bytes(seq):
    raw = seq.tobytes()
    return min(len(lzma.compress(raw, preset=9)), len(bz2.compress(raw, 9)))


prices = pd.read_csv(HERE / "data" / "sp500_prices.csv", index_col=0, parse_dates=True)
prices = prices.loc[prices["SPY"].notna()].drop(columns="^VIX")
rets = prices.pct_change(fill_method=None)
rets = rets.where(rets.abs() < 1.0)

results = {}
for kind in ("direction", "size"):
    tot = defaultdict(float)
    pooled = defaultdict(float)   # one shared model across all stocks: summed counts
    n_syms = 0
    for t in rets.columns:
        seq, a = symbolize(rets[t], kind)
        if len(seq) < 1000:
            continue
        n_syms += len(seq)
        shuffles = [rng.permutation(seq) for _ in range(N_SHUFFLES)]
        for k in ORDERS:
            c_real = context_counts(seq, k, a)
            c_shuf = [context_counts(s, k, a) for s in shuffles]
            tot[f"real_k{k}"] += kt_bits(c_real)
            tot[f"shuf_k{k}"] += np.mean([kt_bits(c) for c in c_shuf])
            pooled[f"real_k{k}"] = pooled[f"real_k{k}"] + c_real
            for i, c in enumerate(c_shuf):
                pooled[f"shuf{i}_k{k}"] = pooled[f"shuf{i}_k{k}"] + c
        tot["real_lib"] += 8 * lib_bytes(seq)
        tot["shuf_lib"] += 8 * np.mean([lib_bytes(s) for s in shuffles])
    res = {"symbols": n_syms, "max_bits_per_symbol": float(np.log2(a))}
    for k in ORDERS:
        real, shuf = tot[f"real_k{k}"] / n_syms, tot[f"shuf_k{k}"] / n_syms
        res[f"order_{k}"] = {"real_bits": real, "shuffled_bits": shuf, "saving_pct": 100 * (1 - real / shuf)}
    for k in ORDERS:
        real = kt_bits(pooled[f"real_k{k}"]) / n_syms
        shufs = [kt_bits(pooled[f"shuf{i}_k{k}"]) / n_syms for i in range(N_SHUFFLES)]
        res[f"pooled_order_{k}"] = {"real_bits": real, "shuffled_bits": float(np.mean(shufs)),
                                    "shuffled_sd": float(np.std(shufs)),
                                    "saving_pct": 100 * (1 - real / np.mean(shufs))}
    real, shuf = tot["real_lib"] / n_syms, tot["shuf_lib"] / n_syms
    res["lzma_bz2"] = {"real_bits": real, "shuffled_bits": shuf, "saving_pct": 100 * (1 - real / shuf)}
    best = min(ORDERS, key=lambda k: tot[f"real_k{k}"])
    res["best_order"] = best
    results[kind] = res
    print(f"\n== {kind} ({n_syms:,} symbols, max {np.log2(a):.2f} bits each)")
    for k in ORDERS:
        r = res[f"order_{k}"]
        print(f"  look back {k} days: real {r['real_bits']:.4f}  shuffled {r['shuffled_bits']:.4f}  saving {r['saving_pct']:+.2f}%")
    for k in ORDERS:
        r = res[f"pooled_order_{k}"]
        print(f"  POOLED, look back {k}: real {r['real_bits']:.4f}  shuffled {r['shuffled_bits']:.4f}"
              f" (sd {r['shuffled_sd']:.5f})  saving {r['saving_pct']:+.2f}%")
    r = res["lzma_bz2"]
    print(f"  lzma/bz2:          real {r['real_bits']:.4f}  shuffled {r['shuffled_bits']:.4f}  saving {r['saving_pct']:+.2f}%")

# What is the sequence pattern worth? Only the bits saved versus shuffled data
# come from the ORDER of days (the rest is just "up days are a bit more common").
# Solve 1 - H(p) = saved bits for p: the next-day accuracy that pattern implies.
d = results["direction"]
best = min(ORDERS, key=lambda k: d[f"pooled_order_{k}"]["real_bits"])
saved = d[f"pooled_order_{best}"]["shuffled_bits"] - d[f"pooled_order_{best}"]["real_bits"]
grid = np.linspace(0.5, 0.6, 100001)[1:]
h = -(grid * np.log2(grid) + (1 - grid) * np.log2(1 - grid))
p = float(grid[np.argmin(np.abs((1 - h) - saved))]) if saved > 0 else 0.5
avg_move = float(rets.abs().stack().mean())
results["direction_pattern"] = {
    "best_pooled_order": best,
    "bits_saved_per_day": saved,
    "implied_accuracy": p,
    "avg_abs_daily_move": avg_move,
    "implied_edge_per_trade": (2 * p - 1) * avg_move,
}
up = (rets > 0).where(rets.notna())
prev = up.shift(1)
results["direction_pattern"]["p_up_after_up"] = float(up[prev == 1].stack().mean())
results["direction_pattern"]["p_up_after_down"] = float(up[prev == 0].stack().mean())
print(f"\ndirection pattern: saves {saved:.5f} bits/day -> implied accuracy {p:.2%}, "
      f"edge per trade ~{(2 * p - 1) * avg_move:.3%} (avg move {avg_move:.2%})")
print(f"P(up | yesterday up) = {results['direction_pattern']['p_up_after_up']:.2%}, "
      f"P(up | yesterday down) = {results['direction_pattern']['p_up_after_down']:.2%}")

OUT.mkdir(exist_ok=True)
with open(OUT / "compression_results.json", "w") as f:
    json.dump(results, f, indent=2)
