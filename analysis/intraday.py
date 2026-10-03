"""Do well-known intraday (day-trading) patterns make money after costs?

Two datasets from fetch_intraday.py:
  hourly  ~730 trading days (Nov 2023 - Oct 2026): bars 9:30, 10:30 ... 15:30 (last = final half hour)
  5-min   ~60 trading days (Jul - Oct 2026)

Every pattern is fixed in advance (no tuning). Each dataset is split in half by
date to check that a result holds in both halves. Returns are per trade, in
basis points (1 bp = 0.01%), BEFORE costs; the break-even cost is the
per-side cost (spread + fees + slippage) that would wipe the edge out.
t-stats average trades within each day first, because trades on the same day
move together.

Hourly patterns
  momentum_1h       sign of (yesterday close -> 10:30) predicts the last half hour
                    (Gao, Han, Li & Zhou 2018, "Market intraday momentum")
  gap_fade          gap of 0.5%+ at the open: bet against it, open -> close
  orb_1h            break of the first hour's high/low (on an hourly close): follow it to the close
  reversal_1h       each hour, buy the 10% of stocks that fell most, short the 10% that rose most, hold 1 hour
5-minute patterns
  momentum_30m      sign of (yesterday close -> 10:00) predicts 15:30 -> 16:00 (the paper's exact spec)
  orb_15m           break of the first 15 minutes' range: follow it, stop at the other side, exit at the close
  vwap_reversion    price 1+ hour-sigma away from the day's VWAP after 10:00: bet back toward it for 30 min
  reversal_5m       every 5 minutes, buy the decile that fell most, short the decile that rose most

Outputs: output/intraday_results.json, output/intraday.png
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from common import BLUE, GRAY, ORANGE, OUT, style

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ETFS = ["SPY", "QQQ", "IWM"]
COSTS = {"pro (1 bp/side)": 1.0, "retail (3 bp/side)": 3.0}
DECILE = 0.10


def load(name, bars_per_day):
    df = pd.read_parquet(HERE / "data" / f"{name}.parquet")
    df = df[df.time.dt.strftime("%H:%M") != "16:00"]                      # drop closing-print stubs
    df["date"] = df.time.dt.tz_localize(None).dt.normalize()
    df["bar"] = df.groupby(["ticker", "date"]).cumcount()
    full = df.groupby(["ticker", "date"]).bar.transform("size") == bars_per_day
    df = df[full]
    cube = {f: df.pivot_table(index=["date", "bar"], columns="ticker", values=f) for f in ("open", "high", "low", "close", "volume")}
    days = cube["close"].index.get_level_values(0).unique()
    arr = {f: cube[f].reindex(pd.MultiIndex.from_product([days, range(bars_per_day)])).to_numpy(copy=True).reshape(len(days), bars_per_day, -1)
           for f in cube}
    tickers = cube["close"].columns
    prev_close = np.vstack([np.full((1, len(tickers)), np.nan), arr["close"][:-1, -1, :]])
    gap = arr["open"][:, 0, :] / prev_close - 1
    bad = np.abs(gap) > 0.2                                                 # splits / broken data
    for f in arr:
        arr[f][bad[:, None, :].repeat(bars_per_day, 1)] = np.nan
    return days, tickers, arr, np.where(bad, np.nan, prev_close)


def summarize(trades, split_date):
    """trades: DataFrame(date, ret_bps). Per-trade mean, win rate, day-clustered t, both halves."""
    out = {}
    for name, t in (("all", trades), ("first_half", trades[trades.date < split_date]),
                    ("second_half", trades[trades.date >= split_date])):
        t = t.dropna()
        if len(t) < 10:
            out[name] = None
            continue
        daily = t.groupby("date").ret_bps.mean()
        out[name] = {
            "trades": len(t), "days": len(daily),
            "gross_bps_per_trade": float(t.ret_bps.mean()),
            "win_rate": float((t.ret_bps > 0).mean()),
            "t_stat_by_day": float(daily.mean() / daily.std() * np.sqrt(len(daily))) if daily.std() > 0 else float("nan"),
            "break_even_cost_bps_per_side": float(t.ret_bps.mean() / 2),
        }
        out[name].update({f"net_bps_{k}": float(t.ret_bps.mean() - 2 * c) for k, c in COSTS.items()})
    return out


def frame(days, mask, ret):
    """Collect (date, return in bps) for every True in mask (days x tickers)."""
    d_idx, _ = np.nonzero(mask & ~np.isnan(ret))
    return pd.DataFrame({"date": days[d_idx], "ret_bps": 1e4 * ret[mask & ~np.isnan(ret)]})


def momentum(days, tickers, a, prev_close, first_bar_end, last_bar):
    sig = a["close"][:, first_bar_end, :] / prev_close - 1
    last = a["close"][:, last_bar, :] / a["open"][:, last_bar, :] - 1
    ret = np.sign(sig) * last
    is_etf = np.isin(tickers, ETFS)[None, :].repeat(len(days), 0)
    mask = ~np.isnan(sig) & (sig != 0)
    return frame(days, mask & is_etf, ret), frame(days, mask & ~is_etf, ret)


def orb(days, a, range_bars, use_stop):
    hi = np.nanmax(a["high"][:, :range_bars, :], axis=1)
    lo = np.nanmin(a["low"][:, :range_bars, :], axis=1)
    close = a["close"]
    n_days, n_bars, n_tk = close.shape
    ret = np.full((n_days, n_tk), np.nan)
    for d in range(n_days):
        c = close[d]
        for s in range(n_tk):
            col = c[range_bars:, s]
            up = np.flatnonzero(col > hi[d, s])
            dn = np.flatnonzero(col < lo[d, s])
            first = min(up[0] if len(up) else 1e9, dn[0] if len(dn) else 1e9)
            if first == 1e9 or first >= len(col) - 1:
                continue
            side = 1 if (len(up) and up[0] == first) else -1
            entry = col[first]
            path = side * (col[first + 1:] / entry - 1)
            exit_ret = path[-1]
            if use_stop:
                stop = side * ((lo[d, s] if side == 1 else hi[d, s]) / entry - 1)
                hit = np.flatnonzero(path <= stop)
                if len(hit):
                    exit_ret = path[hit[0]]
            ret[d, s] = exit_ret
    return frame(days, ~np.isnan(ret), ret)


def cross_reversal(days, a):
    """Each bar: long the decile with the lowest return over that bar, short the highest, hold the next bar."""
    o, c = a["open"], a["close"]
    bar_ret = c / o - 1
    rows = []
    for d in range(len(days)):
        for b in range(c.shape[1] - 1):
            r_now, r_next = bar_ret[d, b], c[d, b + 1] / c[d, b] - 1
            ok = ~np.isnan(r_now) & ~np.isnan(r_next)
            if ok.sum() < 50:
                continue
            x, y = r_now[ok], r_next[ok]
            n = max(1, int(len(x) * DECILE))
            order = np.argsort(x)
            rows.append((days[d], 1e4 * (y[order[:n]].mean() - y[order[-n:]].mean()) / 2))   # per $ of gross
    return pd.DataFrame(rows, columns=["date", "ret_bps"])


def gap_fade(days, a, prev_close):
    gap = a["open"][:, 0, :] / prev_close - 1
    day = a["close"][:, -1, :] / a["open"][:, 0, :] - 1
    return frame(days, np.abs(gap) >= 0.005, -np.sign(gap) * day)


def vwap_reversion(days, a):
    o, h, lo, c, v = (a[k] for k in ("open", "high", "low", "close", "volume"))
    typical = (h + lo + c) / 3
    vwap = np.nancumsum(typical * v, axis=1) / np.nancumsum(v, axis=1)
    r5 = np.log(c[:, 1:, :] / c[:, :-1, :])
    sigma_hour = np.nanstd(r5, axis=1, keepdims=True) * np.sqrt(12)             # one-hour volatility, per day
    sigma_hour = np.vstack([np.full((1, 1, c.shape[2]), np.nan), sigma_hour[:-1]])  # use YESTERDAY's: no look-ahead
    dev = np.log(c / vwap)
    hold = 6
    rows = []
    for d in range(c.shape[0]):
        for s in range(c.shape[2]):
            b = 6                                                               # from 10:00
            while b < c.shape[1] - hold:
                z = dev[d, b, s] / sigma_hour[d, 0, s]
                if np.isfinite(z) and abs(z) >= 1:
                    rows.append((days[d], 1e4 * -np.sign(z) * (c[d, b + hold, s] / c[d, b, s] - 1)))
                    b += hold
                else:
                    b += 1
    return pd.DataFrame(rows, columns=["date", "ret_bps"])


# ---------------------------------------------------------------- run
results = {}
h_days, h_tk, h, h_prev = load("intraday_60m", 7)
h_split = h_days[len(h_days) // 2]
etf, stk = momentum(h_days, h_tk, h, h_prev, first_bar_end=0, last_bar=6)
results["momentum_1h_etfs"] = summarize(etf, h_split)
results["momentum_1h_stocks"] = summarize(stk, h_split)
results["gap_fade"] = summarize(gap_fade(h_days, h, h_prev), h_split)
results["orb_1h"] = summarize(orb(h_days, h, range_bars=1, use_stop=False), h_split)
results["reversal_1h"] = summarize(cross_reversal(h_days, h), h_split)

m_days, m_tk, m, m_prev = load("intraday_5m", 78)
m_split = m_days[len(m_days) // 2]
# first half hour = yesterday's close -> close of the 9:55 bar; last half hour = 15:30 open -> 15:55 close
sig = m["close"][:, 5, :] / m_prev - 1
last = m["close"][:, 77, :] / m["open"][:, 72, :] - 1
is_etf = np.isin(m_tk, ETFS)[None, :].repeat(len(m_days), 0)
mask = ~np.isnan(sig) & (sig != 0)
results["momentum_30m_etfs"] = summarize(frame(m_days, mask & is_etf, np.sign(sig) * last), m_split)
results["momentum_30m_stocks"] = summarize(frame(m_days, mask & ~is_etf, np.sign(sig) * last), m_split)
results["orb_15m_with_stop"] = summarize(orb(m_days, m, range_bars=3, use_stop=True), m_split)
results["vwap_reversion"] = summarize(vwap_reversion(m_days, m), m_split)
results["reversal_5m"] = summarize(cross_reversal(m_days, m), m_split)

meta = {"hourly": [str(h_days[0].date()), str(h_days[-1].date()), len(h_days), str(h_split.date())],
        "five_min": [str(m_days[0].date()), str(m_days[-1].date()), len(m_days), str(m_split.date())]}
OUT.mkdir(exist_ok=True)
with open(OUT / "intraday_results.json", "w") as f:
    json.dump({"meta": meta, "results": results}, f, indent=2)

# ---------------------------------------------------------------- chart: gross edge per trade vs costs, both halves
fig, ax = plt.subplots(figsize=(9, 4.6))
names = list(results)
y = np.arange(len(names))[::-1]
for i, n in enumerate(names):
    for half, c, off in (("first_half", BLUE, 0.17), ("second_half", ORANGE, -0.17)):
        v = results[n][half]
        if v:
            ax.barh(y[i] + off, v["gross_bps_per_trade"], 0.3, color=c, label=half.replace("_", " ") if i == 0 else None)
for k, c in COSTS.items():
    ax.axvline(2 * c, color=GRAY, lw=0.8, ls="--" if c > 1 else ":")
    ax.text(2 * c + 0.15, y[0], f"{k}\nround trip", fontsize=6.5, color="#52514e", ha="left", va="center")
ax.axvline(0, color=GRAY, lw=0.8)
ax.set_yticks(y, [n.replace("_", " ") for n in names], fontsize=8)
ax.set_xlabel("average profit per trade BEFORE costs (basis points; 1 bp = 0.01%)", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="lower right")
style(ax, "Intraday patterns: edge per trade vs the cost of trading (bars must clear the dashed lines)")
fig.tight_layout(); fig.savefig(OUT / "intraday.png", dpi=150); plt.close(fig)

print(f"hourly {meta['hourly']}, 5-min {meta['five_min']}")
print(f"\n{'pattern':22s} {'trades':>8s} | {'gross bp 1st':>12s} {'t':>5s} | {'gross bp 2nd':>12s} {'t':>5s} | "
      f"{'win%':>5s} {'break-even bp/side':>18s}")
for n, v in results.items():
    a, f1, f2 = v["all"], v["first_half"], v["second_half"]
    print(f"{n:22s} {a['trades']:8d} | {f1['gross_bps_per_trade']:+12.2f} {f1['t_stat_by_day']:5.1f} | "
          f"{f2['gross_bps_per_trade']:+12.2f} {f2['t_stat_by_day']:5.1f} | {a['win_rate']:5.1%} "
          f"{a['break_even_cost_bps_per_side']:+18.2f}")
