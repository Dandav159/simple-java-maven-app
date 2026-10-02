"""Shared settings and helpers for the analysis scripts."""
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "output"
TRAIN = ("2014-01-01", "2020-12-31")
TEST = ("2021-01-01", "2026-12-31")
ANN = 252
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"


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


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e8e7e2", linewidth=0.6)
    ax.set_axisbelow(True)
