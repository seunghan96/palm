#!/usr/bin/env python3
"""The adapter capacity curve, as a figure rather than only as a table.

The table reports a gain per rank, but the shape of that sequence is the actual
finding, namely that it rises to a peak at moderate capacity and falls at both
ends. A column of numbers hides a shape, so the same quantities are plotted.

Gains are in units of 1e-3 because the raw ICs sit near 1e-2, and the interval
around each point is a bootstrap over checkpoints, so the reader can see which
ranks are separated from zero and which are not."""
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS

CACHE = os.path.join(RESULTS, "rank_curve.csv")
RAW = os.path.join(RESULTS, "rank_raw.csv")

GREEN = "#2c7a4b"
BLUE = "#2c5f9e"
RED = "#c0504d"
GREY = "#8c8c8c"
COL = 3.33

# A geometric ladder, so the axis is evenly spaced in log2 and no rank looks
# missing. Ranks 3 and 6 were also run and sit between their neighbours, but
# plotting them turns a clean ladder into a list with gaps at 5 and 7, so they
# are reported in the table instead.
RANKS = [(1, "palm_r1"), (2, "palm_r2"), (4, "palm_r4"), (8, "palm_r8"),
         (16, "palm_r16"), (32, "palm_r32")]

import main_figure as MF


def ci(x, reps=4000):
    """Bootstrap mean and its 95 percent interval over checkpoints."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(0)
    b = np.array([rng.choice(x, len(x), replace=True).mean() for _ in range(reps)])
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), len(x)


def build():
    """Every rank on the same checkpoints.

    A rank that has run on part of the pool would otherwise enter the curve
    with a wider interval for a reason that has nothing to do with rank, and
    the points would not be comparable to each other. So the curve is drawn on
    the checkpoints every rank has in common, and a rank that is still thin is
    left out rather than allowed to shrink that set for everyone."""
    base = MF.collect("base")
    got = {r: {m: hi - lo for m, lo, hi in MF.paired(base, sub)}
           for r, sub in RANKS}
    got = {r: v for r, v in got.items() if len(v) >= 35}
    shared = set.intersection(*[set(v) for v in got.values()])
    print(f"  {len(shared)} checkpoints shared by ranks {sorted(got)}")
    rows, raw = [], []
    for r in sorted(got):
        pairs = [(m, 0.0, got[r][m]) for m in shared]
        d = [hi - lo for _, lo, hi in pairs]
        if len(d) >= 3:
            m, lo, hi, n = ci(d)
            rows.append({"rank": r, "gain": m, "lo": lo, "hi": hi, "n": n})
            raw.extend({"rank": r, "model": m_, "gain": h - l}
                       for m_, l, h in pairs)
            print(f"  rank {r:2d}  n={n:3d}  gain {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]")
    t = pd.DataFrame(rows)
    t.to_csv(CACHE, index=False)
    pd.DataFrame(raw).to_csv(RAW, index=False)
    return t


def draw(t):
    fig, a = plt.subplots(figsize=(COL, 1.95))
    x = np.log2(t["rank"].values)
    g, lo, hi = (t[c].values * 1e3 for c in ("gain", "lo", "hi"))

    # the rule at zero is the comparison the whole table makes, so both sides of
    # it are named. A strip of the losing side is always shown, even when no
    # rank falls there, so that the reader can see the sign convention.
    top = max(hi.max() * 1.40, 1.6)
    bot = min(lo.min() * 1.20, -0.21 * top)
    a.axhspan(0, top, color=BLUE, alpha=0.09, zorder=0)
    a.axhspan(bot, 0, color=GREY, alpha=0.16, zorder=0)
    a.axhline(0, color="#555555", lw=0.9, zorder=2)
    a.text(x[0] - 0.30, top * 0.90, "Improvement", fontsize=6.6, color=BLUE,
           ha="left", va="top", fontweight="bold")
    a.text(x[0] - 0.30, bot / 2.0, "Degradation", fontsize=6.6, color="#666666",
           ha="left", va="center", fontweight="bold")

    a.fill_between(x, lo, hi, color=GREEN, alpha=0.16, zorder=2)
    a.plot(x, g, color=GREEN, lw=1.6, marker="o", ms=4.5, zorder=4,
           markerfacecolor="white", markeredgewidth=1.3)
    best = int(np.argmax(g))
    a.scatter([x[best]], [g[best]], s=46, color=GREEN, zorder=5)

    a.set_xticks(x)
    a.set_xticklabels([str(r) for r in t["rank"]], fontsize=6.8)
    a.set_xlim(x[0] - 0.35, x[-1] + 0.35)
    a.set_ylim(bot, top)
    a.tick_params(labelsize=6.8, length=2)
    a.set_xlabel("Adapter Rank", fontsize=7.4, labelpad=2)
    a.set_ylabel("Gain in IC ($\\times 10^{-3}$)", fontsize=7.4, labelpad=2)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    # a title over a subtitle, the second set smaller and grey so that the
    # sample the curve averages over is stated without competing with the claim
    n = int(t["n"].max())
    a.text(0.5, 1.155, "Effect of the Adapter Capacity", transform=a.transAxes,
           ha="center", va="bottom", fontsize=8.0)
    a.text(0.5, 1.035, f"Averaged across {n} checkpoints of 3 backbones",
           transform=a.transAxes, ha="center", va="bottom", fontsize=6.3,
           color="#5a5a5a")
    fig.savefig(f"{FIGS}/rank_curve.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("  wrote rank_curve.pdf")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    if os.path.exists(CACHE) and "--refresh" not in sys.argv:
        t = pd.read_csv(CACHE)
    else:
        t = build()
    draw(t)
