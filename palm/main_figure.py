#!/usr/bin/env python3
"""The paper's main figure: (a) each checkpoint before and after PALM, and
(b) the IC of the checkpoints as released against their staleness, demeaned
within suite. Next open -> next close target. Writes figs/ws_main.pdf."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import FIGS
import stats

A = "@oc1"
MAIN, BASE = os.environ.get("MAIN", "sel_ps"), "base_test"
COL = {"ChronoGPT": "#1f77b4", "ChronoGPT-Instruct": "#ff7f0e", "DatedGPT": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "font.family": "serif"})

b = stats.load(BASE + A)
a = stats.load(MAIN + A)
ms = sorted(m for m in b if stats.suite(m) in COL and m in a)
st = np.array([stats.EV - stats.cutoff(m) for m in ms])
su = np.array([stats.suite(m) for m in ms])
icb = np.array([b[m].ic.mean() for m in ms]) * 1e3
ica = []
for m in ms:
    j = b[m].index.intersection(a[m].index)
    ica.append(a[m].ic.reindex(j).mean() * 1e3)
    icb[ms.index(m)] = b[m].ic.reindex(j).mean() * 1e3
ica = np.array(ica)

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.25, 1]})
ax = ax[::-1]   # PALM first (left), staleness second (right)
dm = icb.copy()
for s in COL:
    k = su == s
    if k.any():
        dm[k] = icb[k] - icb[k].mean()
        ax[0].scatter(st[k], dm[k], s=14, color=COL[s], label=s)
r = stats.slope(BASE + A, "ic")
xs = np.linspace(st.min(), st.max(), 10)
ax[0].plot(xs, r["slope"] * 1e3 * (xs - st.mean()), color="black", lw=1)
ax[0].axhline(0, color="grey", lw=0.5, ls=":")
ax[0].set_xlabel("Staleness at the decision date (years)")
ax[0].set_ylabel("IC, demeaned ($\\times10^{-3}$)")
ax[0].set_title(f"(b) Slope {r['slope'] * 1e3:+.3f}$\\times10^{{-3}}$/year", fontsize=9)
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=7, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))

order = np.argsort(icb)
for i, k in enumerate(order):
    ax[1].plot([icb[k], ica[k]], [i, i], color=COL[su[k]], lw=0.8)
    ax[1].scatter(icb[k], i, s=10, facecolors="white", edgecolors=COL[su[k]], lw=0.8)
    ax[1].scatter(ica[k], i, s=10, color=COL[su[k]])
ax[1].axvline(0, color="grey", lw=0.5, ls=":")
ax[1].set_yticks([])
ax[1].set_xlabel("IC ($\\times10^{-3}$)")
ax[1].set_ylabel("Checkpoints, sorted by baseline", labelpad=8)
w = int((ica > icb).sum())
ax[1].set_title(f"(a) Improves {w} of {len(ms)} checkpoints", fontsize=9)
ax[1].legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="w/o PALM"),
                      Line2D([], [], marker="o", ls="", color="black", label="w/ PALM")],
             fontsize=7, frameon=False, loc="lower right")
for x in ax:
    x.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out = os.path.join(FIGS, "ws_main.pdf")
fig.savefig(out, bbox_inches="tight", pad_inches=0)
print("wrote", out, MAIN, BASE, len(ms))
