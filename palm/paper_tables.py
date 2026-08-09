#!/usr/bin/env python3
"""Every number the result tables carry, printed as LaTeX rows.

The tables were hand-edited for a while, which was fine when there was one of
them and wrong once there were six that had to agree. This reads metrics3.csv
once and prints the rows, so a change to the pool changes every table together.

One suite is held out of the pooled numbers. PIT-4B releases its vintages as
monthly snapshots of a single continuous run rather than as models trained
separately per cutoff, so a later vintage there has taken more optimizer steps
as well as read more text. That is a different quantity from the one the rest of
the rest of the pool measures, so it is reported on its own instead."""
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statsmodels.formula.api as smf

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS

import main_figure as MF

HELD = "PIT-4B"
D = pd.read_csv(METRICS)
D["cut"] = D.model.map(MF.cutoff)
P = D[D.suite != HELD]


def wins(g, a, b):
    d = g[a] - g[b]
    return f"{int((d > 0).sum())}/{len(g)} ({round(100 * (d > 0).mean())}\\%)"


def share(g, a, b):
    """Just the percentage, for a table too narrow to carry the fraction."""
    return f"{round(100 * ((g[a] - g[b]) > 0).mean())}\\%"


def gain(g, a, b, sc, dp=2):
    return f"{((g[a] - g[b]) * sc).mean():+.{dp}f}"


def pv(g, a, b):
    d = (g[a] - g[b]).values
    if len(d) < 5:
        return "---"
    p = wilcoxon(d).pvalue
    return "$<$0.001" if p < 0.001 else f"{p:.3f}".rstrip("0")


def block(g):
    """One backbone, both rows of the - / + PALM pair."""
    def bold(x, y, f):
        return (f"\\textbf{{{f(x)}}}", f(y)) if x > y else (f(x), f"\\textbf{{{f(y)}}}")
    ic = bold(g.ic_b.mean() * 1e3, g.ic_a.mean() * 1e3, lambda v: f"{v:.2f}")
    rt = bold(g.ret_b.mean() * 100, g.ret_a.mean() * 100, lambda v: f"{v:.2f}")
    sh = bold(g.sh_b.mean(), g.sh_a.mean(), lambda v: f"{v:.2f}")
    return ic, rt, sh


# ------------------------------------------------------- Table 4, staleness
def staleness():
    """Every eligible vintage, not only the ones that also have an adapter run.

    The question here concerns the base checkpoint alone, so restricting the
    sample to the paired subset would drop vintages for no reason. This reads
    the same series the figure plots, so the two agree by construction."""
    print("\n% ---- tab:fresh")
    base = MF.collect("base")
    b = pd.DataFrame([dict(s=MF.suite(m), stale=MF.EVAL - MF.cutoff(m), ic=v.mean())
                      for m, v in base.items()])
    for s in sorted(b.s.unique()):
        r = smf.ols("ic ~ stale", data=b[b.s == s]).fit()
        print(f"{s:18s} & {r.params['stale']:+.5f} & {r.pvalues['stale']:.2f} \\\\")
    r = smf.ols("ic ~ stale + C(s)", data=b).fit()
    print(f"Pooled             & {r.params['stale']:+.5f} & {r.pvalues['stale']:.2f} \\\\"
          f"   % n={len(b)}")


# --------------------------------------------------- Table 5, effectiveness
def effectiveness():
    print("\n% ---- tab:adapt")
    r = P[P.variant == "r2"]
    for s in sorted(r.suite.unique()) + ["All backbones"]:
        g = r if s == "All backbones" else r[r.suite == s]
        ic, rt, sh = block(g)
        print(f"\\multirow{{2}}{{*}}{{{s}}} & $-$ & {ic[0]} & {rt[0]} & {sh[0]} & "
              f"\\multirow{{2}}{{*}}{{{wins(g, 'ic_a', 'ic_b')}}} & "
              f"\\multirow{{2}}{{*}}{{{pv(g, 'ic_a', 'ic_b')}}} \\\\")
        print(f" & $+$ PALM & {ic[1]} & {rt[1]} & {sh[1]} & & \\\\")


# ------------------------------------------------------- Table 6, rank curve
def ranks():
    print("\n% ---- tab:rank")
    for r, v in [(1, "r1"), (2, "r2"), (3, "rank3"), (4, "r4"), (6, "rank6"),
                 (8, "r8"), (12, "rank12"), (16, "r16"), (24, "rank24"),
                 (32, "rank32")]:
        g = P[P.variant == v]
        if len(g) < 5:
            print(f"{r} & \\tbd & \\tbd & \\tbd & \\tbd & \\tbd & \\tbd \\\\"
                  f"   % {len(g)} paired so far")
            continue
        print(f"{r} & {gain(g,'ic_a','ic_b',1e3)} & {share(g,'ic_a','ic_b')} & "
              f"{gain(g,'ret_a','ret_b',100)} & {share(g,'ret_a','ret_b')} & "
              f"{gain(g,'sh_a','sh_b',1)} & {share(g,'sh_a','sh_b')} \\\\")


# ----------------------------------------------------- Table 8, retraining
def horizon():
    print("\n% ---- tab:horizon")
    g = P[P.variant == "r2"].copy()
    cuts = [(0, 3.5, "0 to 3 years"), (3.5, 6.5, "4 to 6 years"),
            (6.5, 10.5, "7 to 10 years"), (10.5, 99, "11 years or more")]
    for lo, hi, name in cuts:
        h = g[(g.stale >= lo) & (g.stale < hi)]
        if not len(h):
            continue
        print(f"{name:16s} & {h.ic_b.mean()*1e3:.2f} & {h.ic_a.mean()*1e3:.2f} & "
              f"{wins(h,'ic_a','ic_b')} \\\\")
    g["gn"] = g.ic_a - g.ic_b
    for lab, col in [("$-$", "ic_b"), ("$+$ PALM", "ic_a"), ("Gain", "gn")]:
        r = smf.ols(f"{col} ~ stale + C(suite)", data=g).fit()
        print(f"{lab:9s} & \\multicolumn{{2}}{{c}}{{${r.params['stale']:+.6f}$}} & "
              f"$p = {r.pvalues['stale']:.2f}$ \\\\")


# ------------------------------------------------- Table 7, capacity vs width
def width():
    print("\n% ---- tab:width")
    W = {"ChronoGPT": 1536, "ChronoGPT-Instruct": 1536, "DatedGPT": 2048,
         "PIT-4B": 4096}
    for s in ["ChronoGPT", "ChronoGPT-Instruct", "DatedGPT", "PIT-4B"]:
        cells = []
        for v in ["r1", "r2", "r4", "r8", "r16"]:
            g = D[(D.variant == v) & (D.suite == s)]
            cells.append(((g.ic_a - g.ic_b) * 1e3).mean() if len(g) else np.nan)
        top = int(np.nanargmax(cells))
        out = [f"\\textbf{{{c:+.2f}}}" if i == top else f"{c:+.2f}"
               for i, c in enumerate(cells)]
        print(f"{s:18s} & {W[s]} & " + " & ".join(out) + " \\\\")


# ------------------------------------------------------ Table 10, sensitivity
SENS = [("Adaptation schedule", [("One pass over the window", "r2"),
                                 ("One pass per calendar year", "incr")]),
        ("Decision date", [("2017", "d2017"), ("2018", "d2018"), ("2019", "d2019"),
                           ("2020", "d2020"), ("2021", "r2"),
                           ("2022", "d2022"), ("2023", "d2023")]),
        ("Share of eligible text", [("2\\%", "frac2"), ("5\\%", "frac5"),
                                    ("10\\%", "frac10"), ("25\\%", "frac25"),
                                    ("50\\%", "frac50"), ("75\\%", "frac75"),
                                    ("100\\%", "r2")]),
        ("Training budget", [("100 steps", "steps100"),
                             ("200 steps", "steps200"), ("300 steps", "steps300"),
                             ("600 steps", "r2"), ("900 steps", "steps900"),
                             ("1200 steps", "steps1200"),
                             ("1800 steps", "steps1800"),
                             ("2400 steps", "steps2400"),
                             ("Rate $5\\times10^{-6}$", "lr5e6"),
                             ("Rate $2\\times10^{-5}$", "lr2e5"),
                             ("Rate $5\\times10^{-5}$", "lr5e5"),
                             ("Rate $2\\times10^{-4}$", "lr2e4"),
                             ("Rate $3\\times10^{-4}$", "lr3e4")]),
        ("Adapted projections", [("Attention only", "attn"),
                                 ("Feed forward only", "mlp"),
                                 ("Attention and feed forward", "r2")]),
        ("Tokens read per article", [("64 tokens", "len64"),
                                     ("128 tokens", "len128"),
                                     ("256 tokens", "r2"),
                                     ("512 tokens", "len512"),
                                     ("1024 tokens", "len1024")]),
        ("Random seed", [("Seed 0", "r2"), ("Seed 1", "seed1"),
                         ("Seed 2", "seed2"), ("Seed 3", "seed3"),
                         ("Seed 4", "seed4"), ("Seed 5", "seed5"),
                         ("Seed 6", "seed6")]),
        ("Control: shuffled adaptation text",
         [("At the reported rank", "shuf2"), ("At rank 8", "shuffled")]),
        ("Ceiling: eligibility ignored",
         [("Text past the date", "oracle")])]


# What the main text keeps, in the order a sceptic would attack the result.
# The control comes first in importance, since it is the only entry that can
# falsify the claim rather than qualify it. Everything the full table carries is
# in the appendix, so nothing is lost by leaving it out here.
SENS_MAIN = [
    ("Training budget", [("200 steps", "steps200"),
                         ("600 steps, the reported setting", "r2"),
                         ("1200 steps", "steps1200"),
                         ("Rate $2\\times10^{-5}$", "lr2e5")]),
    ("Decision date", [("2019", "d2019"), ("2022", "d2022")]),
    ("Share of eligible text", [("10\\%", "frac10")]),
    ("Adapted projections", [("Attention only", "attn"),
                             ("Feed forward only", "mlp")]),
]

# Neither of these is a setting we would ship. One destroys the text the update
# reads and the other lets it read text the eligibility rule forbids, so they
# bound the result from below and from above rather than varying it.
BOUNDS = [
    ("The adaptation text is destroyed",
     [("Scrambled, at rank 2", "shuf2"),
      ("Scrambled, at rank 8", "shuffled")]),
    ("The eligibility rule is ignored",
     [("Text past the date", "oracle")]),
]


def emit_sens(groups, ncol=7):
    """Every measure carries its own win share, so the columns are Gain and Wins
    three times over rather than one Wins standing for all three."""
    for grp, items in groups:
        print(f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{{grp}}}}} \\\\")
        print(f"\\cmidrule(lr){{1-{ncol}}}")
        for name, v in items:
            g = P[P.variant == v]
            if len(g) < 5:
                print(f"{name}" + " & \\tbd" * (ncol - 1) + " \\\\"
                      f"   % {len(g)} paired so far")
                continue
            print(f"{name} & {gain(g,'ic_a','ic_b',1e3)} & {share(g,'ic_a','ic_b')} & "
                  f"{gain(g,'ret_a','ret_b',100)} & {share(g,'ret_a','ret_b')} & "
                  f"{gain(g,'sh_a','sh_b',1)} & {share(g,'sh_a','sh_b')} \\\\")


# Four questions that used to share one table. Each now carries its own, next
# to the paragraph that asks it, because a reader looking for the budget should
# not have to find it inside a block about decision dates.
SPLIT = {
    "budget": [("200 steps", "steps200"), ("600 steps", "r2"),
               ("1200 steps", "steps1200"), ("Rate $2\\times10^{-5}$", "lr2e5")],
    "date":   [("2019", "d2019"), ("2021", "r2"), ("2022", "d2022")],
    "proj":   [("Attention only", "attn"), ("Feed forward only", "mlp"),
               ("Attention and feed forward", "r2")],
    "text":   [("10\\%", "frac10"), ("25\\%", "frac25"),
               ("50\\%", "frac50"), ("100\\%", "r2")],
}


def split():
    """Every row of a table averages the same checkpoints.

    A variant that failed on one vintage would otherwise enter its table over a
    slightly different pool than the row above it, which turns a comparison
    between settings into a comparison between samples. The decision date is the
    exception, since moving that date is meant to move the pool."""
    for key, items in SPLIT.items():
        print(f"\n% ---- tab:{key}")
        keep = None
        if key != "date":
            keep = set.intersection(*[set(P[P.variant == v].model) for _, v in items])
        for name, v in items:
            g = P[P.variant == v]
            if keep is not None:
                g = g[g.model.isin(keep)]
            if len(g) < 5:
                print(f"{name} & \\tbd & \\tbd & \\tbd & \\tbd \\\\")
                continue
            print(f"{name} & {gain(g,'ic_a','ic_b',1e3)} & {share(g,'ic_a','ic_b')} & "
                  f"{gain(g,'ret_a','ret_b',100)} & {share(g,'ret_a','ret_b')} & "
                  f"{gain(g,'sh_a','sh_b',1)} & {share(g,'sh_a','sh_b')} \\\\")


def sens_main():
    print("\n% ---- tab:sens (main text subset)")
    emit_sens(SENS_MAIN)


def bounds():
    print("\n% ---- tab:bounds")
    emit_sens(BOUNDS)


def sens():
    """Five columns, not seven.

    The wins on the return and on the Sharpe ratio agree with each other to
    within a checkpoint on almost every row, because both read the sign of the
    same daily series, so carrying both spent a third of the table's width on a
    column that repeats its neighbour."""
    print("\n% ---- tab:sens")
    for grp, items in SENS:
        print(f"\\multicolumn{{5}}{{l}}{{\\textit{{{grp}}}}} \\\\")
        print("\\cmidrule(lr){1-5}")
        for name, v in items:
            g = P[P.variant == v]
            if len(g) < 5:
                print(f"{name} & \\tbd & \\tbd & \\tbd & \\tbd \\\\"
                      f"   % {len(g)} paired so far")
                continue
            print(f"{name} & {gain(g,'ic_a','ic_b',1e3)} & {share(g,'ic_a','ic_b')} & "
                  f"{gain(g,'ret_a','ret_b',100)} & {gain(g,'sh_a','sh_b',1)} \\\\")


# ------------------------------------- the held-out suite, analysed on its own
def held():
    print(f"\n% ---- {HELD}, the continuously pretrained suite")
    h = D[D.suite == HELD]
    r2 = h[h.variant == "r2"].sort_values("cut")
    sl = smf.ols("ic_b ~ cut", data=r2).fit()
    span = r2.cut.max() - r2.cut.min()
    print(f"  vintages {len(r2)}  cutoffs {r2.cut.min():.0f}-{r2.cut.max():.0f}")
    print(f"  run slope {sl.params['cut']*1e3:+.2f}/yr  p={sl.pvalues['cut']:.2f}"
          f"  over {span:.0f} yr = {sl.params['cut']*1e3*span:+.2f}")
    print(f"  spread across its own vintages {r2.ic_b.std()*1e3:.2f}")
    for v in ["r1", "r2", "r4", "r8", "r16"]:
        g = h[h.variant == v]
        if len(g):
            print(f"  PALM {v:4s} {gain(g,'ic_a','ic_b',1e3)}  {wins(g,'ic_a','ic_b')}")
    old, new = r2.iloc[0], r2.iloc[-1]
    for v in ["r2", "r8", "r16"]:
        a = h[(h.variant == v) & (h.model == old.model)]
        if len(a):
            print(f"  oldest {old.cut:.0f} + PALM {v:3s} {a.ic_a.iloc[0]*1e3:5.2f}"
                  f"   vs newest {new.cut:.0f} alone {new.ic_b*1e3:5.2f}")


if __name__ == "__main__":
    print(f"pool: {P[P.variant=='r2'].shape[0]} paired checkpoints over "
          f"{P[P.variant=='r2'].suite.nunique()} suites, {HELD} held out")
    staleness(); effectiveness(); ranks(); width(); horizon()
    sens_main(); bounds(); split(); sens(); held()
