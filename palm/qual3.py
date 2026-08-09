#!/usr/bin/env python3
"""Three qualitative views, for the claims that a four row table cannot show.

Two of the claims are shapes rather than numbers.
The gain accrues through the window instead of arriving in one episode, it does
not fade as the base checkpoint ages, and it lands on the material a vintage
could not have read. Each is drawn here.

  g1  the daily gain accumulated over the evaluation window
  g3  the vocabulary the update moves, read off one checkpoint's scored panel
"""
import glob
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS

RES = RESULTS
BIG = f"{RES}/big"
COL, GREEN, BLUE, RED, GREY = 3.33, "#2c7a4b", "#2c5f9e", "#c0504d", "#8c8c8c"

import main_figure as MF

# Words that carry no subject matter, plus the wire boilerplate that appears in
# most headlines. Without this the ranking is a frequency list.
STOP = set("""the a an and or of to in for on at by with is are was were be been being as from that
this these those it its it's has have had will would can could may might should must do does did not
no more most less least up down over under after before new news stock stocks share shares market
markets said says say report reports says year years day days week weeks month months time company
companies inc corp ltd co group holdings you your we our they their he she his her but if than then
into out about against between during through above below off again once here there when where why
how all any both each few other some such only own same so too very just now also one two three
four five first second next last top best worst high low higher lower buy sell hold rating price
target update updates via amid ahead vs percent million billion trillion quarter""".split())

# Terms that pass every statistical filter and still say nothing about subject
# matter. They are relational or quantitative words that ride along with the
# style of a document rather than with what it is about, and leaving them in
# turns a claim about vocabulary into a list the reader cannot interpret.
STOP |= set("""dedicated relates comparative weight actual duration crossing crowds
assessing remarks implying dates forecasted studying possessing disclosing breakdown
maintains maintained relating relative respective dedicated""".split())


def save(fig, name):
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


# ------------------------------------------------------------------------ g1
def g1():
    """The daily difference, accumulated. A steady climb says the gain is not
    one episode, which a table of four calendar years cannot distinguish from
    a single good quarter."""
    p = f"{RES}/daily_pairs.parquet"
    if not os.path.exists(p):
        base, adapted = MF.collect("base"), MF.collect("palm_r2")
        rows = []
        for m, av in adapted.items():
            if m not in base:
                continue
            bv = base[m]
            for t in bv.index.intersection(av.index):
                rows.append({"model": m, "date": t, "alone": bv[t], "palm": av[t]})
        pd.DataFrame(rows).to_parquet(p)
        print(f"    built {os.path.basename(p)} from the scored panels")
    d = pd.read_parquet(p)
    p = d.groupby("date")[["alone", "palm"]].mean()
    p.index = pd.to_datetime(p.index)
    p = p.sort_index()
    cum = (p.palm - p.alone).cumsum()

    fig, a = plt.subplots(figsize=(COL, 1.9))
    a.fill_between(cum.index, 0, cum.values, color=GREEN, alpha=0.20)
    a.plot(cum.index, cum.values, color=GREEN, lw=1.5)
    a.axhline(0, color="#555555", lw=0.9)
    a.set_ylabel("Cumulative Gain in IC", fontsize=7.4, labelpad=2)
    a.xaxis.set_major_locator(mdates.YearLocator())
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.tick_params(labelsize=6.8, length=2)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.text(0.5, 1.03, "The Gain Accrues Through the Window",
           transform=a.transAxes, ha="center", va="bottom", fontsize=8.0)
    save(fig, "g1_cumulative")


# ------------------------------------------------------------------------ g3
CUTS = ("2013", "2015", "2017", "2019")


def g3(k=9, mindf=250, ratio=2.0):
    """Which vocabulary the update moves, kept to the words it could not know.

    Two things make a word ranking on one checkpoint hard to trust. A term can
    move by chance, and a term that moves may be ordinary language rather than
    anything the vintage was missing. So a word has to move the same way on
    every checkpoint we look at, and it has to be a word whose use in the panel
    rose sharply after that checkpoint's cutoff. What is left is vocabulary the
    vintage demonstrably could not have read, which is the claim being made."""
    panel = pd.read_parquet(PANEL,
                            columns=["aid", "text", "date"])
    panel["y"] = pd.to_datetime(panel.date).dt.year

    per, freq = {}, {}
    for cut in CUTS:
        hits = [x for x in glob.glob(f"{BIG}/palm_r2/*.parquet")
                if f"chrono-gpt-v1-{cut}" in x]
        if not hits:
            continue
        f = hits[0]
        m = os.path.basename(f)
        if not os.path.exists(f"{BIG}/base/{m}"):
            continue
        a = pd.read_parquet(f, columns=["aid", "score"])
        b = pd.read_parquet(f"{BIG}/base/{m}", columns=["aid", "score"])
        d = a.merge(b, on="aid", suffixes=("_a", "_b")).merge(panel, on="aid")
        d = d[d.y >= 2021]
        for c in ("score_a", "score_b"):
            d[c] = (d[c] - d[c].mean()) / d[c].std()
        acc = {}
        for t, sh in zip(d.text.values, (d.score_a - d.score_b).values):
            for w in set(re.findall(r"[a-z]{3,}", str(t).lower())):
                if w not in STOP:
                    acc.setdefault(w, []).append(sh)
        per[cut] = {w: float(np.mean(v)) for w, v in acc.items() if len(v) >= mindf}
        print(f"    {cut}: {len(per[cut])} terms above the document floor")

        # how much more often the word is used after that cutoff than before,
        # measured on the panel rather than assumed
        y = int(cut)
        pre, post = panel[panel.y <= y], panel[panel.y > y]
        if len(pre) and len(post):
            def share(fr):
                c = {}
                for t in fr.text.values:
                    for w in set(re.findall(r"[a-z]{3,}", str(t).lower())):
                        c[w] = c.get(w, 0) + 1
                return {w: v / len(fr) for w, v in c.items()}
            a_, b_ = share(pre), share(post)
            freq[cut] = {w: (b_.get(w, 0.0) + 1e-6) / (a_.get(w, 0.0) + 1e-6)
                         for w in per[cut]}

    common = set.intersection(*[set(v) for v in per.values()])
    rows = []
    for w in common:
        v = [per[c][w] for c in per]
        if min(v) > 0 or max(v) < 0:                 # same sign everywhere
            r = np.mean([freq[c].get(w, 1.0) for c in freq if w in freq[c]])
            rows.append((w, float(np.mean(v)), float(r)))
    st = pd.DataFrame(rows, columns=["word", "mean", "ratio"])
    kept = st[st.ratio >= ratio].sort_values("mean")
    print(f"    {len(common)} shared terms, {len(st)} move the same way on every "
          f"checkpoint, {len(kept)} of those also rose {ratio:g}x after the cutoff")
    st = kept
    sel = pd.concat([st.head(k), st.tail(k)])

    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(sel))
    c = [GREEN if v > 0 else RED for v in sel["mean"]]
    ax.barh(y, sel["mean"], height=0.68, color=c, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sel.word, fontsize=6.4)
    ax.axvline(0, color="#555555", lw=0.9)
    ax.set_xlabel("Mean Shift in the Score (Std. Dev.)", fontsize=7.4, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.8, length=2)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.text(0.5, 1.105, "The Vocabulary the Update Moves", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8.0)
    ax.text(0.5, 1.020, "Terms whose use rose after the cutoff, on four checkpoints",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=6.2,
            color="#5a5a5a")
    save(fig, "g3_words")
    print(sel.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    g1(); g3()
