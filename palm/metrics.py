#!/usr/bin/env python3
"""All three reported metrics, for every variant, in one pass.

The paper reports an information coefficient, an annualized long-short return
and a Sharpe ratio, but only the headline table carried all three. There was no
reason for that beyond table width, so this computes the full set once and
writes it out, and the tables read it back.

For a scored panel the three follow from the same daily series. The IC is the
cross-sectional Spearman correlation with the realized return on each date. The
portfolio takes the sign of the score demeaned within the date, so the return is
the equally weighted long-short return, and the Sharpe ratio annualizes it."""
import glob
import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from paths import ROOT, DATA, RESULTS, BIG, FIGS, PANEL, METRICS

OUT = os.environ.get("PIT_OUT", METRICS)
# t encodes December as the next year, so a window opened at ev still holds
# that December, which a vintage cut at its end has already read. EDGE drops it.
EDGE = float(os.environ.get("PIT_EDGE", "0")) / 12.0

import main_figure as MF

PAN = pd.read_parquet(PANEL,
                      columns=["aid", "date", "fwd_ret"])
PAN["t"] = pd.to_datetime(PAN.date).dt.year + pd.to_datetime(PAN.date).dt.month / 12.0

# Which arms to read, and where each one lives. Keeping this in a file rather
# than in the source means a new arm is a line of text, and means the grid can
# list arms that have not been run yet without breaking the pass.
ARMS_FILE = os.environ.get(
    "PALM_ARMS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "configs", "arms.txt"))


def read_arms(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            name, sub, ev = line.split()
            out[name] = (sub, float(ev))
    return out


VARIANTS = read_arms(ARMS_FILE)


def three(path, ev):
    """Daily IC and daily long-short return for one scored panel.

    A worker killed mid-write leaves a parquet that cannot be opened, and one
    such file used to end the whole pass. The file is reported and skipped
    instead, so a single bad write costs one checkpoint rather than the run."""
    try:
        d = pd.read_parquet(path, columns=["aid", "score"])
    except Exception as e:
        print(f"    unreadable, skipping: {path}  ({str(e)[:50]})", flush=True)
        return None
    if d.score.std() < 1e-9:
        return None
    d = d.merge(PAN, on="aid")
    d = d[d.t >= ev + EDGE]
    if len(d) < 5000:
        return None
    d["w"] = np.sign(d.score - d.groupby("date").score.transform("mean"))
    ic = d.groupby("date").apply(
        lambda g: g.score.corr(g.fwd_ret, method="spearman")
        if len(g) >= 8 and g.score.nunique() >= 4 else np.nan).dropna()
    r = d.groupby("date").apply(
        lambda g: (g.w * g.fwd_ret).sum() / np.abs(g.w).sum()
        if np.abs(g.w).sum() > 0 else np.nan).dropna()
    return (ic, r) if len(ic) >= 150 else None


def collect(sub, ev):
    out = {}
    for f in glob.glob(f"{BIG}/{sub}/*.parquet"):
        m = os.path.basename(f)[:-8]
        c = MF.cutoff(m)
        if not MF.suite(m) or c is None or c > ev:
            continue
        v = three(f, ev)
        if v is not None:
            out[m] = v
    return out


if __name__ == "__main__":
    cache, rows = {}, []
    for name, (sub, ev) in VARIANTS.items():
        if not os.path.isdir(f"{BIG}/{sub}"):
            continue
        if ev not in cache:
            cache[ev] = collect("base_full", ev)
            print(f"  baselines at d={ev:.0f}: {len(cache[ev])}", flush=True)
        base = cache[ev]
        ad = base if name == "base" else collect(sub, ev)
        for m, (ia, ra) in ad.items():
            if m not in base:
                continue
            ib, rb = base[m]
            j = ib.index.intersection(ia.index)
            k = rb.index.intersection(ra.index)
            if len(j) < 150:
                continue
            rows.append(dict(
                variant=name, model=m, suite=MF.suite(m), ev=ev,
                stale=ev - MF.cutoff(m), n_days=len(j),
                ic_b=ib[j].mean(), ic_a=ia[j].mean(),
                ret_b=252 * rb[k].mean(), ret_a=252 * ra[k].mean(),
                sh_b=np.sqrt(252) * rb[k].mean() / rb[k].std(ddof=1),
                sh_a=np.sqrt(252) * ra[k].mean() / ra[k].std(ddof=1)))
        print(f"  {name:10s} {sub:16s} paired {sum(1 for x in rows if x['variant']==name)}",
              flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT} with {len(rows)} rows")
