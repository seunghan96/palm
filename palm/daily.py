#!/usr/bin/env python3
"""Scored panel -> one row per date: IC, long-short return, turnover.

Every number the paper reports is a function of these daily series, so they
are computed once per (arm, checkpoint) and cached. The conventions:
  - the cross-section is the article, not the firm
  - a date counts only with >= 8 articles and >= 4 distinct scores
  - position = sign(score - mean score that date), return = sum(w r)/sum|w|
  - turnover: the book is netted to one weight per ticker and date
    (w_sym = sum of its article signs / sum|w|), and turnover is
    sum_sym |w_sym,t - w_sym,t-1|, i.e. two-sided, so a one-way cost of c
    per unit traded costs c * turnover that day.

Two environment variables pick the target and the cross-section:
  ALIGN=oc1   open(D+1) -> close(D+1), the paper's primary target
  ALIGN=on1   close(D)  -> open(D+1), the overnight gap
  ALIGN=lag1  close(D+1) -> close(D+2), one session later
  (unset)     close(D)  -> close(D+1), the panel's own column
  AGG=firm    average the scores of one ticker on one date before ranking
  AGG=dedup   drop texts repeated verbatim on the same date, keeping the first

    ALIGN=oc1 python3 palm/daily.py results/big/<arm> [...]
writes results/cache/<arm>[@align][@agg]/<model>.parquet."""
import glob, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PANEL, SHIFT, CACHE

PAN = pd.read_parquet(PANEL, columns=["aid", "date", "symbol", "fwd_ret"])
ALIGN = os.environ.get("ALIGN", "")
if ALIGN in ("lag1", "on1", "oc1"):
    sh = pd.read_parquet(SHIFT)[["aid", f"fwd_ret_{ALIGN}"]]
    PAN = PAN.drop(columns="fwd_ret").merge(
        sh.rename(columns={f"fwd_ret_{ALIGN}": "fwd_ret"}), on="aid").dropna()
elif ALIGN:
    sys.exit(f"unknown ALIGN {ALIGN}")

AGG = os.environ.get("AGG", "")
if AGG == "dedup":
    _t = pd.read_parquet(PANEL, columns=["aid", "date", "text"])
    _t["k"] = _t.text.str.strip().str.lower()
    KEEP = set(_t.loc[~_t.duplicated(["date", "k"]), "aid"])
elif AGG and AGG != "firm":
    sys.exit(f"unknown AGG {AGG}")


def series(path):
    d = pd.read_parquet(path, columns=["aid", "score"])
    if d.score.std() < 1e-9:
        return None
    d = d.merge(PAN, on="aid")
    if AGG == "dedup":
        d = d[d.aid.isin(KEEP)]
    elif AGG == "firm":
        d = d.groupby(["date", "symbol"], as_index=False).agg(score=("score", "mean"),
                                                             fwd_ret=("fwd_ret", "first"))
    g = d.groupby("date")
    ok = (g.score.transform("size") >= 8) & (g.score.transform("nunique") >= 4)
    d = d[ok].copy()
    d["w"] = np.sign(d.score - d.groupby("date").score.transform("mean"))
    rows = []
    prev = None
    for dt, x in d.groupby("date", sort=True):
        ic = x.score.corr(x.fwd_ret, method="spearman")
        aw = np.abs(x.w).sum()
        r = (x.w * x.fwd_ret).sum() / aw if aw > 0 else np.nan
        book = (x.groupby("symbol").w.sum() / aw) if aw > 0 else None
        if book is not None and prev is not None:
            u = book.index.union(prev.index)
            to = (book.reindex(u, fill_value=0) - prev.reindex(u, fill_value=0)).abs().sum()
        else:
            to = np.nan
        prev = book
        rows.append((dt, ic, r, to, len(x), int((x.w > 0).sum()), int((x.w < 0).sum())))
    return pd.DataFrame(rows, columns=["date", "ic", "ret", "turnover", "n", "n_long", "n_short"])


def cache(path, arm):
    m = os.path.basename(path)[:-8]
    tag = "".join("@" + x for x in (ALIGN, AGG) if x)
    out = f"{CACHE}/{arm}{tag}/{m}.parquet"
    if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(path):
        return out
    s = series(path)
    if s is None:
        return None
    os.makedirs(os.path.dirname(out), exist_ok=True)
    s.to_parquet(out, index=False)
    return out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    jobs = []
    for d in sys.argv[1:]:
        arm = os.path.basename(d.rstrip("/"))
        for f in sorted(glob.glob(f"{d}/*.parquet")):
            jobs.append((f, arm))
    if not jobs:
        sys.exit("no scored panels under " + " ".join(sys.argv[1:]))
    n = int(os.environ.get("NPROC", min(16, os.cpu_count() or 1)))
    with ProcessPoolExecutor(n) as ex:
        for (f, arm), r in zip(jobs, ex.map(cache, *zip(*jobs))):
            print(("ok  " if r else "skip"), arm, os.path.basename(f), flush=True)
