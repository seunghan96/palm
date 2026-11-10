#!/usr/bin/env python3
"""The alternative targets each article-day can be scored against.

99.6% of FNSPID timestamps in the panel are exactly 00:00 UTC, so they carry a
date and no time. The panel's own target, close(D) to close(D+1), therefore
credits almost every article to its own date, and an article published after
the close of D could not have been traded at close(D). This script splits and
shifts that return without touching any model:

    fwd_ret_on1   close(D)   -> open(D+1)    the overnight gap
    fwd_ret_oc1   open(D+1)  -> close(D+1)   tradable at the next open; the
                                             paper's primary target
    fwd_ret_lag1  close(D+1) -> close(D+2)   one full session later

    python3 palm/shift_returns.py

reads data/panel.parquet and data/full_history.zip, writes
data/fwd_ret_shift.parquet (aid, fwd_ret_lag1, fwd_ret_on1, fwd_ret_oc1)."""
import os, sys, zipfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, PANEL, SHIFT

p = pd.read_parquet(PANEL, columns=["aid", "date", "symbol", "fwd_ret"])
syms = set(p.symbol)
z = zipfile.ZipFile(f"{DATA}/full_history.zip")
px = {}
for n in z.namelist():
    if not (n.startswith("full_history/") and n.endswith(".csv")):
        continue
    s = os.path.basename(n)[:-4].upper()
    if s not in syms:
        continue
    d = pd.read_csv(z.open(n), usecols=["date", "open", "close", "adj close"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna().sort_values("date")
    d = d[(d["adj close"] > 0) & (d["open"] > 0) & (d["close"] > 0)]
    px[s] = d.set_index("date")

out, chk = [], []
for s, g in p.groupby("symbol"):
    if s not in px:
        continue
    full = px[s]
    pr = full["adj close"]
    idx = pr.index
    pos = idx.searchsorted(g.date.values, side="left")      # D itself, if it traded
    ok = (pos + 2 < len(idx)) & (pos < len(idx))
    pos = np.where(ok, pos, 0)
    same = idx[pos] == g.date.values                          # D is a trading day
    p0, p1, p2 = pr.values[pos], pr.values[np.minimum(pos + 1, len(idx) - 1)], \
        pr.values[np.minimum(pos + 2, len(idx) - 1)]
    r1 = np.where(ok & same, p1 / p0 - 1, np.nan)             # the panel's own target
    r2 = np.where(ok & same, p2 / p1 - 1, np.nan)             # one session later
    # split close(D)->close(D+1) into the overnight gap and the D+1 session
    # (unadjusted prices within a day, which no split straddles)
    o1 = full["open"].values[np.minimum(pos + 1, len(idx) - 1)]
    c0 = full["close"].values[pos]
    c1 = full["close"].values[np.minimum(pos + 1, len(idx) - 1)]
    on = np.where(ok & same, o1 / c0 - 1, np.nan)
    oc = np.where(ok & same, c1 / o1 - 1, np.nan)
    out.append(pd.DataFrame({"aid": g.aid.values, "fwd_ret_lag1": r2,
                             "fwd_ret_on1": on, "fwd_ret_oc1": oc}))
    chk.append(pd.DataFrame({"aid": g.aid.values, "r1": r1, "fwd_ret": g.fwd_ret.values}))
o = pd.concat(out)
c = pd.concat(chk).dropna()
print(f"rows {len(o):,}, with a shifted return {o.fwd_ret_lag1.notna().mean():.1%}")
print(o[["fwd_ret_lag1", "fwd_ret_on1", "fwd_ret_oc1"]].describe().T[["count", "mean", "std"]])
print(f"rebuilt close-to-close return vs the panel's: corr {c.r1.corr(c.fwd_ret):.4f}, "
      f"max abs diff {np.abs(c.r1 - c.fwd_ret).max():.2e}")
o.to_parquet(SHIFT, index=False)
print("wrote", SHIFT)
