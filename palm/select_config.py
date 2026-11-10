#!/usr/bin/env python3
"""Pick the adapter setting on the validation window only.

Validation: decision date end of Nov 2017 (t = 2018.0), window Dec 2017 to
Nov 2020, which ends where the test window begins. Target: open(D+1) ->
close(D+1). Rule, fixed before any validation result existed: the setting
with the largest pooled IC gain over the validation checkpoints; a tie
(difference < 0.05e-3) goes to the smaller rank, then to fewer steps. The
paper applies the rule once per backbone (results/selection_by_suite.csv);
the pooled pick is also written (results/selection.csv).

    python3 palm/select_config.py"""
import os, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("NBOOT", "1000")
from paths import CACHE, RESULTS
import stats

ALIGN = "@oc1"


def pick(g):
    best = g.d_ic.max()
    return g[g.d_ic >= best - 0.05].sort_values(["rank", "steps"]).iloc[0]


pool, per = [], []
for d in sorted(os.listdir(CACHE)):
    m = re.fullmatch(r"val_r(\d+)_s(\d+)_lr([0-9e.-]+)" + ALIGN, d)
    if not m:
        continue
    arm = d.replace(ALIGN, "")
    cfg = dict(arm=arm, rank=int(m.group(1)), steps=int(m.group(2)), lr=float(m.group(3)))
    try:
        r = stats.paired(d, "val_base" + ALIGN, ev=2018.0, ev_to=2021.0)
    except Exception as e:
        print("skip", d, e); continue
    r.pop("by_suite")
    pool.append(dict(family="pool", **cfg, **r))
    for s in stats.THREE:
        try:
            rs = stats.paired(d, "val_base" + ALIGN, ev=2018.0, ev_to=2021.0, suites=(s,))
        except Exception as e:
            continue
        rs.pop("by_suite")
        per.append(dict(suite=s, **cfg, n=rs["n_ckpt"], **{k: v for k, v in rs.items() if k != "n_ckpt"}))

if not pool:
    sys.exit("no validation arms under results/cache (run: bash run/02_adapt.sh val)")
pool = pd.DataFrame(pool)
pool.to_csv(f"{RESULTS}/selection.csv", index=False)
p = pick(pool)
print(f"[pool] n_ckpt={int(pool.n_ckpt.max())}")
print(pool[["arm", "n_ckpt", "ic_b", "ic_a", "d_ic", "d_ic_lo95", "d_ic_hi95", "wins_ic"]].round(3).to_string(index=False))
print(f"SELECTED pool: {p.arm}  (rank {p['rank']}, steps {p.steps}, lr {p.lr})")

per = pd.DataFrame(per)
per.to_csv(f"{RESULTS}/selection_by_suite.csv", index=False)
print("\nper backbone:")
for s, g in per.groupby("suite"):
    q = pick(g)
    print(f"  {s:20s} n={int(g.n.max())}  rank {int(q['rank'])}, steps {int(q.steps)}, lr {q.lr}  (gain {q.d_ic:+.2f})")
