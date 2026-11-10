#!/usr/bin/env bash
# A one-minute check that the tree is wired correctly.
#
# Confirms that the configuration parses, that the shipped list of
# article-days is readable, and which inputs and arms are present. Run it
# after cloning, before spending GPU hours.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

python3 - <<'PY'
import os, sys
sys.path.insert(0, "palm")
import pandas as pd
from paths import ROOT, DATA, BIG, CACHE, PANEL, PANEL_IDS, SHIFT
print(f"root        {ROOT}")
ids = pd.read_parquet(PANEL_IDS)
print(f"panel ids   {len(ids):,} article-days, {ids.symbol.nunique():,} tickers, "
      f"{ids.date.min().date()} to {ids.date.max().date()}")
for name, p in (("news csv", f"{DATA}/nasdaq_exteral_data.csv"), ("prices", f"{DATA}/full_history.zip"),
                ("panel", PANEL), ("targets", SHIFT)):
    print(f"{name:11s} {'found' if os.path.exists(p) else 'MISSING'}  {p}")
ck = [l.split() for l in open("configs/checkpoints.txt") if l.split("#")[0].strip()]
print(f"checkpoints {len(ck)} listed, {sum(1 for c in ck if c[3] == 'yes')} eligible, "
      f"{sum(1 for c in ck if c[4] == 'yes')} in the validation pool")
st = [l.split() for l in open("configs/settings.txt") if l.split("#")[0].strip()]
print("settings    " + "; ".join(f"{s[0]} r{s[1]}/{s[2]} steps" for s in st))
arms = [l.split()[0] for l in open("configs/arms.txt") if l.split("#")[0].strip()]
have = {a: len([f for f in os.listdir(os.path.join(BIG, a)) if f.endswith(".parquet")])
        for a in arms if os.path.isdir(os.path.join(BIG, a))}
print(f"arms        {len(have)} of {len(arms)} present: " +
      (", ".join(f"{a} ({n})" for a, n in have.items()) or "none"))
PY
