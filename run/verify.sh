#!/usr/bin/env bash
# A one-minute check that the tree is wired correctly.
#
# Confirms that the arm list parses, that the panel is readable, and that the
# table emitter produces the headline row. Run it after cloning, before
# spending GPU hours.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

python3 - <<'PY'
import os, sys
sys.path.insert(0, "palm")
from paths import ROOT, DATA, RESULTS, BIG, PANEL, METRICS
print(f"root      {ROOT}")
for name, p in (("panel", PANEL), ("metrics", METRICS)):
    print(f"{name:9s} {'found' if os.path.exists(p) else 'MISSING'}  {p}")
arms = [l.split()[0] for l in open("configs/arms.txt")
        if l.split("#")[0].strip()]
have = [a for a in arms
        if os.path.isdir(os.path.join(BIG, dict(
            (l.split()[0], l.split()[1]) for l in open("configs/arms.txt")
            if l.split("#")[0].strip())[a]))]
print(f"arms      {len(have)} of {len(arms)} present on disk")
ck = [l for l in open("configs/checkpoints.txt") if l.split("#")[0].strip()]
print(f"checkpoints {len(ck)} listed, {sum(1 for l in ck if l.split()[3]=='yes')} eligible")
PY

if [ -f "$PALM_ROOT/results/metrics.csv" ]; then
  echo "--- headline row ---"
  python3 palm/paper_tables.py 2>/dev/null | sed -n '/tab:adapt/,/^$/p' | tail -3
fi
