#!/usr/bin/env bash
# Turn the scored panels into the daily series every number is computed from.
#
# For each (arm, checkpoint) this writes one row per date: the information
# coefficient, the long-short return and the turnover, under
# results/cache/<arm>@oc1/. The primary target is open(D+1) -> close(D+1).
# The main arm and the base arm are also cached under the three other
# alignments (the alignment table) and with the two alternative
# cross-sections (the robustness table). Reads no model; minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT
: "${MAIN:=sel_ps}"
BIG="$PALM_ROOT/results/big"

arms=()
for d in "$BIG"/*/; do [ -d "$d" ] && arms+=("$d"); done
[ ${#arms[@]} -gt 0 ] || { echo "nothing under $BIG" >&2; exit 1; }

ALIGN=oc1 python3 palm/daily.py "${arms[@]}"
for al in "" on1 lag1; do
  ALIGN=$al python3 palm/daily.py "$BIG/$MAIN" "$BIG/base_test"
done
for agg in firm dedup; do
  ALIGN=oc1 AGG=$agg python3 palm/daily.py "$BIG/$MAIN" "$BIG/base_test"
done

if ls "$BIG" | grep -q '^val_r'; then
  python3 palm/select_config.py
fi
python3 palm/audit.py > /dev/null && echo "wrote $PALM_ROOT/results/audit.json"
echo "daily series under $PALM_ROOT/results/cache"
