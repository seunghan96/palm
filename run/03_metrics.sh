#!/usr/bin/env bash
# Turn the scored panels into one row per (checkpoint, arm).
#
# For each arm this computes the information coefficient, the annualized return
# of the long-short book, and its Sharpe ratio, all on the days that arm and
# the base arm both scored. The result is results/metrics.csv, which every
# table and figure reads.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT
python3 palm/metrics.py
echo "wrote $PALM_ROOT/results/metrics.csv"
