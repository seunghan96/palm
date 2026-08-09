#!/usr/bin/env bash
# Emit every number and every figure the paper contains.
#
# The LaTeX rows go to stdout, grouped by the table label they belong to. The
# figures are written to figs/ as PDFs. Nothing here reads a model, so it runs
# on a laptop in under a minute once results/metrics.csv exists.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

echo "### tables"
python3 palm/paper_tables.py

echo "### figures"
python3 palm/main_figure.py     # staleness and the paired comparison
python3 palm/rank_curve.py      # the adapter capacity curve
python3 palm/qual3.py           # cumulative gain and the vocabulary the update moves
echo "wrote $PALM_ROOT/figs"
