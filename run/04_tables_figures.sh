#!/usr/bin/env bash
# Emit every number, table and figure the paper contains.
#
# Table bodies go to results/tables/*.tex, the macros the prose uses to
# results/numbers.tex, everything to results/report.json, and the main figure
# to figs/ws_main.pdf. Nothing here reads a model. NBOOT (default 2,000) is
# the number of bootstrap draws behind every interval.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

python3 palm/report.py
python3 palm/main_figure.py
