#!/usr/bin/env bash
# Rebuild the paper's panel and its targets from the FNSPID release.
#
# Download these two files from FNSPID first and put them in data/:
#   nasdaq_exteral_data.csv   the news, with UTC timestamps (23 GB)
#   full_history.zip          the daily price history
#   https://huggingface.co/datasets/Zihan1004/FNSPID
#
# The paper scores 252,000 article-days, 25,200 per calendar year from 2014 to
# 2023. data/panel_ids.parquet ships with the repository and names every one of
# them (timestamp, ticker, SHA-1 of the text); restore_panel.py attaches the
# text from the FNSPID file and refuses to write if any row is missing. The
# second step derives the paper's primary target, open(D+1) -> close(D+1), and
# the other alignments from the price history.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

python3 palm/restore_panel.py --news "$PALM_ROOT/data/nasdaq_exteral_data.csv"
python3 palm/shift_returns.py
python3 palm/audit.py > /dev/null && echo "wrote $PALM_ROOT/results/audit.json"
