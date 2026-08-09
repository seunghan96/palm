#!/usr/bin/env bash
# Build the article-day panel from FNSPID.
#
# Download these two files from the FNSPID release first and put them in data/:
#   nasdaq_exteral_data.csv   the news, with UTC timestamps
#   full_history.zip          the daily price history
#
# An article is credited to a trading day only when its timestamp falls before
# that day's close, taken conservatively at 20:00 UTC. The target is the return
# of the following session, so no article is ever scored against a bar it could
# have moved.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT

python3 palm/build_panel.py \
  --news    "$PALM_ROOT/data/nasdaq_exteral_data.csv" \
  --prices  "$PALM_ROOT/data/full_history.zip" \
  --year-min 2014 --year-max 2023 \
  --top-symbols 1200 --max-per-symbol-year 60 \
  --seed 0 \
  --out "$PALM_ROOT/data/panel.parquet"
