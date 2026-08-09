#!/usr/bin/env bash
# Score every checkpoint without an adapter. This is the "-" arm of every table.
#
# One forward pass per article, no sampling, so a checkpoint returns identical
# scores on identical input. Roughly 20 minutes per checkpoint on one L40S for
# the 1.5B models and about an hour for PIT-4B.
#
#   bash run/01_score_base.sh            every eligible checkpoint
#   bash run/01_score_base.sh ChronoGPT  one suite only
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT
only="${1:-}"

grep -v '^#' configs/checkpoints.txt | grep -v '^$' | while read -r suite model cutoff eligible; do
  [ -n "$only" ] && [ "$suite" != "$only" ] && continue
  out="$PALM_ROOT/results/big/base/${model//\//_}.parquet"
  [ -f "$out" ] && { echo "have  $model"; continue; }
  echo "score $model"
  python3 palm/score_news.py \
    --model "$model" \
    --panel "$PALM_ROOT/data/panel.parquet" \
    --head cloze --batch-size 8 \
    --out "$out"
done
