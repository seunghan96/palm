#!/usr/bin/env bash
# Fit PALM on each eligible checkpoint and score the result.
#
# The adapter reads only text published after the checkpoint's cutoff and
# before the decision date, which is what --cutoff and --eval-from bound. Every
# pretrained weight stays frozen. Defaults here are the reported setting:
# rank 2, 600 steps, learning rate 1e-4, 256 tokens per article.
#
#   bash run/02_adapt.sh                 the reported setting
#   RANK=8 STEPS=1200 bash run/02_adapt.sh   a sensitivity arm
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT
: "${RANK:=2}" "${STEPS:=600}" "${LR:=1e-4}" "${EVAL_FROM:=2021.0}" "${ARM:=palm_r$RANK}"
only="${1:-}"

grep -v '^#' configs/checkpoints.txt | grep -v '^$' | while read -r suite model cutoff eligible; do
  [ "$eligible" = "yes" ] || continue
  [ -n "$only" ] && [ "$suite" != "$only" ] && continue
  out="$PALM_ROOT/results/big/$ARM/${model//\//_}.parquet"
  [ -f "$out" ] && { echo "have  $model"; continue; }
  echo "adapt $model  rank=$RANK steps=$STEPS lr=$LR"
  python3 palm/adapt.py \
    --model "$model" --panel "$PALM_ROOT/data/panel.parquet" \
    --cutoff "$cutoff" --eval-from "$EVAL_FROM" \
    --rank "$RANK" --steps "$STEPS" --lr "$LR" \
    --batch-size 4 --head cloze --score-batch 4 \
    --out "$out"
done
