#!/usr/bin/env bash
# Run one arm of configs/arms.txt on every eligible checkpoint.
#
#   bash run/02_adapt.sh                     ARM=sel_ps, PALM at the reported setting
#   ARM=shuf_ps bash run/02_adapt.sh         any other arm of configs/arms.txt
#   ARM=sel_ps bash run/02_adapt.sh DatedGPT one suite only
#   bash run/02_adapt.sh val                 the validation grid (decision date 2018.0)
#   ALL=1 ARM=base_test bash run/02_adapt.sh also the held-out and ineligible vintages
#
# The per-backbone setting comes from configs/settings.txt. The adapter reads
# only text published after the checkpoint's cutoff and before the decision
# date, which is what --cutoff and --eval-from bound, and every pretrained
# weight stays frozen. Outputs already on disk are skipped, so a run is safe
# to interrupt and resume. SAVE_ADAPTERS=1 also keeps each trained adapter
# under results/adapters/<arm>/<model>/.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PALM_ROOT:=$PWD}"; export PALM_ROOT
: "${ARM:=sel_ps}" "${ALL:=}" "${SAVE_ADAPTERS:=}"
only="${1:-}"

setting() {   # <suite> -> "--rank R --steps S --lr LR"
  awk -v s="$1" '!/^#/ && NF==4 && $1==s {printf "--rank %s --steps %s --lr %s", $2, $3, $4}' configs/settings.txt
}
resources() {   # <suite> -> "<batch> <score-batch>"
  case "$1" in PIT-4B|DatedGPT) echo "2 4";; *) echo "4 8";; esac
}
run_one() {     # <suite> <model> <cutoff> <arm> <eval_from> <flags...>
  local suite=$1 model=$2 cutoff=$3 arm=$4 ev=$5; shift 5
  local out="$PALM_ROOT/results/big/$arm/${model//\//_}.parquet"
  if [ -f "$out" ] || [ -f "$out.nodata" ]; then echo "have  $arm $model"; return; fi
  read -r bs sb <<< "$(resources "$suite")"
  local save=()
  if [ -n "$SAVE_ADAPTERS" ]; then save=(--save-adapter "$PALM_ROOT/results/adapters/$arm/${model//\//_}"); fi
  echo "run   $arm $model  $*"
  python3 palm/adapt.py --model "$model" --cutoff "$cutoff" --eval-from "$ev" \
    --batch-size "$bs" --score-batch "$sb" --head cloze "$@" ${save[@]+"${save[@]}"} --out "$out"
}

if [ "$only" = "val" ]; then
  # the validation grid: 14 vintages, decision date 2018.0, scored on 2018.0 <= t < 2021.0
  grep -v '^#' configs/checkpoints.txt | grep -v '^$' | while read -r suite model cutoff eligible val; do
    [ "$val" = "yes" ] || continue
    run_one "$suite" "$model" "$cutoff" val_base 2018.0 --no-train --eval-to 2021.0
    for r in 1 2 4 8 16; do for s in 300 600 1200; do
      run_one "$suite" "$model" "$cutoff" "val_r${r}_s${s}_lr1e-4" 2018.0 --rank $r --steps $s --lr 1e-4 --eval-to 2021.0
    done; done
    for r in 2 4 8; do for lr in 5e-5 2e-4; do
      run_one "$suite" "$model" "$cutoff" "val_r${r}_s600_lr${lr}" 2018.0 --rank $r --steps 600 --lr $lr --eval-to 2021.0
    done; done
  done
  exit 0
fi

flags=$(awk -v a="$ARM" '!/^#/ && $1==a {$1=""; print substr($0, 2)}' configs/arms.txt)
[ -n "$flags" ] || { echo "arm $ARM is not in configs/arms.txt" >&2; exit 1; }

grep -v '^#' configs/checkpoints.txt | grep -v '^$' | while read -r suite model cutoff eligible val; do
  if [ -z "$ALL" ] && [ "$eligible" != "yes" ]; then continue; fi
  [ -n "$only" ] && [ "$suite" != "$only" ] && continue
  f="$flags"
  if [[ "$f" == *"{PS}"* ]]; then
    ps=$(setting "$suite")
    if [ -z "$ps" ]; then echo "skip  $ARM $model (no setting for $suite)"; continue; fi
    f="${f//\{PS\}/$ps}"
  fi
  if [[ "$f" == *"{DOM}"* ]]; then
    # the same length of text before the cutoff as PALM reads after it
    dom=$(python3 -c "c=$cutoff; lo=max(2014.0, c-(2021.0-c)); print(f'--fit-window {lo} {c}' if c>=2015.0 else '')")
    if [ -z "$dom" ]; then echo "skip  $ARM $model (no FNSPID before its cutoff)"; continue; fi
    f="${f//\{DOM\}/$dom}"
  fi
  # shellcheck disable=SC2086
  run_one "$suite" "$model" "$cutoff" "$ARM" 2021.0 $f
done
