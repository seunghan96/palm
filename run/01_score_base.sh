#!/usr/bin/env bash
# Score every eligible checkpoint as released. This is the "Base" row of every
# table, and the arm every other arm is paired with.
#
# It goes through adapt.py with --no-train rather than through a separate
# scorer, so that a paired comparison holds the scored articles and their
# batching fixed. One forward pass per article, no sampling, so a checkpoint
# returns identical scores on identical input.
#
#   bash run/01_score_base.sh            the pooled 44
#   bash run/01_score_base.sh ChronoGPT  one suite only
#   ALL=1 bash run/01_score_base.sh      also the held-out and ineligible vintages
set -euo pipefail
cd "$(dirname "$0")/.."
ARM=base_test exec bash run/02_adapt.sh "${1:-}"
