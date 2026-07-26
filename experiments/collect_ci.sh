#!/bin/bash
# Copy finished sandbox results into the canonical ci_runs layout consumed
# by ci_aggregate.py / score_table4.py.
set -euo pipefail
ROOT=/raid/icy/jarvis-cathy
for arm in singleround det_image_only det_image_and_text det_text_only; do
  for k in 1 2 3; do
    src=$ROOT/experiments/sandboxes/${arm}_r${k}/result.json
    dst=$ROOT/experiments/results/table4-cvbench/$arm/ci_runs/repeat$k
    if [ -f "$src" ]; then
      mkdir -p "$dst"
      cp "$src" "$dst/result.json"
      n=$(python3 -c "import json;print(len(json.load(open('$src'))))")
      echo "${arm}_r${k}: $n items collected"
    else
      echo "${arm}_r${k}: no result yet"
    fi
  done
done
