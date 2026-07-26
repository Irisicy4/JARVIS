#!/bin/bash
# Run one CI repeat in its sandbox: run_ci.sh <arm> <repeat>
# Faithful to the published conditions: baseline = default config
# (single-round, legacy det encoding); det arms = det_encoding layer +
# multiround (the published runs had multiround on for det arms — see
# judge-exp-hugginggpt.md §1). rerun_local.yaml only redirects the tool
# server port.
set -euo pipefail
ARM=$1; K=$2
ROOT=/raid/icy/jarvis-cathy
PY=/raid/cathy/miniconda3/envs/jarvis/bin/python
# /tmp/data-gym-cache is owned by another user on this box
export TIKTOKEN_CACHE_DIR=/raid/icy/iris/.cache/tiktoken
D=$ROOT/experiments/sandboxes/${ARM}_r${K}
CFGS=(--config configs/config.default.yaml --config configs/rerun_local.yaml)
case $ARM in
  singleround) ;;
  det_image_only|det_image_and_text|det_text_only)
    CFGS+=(--config configs/${ARM}.yaml --config configs/multiround.yaml)
    ;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac
cd "$D"
exec "$PY" run_blink.py cvbench_first100.jsonl result.json "${CFGS[@]}"
