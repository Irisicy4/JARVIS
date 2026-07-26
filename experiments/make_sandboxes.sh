#!/bin/bash
# Create per-run sandbox dirs for the Table-4 CI reruns.
# Each sandbox has its own result file, logs/ and configs/ (run_blink writes
# merged tmp configs there); public/images is a symlink to the main server
# dir's public/images so the shared models_server (which resolves relative
# paths against its own cwd) can read overlays drawn by any run. Collisions
# are prevented by the full-uuid artifact-name patch.
set -euo pipefail
ROOT=/raid/icy/jarvis-cathy
SERVER=$ROOT/hugginggpt/server
SB=$ROOT/experiments/sandboxes

for arm in singleround det_image_only det_image_and_text det_text_only; do
  for k in 1 2 3; do
    d=$SB/${arm}_r${k}
    mkdir -p "$d/logs" "$d/public/audios" "$d/public/videos" "$d/configs"
    for f in awesome_chat.py run_blink.py evaluate.py get_token_ids.py demos data models; do
      ln -sfn "$SERVER/$f" "$d/$f"
    done
    ln -sfn "$ROOT/experiments/cvbench_first100.jsonl" "$d/cvbench_first100.jsonl"
    ln -sfn /raid/cathy/dataset/CVBench "$d/public/CVBench"
    ln -sfn "$SERVER/public/examples" "$d/public/examples"
    ln -sfn "$SERVER/public/images" "$d/public/images"
    for c in config.default.yaml rerun_local.yaml det_image_only.yaml det_image_and_text.yaml det_text_only.yaml multiround.yaml; do
      cp "$SERVER/configs/$c" "$d/configs/$c"
    done
  done
done
echo "sandboxes ready under $SB"
