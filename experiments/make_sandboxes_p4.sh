#!/bin/bash
# Sandboxes for the Phase-4 single-tool encoding sweeps (RUNBOOK §3).
# Differences vs make_sandboxes.sh: per-family input file; isolation arms
# get a PRIVATE data/ dir whose p0_models.jsonl registers ONLY the tool
# under test (dpt-large for depth, detr-resnet-50-panoptic for seg).
set -euo pipefail
ROOT=/raid/icy/jarvis-cathy
SERVER=$ROOT/hugginggpt/server
SB=$ROOT/experiments/sandboxes

DEPTH_ARMS="depth_gray depth_plasma depth_turbo depth_stock"
SEG_ARMS="seg_overlay_base seg_opacity100 seg_color_by_instance seg_contour_only seg_polygon_text seg_mask_only seg_stock"
REFSEG_ARMS="refseg_overlay_base refseg_opacity100 refseg_color_by_instance refseg_contour_only refseg_polygon_text refseg_mask_only refseg_stock"

mk() { # arm k input_file p0_variant(full|depth|seg)
  local arm=$1 k=$2 input=$3 p0=$4
  local d=$SB/${arm}_r${k}
  mkdir -p "$d/logs" "$d/public/audios" "$d/public/videos" "$d/configs" "$d/data"
  for f in awesome_chat.py run_blink.py evaluate.py get_token_ids.py demos models; do
    ln -sfn "$SERVER/$f" "$d/$f"
  done
  ln -sfn "$ROOT/experiments/$input" "$d/$input"
  ln -sfn /raid/cathy/dataset/CVBench "$d/public/CVBench"
  ln -sfn "$SERVER/public/examples" "$d/public/examples"
  ln -sfn "$SERVER/public/images" "$d/public/images"
  cp "$SERVER"/configs/*.yaml "$d/configs/" 2>/dev/null || true
  case $p0 in
    depth) ln -sfn "$SERVER/data/p0_models.depth_only.jsonl" "$d/data/p0_models.jsonl";;
    seg)   ln -sfn "$SERVER/data/p0_models.seg_only.jsonl"   "$d/data/p0_models.jsonl";;
    full)  ln -sfn "$SERVER/data/p0_models.jsonl"            "$d/data/p0_models.jsonl";;
  esac
}

for k in 1 2 3; do
  for arm in $DEPTH_ARMS; do
    p0=depth; [ "$arm" = depth_stock ] && p0=full
    mk "$arm" "$k" cvbench_depth100.jsonl "$p0"
  done
  for arm in $SEG_ARMS; do
    p0=seg; [ "$arm" = seg_stock ] && p0=full
    mk "$arm" "$k" cvbench_count100.jsonl "$p0"
  done
  for arm in $REFSEG_ARMS; do
    p0=seg; [ "$arm" = refseg_stock ] && p0=full
    mk "$arm" "$k" cvbench_relation100.jsonl "$p0"
  done
done
echo "P4 sandboxes ready"
