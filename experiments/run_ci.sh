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
INPUT=cvbench_first100.jsonl
CFGS=(--config configs/config.default.yaml --config configs/rerun_local.yaml)
case $ARM in
  singleround) ;;
  det_image_only|det_image_and_text|det_text_only)
    CFGS+=(--config configs/${ARM}.yaml --config configs/multiround.yaml)
    ;;
  # De-confound arms (RUNBOOK §3.1): encoding without multiround (D1-D3)
  sr_det_image_only|sr_det_image_and_text|sr_det_text_only)
    CFGS+=(--config configs/${ARM#sr_}.yaml)
    ;;
  # D4: multiround without det_encoding
  mr_baseline)
    CFGS+=(--config configs/multiround.yaml)
    ;;
  # Cleaned text encoding (audit follow-up): convention header, top-5,
  # 2-decimal coords + confidence, raw pixel dump suppressed
  det_text_clean)
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_clean.yaml --config configs/multiround.yaml)
    ;;
  sr_det_text_clean)
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_clean.yaml)
    ;;
  # P4-A depth encodings: single-tool (dpt-large), 3D slice, multiround
  depth_gray|depth_plasma|depth_turbo)
    INPUT=cvbench_depth100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_depth.yaml --config configs/${ARM}.yaml
          --config configs/multiround.yaml)
    ;;
  depth_stock)
    INPUT=cvbench_depth100.jsonl
    ;;
  # P4-B semantic-seg encodings: single-tool (detr-panoptic), Count slice
  seg_overlay_base|seg_opacity100|seg_color_by_instance|seg_contour_only|seg_polygon_text|seg_mask_only)
    INPUT=cvbench_count100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/${ARM}.yaml
          --config configs/multiround.yaml)
    ;;
  # P4-B debugged: FORCED seg plan in round 1 (planner bypass — it plans
  # unregistered VQA/detection for counting questions otherwise)
  fseg_overlay_base|fseg_opacity100|fseg_color_by_instance|fseg_contour_only|fseg_polygon_text|fseg_mask_only)
    INPUT=cvbench_count100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#fseg_}.yaml --config configs/multiround.yaml)
    ;;
  seg_stock)
    INPUT=cvbench_count100.jsonl
    ;;
  # P4-C referring-seg encodings: masks filtered to referred objects, Relation slice
  refseg_overlay_base|refseg_opacity100|refseg_color_by_instance|refseg_contour_only|refseg_polygon_text|refseg_mask_only)
    INPUT=cvbench_relation100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/seg_${ARM#refseg_}.yaml
          --config configs/seg_ref.yaml --config configs/multiround.yaml)
    ;;
  refseg_stock)
    INPUT=cvbench_relation100.jsonl
    ;;
  # DA-2K point-pair depth probe (single-tool dpt, multiround)
  da2k_stock)
    INPUT=da2k_100.jsonl
    ;;
  da2k_gray|da2k_plasma|da2k_turbo)
    INPUT=da2k_100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_depth.yaml --config configs/depth_${ARM#da2k_}.yaml
          --config configs/multiround.yaml)
    ;;
  # DA-2K 300 balanced (cross-framework family, sibling ASK 1)
  da300_stock)
    INPUT=da2k_300balanced.jsonl
    ;;
  da300_gray|da300_plasma|da300_turbo)
    INPUT=da2k_300balanced.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_depth.yaml --config configs/depth_${ARM#da300_}.yaml
          --config configs/multiround.yaml)
    ;;
  # TallyQA-complex counting (forced-plan seg)
  tqa_stock)
    INPUT=tallyqa_complex100.jsonl
    ;;
  tqa_fseg_polygon_text|tqa_fseg_overlay_base|tqa_fseg_mask_only)
    INPUT=tallyqa_complex100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#tqa_fseg_}.yaml --config configs/multiround.yaml)
    ;;
  *) echo "unknown arm $ARM" >&2; exit 1;;
esac
cd "$D"
exec "$PY" run_blink.py "$INPUT" result.json "${CFGS[@]}"
