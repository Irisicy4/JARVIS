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
  # DA-2K 300 FIXED probe: markers re-stamped on the depth map + tool and
  # source images attached as real pixels to the answer call (single-round)
  da300f_stock)
    INPUT=da2k_300balanced.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/depth_probe_fix.yaml)
    ;;
  da300f_gray|da300f_plasma|da300f_turbo)
    INPUT=da2k_300balanced.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_depth.yaml --config configs/depth_${ARM#da300f_}.yaml
          --config configs/depth_probe_fix.yaml)
    ;;
  # ===== DEADLINE PRIORITY ARMS (full-size benchmarks) =====
  # P1: canonical single-round detection arms at FULL CV-Bench 500
  f500_baseline)
    INPUT=cvbench_full500.jsonl
    ;;
  f500_image_only|f500_image_and_text|f500_text_only)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_${ARM#f500_}.yaml)
    ;;
  f500_text_pixel)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_pixel.yaml)
    ;;
  # P1b: single-tool forced-plan detection at full 500 (one tool call per
  # item, pixels attached) — deadline-feasible clean encoding comparison
  f500s_baseline)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_probe_fix.yaml)
    ;;
  f500s_image_only|f500s_image_and_text|f500s_text_only)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_${ARM#f500s_}.yaml --config configs/isolate_det.yaml
           --config configs/det_probe_fix.yaml)
    ;;
  f500s_text_pixel)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_pixel.yaml
           --config configs/isolate_det.yaml --config configs/det_probe_fix.yaml)
    ;;
  # Stage-I-aligned canonical detection text (pixel xyxy, 1dp, one JSON per
   # line, verbatim preamble) — 100-item set for direct comparison to the
   # published/board arms, and full-500 variant
  canon_text_r|canon_text)
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_canon.yaml
           --config configs/isolate_det.yaml --config configs/det_probe_fix.yaml)
    ;;
  canon_text500)
    INPUT=cvbench_full500.jsonl
    CFGS+=(--config configs/det_text_only.yaml --config configs/det_text_canon.yaml
           --config configs/isolate_det.yaml --config configs/det_probe_fix.yaml)
    ;;
  # P2: instance-seg encodings on COCO-Count-Crowded 300 (forced plan, isolated)
  cc_stock)
    INPUT=cococount300.jsonl
    ;;
  cc_overlay_base|cc_color_by_instance|cc_polygon_text|cc_mask_only)
    INPUT=cococount300.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#cc_}.yaml --config configs/attach_images.yaml)
    ;;
  # Easy COCO-Count (142, no same-class overlap) — difficulty-only contrast
  cce_stock)
    INPUT=cococount_easy142.jsonl
    ;;
  cce_overlay_base|cce_color_by_instance|cce_polygon_text|cce_mask_only)
    INPUT=cococount_easy142.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#cce_}.yaml --config configs/attach_images.yaml)
    ;;
  # SEMANTIC axis: largest-region-area on COCO panoptic (250, 65% stuff winners)
  ca_stock)
    INPUT=cocoarea250.jsonl
    ;;
  ca_overlay_base|ca_color_by_instance|ca_polygon_text|ca_mask_only)
    INPUT=cocoarea250.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#ca_}.yaml --config configs/attach_images.yaml)
    ;;
  # REFERRING: separate(best) | contour(middle) | overlay+fill base(worst)
  rsF_separate_green|rsF_contour_only|rsF_overlay_base)
    INPUT=cvbench_relation100.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#rsF_}.yaml --config configs/seg_ref.yaml
          --config configs/attach_images.yaml)
    ;;
  rsF_stock)
    INPUT=cvbench_relation100.jsonl
    ;;
  # NYU-pairs depth (sensor GT, cluttered indoor) - 4 arms
  nyu_stock)
    INPUT=nyupairs250.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/attach_images.yaml)
    ;;
  nyu_plasma|nyu_turbo|nyu_gray)
    INPUT=nyupairs250.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_depth.yaml --config configs/depth_${ARM#nyu_}.yaml
          --config configs/depth_probe_fix.yaml)
    ;;
  # ===== FINAL Stage-I best/mid/worst (Fig-4 median ranking) =====
  # DETECTION: xyxy(best) | box+label overlay(mid) | separate=boxes on canvas(worst)
  dF_mid)
    CFGS+=(--config configs/det_image_only.yaml --config configs/isolate_det.yaml
           --config configs/attach_images.yaml)
    ;;
  # INSTANCE/COUNT on easy cococount: colour-by-instance(best) | mask-only(mid) | polygon(worst)
  ccF_color_by_instance|ccF_mask_only|ccF_polygon_text|ccF_separate_green)
    INPUT=cococount_easy142.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#ccF_}.yaml --config configs/attach_images.yaml)
    ;;
  # SEMANTIC/AREA: matrix text(best) | separate=opaque mask canvas(mid) | overlay base(worst)
  caF_matrix_text|caF_mask_only|caF_overlay_base|caF_separate_green)
    INPUT=cocoarea250.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#caF_}.yaml --config configs/attach_images.yaml)
    ;;
  # ===== Stage-I BEST / MIDDLE / WORST design (3 arms + no-tool per axis) =====
  # DETECTION (CV-Bench 100): best=boxes no labels, mid=text xyxy pixel, worst=box+label on canvas
  d3_best)
    CFGS+=(--config configs/det_image_only.yaml --config configs/det_box_nolabel.yaml
           --config configs/isolate_det.yaml --config configs/det_probe_fix.yaml)
    ;;
  d3_worst)
    CFGS+=(--config configs/det_image_only.yaml --config configs/det_box_canvas.yaml
           --config configs/isolate_det.yaml --config configs/det_probe_fix.yaml)
    ;;
  # INSTANCE-SEG counting (COCO-Count easy 142): best=canvas+bbox, mid=polygon, worst=plain overlay
  cce_canvas_bbox|cce_plain_overlay)
    INPUT=cococount_easy142.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#cce_}.yaml --config configs/attach_images.yaml)
    ;;
  # SEMANTIC area (COCO-Area 250): best=matrix text, mid=polygon, worst=plain overlay
  ca_matrix_text|ca_plain_overlay)
    INPUT=cocoarea250.jsonl
    CFGS=(--config configs/config.default.yaml --config configs/rerun_local2.yaml
          --config configs/isolate_seg.yaml --config configs/forced_seg.yaml
          --config configs/seg_${ARM#ca_}.yaml --config configs/attach_images.yaml)
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
