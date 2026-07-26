# Experiments manifest — HuggingGPT (JARVIS) side of Vision-Judge Stage II

Recovered working copy of `/raid/cathy/JARVIS` (branch `cathy-work`,
import commit on top of cathy's `50166ca`). Model weights and the
BLINK/CVBench image dirs are symlinks (weights →
`/raid/cathy/JARVIS/hugginggpt/server/models`, images →
`/raid/cathy/dataset/{BLINK_images,CVBench}`); everything else is a
verbatim copy, so `hugginggpt/server` scripts run in place unmodified.

## Table 4 (paper), HuggingGPT column — provenance

Published: single-round 57.00 / +det image-only 60.00 / +det image+text
61.00 / +det text-only-xyxy 63.00, CV-Bench, controller Qwen2.5-VL-72B.

**Exact reproduction from disk** (`python experiments/score_table4.py`):
score the four merged result files on the **first 100 items of
`cvbench_data_500sample.jsonl`** with `evaluate.py`'s extraction logic →
57 / 60 / 61 / 63, bit-exact. So **n = 100**, not 500.

| Arm (paper) | Original file (hugginggpt/server/) | Organized copy (results/table4-cvbench/) | acc on first-100 |
|---|---|---|---|
| single round, no det | `result_cvbench_500.json` | `singleround/final_merged_500.json` | 57/100 |
| + det, image-only | `result_cvbench_500_det_image_only.json` | `det_image_only/final_merged_166.json` | 60/100 |
| + det, image+text | `result_cvbench_500_det_image_and_text.json` | `det_image_and_text/final_merged_166.json` | 61/100 |
| + det, text-only xyxy | `result_cvbench_500_det_text_only.json` | `det_text_only/final_merged_166.json` | 63/100 |

Pipeline that produced those files (all timestamps 2026-05-07):

1. ~05:00–08:53 — `run_blink.py cvbench_data_500sample.jsonl <out> --config
   configs/config.default.yaml [--config configs/det_X.yaml --config
   configs/multiround.yaml]` against local vLLM Qwen2.5-VL-72B-Instruct
   (`localhost:8001`) + local `models_server` tools. The three det runs
   were stopped at ~166/500 items (first 100 complete — enough for the
   paper's n=100); single-round completed 500.
   Pre-merge first-100 snapshots: `tmp_eval_100_cvbench_*` →
   `<arm>/premerge_first100.json` (56 / 58 / 58 / 61).
2. 08:58 — failed-item lists `retry_cvbench_*.jsonl`.
3. 10:24–10:38 — backfill reruns `rerun_result_cvbench_*.json` →
   `<arm>/rerun_backfill.json` (2 / 1 / 8 / 5 items).
4. 10:48 — backfills merged into the `result_cvbench_500*` files **plus 4
   `_manual_answer` audit injections** for unparsed-but-implicit answers:
   `Relation_511`→A (gt A, correct), `Relation_456`→A (gt A, correct),
   `Distance_2298`→A (gt A, correct), `Count_187`→A (**gt D — scored
   wrong**, honest injection). Net effect: 56→57 (singleround),
   58→60 (image_only), 58→61 (image_and_text, backfill only),
   61→63 (text_only, backfill only).

⚠️ Caveats found during the audit (details + wiring table in
`judge-exp-hugginggpt.md`):
- The det arms ran **multi-round (max 3)**; the baseline is single-round —
  the paper's det-vs-baseline delta confounds encoding with multi-round.
- The "no detection tool" baseline **still had the det tool registered**;
  the planner invoked object-detection on 134/497 samples (legacy
  encoding).
- In *all* arms the controller receives the raw pixel-coord `predicted`
  bbox JSON as text; "image-only" is not image-only.
- The controller sees actual image pixels **only** in multi-round
  reflection prompts (rounds ≥ 2), and only for arms that generate an
  overlay (image_only, image_and_text).

## Other artifacts (kept at original paths, not reorganized)

| File group (hugginggpt/server/) | What it is | Status |
|---|---|---|
| `result_blink_500*.json`, `accuracy_blink_500.json`, `tmp_eval_100_blink_*`, `rerun_result_blink_*` | BLINK mirror of the CV-Bench campaign (same 4 arms, same May-7 pipeline) | complete chain on disk; not part of Table 4 |
| `accuracy_cvbench_100_multiround.json` (53%), `accuracy_cvbench_100_run2.json` (52%) | April multi-round / repeat experiments on the older `cvbench_data_100sample.jsonl` (a *different* 100-set than Table 4's first-100) | superseded |
| `accuracy_blink_det{,_multiround,_separate,_xywh,_xyxy}.json` | April bbox-format pilots (legacy `bbox_*` config flags) on `BLINK_det.jsonl` | pilots, superseded by det_encoding arms |
| `accuracy_blink_10_multiround_vlm*.json` | May-7 early-morning 10-sample multiround-VLM smoke tests | smoke tests |
| `result.json`, `*_1sample_tmp*`, `*small*` | scratch/smoke | ignore |
| `configs/tmp*.yaml` (35 files) | ephemeral merged configs from `run_blink.py --config` layering; were root-owned & unreadable at copy time | **missing**; regenerable from named layers |
| `logs/debug.log` (101M), `logs/log_success.jsonl` (13M), `logs/log_fail.jsonl` | full controller message traces for 2026-05-07 (evidence base for the wiring audit) | on disk, gitignored |
| `logs/run_*_500_det_*.log`, `logs/rerun_*.log` | per-run stdout (sample order, retries) | committed |

## Rerun infrastructure (this recovery)

- `cvbench_first100.jsonl` — the exact 100-item eval set (first 100 of
  `cvbench_data_500sample.jsonl`).
- `score_table4.py` — canonical scorer (reproduces the paper column
  bit-exact; `--dir` scores rerun outputs in the same layout).
- CI reruns land in `results/table4-cvbench/<arm>/ci_runs/repeatK/`.
