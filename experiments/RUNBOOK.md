# RUNBOOK — HuggingGPT Table-4: original runs, CI rerun, planned experiments

Companion to `judge-exp-hugginggpt.md` (findings/status) and `README.md`
(artifact manifest). This file is the *operational* record: exactly what
ran, with which configs, and where results live.

---

## 1. Original runs (cathy, 2026-05-07) — what produced 57/60/61/63

### 1.1 Stack

| Component | Value |
|---|---|
| Repo | `/raid/cathy/JARVIS` @ `50166ca` + uncommitted work (now imported as `cathy-work@fbe5a55`) |
| Python env | `/raid/cathy/miniconda3/envs/jarvis` (py3.8) — both `run_blink.py` and `models_server.py` |
| Controller | vLLM v0.17.0 `Qwen/Qwen2.5-VL-72B-Instruct`, **TP=4**, `max_model_len=32768`, bf16, port **8001**, served name `Qwen2.5-VL-72B-Instruct` (from `logs/vllm_72b.log`) |
| Tool server | `models_server.py --config configs/config.default.yaml`, port **8005**, `device: cuda:0`, `local_deployment: full`, `inference_mode: local` |
| Benchmark | `cvbench_data_500sample.jsonl` (500 items); **scored on the first 100** |
| LLM sampling | `temperature: 0` (hardcoded in `send_request`); `logit_bias: {parse_task: 0.1, choose_model: 5}` |

### 1.2 Per-arm run config (effective merged YAML keys)

`run_blink.py` merges `--config` layers left→right into an ephemeral
`configs/tmp*.yaml` (those 35 files were root-owned/unreadable at copy
time — reconstructed below from `_det_encoding` markers in results and
round-2/3 traces in `logs/debug.log`).

| Arm | Config layers (reconstructed) | `det_encoding` | `multi_round.enabled` | `max_rounds` | `bbox_*` legacy flags |
|---|---|---|---|---|---|
| singleround (baseline) | `config.default.yaml` | `null` (legacy) | false | 1 | all false |
| det image_only | `config.default.yaml` + `det_image_only.yaml` + `multiround.yaml` | `image_only` | **true** | 3 | all false |
| det image_and_text | `config.default.yaml` + `det_image_and_text.yaml` + `multiround.yaml` | `image_and_text` | **true** | 3 | all false |
| det text_only | `config.default.yaml` + `det_text_only.yaml` + `multiround.yaml` | `text_only` | **true** | 3 | all false |

Common keys from `config.default.yaml` (as of the runs): `model:
Qwen2.5-VL-72B-Instruct`, `use_completion: false`, `openai.base_url:
http://localhost:8001`, `local_inference_endpoint: localhost:8005`,
`inference_mode: local`, `num_candidate_models: 5`.

Command shape (per arm):
```
python run_blink.py cvbench_data_500sample.jsonl <result_file> \
  --config configs/config.default.yaml [--config configs/det_<enc>.yaml --config configs/multiround.yaml]
```

### 1.3 Result chain per arm (all under `hugginggpt/server/`)

| Arm | Raw run output | First-100 snapshot (pre-merge) | Retry list | Backfill output | Final merged (scored for paper) | Paper # |
|---|---|---|---|---|---|---|
| singleround | `result_cvbench_500.json` (500/500 done) | `tmp_eval_100_cvbench_singleround.json` (56) | `retry_cvbench_singleround.jsonl` | `rerun_result_cvbench_singleround.json` (2 items) | `result_cvbench_500.json` (merged in place, +1 manual) | **57** |
| det image_only | `result_cvbench_500_det_image_only.json` (**166**/500, stopped) | `tmp_eval_100_...det_image_only.json` (58) | `retry_cvbench_image_only.jsonl` | `rerun_result_cvbench_image_only.json` (1) | same file, merged (+2 manual) | **60** |
| det image_and_text | `result_cvbench_500_det_image_and_text.json` (166/500) | `tmp_eval_100_...det_image_and_text.json` (58) | `retry_cvbench_image_and_text.jsonl` | `rerun_result_cvbench_image_and_text.json` (8) | same file, merged (+0 manual scored correct) | **61** |
| det text_only | `result_cvbench_500_det_text_only.json` (166/500) | `tmp_eval_100_...det_text_only.json` (61) | `retry_cvbench_text_only.jsonl` | `rerun_result_cvbench_text_only.json` (5) | same file, merged | **63** |

Scoring = `evaluate.py` extraction logic on first-100 IDs, incl. 4
`_manual_answer` injections (3 correct: Relation_511, Relation_456,
Distance_2298 → all gt=A; 1 wrong: Count_187 manual=A vs gt=D).
Reproduce: `python experiments/score_table4.py` → 57/60/61/63 bit-exact.
Organized copies: `experiments/results/table4-cvbench/<arm>/{final_merged_*,premerge_first100,rerun_backfill}.json`.

A parallel BLINK campaign (same arms, `BLINK_All_Tasks_500sample.jsonl` →
`result_blink_500*.json`) ran in the same window; not part of Table 4.

---

## 2. CI rerun (this recovery, 2026-07-26) — 4 arms × 3 repeats, n=100

### 2.1 Stack & deviations from §1

| Component | This rerun | Deviation vs original |
|---|---|---|
| Repo | `/raid/icy/jarvis-cathy` @ `cathy-work` (e4ddf5e) | code identical except: full-uuid artifact names (collision safety, commit e4ddf5e) |
| Python env | same `/raid/cathy/miniconda3/envs/jarvis` | none (plus `TIKTOKEN_CACHE_DIR=/raid/icy/iris/.cache/tiktoken` — /tmp cache owned by another user) |
| Controller | **shared existing** vLLM `Qwen2.5-VL-72B-Instruct` @ :8001 (SpAgent fleet) | **TP=2, max_model_len=16384** vs TP=4/32768; server co-serves the SpAgent campaign (no two free GPUs available: GPUs 0-5,7 occupied) |
| Tool server | `models_server.py --config configs/config.models_server.rerun.yaml` → port **9005**, GPU **6** | port 8005→9005; runwayml SD-v1.5 weights restored from official HF mirror into `/raid/icy/iris/.cache` (cathy's HF cache deleted; only affects text-to-image/controlnet tasks) |
| Benchmark | `experiments/cvbench_first100.jsonl` = exact first-100 of the 500 file | n=100 directly (original ran 500/166 and scored first 100) |
| Concurrency | 4 parallel `run_blink` processes (xargs -P4) | original ran ~8 arms concurrently |

### 2.2 Per-run layout & commands

Sandboxes: `experiments/sandboxes/<arm>_r<k>/` (k=1..3), created by
`experiments/make_sandboxes.sh` — private `result.json`, `logs/`,
`configs/` (incl. ephemeral merged tmp yaml); symlinked
`awesome_chat.py, run_blink.py, evaluate.py, get_token_ids.py, demos/,
data/, models/`; `public/CVBench → /raid/cathy/dataset/CVBench`;
`public/images → <repo>/hugginggpt/server/public/images` (shared with
models_server so relative paths resolve; full-uuid names prevent
collisions).

| Arm | `run_ci.sh` config layers | Matches §1.2 arm |
|---|---|---|
| singleround | `config.default.yaml` + `rerun_local.yaml` | baseline |
| det_image_only | + `det_image_only.yaml` + `multiround.yaml` | det image_only |
| det_image_and_text | + `det_image_and_text.yaml` + `multiround.yaml` | det image_and_text |
| det_text_only | + `det_text_only.yaml` + `multiround.yaml` | det text_only |

`rerun_local.yaml` overrides **only** `local_inference_endpoint.port: 9005`.

Driver: `experiments/launch_ci_fleet.sh` (resumable; re-running it
backfills failed items via `run_blink.py`'s resume+retry). Collect:
`experiments/collect_ci.sh` → `results/table4-cvbench/<arm>/ci_runs/repeat<k>/result.json`.
Score: `score_table4.py --dir ...`. Aggregate:
`experiments/ci_aggregate.py` (per-run bootstrap CI + across-repeat t-CI +
pooled exact McNemar between arms; summary-vs-items consistency guard).

### 2.3 Status

Fleet launched 05:05–05:14; 12 runs (4 lanes). Progress and final numbers
tracked in `judge-exp-hugginggpt.md` §3. Backfill pass + aggregation after
all repeats land.

---

## 3. New experiment plans (Phase 4)

Design rules carried over from the SpAgent side (judge-exp.md): 3 repeats
each; temperature 0; sandboxes; **single-tool isolation** for encoding
sweeps (register ONLY the tool variant under test in `data/p0_models.jsonl`
/ task list so the planner cannot route around it); paired per-item
analysis over shared IDs.

### 3.1 De-confound arm (highest priority — direct paper check)

The published det arms differ from baseline in encoding AND rounds (§1.2).
One extra axis separates them:

| Run | Config layers | det_encoding | rounds | Purpose |
|---|---|---|---|---|
| D1 | default + `det_image_only.yaml` | image_only | 1 | encoding effect without multiround |
| D2 | default + `det_image_and_text.yaml` | image_and_text | 1 | " |
| D3 | default + `det_text_only.yaml` | text_only | 1 | " |
| D4 | default + `multiround.yaml` | null (legacy) | ≤3 | multiround effect without det_encoding |

3 repeats × 4 runs, n=100. Compare {D1-D3} vs published det arms (round
effect) and vs baseline (pure encoding effect); D4 isolates the round
effect on the baseline. Note: in round 1 the controller never sees images
(§2 wiring table in judge-exp-hugginggpt.md), so D1 vs D2 vs D3 measures
the *text+tool-chaining* channel only — itself a finding worth reporting.

> **Priority directive (operator, 2026-07-26):** run the encoding tests as
> single-tool experiments — exactly ONE tool per category (the most
> confident tool of that category: `Intel/dpt-large` for depth,
> `facebook/detr-resnet-50-panoptic` for segmentation), on a CV-Bench
> slice that the tool should provably help; referring segmentation and
> semantic segmentation tested separately; report per-encoding accuracy.
> Implemented as §3.2 (depth, 3D slice), §3.3 (semantic seg, Count slice),
> §3.3b (referring seg, Relation slice); each family includes a
> `*_stock` control arm (default toolset, single-round — the Table-4
> baseline config restricted to the slice) to establish the tool's benefit.
> Isolation = restricted `tprompt.parse_task` (single task option) +
> private `data/p0_models.jsonl` registering only the tool under test.
> Encoding parameters travel per-request (`colormap` / `seg_encoding` /
> `text`) from the per-run config to a second models_server (:9006, GPU 6)
> running the new formatters, so the original :9005 server and in-flight
> Phase-3 runs are untouched.

### 3.2 Depth-colormap sweep (mirrors SpAgent B runs)

Current wiring: `models_server.py` `Intel/dpt-large` returns the HF
pipeline's grayscale depth PIL (`output['depth']`) saved as
`public/images/<uuid>.jpg`; controller sees only the path text in round 1,
pixels in reflection rounds; chained tools may consume it.

Implementation: add `depth_colormap: gray|plasma|turbo` config key; in the
dpt-large branch apply `matplotlib.cm` colormap to the normalized depth
array before save. New layers `configs/depth_{gray,plasma,turbo}.yaml`.

| Run | Layers | depth_colormap | multi_round | Tool registration |
|---|---|---|---|---|
| DC1 | default + `depth_gray.yaml` + `multiround.yaml` + single-tool | gray (≈current) | ≤3 | depth-estimation ONLY |
| DC2 | ... `depth_plasma.yaml` ... | plasma | ≤3 | " |
| DC3 | ... `depth_turbo.yaml` ... | turbo | ≤3 | " |

Benchmark: CV-Bench Depth+Distance slice of the 500 (the 3D tasks;
first-100 contains ~35). Use n=100 depth/distance-only subset drawn from
the 500 for power. Multiround ON so the colormapped image actually reaches
the controller (otherwise the colormap is invisible to the VLM in this
framework — unlike SpAgent). 3 encodings × 3 repeats = 9 runs.
Hypothesis (Stage-I transfer): plasma ≥ turbo > gray at judging, but
SpAgent Stage-II found turbo/gray > plasma — HuggingGPT adjudicates.

### 3.3 Segmentation-encoding sweep (judge-exp.md §N1 mirror)

Backend: `facebook/detr-resnet-50-panoptic` (`image-segmentation` task in
`models_server.py`; returns per-segment masks). Add
`seg_encoding: S1..S6` key + formatter in the image-segmentation branch of
`awesome_chat.py` (same fork pattern as `det_encoding`).

| ID | Encoding | Image output | Text output | Layer file |
|---|---|---|---|---|
| S1 | overlay-base | α=0.5 colored masks on image + labels | none | `seg_overlay.yaml` |
| S2 | opacity-100 | α=1.0 masks (image occluded) | none | `seg_opaque.yaml` |
| S3 | color-by-instance | one color per instance, α=0.5 | none | `seg_instance.yaml` |
| S4 | contour-only | boundary polylines only | none | `seg_contour.yaml` |
| S5 | polygon-text | none | JSON: label + normalized polygon coords | `seg_polytext.yaml` |
| S6 | mask-only (neg ctrl) | masks on black background | none | `seg_maskonly.yaml` |

Benchmark: CV-Bench Count slice (`cvbench_count100.jsonl`) — counting =
count instance masks; single-tool isolation (detr-panoptic ONLY);
multiround ON (same visibility rationale as §3.2). 6 encodings × 3
repeats + `seg_stock` ×3 = 21 runs. Hypotheses H1-H3 as in
judge-exp.md §N1.

### 3.3b Referring-segmentation sweep (Relation slice)

Same 6 encodings and isolation as §3.3, plus `seg_referring: true`: the
user request text is passed to the seg tool and masks are filtered to
segments whose label occurs in the request (fallback: all masks when no
label matches) — a referring-style variant of the same most-confident
segmentation tool (stock JARVIS has no text-conditioned segmenter).
Benchmark: CV-Bench Relation slice (`cvbench_relation100.jsonl`), whose
questions refer to specific named objects. Arms `refseg_<enc>` ×3 +
`refseg_stock` ×3 = 21 runs.

### 3.4 Cross-framework consistency table (analysis only, zero GPU)

Join this repo's CI means/CIs with
`/raid/icy/spagent-william/experiments/` rerun results on the shared axes:

| Axis | HuggingGPT arms | SpAgent arms |
|---|---|---|
| det: image-only vs image+text vs text-xyxy | §2 reruns (+ §3.1 de-confounded) | A1/A2/A3(+A3') |
| depth: gray vs plasma vs turbo | §3.2 | B1-3 |
| seg: S1-S6 | §3.3 | N1 |

Deliverable: per-axis encoding *ordering* with CIs per framework +
Kendall-τ agreement; the paper's claim is that the ordering is
framework-consistent. Paired per-item stats within framework only (item
sets differ across frameworks).
