# Judge-Transfer Experiments — HuggingGPT side (Design & Status)

Companion to `/raid/icy/spagent-william/experiments/judge-exp.md` (SpAgent
side). This repo (`/raid/icy/jarvis-cathy`, branch `judge-work`) is the
recovered HuggingGPT Stage-II kitchen: the paper's Table-4 HuggingGPT
column (57 / 60 / 61 / 63, CV-Bench, controller Qwen2.5-VL-72B) was
produced here. This doc tracks (a) fidelity vs the paper, (b) CI reruns,
(c) the encoding-sweep expansion, (d) the cross-framework comparison.

---

## 1. Fidelity: does this repo match the paper?

| Paper element | This repo | Verdict |
|---|---|---|
| Table 4 HuggingGPT 57.00/60.00/61.00/63.00 | `result_cvbench_500*.json` scored on the first 100 of `cvbench_data_500sample.jsonl` → **57/60/61/63 bit-exact** (`experiments/score_table4.py`) | ✅ located; **n=100** |
| "Single round, no detection tool" (baseline) | det tool WAS registered; planner invoked object-detection on 134/497 samples (legacy encoding: overlay drawn + raw bbox text) | ⚠️ baseline is "default single-round HuggingGPT", not "no det tool" |
| "+ Det. tool" arms | `det_encoding: image_only / image_and_text / text_only` **plus `multiround` (max 3 rounds)** — evidenced by round-2/3 traces in `logs/debug.log` and duplicate-plan structure in ~30% of det-arm samples (0% in baseline) | 🚨 det arms differ from baseline in TWO factors (encoding AND rounds) |
| Det arms item count | det runs stopped at 166/500; first 100 complete | ⚠️ fine for n=100 claim |
| Final numbers include curation | backfill reruns (2/1/8/5 items) + 4 `_manual_answer` injections (3 scored correct, 1 scored wrong) | ⚠️ documented in README; reruns must replicate the backfill step (not the manual audit) |

## 2. Wiring audit — what the controller ACTUALLY receives, per arm

All controller stages (`parse_task`, `choose_model`, `response_results`)
send plain-text messages; tool outputs are serialized into an assistant
"workflow JSON" message. Image pixels reach the VLM **only** via
`build_reflection_prompt` (multi-round, rounds ≥ 2) as base64
`image_url` parts. Verified end-to-end in `awesome_chat.py`
(`object-detection` branch @ ~line 657, `response_results` @ 440,
`build_reflection_prompt` @ 1123) and against live traces in
`logs/debug.log`.

| Channel → controller | baseline (legacy) | image_only | image_and_text | text_only |
|---|---|---|---|---|
| raw `predicted` bbox JSON (absolute px, xyxy) as text | ✅ | ✅ (!) | ✅ (!) | ✅ |
| normalized [0,1] xyxy `description` text | ❌ | ❌ | ✅ | ✅ |
| overlay image drawn & path in results text | ✅ | ✅ | ✅ | ❌ |
| overlay pixels attached to VLM (reflection, r≥2) | ❌ (single-round) | ✅ | ✅ | ❌ (no overlay) |
| overlay consumed by chained tools (VQA on `<GENERATED>` image) | ✅ when planned | ✅ when planned | ✅ when planned | n/a (original image) |
| rounds | 1 | ≤3 | ≤3 | ≤3 |

Key takeaways:
- **image_only is not image-only**: raw bbox text is always present
  (`results = {"predicted": predicted}` is unconditional). The arm's real
  contrast vs image_and_text is only the normalized-description line.
- **text_only works as designed** (unlike SpAgent's broken text arm): the
  `description` field does reach the controller via the workflow JSON in
  `response_results` and via reflection text.
- The encoding manipulation is **stronger for the tools than for the
  controller**: chained VQA sees overlay vs original pixels; the
  controller mostly sees different text.

## 3. CI reruns (Phase 3) — 4 arms × 3 repeats, n=100

Protocol: `experiments/cvbench_first100.jsonl`, temperature 0, faithful
configs (det arms keep the multiround confound — we replicate the
*published* condition), per-repeat sandbox dirs (own result/logs/tmp
configs; shared `public/images` made collision-safe by full-uuid image
names), one shared `models_server`, fresh vLLM Qwen2.5-VL-72B-Instruct
controller. Backfill failed items via `run_blink.py` resume (same
mechanism as the original), then score with
`score_table4.py --dir <ci_run>`; aggregate with across-repeat t-CI +
per-item bootstrap + McNemar paired tests on shared items.

All 12 runs complete, 0 hard failures after backfill (the `replace_slot`
backslash fix, commit c8b7be3, eliminated the deterministic Depth-sample
failures present in the original runs too).

| Arm | Reported | Rerun repeats | Mean [95% t-CI] | Verdict |
|---|---|---|---|---|
| singleround | 57.00 | 58/57/57 | **57.33 [55.90, 58.77]** | ✅ reproduces |
| det image_only | 60.00 (+3) | 60/55/58 | 57.67 [51.42, 63.91] | ❌ delta does not reproduce (mean ≈ baseline; 60 = best-of-3) |
| det image_and_text | 61.00 (+4) | 57/55/59 | 57.00 [52.03, 61.97] | ❌ delta does not reproduce |
| det text_only | 63.00 (+6) | 61/58/55 | 58.00 [50.55, 65.45] | ❌ delta does not reproduce (63 inside wide CI, but see McNemar) |

Paired McNemar (pooled over repeats, shared items): **every pair p ≥ 0.79**
— no paired evidence for any det-arm advantage over the baseline in the
published (multiround) configuration. Discordant counts are balanced
(e.g. singleround vs text_only 20/22). The paper's +3/+4/+6 column is
best read as run-to-run noise (±3pp at n=100, temperature 0) on top of
curation (backfill + manual answers) that favoured the det arms.

The one surviving encoding signal is in the de-confounded single-round
arms (§4a): text_only 59.0 > image_and_text 57.0 > image_only 55.7 —
same ordering as the paper, text_only vs image_only p=0.087.

Note per-run bootstrap CIs are ±10pp at n=100: the paper's column
resolves nothing smaller than ~7pp at this sample size.

> **Canonical reproduction protocol (operator decision, 2026-07-26):**
> the SINGLE-ROUND arms (`singleround` baseline + `sr_det_*`) are the
> recommended configuration for reproducing/extending the Table-4
> comparison — no multiround confound, deterministic control flow, and
> the encoding acts through exactly one channel (tool text in the
> response prompt). The multiround arms are kept as the faithful
> replication of the published condition only. Exception: seg/depth
> IMAGE-encoding sweeps still need multiround, since round 1 never
> attaches pixels to the controller.

## 4a. Phase-4 results — de-confound + single-tool encoding sweeps

All arms 3 repeats, n=100, temperature 0; `*_stock` = default toolset
single-round (Table-4 baseline config) on the same slice. Full data:
`results/p4-encodings/`; aggregation: `score_p4.py`.

**De-confound (first-100 set):**

| Arm | Mean [95% t-CI] | Note |
|---|---|---|
| sr_det_image_only | 55.67 [43.42, 67.91] | encoding without multiround |
| sr_det_image_and_text | 57.00 [50.43, 63.57] | |
| sr_det_text_only | **59.00 [56.52, 61.48]** | ordering matches paper; vs image_only p=0.087 |
| mr_baseline (multiround, no det_encoding) | 52.57 [46.71, 58.42] | multiround alone HURTS (vs sr_det_text_only p=0.026) |

**Depth (3D slice, dpt-large only, multiround):** stock 56.67 <
turbo 62.00 < gray 63.00 < **plasma 64.00 [59.70, 68.30]**. Tool provably
helps (plasma vs stock McNemar p=0.035; gray p=0.079). Plasma-first
**matches Stage-I** and **opposes SpAgent Stage-II** (turbo>gray>plasma
there) — the depth-colormap ordering is not framework-stable, though
between-colormap gaps here are individually non-significant.

**Semantic seg (Count slice, detr-panoptic only, multiround):** total
collapse — every encoding 22.7–24.0 vs stock 62.0 (all McNemar p<1e-4
vs stock; encodings mutually indistinguishable; mask_only negative
control NOT worse than overlay). The encoding axis is unmeasurable:
panoptic masks carry no usable counting signal for a controller that is
text-blind in round 1. Stage-I→II transfer for semantic seg is undefined
in HuggingGPT; the paper's §5.2 flip-prediction is not borne out here.

**Referring seg (Relation slice, label-filtered masks, multiround):**
stock 60.67; image encodings all ~51 (each p≤0.001 below stock);
**polygon_text 59.00 [54.70, 63.30]** ≈ stock and significantly above
every image encoding (p≈0.005–0.008 pairwise). Text encodings dominate
segmentation output in this framework — consistent with the Stage-I
polygon-text signal and with the round-1 wiring (controller sees text
only).

**Audit follow-up arms (running):** det_text_clean / sr_det_text_clean —
convention-explicit, W×H-stated, top-5, 2-decimal, no raw pixel dump —
directly test the "distracting text" hypothesis against det_text_only /
sr_det_text_only.

## 4. Expansion (Phase 4)

- Depth-colormap sweep (gray/plasma/turbo) via `depth-estimation` output
  formatter — mirrors SpAgent B runs.
- Segmentation-encoding sweep per judge-exp.md §N1 (6 encodings,
  single-tool isolation: register ONLY the variant under test).
- **De-confound arm**: det arms re-run single-round (multiround off) to
  separate encoding effect from round effect — directly tests whether the
  paper's +3/+4/+6 ordering survives without the extra rounds.
- Cross-framework table: HuggingGPT vs SpAgent rerun CIs (encoding
  ordering consistency = the paper's core claim).

## 5. Infra & bookkeeping

- Fleet etiquette: SpAgent campaign may hold GPUs 0-5,7 and ports
  8001-8004, 20019-20025, 30010. This side uses free GPUs only and ports
  in the 9000s/21000s. Controller: vLLM
  (`/raid/miniconda3/envs/qwenvl3_vllm/bin/vllm`) serve
  Qwen/Qwen2.5-VL-72B-Instruct, TP=2, HF cache
  `/raid/icy/iris/.cache/huggingface`.
- `models_server.py` (tools) needs one GPU (`device` in config).
- Original runs' `base_url: localhost:8001` — rerun configs override to
  this side's port; nothing hardcoded touched.
- No API keys on disk; git identity IcyWang <irisicy@outlook.com>; no
  co-author lines.
