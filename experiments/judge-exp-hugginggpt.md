# Judge-Transfer Experiments — HuggingGPT side (Design & Status)

Companion to `/raid/icy/spagent-william/experiments/judge-exp.md` (SpAgent
side). This repo (`/raid/icy/jarvis-cathy`, branch `cathy-work`) is the
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

| Arm | Reported | Rerun status | Mean [95% t-CI] |
|---|---|---|---|
| singleround | 57.00 | pending | — |
| det image_only | 60.00 | pending | — |
| det image_and_text | 61.00 | pending | — |
| det text_only | 63.00 | pending | — |

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
