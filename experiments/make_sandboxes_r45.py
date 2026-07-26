#!/usr/bin/env python3
"""Create repeat-4/5 sandboxes for every arm, load-balanced across the
three 72B vLLM endpoints (:8001/:8002/:8003) and, for server-side
encoding families, across the two new-code models_servers (:9006/:9007).

Each sandbox owns its configs/, so the endpoint routing is done by
patching the sandbox's config.default.yaml (openai.base_url) and
rerun_local2.yaml (tool port) — run_ci.sh needs no changes.
"""
import os
import shutil
import yaml

ROOT = "/raid/icy/jarvis-cathy"
SERVER = f"{ROOT}/hugginggpt/server"
SB = f"{ROOT}/experiments/sandboxes"

# arm -> (input file, p0 registry)
DET = {a: ("cvbench_first100.jsonl", "full") for a in [
    "singleround", "det_image_only", "det_image_and_text", "det_text_only",
    "det_text_clean", "sr_det_image_only", "sr_det_image_and_text",
    "sr_det_text_only", "sr_det_text_clean", "mr_baseline"]}
DEPTH = {a: ("cvbench_depth100.jsonl", "full" if a == "depth_stock" else "depth")
         for a in ["depth_stock", "depth_gray", "depth_plasma", "depth_turbo"]}
SEG = {a: ("cvbench_count100.jsonl", "full" if a == "seg_stock" else "seg")
       for a in ["seg_stock", "seg_overlay_base", "seg_opacity100",
                 "seg_color_by_instance", "seg_contour_only",
                 "seg_polygon_text", "seg_mask_only",
                 "fseg_overlay_base", "fseg_opacity100",
                 "fseg_color_by_instance", "fseg_contour_only",
                 "fseg_polygon_text", "fseg_mask_only"]}
REF = {a: ("cvbench_relation100.jsonl", "full" if a == "refseg_stock" else "seg")
       for a in ["refseg_stock", "refseg_overlay_base", "refseg_opacity100",
                 "refseg_color_by_instance", "refseg_contour_only",
                 "refseg_polygon_text", "refseg_mask_only"]}
ARMS = {**DET, **DEPTH, **SEG, **REF}

VLLM = ["http://localhost:8001", "http://localhost:8002", "http://localhost:8003"]
TOOL2 = [9006, 9007]  # new-code models_servers

jobs = []
i = 0
for arm, (inp, p0) in ARMS.items():
    for k in (4, 5):
        d = f"{SB}/{arm}_r{k}"
        for sub in ("logs", "public/audios", "public/videos", "configs", "data"):
            os.makedirs(f"{d}/{sub}", exist_ok=True)
        for f in ("awesome_chat.py", "run_blink.py", "evaluate.py",
                  "get_token_ids.py", "demos", "models"):
            dst = f"{d}/{f}"
            if not os.path.islink(dst):
                os.symlink(f"{SERVER}/{f}", dst)
        for name, target in [
                (inp, f"{ROOT}/experiments/{inp}"),
                ("public/CVBench", "/raid/cathy/dataset/CVBench"),
                ("public/examples", f"{SERVER}/public/examples"),
                ("public/images", f"{SERVER}/public/images")]:
            dst = f"{d}/{name}"
            if not os.path.islink(dst):
                os.symlink(target, dst)
        p0_file = {"full": "p0_models.jsonl", "depth": "p0_models.depth_only.jsonl",
                   "seg": "p0_models.seg_only.jsonl"}[p0]
        dst = f"{d}/data/p0_models.jsonl"
        if not os.path.islink(dst):
            os.symlink(f"{SERVER}/data/{p0_file}", dst)
        for cf in os.listdir(f"{SERVER}/configs"):
            if cf.endswith(".yaml") and not cf.startswith("tmp"):
                shutil.copy(f"{SERVER}/configs/{cf}", f"{d}/configs/{cf}")
        # endpoint routing
        cfg_path = f"{d}/configs/config.default.yaml"
        c = yaml.safe_load(open(cfg_path))
        c["openai"]["base_url"] = VLLM[i % len(VLLM)]
        yaml.safe_dump(c, open(cfg_path, "w"), default_flow_style=False, sort_keys=False)
        with open(f"{d}/configs/rerun_local2.yaml", "w") as f:
            f.write("local_inference_endpoint:\n  host: localhost\n"
                    f"  port: {TOOL2[i % len(TOOL2)]}\n")
        jobs.append((arm, k, VLLM[i % len(VLLM)], TOOL2[i % len(TOOL2)]))
        i += 1

print(f"{len(jobs)} sandboxes ready")
for arm, k, v, t in jobs[:6]:
    print(" e.g.", arm, k, v, t)
