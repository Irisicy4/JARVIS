#!/usr/bin/env python3
"""Reproduce the paper's Table-4 HuggingGPT column from on-disk artifacts.

The published numbers (57 / 60 / 61 / 63 on CV-Bench, controller
Qwen2.5-VL-72B) are the four merged result files scored on the FIRST 100
items of cvbench_data_500sample.jsonl, using the same answer-extraction
logic as hugginggpt/server/evaluate.py (including `_manual_answer`
audit injections).

Usage:  python experiments/score_table4.py           # from repo root
        python experiments/score_table4.py --dir X   # score rerun dirs laid
                                                     # out like results/table4-cvbench
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "hugginggpt", "server")
sys.path.insert(0, SERVER)
from evaluate import extract_answer, is_api_failure  # noqa: E402

ARMS = ["singleround", "det_image_only", "det_image_and_text", "det_text_only"]
PAPER = {"singleround": 57, "det_image_only": 60,
         "det_image_and_text": 61, "det_text_only": 63}


def first100_ids():
    path = os.path.join(SERVER, "cvbench_data_500sample.jsonl")
    with open(path) as f:
        return [json.loads(line)["id"] for line in list(f)[:100]]


def score(entries, ids=None):
    """Return (correct, n, api_failed, per_item dict id->bool|None)."""
    keep = set(ids) if ids is not None else None
    correct = failed = n = 0
    per_item = {}
    for e in entries:
        if keep is not None and e["id"] not in keep:
            continue
        n += 1
        r = e.get("response", {})
        msg = r.get("message", "") if isinstance(r, dict) else ""
        manual = r.get("_manual_answer") if isinstance(r, dict) else None
        if is_api_failure(msg) and not manual:
            failed += 1
            per_item[e["id"]] = None
            continue
        pred = (manual or extract_answer(msg) or "").strip().upper()
        ok = bool(pred) and pred == e.get("ground_truth", "").strip().upper()
        correct += int(ok)
        per_item[e["id"]] = ok
    return correct, n, failed, per_item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "table4-cvbench"),
        help="dir with <arm>/final_merged_*.json")
    ap.add_argument("--file-glob", default="final_merged_",
                    help="prefix of the result file inside each arm dir")
    args = ap.parse_args()

    ids = first100_ids()
    print(f"{'arm':<22}{'correct':>8}{'n':>5}{'acc%':>8}{'api_fail':>9}{'paper':>7}")
    for arm in ARMS:
        d = os.path.join(args.dir, arm)
        if not os.path.isdir(d):
            print(f"{arm:<22}  (missing)")
            continue
        fname = next((f for f in sorted(os.listdir(d))
                      if f.startswith(args.file_glob)), None)
        if fname is None:
            print(f"{arm:<22}  (no result file)")
            continue
        entries = json.load(open(os.path.join(d, fname)))
        c, n, fl, _ = score(entries, ids)
        flag = "" if PAPER[arm] == c else "  <-- differs"
        print(f"{arm:<22}{c:>8}{n:>5}{c / n * 100:>8.2f}{fl:>9}{PAPER[arm]:>7}{flag}")


if __name__ == "__main__":
    main()
