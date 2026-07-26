#!/usr/bin/env python3
"""Score & aggregate the Phase-4 single-tool encoding sweeps.

Families (one most-confident tool each, per RUNBOOK §3):
  depth  — Intel/dpt-large, 3D slice (cvbench_depth100.jsonl)
  seg    — detr-resnet-50-panoptic, Count slice (cvbench_count100.jsonl)
  refseg — same tool, label-filtered masks, Relation slice

For each arm: per-repeat accuracy, across-repeat t-CI; within-family
pairwise McNemar (pooled over repeats) against the family's *_stock
control and between encodings.
"""
import glob
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_table4 import score  # noqa: E402
from ci_aggregate import t_interval, binom_two_sided_p  # noqa: E402

SB = os.path.join(HERE, "sandboxes")

FAMILIES = {
    "depth": ["depth_stock", "depth_gray", "depth_plasma", "depth_turbo"],
    "seg": ["seg_stock", "seg_overlay_base", "seg_opacity100",
            "seg_color_by_instance", "seg_contour_only", "seg_polygon_text",
            "seg_mask_only"],
    "refseg": ["refseg_stock", "refseg_overlay_base", "refseg_opacity100",
               "refseg_color_by_instance", "refseg_contour_only",
               "refseg_polygon_text", "refseg_mask_only"],
    "deconfound": ["sr_det_image_only", "sr_det_image_and_text",
                   "sr_det_text_only", "mr_baseline"],
}


def load_arm(arm):
    reps = {}
    for k in (1, 2, 3, 4, 5):
        f = os.path.join(SB, f"{arm}_r{k}", "result.json")
        if not os.path.exists(f):
            continue
        entries = json.load(open(f))
        c, n, fl, per_item = score(entries, ids=None)
        reps[k] = {"correct": c, "n": n, "failed": fl, "items": per_item}
    return reps


def mcnemar(arm_a, arm_b, data):
    wins_a = wins_b = 0
    for k in set(data[arm_a]) & set(data[arm_b]):
        pa, pb = data[arm_a][k]["items"], data[arm_b][k]["items"]
        for i in set(pa) & set(pb):
            if pa[i] is None or pb[i] is None:
                continue
            if pa[i] and not pb[i]:
                wins_a += 1
            elif pb[i] and not pa[i]:
                wins_b += 1
    return wins_a, wins_b, binom_two_sided_p(wins_a, wins_a + wins_b)


def main():
    only = sys.argv[1:] or list(FAMILIES)
    for famname in only:
        arms = FAMILIES[famname]
        print(f"\n=== {famname} ===")
        print(f"{'arm':<26}{'repeats (correct/n)':<30}{'mean% [95% t-CI]':<24}{'fail'}")
        data = {}
        for arm in arms:
            reps = load_arm(arm)
            if not reps:
                print(f"{arm:<26}(no runs)")
                continue
            data[arm] = reps
            accs = [r["correct"] / r["n"] for r in reps.values() if r["n"]]
            runs = "/".join(f"{r['correct']}/{r['n']}" for r in reps.values())
            fails = sum(r["failed"] for r in reps.values())
            if len(accs) >= 2:
                m, lo, hi = t_interval(accs)
                ci = f"{m*100:.2f} [{lo*100:.2f}, {hi*100:.2f}]"
            else:
                ci = f"{accs[0]*100:.2f} (1 rep)" if accs else "-"
            complete = all(r["n"] >= 100 for r in reps.values()) and len(reps) == 5
            mark = "" if complete else "  (INCOMPLETE)"
            print(f"{arm:<26}{runs:<30}{ci:<24}{fails}{mark}")

        stock = arms[0]
        if stock in data and len(data) > 1:
            print(f"  paired McNemar vs {stock} (pooled repeats):")
            for arm in arms[1:]:
                if arm in data:
                    a, b, p = mcnemar(arm, stock, data)
                    print(f"    {arm:<26} wins {a:>3} losses {b:>3}  p={p:.4f}")
            encs = [a for a in arms[1:] if a in data]
            if len(encs) > 1:
                print("  pairwise between encodings:")
                for x, y in itertools.combinations(encs, 2):
                    a, b, p = mcnemar(x, y, data)
                    print(f"    {x} vs {y}: {a}/{b} p={p:.4f}")


if __name__ == "__main__":
    main()
