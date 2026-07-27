#!/usr/bin/env python3
"""Aggregate the Table-4 CI reruns (adapted from spagent-william's
experiments/ci_aggregate.py).

Three views:
1. Per-run per-item bootstrap CI (dataset sampling noise), with a
   summary-vs-items consistency guard.
2. Across-repeat mean +/- t-interval (run-to-run stochasticity).
3. Paired per-item comparison between arms (exact-binomial McNemar over
   pooled (item, repeat) pairs on shared scored items).

Layout expected: results/table4-cvbench/<arm>/ci_runs/repeat<K>/result.json
"""
import glob
import itertools
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_table4 import ARMS, PAPER, first100_ids, score  # noqa: E402

RESULTS = os.path.join(HERE, "results", "table4-cvbench")


def bootstrap_ci(items, iters=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(items)
    stats = sorted(
        sum(items[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters)
    )
    return stats[int(alpha / 2 * iters)], stats[int((1 - alpha / 2) * iters) - 1]


def t_interval(vals, alpha=0.05):
    """mean, mean-sd, mean+sd across repeats. NOT a t-interval: with 2-5
    repeats a t-CI multiplies sd by 4.3-12.7, manufacturing false precision.
    Descriptive only; significance comes from the paired per-item tests."""
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1))
    return m, m - sd, m + sd


def binom_two_sided_p(k, n):
    """Exact two-sided binomial test p-value for k successes of n at p=0.5."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) / 2 ** n
    p_k = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= p_k + 1e-12))


def load_repeats(arm):
    """Return list of (repeat_tag, per_item dict, correct, n, failed)."""
    out = []
    ids = first100_ids()
    for rep in sorted(glob.glob(os.path.join(RESULTS, arm, "ci_runs", "repeat*/"))):
        f = os.path.join(rep, "result.json")
        if not os.path.exists(f):
            continue
        entries = json.load(open(f))
        c, n, fl, per_item = score(entries, ids)
        # consistency guard: summary accuracy must equal per-item mean
        scored = [v for v in per_item.values() if v is not None]
        if scored and abs(sum(scored) / len(per_item) * len(per_item) - c) > 1e-9:
            print(f"  !! {arm}/{os.path.basename(rep.rstrip('/'))}: "
                  f"summary/items inconsistent — skipping bootstrap")
            per_item = None
        out.append((os.path.basename(rep.rstrip("/")), per_item, c, n, fl))
    return out


def main():
    print(f"{'arm':<22}{'paper':>6}  repeats (correct/100)          mean [95% t-CI]")
    print("-" * 78)
    all_items = {}
    for arm in ARMS:
        reps = load_repeats(arm)
        if not reps:
            print(f"{arm:<22}{PAPER[arm]:>6}  (no ci_runs)")
            continue
        accs = [c / n for _, _, c, n, _ in reps]
        m, lo, hi = t_interval(accs)
        runs = "/".join(str(c) for _, _, c, _, _ in reps)
        fails = sum(fl for *_, fl in reps)
        print(f"{arm:<22}{PAPER[arm]:>6}  {runs:<28}  "
              f"{m * 100:.2f} [{lo * 100:.2f}, {hi * 100:.2f}]  (api_fail tot {fails})")
        for tag, per_item, c, n, _ in reps:
            if per_item:
                blo, bhi = bootstrap_ci(
                    [1 if v else 0 for v in per_item.values()])
                print(f"    {tag}: {c}/{n}  bootstrap [{blo * 100:.2f}, {bhi * 100:.2f}]")
                all_items.setdefault(arm, {})[tag] = per_item

    print("\nPaired McNemar (pooled over repeats, shared scored items):")
    print(f"{'pair':<44}{'b(A>B)':>7}{'c(B>A)':>7}{'p':>9}")
    for a, b in itertools.combinations([x for x in ARMS if x in all_items], 2):
        wins_a = wins_b = 0
        for tag in set(all_items[a]) & set(all_items[b]):
            pa, pb = all_items[a][tag], all_items[b][tag]
            for i in set(pa) & set(pb):
                if pa[i] is None or pb[i] is None:
                    continue
                if pa[i] and not pb[i]:
                    wins_a += 1
                elif pb[i] and not pa[i]:
                    wins_b += 1
        p = binom_two_sided_p(wins_a, wins_a + wins_b)
        print(f"{a + ' vs ' + b:<44}{wins_a:>7}{wins_b:>7}{p:>9.4f}")


if __name__ == "__main__":
    main()
