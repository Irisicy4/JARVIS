#!/usr/bin/env python3
"""Build tallyqa_complex100.jsonl: 100 complex (is_simple=False) TallyQA
test items as CV-Bench-style MCQ, images saved to
hugginggpt/server/public/TallyQA/. Deterministic (seed 0)."""
import json
import os
import random

from datasets import load_dataset

ROOT = "/raid/icy/jarvis-cathy"
IMGDIR = f"{ROOT}/hugginggpt/server/public/TallyQA"
os.makedirs(IMGDIR, exist_ok=True)
rng = random.Random(0)

ds = load_dataset("vikhyatk/tallyqa-test", streaming=True, split="test")
items = []
img_idx = 0
for ex in ds:
    complex_qs = [q for q in ex["qa"] if not q["is_simple"]]
    if not complex_qs:
        continue
    q = rng.choice(complex_qs)  # one complex question per image
    ans = int(q["answer"])
    img_name = f"tallyqa_{img_idx:04d}.jpg"
    ex["image"].convert("RGB").save(f"{IMGDIR}/{img_name}", quality=92)
    # MCQ: gt + 3 nearest distinct non-negative distractors, shuffled
    cands = [ans]
    delta = 1
    while len(cands) < 4:
        for d in (ans - delta, ans + delta):
            if d >= 0 and d not in cands and len(cands) < 4:
                cands.append(d)
        delta += 1
    rng.shuffle(cands)
    letters = "ABCD"
    gt_letter = letters[cands.index(ans)]
    choice_txt = "".join(f"({letters[i]}) {c}\n" for i, c in enumerate(cands))
    items.append({
        "id": f"TallyQA_{img_idx:04d}",
        "image": [f"TallyQA/{img_name}"],
        "video": [],
        "conversations": [
            {"from": "human",
             "value": f"{q['question']}\nSelect from the following choices:\n{choice_txt}"},
            {"from": "gpt", "value": gt_letter},
        ],
        "task": "TallyQA_complex",
        "answer_int": ans,
        "data_source": q.get("data_source", ""),
    })
    img_idx += 1
    if img_idx >= 100:
        break

out = f"{ROOT}/experiments/tallyqa_complex100.jsonl"
with open(out, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(f"wrote {out} ({len(items)} items), images in {IMGDIR}")
