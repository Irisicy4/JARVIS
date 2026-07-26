#!/usr/bin/env python3
"""Build da2k_100.jsonl: 100 DA-2K point-pairs as MCQ. Point coordinates
are [y, x]. Marker A (red) = point1, marker B (blue) = point2; the model
picks which marked point is closer to the camera. Deterministic seed 0,
stratified across scene categories."""
import json
import os
import random
from collections import defaultdict

from PIL import Image, ImageDraw

BASE = ("/tmp/claude-1007/-raid-icy-hugginggpt/"
        "0fa4b96b-9a0c-46d6-b868-e57df36ceb1f/scratchpad/da2k/DA-2K")
ROOT = "/raid/icy/jarvis-cathy"
IMGDIR = f"{ROOT}/hugginggpt/server/public/DA2K"
os.makedirs(IMGDIR, exist_ok=True)
rng = random.Random(0)

anns = json.load(open(f"{BASE}/annotations.json"))
by_cat = defaultdict(list)
for img, alist in anns.items():
    by_cat[img.split("/")[1]].append((img, alist[0]))

# stratified round-robin sample of 100
order = []
cats = sorted(by_cat)
for c in cats:
    rng.shuffle(by_cat[c])
idx = 0
while len(order) < 100:
    c = cats[idx % len(cats)]
    if by_cat[c]:
        order.append(by_cat[c].pop())
    idx += 1

items = []
for i, (img, a) in enumerate(order):
    im = Image.open(f"{BASE}/{img}").convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    r = max(6, int(min(W, H) * 0.012))
    # annotations always put the closer point in point1 — randomize which
    # point gets marker A so the answer key is balanced
    swap = rng.random() < 0.5
    pA, pB = (a["point2"], a["point1"]) if swap else (a["point1"], a["point2"])
    closer_is_A = (a["closer_point"] == "point1") != swap
    for (y, x), color, lbl in [(pA, (255, 40, 40), "A"),
                               (pB, (40, 90, 255), "B")]:
        d.ellipse((x - r, y - r, x + r, y + r), fill=color,
                  outline=(255, 255, 255), width=max(2, r // 3))
        d.text((x + r + 3, y - r), lbl, fill=color,
               stroke_width=2, stroke_fill=(255, 255, 255))
    name = f"da2k_{i:04d}.jpg"
    im.save(f"{IMGDIR}/{name}", quality=92)
    gt = "A" if closer_is_A else "B"
    items.append({
        "id": f"DA2K_{i:04d}",
        "image": [f"DA2K/{name}"],
        "video": [],
        "conversations": [
            {"from": "human",
             "value": ("Two points are marked on the image: point A (red circle) "
                       "and point B (blue circle). Which marked point is closer "
                       "to the camera?\nSelect from the following choices:\n"
                       "(A) point A\n(B) point B\n")},
            {"from": "gpt", "value": gt},
        ],
        "task": "DA2K_" + img.split("/")[1],
        "source_image": img,
    })

out = f"{ROOT}/experiments/da2k_100.jsonl"
with open(out, "w") as f:
    for it in items:
        f.write(json.dumps(it) + "\n")
print(f"wrote {out} ({len(items)} items), images in {IMGDIR}")
