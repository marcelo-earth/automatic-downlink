"""Build exp6d training set: rebalance to make CRITICAL the dominant class.

Strategy:
- CRITICAL: 3x upsample (10 → 30)
- HIGH: keep all (19)
- LOW: keep all (6)
- MEDIUM: cut to 6 (from 22), keeping the most distinctive non-wildfire samples
  to avoid the wildfire-MEDIUM over-representation that caused MEDIUM collapse.

Output: training/data/exp6d_train.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

TRAIN_IN = Path("training/data/exp6_train.jsonl")
TRAIN_OUT = Path("training/data/exp6d_train.jsonl")

CRITICAL_UPSAMPLE = 3
MEDIUM_KEEP = 6

random.seed(42)


def get_priority(sample: dict) -> str:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            c = msg["content"]
            text = c if isinstance(c, str) else " ".join(x.get("text", "") for x in c)
            m = re.search(r'"priority":\s*"([A-Z]+)"', text)
            if m:
                return m.group(1)
    return "UNKNOWN"


def get_hazard(sample: dict) -> str:
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            c = msg["content"]
            text = (c if isinstance(c, str) else " ".join(x.get("text", "") for x in c)).lower()
            if "wildfire" in text or "burn" in text or "active fire" in text:
                return "wildfire"
            if "flood" in text or "inundat" in text:
                return "flood"
            if "landslide" in text or "debris" in text:
                return "landslide"
    return "other"


samples: dict[str, list] = {"CRITICAL": [], "HIGH": [], "LOW": [], "MEDIUM": []}

with open(TRAIN_IN) as f:
    for line in f:
        s = json.loads(line)
        p = get_priority(s)
        if p in samples:
            samples[p].append(s)

print("Input distribution:")
for k, v in samples.items():
    print(f"  {k}: {len(v)}")

# Upsample CRITICAL
critical = samples["CRITICAL"] * CRITICAL_UPSAMPLE
random.shuffle(critical)

# Keep all HIGH and LOW
high = samples["HIGH"]
low = samples["LOW"]

# Trim MEDIUM: prefer flood and landslide over wildfire
medium_non_wildfire = [s for s in samples["MEDIUM"] if get_hazard(s) != "wildfire"]
medium_wildfire = [s for s in samples["MEDIUM"] if get_hazard(s) == "wildfire"]

# Fill MEDIUM_KEEP slots: non-wildfire first, then wildfire if needed
medium_pool = medium_non_wildfire + medium_wildfire
medium = random.sample(medium_pool, min(MEDIUM_KEEP, len(medium_pool)))

final = critical + high + low + medium
random.shuffle(final)

print(f"\nOutput distribution (target):")
from collections import Counter
counts = Counter(get_priority(s) for s in final)
for k in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
    print(f"  {k}: {counts.get(k, 0)}")
print(f"  TOTAL: {len(final)}")

TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
with open(TRAIN_OUT, "w") as f:
    for s in final:
        f.write(json.dumps(s) + "\n")

print(f"\nWrote {len(final)} samples to {TRAIN_OUT}")
