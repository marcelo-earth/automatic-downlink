#!/usr/bin/env python3
"""Prepare Exp 6 training dataset: dual-image (RGB + SWIR) from real Sentinel-2 hazard grid.

Reads:
  - evals/candidates/labels_hazard_grid_v1.jsonl  (teacher labels)
  - evals/candidates/hazard_grid_v1.jsonl         (catalog with image paths + metadata)

Writes:
  - training/data/exp6_train.jsonl  (oldest ~80% of timestamps)
  - training/data/exp6_eval.jsonl   (newest ~20% of timestamps)

Temporal split prevents leakage: Sentinel-2 revisits every ~5 days, so random splitting
would place near-identical captures in both sets. Instead we cut by timestamp.
"""

from __future__ import annotations

import base64
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.triage.prompts import TRIAGE_DUAL_SYSTEM_PROMPT, TRIAGE_DUAL_USER_PROMPT

ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = ROOT / "evals/candidates/hazard_grid_v1.jsonl"
LABELS_PATH = ROOT / "evals/candidates/labels_hazard_grid_v1.jsonl"
DATA_DIR = ROOT / "training/data"


def encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def main() -> None:
    # Build catalog lookup: candidate_group_id -> metadata (from rgb row)
    catalog: dict[str, dict] = {}
    with CATALOG_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if row["view_name"] == "rgb":
                catalog[row["candidate_group_id"]] = row

    # Load labels, drop SKIP
    labels: list[dict] = []
    with LABELS_PATH.open() as f:
        for line in f:
            lab = json.loads(line)
            if lab["priority"] != "SKIP":
                labels.append(lab)

    print(f"Labels after dropping SKIP: {len(labels)}")
    print(Counter(l["priority"] for l in labels))

    # Temporal split: sort unique timestamps, oldest 80% -> train
    timestamps = sorted({catalog[l["candidate_group_id"]]["fetch_timestamp"] for l in labels})
    cutoff_idx = max(1, int(len(timestamps) * 0.8))
    train_timestamps = set(timestamps[:cutoff_idx])
    eval_timestamps = set(timestamps[cutoff_idx:])
    print(f"\nTimestamps: {len(timestamps)} total, {len(train_timestamps)} train, {len(eval_timestamps)} eval")
    print(f"  Train cutoff: {timestamps[0]} – {timestamps[cutoff_idx - 1]}")
    print(f"  Eval window:  {timestamps[cutoff_idx]} – {timestamps[-1]}")

    # Build SFT samples
    train_samples: list[dict] = []
    eval_samples: list[dict] = []

    for lab in labels:
        gid = lab["candidate_group_id"]
        meta = catalog.get(gid)
        if meta is None:
            print(f"  WARNING: no catalog entry for {gid}, skipping")
            continue

        rgb_path = ROOT / meta["image_path"]
        # Derive SWIR path from RGB path
        swir_path = Path(str(rgb_path).replace("__rgb.png", "__swir.png"))

        if not rgb_path.exists() or not swir_path.exists():
            print(f"  WARNING: missing image for {gid}, skipping")
            continue

        hazard_type = meta.get("hazard_type", "unknown")
        assistant_json = json.dumps({
            "description": lab["description"],
            "priority": lab["priority"],
            "reasoning": lab["labeler_notes"],
            "categories": [hazard_type],
        })

        sample = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": TRIAGE_DUAL_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": encode_image(rgb_path)},
                    {"type": "image", "image": encode_image(swir_path)},
                    {"type": "text", "text": TRIAGE_DUAL_USER_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_json}]},
            ],
            "priority": lab["priority"],
            "candidate_group_id": gid,
            "fetch_timestamp": meta["fetch_timestamp"],
        }

        if meta["fetch_timestamp"] in train_timestamps:
            train_samples.append(sample)
        else:
            eval_samples.append(sample)

    # Write JSONL (strip metadata keys before writing — keep only messages)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for path, samples in [
        (DATA_DIR / "exp6_train.jsonl", train_samples),
        (DATA_DIR / "exp6_eval.jsonl", eval_samples),
    ]:
        with path.open("w") as f:
            for s in samples:
                f.write(json.dumps({"messages": s["messages"]}) + "\n")

    # Stats
    print(f"\nTrain: {len(train_samples)}, Eval: {len(eval_samples)}")
    for name, samples in [("Train", train_samples), ("Eval", eval_samples)]:
        dist = Counter(s["priority"] for s in samples)
        print(f"\n{name} distribution:")
        for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            c = dist.get(p, 0)
            pct = c / len(samples) * 100 if samples else 0
            print(f"  {p:10s}: {c:4d} ({pct:5.1f}%)")

    print(f"\nFiles written:")
    print(f"  {DATA_DIR / 'exp6_train.jsonl'}")
    print(f"  {DATA_DIR / 'exp6_eval.jsonl'}")


if __name__ == "__main__":
    main()
