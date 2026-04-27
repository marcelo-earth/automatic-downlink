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

import json
from collections import Counter
from pathlib import Path

from src.triage.prompts import TRIAGE_DUAL_SYSTEM_PROMPT, TRIAGE_DUAL_USER_PROMPT

ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = ROOT / "evals/candidates/hazard_grid_v1.jsonl"
LABELS_PATH = ROOT / "evals/candidates/labels_hazard_grid_v1.jsonl"
DATA_DIR = ROOT / "training/data"

# Paths stored in JSONL are relative to image_root in the training config.
# On Modal the volume mounts at /satellite-vlm/, so image_root="/satellite-vlm"
# and paths here are relative to that mount point.
IMAGE_ROOT_IN_VOLUME = "evals/candidates/hazard_grid_v1"


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

        rgb_local = ROOT / meta["image_path"]
        swir_local = Path(str(rgb_local).replace("__rgb.png", "__swir.png"))

        if not rgb_local.exists() or not swir_local.exists():
            print(f"  WARNING: missing image for {gid}, skipping")
            continue

        # Store volume-relative paths (image_root="/satellite-vlm" in training config)
        rgb_vol_path = f"{IMAGE_ROOT_IN_VOLUME}/{rgb_local.name}"
        swir_vol_path = f"{IMAGE_ROOT_IN_VOLUME}/{swir_local.name}"

        hazard_type = meta.get("hazard_type", "unknown")
        # Use labeler_notes as description — they're concise and don't start with
        # "RGB shows... SWIR confirms..." (which caused the model to use band names as JSON keys).
        assistant_json = json.dumps({
            "description": lab["labeler_notes"],
            "priority": lab["priority"],
            "reasoning": lab["description"],
            "categories": [hazard_type],
        })

        sample = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": TRIAGE_DUAL_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": rgb_vol_path},
                    {"type": "image", "image": swir_vol_path},
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
