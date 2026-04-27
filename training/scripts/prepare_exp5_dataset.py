#!/usr/bin/env python3
"""Prepare Exp 5 training dataset: cleaned captions + new labels + aligned prompt.

Reads:
  - training/data/captions_cleaned.jsonl (GoogleEarth-free captions)
  - training/data/labels_exp5.jsonl (new priority/reasoning/categories)

Writes:
  - training/data/exp5_train.jsonl (90% stratified split)
  - training/data/exp5_eval.jsonl (10% stratified split)

Also writes captions.jsonl and labels.jsonl to the HuggingFace dataset format
for upload to marcelo-earth/VRSBench-satellite-triage-labels.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.triage.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT

DATA_DIR = Path("training/data")


def main() -> None:
    with open(DATA_DIR / "captions_cleaned.jsonl") as f:
        captions = [json.loads(line) for line in f]

    with open(DATA_DIR / "labels_exp5.jsonl") as f:
        labels = [json.loads(line) for line in f]

    assert len(captions) == len(labels), f"Mismatch: {len(captions)} captions vs {len(labels)} labels"

    # Build SFT samples
    by_priority: dict[str, list[dict]] = defaultdict(list)
    for cap, lab in zip(captions, labels):
        assert cap["id"] == lab["id"], f"ID mismatch: {cap['id']} vs {lab['id']}"

        triage_json = json.dumps({
            "description": cap["caption"],
            "priority": lab["priority"],
            "reasoning": lab["reasoning"],
            "categories": lab["categories"],
        })

        sample = {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": TRIAGE_SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "image", "image": cap["image"]},
                    {"type": "text", "text": TRIAGE_USER_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": triage_json}]},
            ],
            "priority": lab["priority"],
        }
        by_priority[lab["priority"]].append(sample)

    # Stratified 90/10 split
    random.seed(42)
    train_samples = []
    eval_samples = []

    for priority, samples in by_priority.items():
        random.shuffle(samples)
        split_idx = max(1, int(len(samples) * 0.9))
        train_samples.extend(samples[:split_idx])
        eval_samples.extend(samples[split_idx:])

    random.shuffle(train_samples)
    random.shuffle(eval_samples)

    # Write train/eval JSONL
    for path, samples in [
        (DATA_DIR / "exp5_train.jsonl", train_samples),
        (DATA_DIR / "exp5_eval.jsonl", eval_samples),
    ]:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    # Write HF dataset files (captions.jsonl + labels.jsonl)
    hf_captions = DATA_DIR / "exp5_captions.jsonl"
    hf_labels = DATA_DIR / "exp5_labels.jsonl"
    with open(hf_captions, "w") as fc, open(hf_labels, "w") as fl:
        for cap, lab in zip(captions, labels):
            fc.write(json.dumps({"image": cap["image"], "caption": cap["caption"]}) + "\n")
            fl.write(json.dumps({
                "id": lab["id"],
                "priority": lab["priority"],
                "reasoning": lab["reasoning"],
                "categories": lab["categories"],
            }) + "\n")

    # Stats
    print(f"Train: {len(train_samples)}, Eval: {len(eval_samples)}")
    for name, samples in [("Train", train_samples), ("Eval", eval_samples)]:
        dist = Counter(s["priority"] for s in samples)
        print(f"\n{name} distribution:")
        for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"]:
            c = dist.get(p, 0)
            print(f"  {p:10s}: {c:6d} ({c/len(samples)*100:5.1f}%)")

    print(f"\nFiles written:")
    print(f"  {DATA_DIR / 'exp5_train.jsonl'}")
    print(f"  {DATA_DIR / 'exp5_eval.jsonl'}")
    print(f"  {hf_captions}")
    print(f"  {hf_labels}")


if __name__ == "__main__":
    main()
