#!/usr/bin/env python3
"""Validate Exp5 labels for completeness and quality."""

import json
from pathlib import Path
from collections import Counter

def main():
    labels_file = Path("/Users/marcelo/Documents/GitHub/automatic-downlink/training/data/labels_exp5.jsonl")
    captions_file = Path("/Users/marcelo/Documents/GitHub/automatic-downlink/training/data/captions_cleaned.jsonl")

    # Load both files
    labels = []
    with open(labels_file, 'r') as f:
        for line in f:
            labels.append(json.loads(line))

    captions = []
    with open(captions_file, 'r') as f:
        for line in f:
            captions.append(json.loads(line))

    print("=== Label Validation ===\n")

    # Check counts match
    print(f"Captions: {len(captions)}")
    print(f"Labels:   {len(labels)}")
    if len(labels) == len(captions):
        print("✓ Counts match\n")
    else:
        print(f"✗ COUNT MISMATCH!\n")
        return

    # Check all IDs are present and in order
    missing_ids = []
    for i in range(len(captions)):
        if i >= len(labels) or labels[i]["id"] != i:
            missing_ids.append(i)

    if not missing_ids:
        print("✓ All IDs present and in order\n")
    else:
        print(f"✗ Missing or out-of-order IDs: {missing_ids[:10]}...\n")

    # Check required fields
    required_fields = ["id", "priority", "reasoning", "categories"]
    invalid_labels = []
    for label in labels:
        if not all(field in label for field in required_fields):
            invalid_labels.append(label["id"])

    if not invalid_labels:
        print("✓ All labels have required fields\n")
    else:
        print(f"✗ {len(invalid_labels)} labels missing fields: {invalid_labels[:10]}...\n")

    # Check priority values
    valid_priorities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"}
    invalid_priorities = []
    for label in labels:
        if label["priority"] not in valid_priorities:
            invalid_priorities.append((label["id"], label["priority"]))

    if not invalid_priorities:
        print("✓ All priorities are valid\n")
    else:
        print(f"✗ {len(invalid_priorities)} invalid priorities: {invalid_priorities[:10]}...\n")

    # Check reasoning uniqueness
    reasoning_strings = [label["reasoning"] for label in labels]
    reasoning_counts = Counter(reasoning_strings)
    duplicates = {r: c for r, c in reasoning_counts.items() if c > 1}

    if not duplicates:
        print("✓ All reasoning strings are unique\n")
    else:
        print(f"✗ {len(duplicates)} duplicate reasoning strings")
        for r, c in list(duplicates.items())[:3]:
            print(f"  '{r}' appears {c} times")
        print()

    # Check category validity
    valid_categories = {
        "urban", "infrastructure", "vegetation", "water", "terrain",
        "disaster", "environmental_change", "cloud_cover", "vehicles",
        "agriculture", "industrial", "residential", "maritime", "military"
    }
    invalid_categories = []
    for label in labels:
        for cat in label["categories"]:
            if cat not in valid_categories:
                invalid_categories.append((label["id"], cat))

    if not invalid_categories:
        print("✓ All categories are valid\n")
    else:
        print(f"✗ {len(invalid_categories)} invalid categories: {invalid_categories[:10]}...\n")

    # Check categories count (should be 1-3)
    invalid_cat_counts = []
    for label in labels:
        if len(label["categories"]) < 1 or len(label["categories"]) > 3:
            invalid_cat_counts.append((label["id"], len(label["categories"])))

    if not invalid_cat_counts:
        print("✓ All labels have 1-3 categories\n")
    else:
        print(f"✗ {len(invalid_cat_counts)} labels with invalid category count\n")

    # Distribution report
    priority_counts = Counter(label["priority"] for label in labels)
    total = len(labels)

    print("=== Distribution ===\n")
    print(f"Total: {total}\n")

    targets = {
        "CRITICAL": (0.5, 2.0),
        "HIGH": (5.0, 10.0),
        "MEDIUM": (40.0, 55.0),
        "LOW": (20.0, 30.0),
        "SKIP": (10.0, 20.0)
    }

    for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"]:
        count = priority_counts[priority]
        pct = (count / total) * 100
        target_min, target_max = targets[priority]

        status = "✓" if target_min <= pct <= target_max else "✗"
        print(f"{status} {priority:10s}: {count:5d} ({pct:5.2f}%)  [target: {target_min}-{target_max}%]")

    print("\n=== Summary ===")
    print("All validation checks passed!" if not (missing_ids or invalid_labels or invalid_priorities or duplicates or invalid_categories or invalid_cat_counts) else "Some validation checks failed!")


if __name__ == "__main__":
    main()
