#!/usr/bin/env python3
"""Strip GoogleEarth / Google Earth boilerplate from VRSBench captions.

Reads captions_to_classify.jsonl → writes captions_cleaned.jsonl.
The caption text changes; id, image, source fields pass through unchanged.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

INPUT = Path("training/data/captions_to_classify.jsonl")
OUTPUT = Path("training/data/captions_cleaned.jsonl")

# Ordered longest-first so greedy matching works correctly.
PREFIX_PATTERNS = [
    "This high-resolution image from Google Earth shows ",
    "This high-resolution image from GoogleEarth shows ",
    "The high-resolution image from GoogleEarth depicts ",
    "The high-resolution image from GoogleEarth shows ",
    "The high-resolution image from GoogleEarth features ",
    "The high-resolution image from GoogleEarth captures ",
    "The image, sourced from GoogleEarth, shows ",
    "The image, sourced from GoogleEarth, depicts ",
    "The image, sourced from GoogleEarth, features ",
    "The image, sourced from GoogleEarth, captures ",
    "The image from GoogleEarth shows ",
    "The image from GoogleEarth depicts ",
    "The image from GoogleEarth captures ",
    "The image from Google Earth shows ",
    "The image from Google Earth depicts ",
]

# Mid-text phrases to remove (order matters: longest first).
MID_TEXT_PHRASES = [
    ", sourced from GoogleEarth,",
    ", sourced from GoogleEarth",
    " sourced from GoogleEarth,",
    " sourced from GoogleEarth",
    " provided by GoogleEarth",
    " provided by Google Earth",
    " from GoogleEarth,",
    " from GoogleEarth",
    " from Google Earth,",
    " from Google Earth",
]

# Catch-all regex for anything that slips through.
FALLBACK_RE = re.compile(r"\bGoogle\s*Earth\b", re.IGNORECASE)


def clean_caption(text: str) -> str:
    # 1) Try prefix stripping
    for prefix in PREFIX_PATTERNS:
        if text.startswith(prefix):
            text = text[len(prefix):]
            text = text[0].upper() + text[1:] if text else text
            break

    # 2) Remove mid-text phrases
    for phrase in MID_TEXT_PHRASES:
        text = text.replace(phrase, "")

    # 3) Fallback regex for remaining mentions
    text = FALLBACK_RE.sub("", text)

    # 4) Clean up whitespace artifacts
    text = re.sub(r"  +", " ", text).strip()
    # Fix orphaned commas/periods at start
    text = re.sub(r"^[,.\s]+", "", text).strip()
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def main() -> None:
    records = []
    with open(INPUT) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} captions")

    # Clean
    changed = 0
    for r in records:
        original = r["caption"]
        r["caption"] = clean_caption(original)
        if r["caption"] != original:
            changed += 1

    print(f"Modified {changed} captions ({changed/len(records)*100:.1f}%)")

    # Validate: zero GoogleEarth/Google Earth remaining
    violations = []
    for r in records:
        if "GoogleEarth" in r["caption"] or "Google Earth" in r["caption"]:
            violations.append(r)

    if violations:
        print(f"\nFAIL: {len(violations)} captions still contain GoogleEarth!")
        for v in violations[:5]:
            print(f"  id={v['id']}: {v['caption'][:120]}")
        sys.exit(1)

    print("PASS: 0 captions contain GoogleEarth or Google Earth")

    # Spot-check: print 10 random cleaned samples
    import random
    random.seed(42)
    samples = random.sample([r for r in records if r["caption"] != records[0]["caption"]], 10)
    print("\n--- Spot-check (10 random cleaned captions) ---")
    for s in samples:
        print(f"  [{s['id']:5d}] {s['caption'][:120]}")

    # Write output
    with open(OUTPUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"\nWritten to {OUTPUT}")


if __name__ == "__main__":
    main()
