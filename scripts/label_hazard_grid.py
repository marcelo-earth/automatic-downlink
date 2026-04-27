"""Prepare the captured hazard grid for multi-agent labeling.

The actual labeling happens via the Claude Agent tool (Sonnet sub-agents).
This script:
1. Reads the capture catalog (hazard_grid_v1.jsonl)
2. Groups RGB + SWIR pairs by candidate_group_id
3. Splits them into batches of N pairs each (one batch -> one agent)
4. Writes batch JSON files that an agent can read and act on
5. Writes a final merge step that combines all per-batch label files

Run labeling by spawning Sonnet agents, one per batch file. Each agent reads
its batch, opens each RGB and SWIR image, writes labels_<batch_id>.jsonl.

Finally run `python scripts/label_hazard_grid.py --merge` to produce
labels_hazard_grid_v1.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CATALOG = REPO_ROOT / "evals" / "candidates" / "hazard_grid_v1.jsonl"
BATCHES_DIR = REPO_ROOT / "evals" / "candidates" / "labeling_batches_v1"
LABELS_DIR = REPO_ROOT / "evals" / "candidates" / "labels_v1"
MERGED_LABELS = REPO_ROOT / "evals" / "candidates" / "labels_hazard_grid_v1.jsonl"


def load_catalog(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def group_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group RGB + SWIR rows by candidate_group_id into a single pair record."""
    groups: dict[str, dict[str, Any]] = {}
    for row in rows:
        gid = row["candidate_group_id"]
        if gid not in groups:
            groups[gid] = {
                "candidate_group_id": gid,
                "location_slug": row["location_slug"],
                "location_name": row["location_name"],
                "hazard_type": row["hazard_type"],
                "lat": row["lat"],
                "lon": row["lon"],
                "fetch_timestamp": row["fetch_timestamp"],
                "event_peak": row.get("event_peak"),
                "event_window": row.get("event_window"),
                "rgb_path": None,
                "swir_path": None,
            }
        if row["view_name"] == "rgb":
            groups[gid]["rgb_path"] = row["image_path"]
        elif row["view_name"] == "swir":
            groups[gid]["swir_path"] = row["image_path"]
    # Keep only pairs that have both views
    return [g for g in groups.values() if g["rgb_path"] and g["swir_path"]]


def make_batches(pairs: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    batches = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        batch_id = f"batch_{i // batch_size:02d}"
        batch_path = BATCHES_DIR / f"{batch_id}.json"
        batch_payload = {
            "batch_id": batch_id,
            "pairs": chunk,
            "output_file": str((LABELS_DIR / f"{batch_id}.jsonl").relative_to(REPO_ROOT)),
        }
        batch_path.write_text(json.dumps(batch_payload, indent=2))
        batches.append({"batch_id": batch_id, "path": str(batch_path), "pair_count": len(chunk)})
    return batches


def merge_labels() -> None:
    if not LABELS_DIR.exists():
        print(f"No labels directory at {LABELS_DIR}")
        return
    all_labels: list[dict[str, Any]] = []
    for label_file in sorted(LABELS_DIR.glob("*.jsonl")):
        with label_file.open() as fh:
            for line in fh:
                if line.strip():
                    all_labels.append(json.loads(line))
    if not all_labels:
        print("No labels found.")
        return
    MERGED_LABELS.parent.mkdir(parents=True, exist_ok=True)
    with MERGED_LABELS.open("w") as fh:
        for row in all_labels:
            fh.write(json.dumps(row) + "\n")
    print(f"Merged {len(all_labels)} labels -> {MERGED_LABELS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--merge", action="store_true", help="Merge per-batch label files")
    args = parser.parse_args()

    if args.merge:
        merge_labels()
        return

    rows = load_catalog(CATALOG)
    pairs = group_pairs(rows)
    print(f"Loaded {len(rows)} catalog rows -> {len(pairs)} (RGB, SWIR) pairs")

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    batches = make_batches(pairs, args.batch_size)
    print(f"Wrote {len(batches)} batch files to {BATCHES_DIR}")
    for b in batches:
        print(f"  {b['batch_id']}: {b['pair_count']} pairs -> {b['path']}")


if __name__ == "__main__":
    main()
