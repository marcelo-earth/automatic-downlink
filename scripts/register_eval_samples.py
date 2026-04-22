"""Validate and append reviewed samples into an evaluation manifest.

Examples:
    .venv/bin/python scripts/register_eval_samples.py \
      --manifest evals/sentinel_eval_v1.jsonl \
      --image-path test_images/sentinel_sahara.png \
      --expected-priority LOW \
      --notes "Routine desert tile."

    .venv/bin/python scripts/register_eval_samples.py \
      --manifest evals/sentinel_eval_v1.jsonl \
      --batch /tmp/reviewed_rows.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_PRIORITIES = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append reviewed rows to an eval manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Target evaluation manifest JSONL.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        help="JSONL file with fully reviewed rows to append.",
    )
    parser.add_argument("--image-path", help="Image path relative to the repo root.")
    parser.add_argument("--expected-priority", help="Priority label for the single-row mode.")
    parser.add_argument("--notes", help="Reviewer note for the single-row mode.")
    parser.add_argument("--id", dest="sample_id", help="Optional stable id.")
    parser.add_argument("--source", default="local", help="Sample source tag.")
    parser.add_argument(
        "--ambiguous",
        action="store_true",
        help="Mark the sample as ambiguous.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace rows with matching ids instead of failing on duplicates.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def validate_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in ("image_path", "expected_priority", "notes") if not row.get(field)]
    if missing:
        raise ValueError(f"Row missing required fields: {', '.join(missing)}")

    priority = str(row["expected_priority"]).upper()
    if priority not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority: {priority}")

    image_path = REPO_ROOT / str(row["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    normalized = {
        "image_path": Path(row["image_path"]).as_posix(),
        "expected_priority": priority,
        "notes": str(row["notes"]).strip(),
    }
    if row.get("id"):
        normalized["id"] = str(row["id"]).strip()
    if row.get("source"):
        normalized["source"] = str(row["source"]).strip()
    if row.get("ambiguous") is not None:
        normalized["ambiguous"] = bool(row["ambiguous"])
    return normalized


def single_row_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if not args.image_path or not args.expected_priority or not args.notes:
        raise ValueError(
            "Single-row mode requires --image-path, --expected-priority, and --notes."
        )
    return validate_row(
        {
            "id": args.sample_id,
            "image_path": args.image_path,
            "expected_priority": args.expected_priority,
            "notes": args.notes,
            "source": args.source,
            "ambiguous": args.ambiguous,
        }
    )


def rows_from_args(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.batch:
        return [validate_row(row) for row in load_jsonl(args.batch.resolve())]
    return [single_row_from_args(args)]


def merge_rows(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    replace: bool,
) -> list[dict[str, Any]]:
    existing_by_id = {row["id"]: row for row in existing if "id" in row}
    merged = existing[:]

    for row in incoming:
        row_id = row.get("id")
        if row_id and row_id in existing_by_id:
            if not replace:
                raise ValueError(f"Duplicate id in manifest: {row_id}")
            merged = [existing_row for existing_row in merged if existing_row.get("id") != row_id]
        merged.append(row)
    return merged


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    incoming = rows_from_args(args)
    existing = load_jsonl(manifest_path)
    merged = merge_rows(existing, incoming, replace=args.replace)
    write_jsonl(manifest_path, merged)
    print(f"Wrote {len(incoming)} row(s) into {manifest_path}")


if __name__ == "__main__":
    main()
