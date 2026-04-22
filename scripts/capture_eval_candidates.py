"""Capture candidate Sentinel-2 eval samples from a local SimSat instance.

This script stages unlabeled candidate images and metadata for later review.
It does not write directly into an evaluation manifest with priorities.

Examples:
    .venv/bin/python scripts/capture_eval_candidates.py --dry-run
    .venv/bin/python scripts/capture_eval_candidates.py --limit 5
    .venv/bin/python scripts/capture_eval_candidates.py --mode current
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(REPO_ROOT))

from src.simsat.client import SimSatClient

DEFAULT_BASE_URL = "http://localhost:9005"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "candidates" / "images"
DEFAULT_CATALOG = REPO_ROOT / "evals" / "candidates" / "sentinel_candidates.jsonl"
DEFAULT_BANDS = ["red", "green", "blue"]

DEMO_LOCATIONS = [
    ("lausanne", "Lausanne, Switzerland", 46.52, 6.63),
    ("amazon", "Amazon rainforest", -3.47, -62.21),
    ("sahara", "Sahara desert", 24.00, 2.00),
    ("lima", "Lima, Peru", -12.04, -77.03),
    ("tokyo", "Tokyo, Japan", 35.68, 139.69),
    ("cape_town", "Cape Town, South Africa", -33.92, 18.42),
    ("greenland", "Greenland ice sheet", 72.00, -40.00),
    ("outback", "Australian outback", -25.27, 134.21),
    ("nile_delta", "Nile delta, Egypt", 30.87, 31.32),
    ("borneo", "Borneo rainforest", 1.50, 110.00),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture candidate Sentinel eval images from SimSat.")
    parser.add_argument(
        "--mode",
        choices=("historical-demo", "current"),
        default="historical-demo",
        help="Capture from demo historical locations or the current live position.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"SimSat base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to save captured images (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
        help=f"Where to append candidate metadata (default: {DEFAULT_CATALOG.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--timestamp",
        help="ISO timestamp for historical fetches. Defaults to current UTC time.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many demo locations to try in historical-demo mode (default: 10).",
    )
    parser.add_argument(
        "--size-km",
        type=float,
        default=5.0,
        help="Sentinel crop size in kilometers (default: 5.0).",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=864000,
        help="Historical lookup window in seconds (default: 864000).",
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=DEFAULT_BANDS,
        help=f"Spectral bands to request (default: {' '.join(DEFAULT_BANDS)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned captures without calling SimSat or writing files.",
    )
    return parser.parse_args()


def historical_timestamp(explicit_timestamp: str | None) -> str:
    if explicit_timestamp:
        return explicit_timestamp
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_candidate_row(
    *,
    candidate_id: str,
    image_path: Path,
    mode: str,
    location_slug: str,
    location_name: str,
    lat: float,
    lon: float,
    fetch_timestamp: str,
    spectral_bands: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "image_path": image_path.relative_to(REPO_ROOT).as_posix(),
        "mode": mode,
        "source": "simsat",
        "location_slug": location_slug,
        "location_name": location_name,
        "lat": lat,
        "lon": lon,
        "fetch_timestamp": fetch_timestamp,
        "spectral_bands": spectral_bands,
        "review_status": "unreviewed",
        "metadata": metadata,
    }


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def capture_historical_demo(args: argparse.Namespace, client: SimSatClient) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fetch_timestamp = historical_timestamp(args.timestamp)
    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for location_slug, location_name, lat, lon in DEMO_LOCATIONS[: args.limit]:
        candidate_id = f"{location_slug}_{fetch_timestamp.replace(':', '').replace('-', '')}"
        output_path = output_dir / f"{candidate_id}.png"

        if args.dry_run:
            rows.append(
                build_candidate_row(
                    candidate_id=candidate_id,
                    image_path=output_path,
                    mode=args.mode,
                    location_slug=location_slug,
                    location_name=location_name,
                    lat=lat,
                    lon=lon,
                    fetch_timestamp=fetch_timestamp,
                    spectral_bands=args.bands,
                    metadata={"dry_run": True},
                )
            )
            continue

        result = client.get_sentinel_historical(
            lon=lon,
            lat=lat,
            timestamp=fetch_timestamp,
            spectral_bands=args.bands,
            size_km=args.size_km,
            window_seconds=args.window_seconds,
        )
        if result.image is None:
            continue
        result.image.save(output_path)
        rows.append(
            build_candidate_row(
                candidate_id=candidate_id,
                image_path=output_path,
                mode=args.mode,
                location_slug=location_slug,
                location_name=location_name,
                lat=lat,
                lon=lon,
                fetch_timestamp=fetch_timestamp,
                spectral_bands=args.bands,
                metadata=asdict(result.metadata),
            )
        )
    return rows


def capture_current(args: argparse.Namespace, client: SimSatClient) -> list[dict[str, Any]]:
    fetch_timestamp = historical_timestamp(args.timestamp)
    output_dir = args.output_dir.resolve()
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        candidate_id = f"current_{fetch_timestamp.replace(':', '').replace('-', '')}"
        output_path = output_dir / f"{candidate_id}.png"
        return [
            build_candidate_row(
                candidate_id=candidate_id,
                image_path=output_path,
                mode=args.mode,
                location_slug="current",
                location_name="Current SimSat position",
                lat=0.0,
                lon=0.0,
                fetch_timestamp=fetch_timestamp,
                spectral_bands=args.bands,
                metadata={"dry_run": True},
            )
        ]

    position = client.get_position()
    result = client.get_sentinel_current(
        spectral_bands=args.bands,
        size_km=args.size_km,
        window_seconds=args.window_seconds,
    )
    if result.image is None:
        return []

    candidate_id = f"current_{fetch_timestamp.replace(':', '').replace('-', '')}"
    output_path = output_dir / f"{candidate_id}.png"
    result.image.save(output_path)
    row = build_candidate_row(
        candidate_id=candidate_id,
        image_path=output_path,
        mode=args.mode,
        location_slug="current",
        location_name="Current SimSat position",
        lat=position.lat,
        lon=position.lon,
        fetch_timestamp=fetch_timestamp,
        spectral_bands=args.bands,
        metadata={
            "satellite_position": asdict(position),
            "sentinel": asdict(result.metadata),
        },
    )
    return [row]


def main() -> None:
    args = parse_args()
    client = SimSatClient(base_url=args.base_url)

    if args.mode == "historical-demo":
        rows = capture_historical_demo(args, client)
    else:
        rows = capture_current(args, client)

    if not rows:
        print("No candidate images captured.")
        return

    if args.dry_run:
        print(json.dumps(rows, indent=2))
        print(f"Dry run only. No files written to {args.output_dir} or {args.catalog}.")
        return

    append_jsonl(args.catalog.resolve(), rows)
    print(f"Captured {len(rows)} candidate image(s).")
    print(f"Images: {args.output_dir.resolve()}")
    print(f"Catalog: {args.catalog.resolve()}")


if __name__ == "__main__":
    main()
