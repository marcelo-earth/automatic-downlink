"""Grid-sample hazard locations across time and space to build the v6 training set.

For each location in the JSONL:
- Compute a list of fetch timestamps covering pre-event, during-event, and post-event
- Offset the center by a set of small spatial jitters (different Sentinel tiles)
- Fetch both RGB and SWIR composites at each (timestamp, offset)

The goal is to produce, per location, a mix of hazard-visible and non-hazard frames.
After labeling with a frontier model those will give the class balance for training.

Usage:
    .venv/bin/python scripts/capture_hazard_grid.py \
        --locations-file evals/locations/hazard_grid_v1.jsonl \
        --timestamps-per-location 4 \
        --offsets-per-location 2 \
        --output-dir evals/candidates/hazard_grid_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.simsat.client import SimSatClient

DEFAULT_BASE_URL = "http://localhost:9005"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "candidates" / "hazard_grid_v1"
DEFAULT_CATALOG = REPO_ROOT / "evals" / "candidates" / "hazard_grid_v1.jsonl"

# One km is ~0.009 degrees latitude. Spatial offsets in degrees.
SPATIAL_OFFSETS = [
    (0.0, 0.0),        # center
    (0.035, 0.035),    # ~4 km NE
    (-0.035, 0.035),   # ~4 km NW
    (0.0, -0.05),      # ~5 km S
]

VIEW_SPECS = [
    ("rgb", ["red", "green", "blue"]),
    ("swir", ["swir16", "nir08", "red"]),
]


def parse_window(window: str) -> tuple[datetime, datetime]:
    start, end = window.split("/")
    return (
        datetime.fromisoformat(start).replace(tzinfo=timezone.utc),
        datetime.fromisoformat(end).replace(tzinfo=timezone.utc),
    )


def plan_timestamps(
    *,
    event_peak: str,
    event_window: str,
    count: int,
) -> list[str]:
    """Return `count` ISO timestamps spanning before, during, and after the event peak.

    Strategy: pick evenly-spaced timestamps inside the event window so each location
    contributes pre-hazard, during-hazard, and post-hazard frames. For the default
    count of 4 the result is:
        - start of window (before / baseline)
        - event peak
        - 1/3 of window past peak (aftermath)
        - end of window (further aftermath)
    """
    peak = datetime.fromisoformat(event_peak).replace(tzinfo=timezone.utc)
    start, end = parse_window(event_window)

    before = start
    during = peak
    early_after = peak + (end - peak) / 3
    late_after = end

    all_points = [before, during, early_after, late_after]
    picks = all_points[:count]
    return [dt.replace(microsecond=0).isoformat().replace("+00:00", "Z") for dt in picks]


def load_locations(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fetch_with_retry(fetch_fn, *, retries: int, retry_delay: float, label: str):
    for attempt in range(1, retries + 2):
        try:
            return fetch_fn()
        except Exception as exc:
            if attempt == retries + 1:
                print(f"  [warn] {label}: failed after {attempt} attempt(s): {exc}")
                return None
            print(f"  [warn] {label}: attempt {attempt} failed: {exc}")
            time.sleep(retry_delay)


def capture_tile(
    *,
    client: SimSatClient,
    location: dict[str, Any],
    timestamp: str,
    offset_idx: int,
    lat_offset: float,
    lon_offset: float,
    size_km: float,
    window_seconds: float,
    output_dir: Path,
    retries: int,
    retry_delay: float,
) -> list[dict[str, Any]] | None:
    slug = location["location_slug"]
    hazard_type = location["hazard_type"]
    lat = float(location["lat"]) + lat_offset
    lon = float(location["lon"]) + lon_offset
    group_id = f"{slug}_ts{timestamp.replace(':','').replace('-','')}_o{offset_idx}"

    rows: list[dict[str, Any]] = []
    for view_name, bands in VIEW_SPECS:
        candidate_id = f"{group_id}__{view_name}"
        out_path = output_dir / f"{candidate_id}.png"
        result = fetch_with_retry(
            lambda: client.get_sentinel_historical(
                lon=lon,
                lat=lat,
                timestamp=timestamp,
                spectral_bands=bands,
                size_km=size_km,
                window_seconds=window_seconds,
            ),
            retries=retries,
            retry_delay=retry_delay,
            label=candidate_id,
        )
        if result is None or result.image is None:
            continue
        result.image.save(out_path)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_group_id": group_id,
                "view_name": view_name,
                "image_path": out_path.relative_to(REPO_ROOT).as_posix(),
                "mode": "hazard-grid",
                "source": "simsat",
                "location_slug": slug,
                "location_name": location["location_name"],
                "hazard_type": hazard_type,
                "lat": lat,
                "lon": lon,
                "lat_offset": lat_offset,
                "lon_offset": lon_offset,
                "offset_idx": offset_idx,
                "fetch_timestamp": timestamp,
                "event_peak": location.get("event_peak"),
                "event_window": location.get("event_window"),
                "spectral_bands": bands,
                "review_status": "unreviewed",
                "metadata": {
                    **asdict(result.metadata),
                    "companion_views": [v for v, _ in VIEW_SPECS],
                },
            }
        )
    return rows if rows else None


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            existing = [json.loads(line) for line in fh if line.strip()]

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in existing + rows:
        cid = str(row.get("candidate_id", ""))
        if not cid:
            continue
        if cid not in merged:
            order.append(cid)
        merged[cid] = row

    with path.open("w", encoding="utf-8") as fh:
        for cid in order:
            fh.write(json.dumps(merged[cid]) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-capture hazard training candidates.")
    parser.add_argument("--locations-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timestamps-per-location", type=int, default=4)
    parser.add_argument("--offsets-per-location", type=int, default=2)
    parser.add_argument("--size-km", type=float, default=5.0)
    parser.add_argument("--window-seconds", type=float, default=864000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--limit-locations", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = SimSatClient(base_url=args.base_url)

    locations = load_locations(args.locations_file.resolve())
    if args.limit_locations:
        locations = locations[: args.limit_locations]

    all_rows: list[dict[str, Any]] = []
    for loc in locations:
        slug = loc["location_slug"]
        timestamps = plan_timestamps(
            event_peak=loc["event_peak"],
            event_window=loc["event_window"],
            count=args.timestamps_per_location,
        )
        offsets = SPATIAL_OFFSETS[: args.offsets_per_location]
        print(
            f"[{slug}] hazard={loc['hazard_type']} "
            f"timestamps={len(timestamps)} offsets={len(offsets)}"
        )
        for ts in timestamps:
            for offset_idx, (lat_off, lon_off) in enumerate(offsets):
                rows = capture_tile(
                    client=client,
                    location=loc,
                    timestamp=ts,
                    offset_idx=offset_idx,
                    lat_offset=lat_off,
                    lon_offset=lon_off,
                    size_km=args.size_km,
                    window_seconds=args.window_seconds,
                    output_dir=output_dir,
                    retries=args.retries,
                    retry_delay=args.retry_delay,
                )
                if rows:
                    all_rows.extend(rows)
                    print(f"  captured {rows[0]['candidate_group_id']}")

    if not all_rows:
        print("No candidates captured.")
        return

    append_jsonl(args.catalog.resolve(), all_rows)
    unique_tiles = {row["candidate_group_id"] for row in all_rows}
    print(
        f"Captured {len(all_rows)} images covering {len(unique_tiles)} tiles "
        f"across {len(locations)} locations."
    )
    print(f"Catalog: {args.catalog.resolve()}")


if __name__ == "__main__":
    main()
