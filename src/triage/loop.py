"""Main triage loop: polls SimSat for images, runs VLM triage, stores decisions."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from PIL import Image

from src.simsat.client import SimSatClient
from src.triage.engine import TriageEngine
from src.triage.model import TriageModel
from src.triage.scenarios import SCENARIOS

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30

# Real disaster sites — burn scars, flood damage, and landslide debris are
# visible in Sentinel-2 imagery months/years after the event.
# Two "normal" scenes at the end provide contrast (should produce SKIP/LOW).
DEMO_LOCATIONS = [
    ("Eaton Fire burn scar, Los Angeles CA (Jan 2025)", 34.24, -117.99),
    ("Palisades Fire burn scar, Los Angeles CA (Jan 2025)", 34.07, -118.55),
    ("Lahaina wildfire burn scar, Maui HI (Aug 2023)", 20.87, -156.68),
    ("Valencia DANA flood plain, Spain (Nov 2024)", 39.47, -0.38),
    ("Derna flash flood debris, Libya (Sep 2023)", 32.76, 22.64),
    ("Kelowna wildfire burn scar, BC Canada (Aug 2023)", 49.89, -119.49),
    ("Enga Province landslide, Papua New Guinea (May 2024)", -5.40, 143.00),
    ("Alexandroupoli wildfire, Greece (Aug 2023)", 41.10, 25.90),
    ("Open Pacific Ocean (contrast — expect SKIP)", -10.00, -140.00),
    ("Greenland ice sheet (contrast — expect SKIP)", 72.00, -40.00),
]


async def run_triage_loop(
    simsat_url: str,
    decisions_store: list[dict],
    poll_interval: float = POLL_INTERVAL,
    profile: str = "default",
    current_analysis: dict | None = None,
    scenario_state: dict | None = None,
) -> None:
    """Run the triage loop forever, appending decisions to the shared store."""
    logger.info("Triage loop starting — loading model...")
    model = TriageModel()

    await asyncio.to_thread(model.load)
    logger.info("Model loaded. Starting triage loop (interval=%ss)", poll_interval)

    engine = TriageEngine(model=model, profile=profile)
    client = SimSatClient(base_url=simsat_url)

    await _wait_for_simsat(client)

    demo_index = 0
    consecutive_errors = 0
    while True:
        if (scenario_state or {}).get("paused", False):
            await asyncio.sleep(0.5)
            continue
        try:
            gen_before = (scenario_state or {}).get("generation", 0)
            decision_dict, demo_index = await asyncio.to_thread(
                _fetch_and_triage, client, engine, demo_index, current_analysis, scenario_state
            )
            gen_after = (scenario_state or {}).get("generation", 0)
            if decision_dict is not None:
                if gen_before == gen_after:
                    decisions_store.append(decision_dict)
                    if len(decisions_store) > 200:
                        del decisions_store[: len(decisions_store) - 200]
                    logger.info(
                        "Decision #%d: %s — %s",
                        len(decisions_store),
                        decision_dict["priority"],
                        decision_dict["description"][:60],
                    )
                else:
                    logger.info("Discarding stale decision — scenario changed mid-inference")
            consecutive_errors = 0
        except Exception:
            consecutive_errors += 1
            backoff = min(30.0, 2.0 ** consecutive_errors)
            logger.exception("Error in triage loop iteration (backoff=%.0fs)", backoff)
            await asyncio.sleep(backoff)
            continue

        if poll_interval > 0:
            await asyncio.sleep(poll_interval)


async def _wait_for_simsat(client: SimSatClient, timeout: float = 120) -> None:
    """Wait for SimSat API to become available."""
    logger.info("Waiting for SimSat API at %s ...", client.base_url)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if await asyncio.to_thread(client.is_healthy):
            logger.info("SimSat API is ready.")
            return
        await asyncio.sleep(3)
    logger.warning("SimSat API not reachable after %ss — will keep trying in loop.", timeout)


def _fetch_and_triage(
    client: SimSatClient,
    engine: TriageEngine,
    demo_index: int,
    current_analysis: dict | None = None,
    scenario_state: dict | None = None,
) -> tuple[dict | None, int]:
    """Fetch one image from SimSat and triage it. Runs in a thread.

    Returns (decision_dict_or_None, next_demo_index).
    """
    # Scenario replay mode takes priority over both live and demo cycling
    if scenario_state and scenario_state.get("active_key"):
        return _fetch_and_triage_scenario(client, engine, scenario_state, current_analysis), demo_index

    # Try live simulation position first
    position = client.get_position()
    is_live = position.lat != 0.0 or position.lon != 0.0 or position.alt != 0.0

    if is_live:
        logger.info(
            "Live satellite at lat=%.2f lon=%.2f alt=%.0f",
            position.lat, position.lon, position.alt,
        )
        result = client.get_sentinel_current()
        swir_result = client.get_sentinel_current(spectral_bands=["swir16", "nir08", "red"])
        if result.image is not None:
            image, swir_image = _trim_nodata_pair(
                result.image,
                swir_result.image if swir_result and swir_result.image is not None else None,
            )
            timestamp = datetime.now(timezone.utc).isoformat()
            decision = engine.analyze(
                image=image,
                timestamp=timestamp,
                position={"lat": position.lat, "lon": position.lon, "alt": position.alt},
                source="sentinel-2",
                swir_image=swir_image,
            )
            d = decision.model_dump(mode="json")
            d["image_b64"] = _image_to_b64(image)
            if swir_image is not None:
                d["swir_b64"] = _image_to_b64(swir_image)
            return d, demo_index

        logger.info("No Sentinel-2 image at live position, falling back to demo location.")

    # Fallback: cycle through demo locations using historical endpoint
    name, lat, lon = DEMO_LOCATIONS[demo_index % len(DEMO_LOCATIONS)]
    next_index = demo_index + 1
    generation = (scenario_state or {}).get("generation", 0)
    logger.info("Demo mode: fetching %s (lat=%.2f, lon=%.2f)", name, lat, lon)

    # Publish ghost row immediately (no image yet) so the UI shows progress
    # while we fetch RGB + SWIR (this can take a few seconds).
    if current_analysis is not None:
        current_analysis.clear()
        current_analysis["generation"] = generation
        current_analysis["location_name"] = name
        current_analysis["position"] = {"lat": lat, "lon": lon}

    now = datetime.now(timezone.utc).isoformat()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_rgb = pool.submit(client.get_sentinel_historical, lon=lon, lat=lat, timestamp=now)
        fut_swir = pool.submit(
            client.get_sentinel_historical,
            lon=lon, lat=lat, timestamp=now, spectral_bands=["swir16", "nir08", "red"],
        )
        result = fut_rgb.result()
        swir_result = fut_swir.result()

    if result.image is None:
        logger.info("No Sentinel-2 image for %s, skipping.", name)
        if current_analysis is not None:
            current_analysis.clear()
        return None, next_index

    image, swir_image = _trim_nodata_pair(
        result.image,
        swir_result.image if swir_result and swir_result.image else None,
    )
    rgb_b64 = _image_to_b64(image)
    swir_b64 = _image_to_b64(swir_image) if swir_image else None
    if current_analysis is not None:
        current_analysis["image_b64"] = rgb_b64
        if swir_b64:
            current_analysis["swir_b64"] = swir_b64

    timestamp = datetime.now(timezone.utc).isoformat()
    inf_start = time.monotonic()

    def _on_partial(text: str) -> None:
        if current_analysis is not None:
            current_analysis["partial_description"] = text

    try:
        decision = engine.analyze(
            image=image,
            timestamp=timestamp,
            position={"lat": lat, "lon": lon, "alt": 550.0},
            source="sentinel-2",
            swir_image=swir_image,
            on_partial=_on_partial if current_analysis is not None else None,
        )
    finally:
        if current_analysis is not None and current_analysis.get("generation") == generation:
            current_analysis.clear()
    inference_seconds = time.monotonic() - inf_start

    d = decision.model_dump(mode="json")
    d["image_b64"] = rgb_b64
    if swir_b64:
        d["swir_b64"] = swir_b64
    d["location_name"] = name
    d["inference_seconds"] = inference_seconds
    return d, next_index


def _fetch_and_triage_scenario(
    client: SimSatClient,
    engine: TriageEngine,
    scenario_state: dict,
    current_analysis: dict | None,
) -> dict | None:
    """Fetch and triage a single frame of an active scenario timeline."""
    key = scenario_state["active_key"]
    scenario = SCENARIOS.get(key)
    if scenario is None:
        scenario_state["active_key"] = None
        return None

    frame_idx = scenario_state.get("frame_index", 0)
    if frame_idx >= len(scenario.frames):
        scenario_state["active_key"] = None
        scenario_state["frame_index"] = 0
        scenario_state["paused"] = True
        return None
    frame = scenario.frames[frame_idx]
    scenario_state["frame_index"] = frame_idx + 1
    generation = scenario_state.get("generation", 0)

    full_name = f"{scenario.name} — {frame.label}"
    logger.info("Scenario: %s @ %s", full_name, frame.timestamp)

    # Show ghost row immediately while we fetch imagery
    if current_analysis is not None:
        current_analysis.clear()
        current_analysis["generation"] = generation
        current_analysis["location_name"] = full_name
        current_analysis["position"] = {"lat": scenario.lat, "lon": scenario.lon}
        current_analysis["frame_timestamp"] = frame.timestamp
        current_analysis["frame_label"] = frame.label

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_rgb = pool.submit(
            client.get_sentinel_historical,
            lon=scenario.lon, lat=scenario.lat, timestamp=frame.timestamp,
            window_seconds=2592000,  # 30-day window for better cloud-free coverage
        )
        fut_swir = pool.submit(
            client.get_sentinel_historical,
            lon=scenario.lon, lat=scenario.lat, timestamp=frame.timestamp,
            spectral_bands=["swir16", "nir08", "red"],
            window_seconds=2592000,
        )
        result = fut_rgb.result()
        swir_result = fut_swir.result()

    if result.image is None:
        logger.info("No Sentinel-2 image for %s, skipping frame.", full_name)
        if current_analysis is not None:
            current_analysis.clear()
        return None

    image, swir_image = _trim_nodata_pair(
        result.image,
        swir_result.image if swir_result and swir_result.image else None,
    )
    rgb_b64 = _image_to_b64(image)
    swir_b64 = _image_to_b64(swir_image) if swir_image else None
    if current_analysis is not None:
        current_analysis["image_b64"] = rgb_b64
        if swir_b64:
            current_analysis["swir_b64"] = swir_b64

    timestamp = datetime.now(timezone.utc).isoformat()
    inf_start = time.monotonic()

    def _on_partial(text: str) -> None:
        if current_analysis is not None:
            current_analysis["partial_description"] = text

    try:
        decision = engine.analyze(
            image=image,
            timestamp=timestamp,
            position={"lat": scenario.lat, "lon": scenario.lon, "alt": 550.0},
            source="sentinel-2",
            swir_image=swir_image,
            on_partial=_on_partial if current_analysis is not None else None,
        )
    finally:
        if current_analysis is not None and current_analysis.get("generation") == generation:
            current_analysis.clear()
    inference_seconds = time.monotonic() - inf_start

    d = decision.model_dump(mode="json")
    d["image_b64"] = rgb_b64
    if swir_b64:
        d["swir_b64"] = swir_b64
    d["location_name"] = full_name
    d["scenario_frame"] = {"label": frame.label, "timestamp": frame.timestamp, "index": frame_idx}
    d["inference_seconds"] = inference_seconds
    return d


def _image_to_b64(image: Image.Image, size: int = 128) -> str:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def _trim_nodata_pair(
    image: Image.Image,
    swir_image: Image.Image | None = None,
) -> tuple[Image.Image, Image.Image | None]:
    """Remove black Sentinel no-data borders before display and inference.

    SimSat/STAC crops can sit on the edge of a Sentinel granule. The returned
    PNG then has valid pixels on one side and solid black no-data on another.
    Trimming only border-wide no-data preserves real dark burn scars while
    avoiding a misleading black strip in both the UI and the VLM input.
    """
    images = [image.convert("RGB")]
    if swir_image is not None:
        images.append(swir_image.convert("RGB"))

    bbox = _nodata_bbox(images)
    if bbox is None:
        return image, swir_image

    width, height = image.size
    left, upper, right, lower = bbox
    if (left, upper, right, lower) == (0, 0, width, height):
        return image, swir_image

    cropped = image.crop(bbox)
    cropped_swir = swir_image.crop(bbox) if swir_image is not None and swir_image.size == image.size else swir_image
    logger.info(
        "Trimmed Sentinel no-data border: %sx%s -> %sx%s",
        width,
        height,
        cropped.size[0],
        cropped.size[1],
    )
    return cropped, cropped_swir


def _nodata_bbox(images: list[Image.Image]) -> tuple[int, int, int, int] | None:
    """Return a shared bbox excluding solid black border rows/columns."""
    if not images:
        return None

    width, height = images[0].size
    if any(img.size != (width, height) for img in images):
        return None

    combined = np.zeros((height, width), dtype=bool)
    for img in images:
        arr = np.asarray(img, dtype=np.uint8)
        # No-data is encoded as near-black across all channels. A real dark
        # burn scar is not cropped unless it forms an entire image border.
        combined |= np.any(arr > 8, axis=2)

    col_valid = combined.mean(axis=0)
    row_valid = combined.mean(axis=1)
    cols = np.flatnonzero(col_valid > 0.02)
    rows = np.flatnonzero(row_valid > 0.02)
    if not len(cols) or not len(rows):
        return None

    left, right = int(cols[0]), int(cols[-1]) + 1
    upper, lower = int(rows[0]), int(rows[-1]) + 1

    # Avoid tiny accidental crops from compression noise; only act on obvious
    # no-data borders.
    if left < 3 and upper < 3 and (width - right) < 3 and (height - lower) < 3:
        return (0, 0, width, height)

    # If the crop would remove most of the image, keep the original. That is a
    # bad acquisition, but preserving it is safer than hallucinating context.
    area_ratio = ((right - left) * (lower - upper)) / float(width * height)
    if area_ratio < 0.35:
        return (0, 0, width, height)

    return (left, upper, right, lower)
