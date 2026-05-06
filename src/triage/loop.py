"""Main triage loop: polls SimSat for images, runs VLM triage, stores decisions."""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import time
from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

from src.simsat.client import SimSatClient
from src.triage.engine import TriageEngine
from src.triage.model import TriageModel

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30

# Real disaster sites — burn scars, flood damage, and landslide debris are
# visible in Sentinel-2 imagery months/years after the event.
# Two "normal" scenes at the end provide contrast (should produce SKIP/LOW).
DEMO_LOCATIONS = [
    ("Eaton Fire burn scar, Los Angeles CA (Jan 2025)", 34.18, -118.03),
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
    while True:
        try:
            decision_dict, demo_index = await asyncio.to_thread(
                _fetch_and_triage, client, engine, demo_index
            )
            if decision_dict is not None:
                decisions_store.append(decision_dict)
                logger.info(
                    "Decision #%d: %s — %s",
                    len(decisions_store),
                    decision_dict["priority"],
                    decision_dict["description"][:60],
                )
        except Exception:
            logger.exception("Error in triage loop iteration")

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
) -> tuple[dict | None, int]:
    """Fetch one image from SimSat and triage it. Runs in a thread.

    Returns (decision_dict_or_None, next_demo_index).
    """
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
            timestamp = datetime.now(timezone.utc).isoformat()
            decision = engine.analyze(
                image=result.image,
                timestamp=timestamp,
                position={"lat": position.lat, "lon": position.lon, "alt": position.alt},
                source="sentinel-2",
                swir_image=swir_result.image if swir_result else None,
            )
            d = decision.model_dump(mode="json")
            d["image_b64"] = _image_to_b64(result.image)
            return d, demo_index

        logger.info("No Sentinel-2 image at live position, falling back to demo location.")

    # Fallback: cycle through demo locations using historical endpoint
    name, lat, lon = DEMO_LOCATIONS[demo_index % len(DEMO_LOCATIONS)]
    next_index = demo_index + 1
    logger.info("Demo mode: fetching %s (lat=%.2f, lon=%.2f)", name, lat, lon)

    now = datetime.now(timezone.utc).isoformat()
    result = client.get_sentinel_historical(lon=lon, lat=lat, timestamp=now)
    swir_result = client.get_sentinel_historical(
        lon=lon, lat=lat, timestamp=now, spectral_bands=["swir16", "nir08", "red"]
    )

    if result.image is None:
        logger.info("No Sentinel-2 image for %s, skipping.", name)
        return None, next_index

    timestamp = datetime.now(timezone.utc).isoformat()
    decision = engine.analyze(
        image=result.image,
        timestamp=timestamp,
        position={"lat": lat, "lon": lon, "alt": 550.0},
        source="sentinel-2",
        swir_image=swir_result.image if swir_result and swir_result.image else None,
    )
    d = decision.model_dump(mode="json")
    d["image_b64"] = _image_to_b64(result.image)
    return d, next_index


def _image_to_b64(image: Image.Image, size: int = 128) -> str:
    thumb = image.copy()
    thumb.thumbnail((size, size))
    buf = BytesIO()
    thumb.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()
