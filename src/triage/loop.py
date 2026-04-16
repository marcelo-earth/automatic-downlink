"""Main triage loop: polls SimSat for images, runs VLM triage, stores decisions."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from datetime import datetime, timezone

from src.simsat.client import SimSatClient
from src.triage.engine import TriageEngine
from src.triage.model import TriageModel

logger = logging.getLogger(__name__)

POLL_INTERVAL = 30

# Interesting Earth locations for demo/fallback when simulation isn't running.
# Each tuple: (name, lat, lon) — covers diverse terrain types.
DEMO_LOCATIONS = [
    ("Lausanne, Switzerland", 46.52, 6.63),
    ("Amazon rainforest", -3.47, -62.21),
    ("Sahara desert", 24.00, 2.00),
    ("Lima, Peru", -12.04, -77.03),
    ("Tokyo, Japan", 35.68, 139.69),
    ("Cape Town, South Africa", -33.92, 18.42),
    ("Greenland ice sheet", 72.00, -40.00),
    ("Australian outback", -25.27, 134.21),
    ("Nile delta, Egypt", 30.87, 31.32),
    ("Borneo rainforest", 1.50, 110.00),
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
        if result.image is not None:
            timestamp = datetime.now(timezone.utc).isoformat()
            decision = engine.analyze(
                image=result.image,
                timestamp=timestamp,
                position={"lat": position.lat, "lon": position.lon, "alt": position.alt},
                source="sentinel-2",
            )
            return decision.model_dump(mode="json"), demo_index

        logger.info("No Sentinel-2 image at live position, falling back to demo location.")

    # Fallback: cycle through demo locations using historical endpoint
    name, lat, lon = DEMO_LOCATIONS[demo_index % len(DEMO_LOCATIONS)]
    next_index = demo_index + 1
    logger.info("Demo mode: fetching %s (lat=%.2f, lon=%.2f)", name, lat, lon)

    result = client.get_sentinel_historical(
        lon=lon,
        lat=lat,
        timestamp=datetime.now(timezone.utc).isoformat(),
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
    )
    return decision.model_dump(mode="json"), next_index
