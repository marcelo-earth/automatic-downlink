"""Ground station dashboard - visualizes triage decisions from the satellite."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.triage.scenarios import SCENARIOS, list_scenarios
from src.triage.schemas import BandwidthStats, Priority, TriageDecision

# Configure logging so our messages show up alongside uvicorn's
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

MAX_DECISIONS = 200

# In-memory stores - shared between dashboard routes and triage loop
_decisions: list[dict] = []
_current_analysis: dict = {}  # populated while inference is running
_scenario_state: dict = {"active_key": None, "frame_index": 0, "generation": 0, "paused": True}

# Environment config
SIMSAT_URL = os.environ.get("SIMSAT_URL", "")
TRIAGE_PROFILE = os.environ.get("TRIAGE_PROFILE", "default")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "30"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the triage loop as a background task when SimSat URL is configured."""
    task = None
    if SIMSAT_URL:
        from src.triage.loop import run_triage_loop
        logger.info("Starting triage loop → %s (profile=%s)", SIMSAT_URL, TRIAGE_PROFILE)
        task = asyncio.create_task(
            run_triage_loop(
                simsat_url=SIMSAT_URL,
                decisions_store=_decisions,
                poll_interval=POLL_INTERVAL,
                profile=TRIAGE_PROFILE,
                current_analysis=_current_analysis,
                scenario_state=_scenario_state,
            )
        )
    else:
        logger.info("No SIMSAT_URL set - dashboard-only mode (POST /api/decisions to add data)")
    yield
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="automatic-downlink",
    description="Ground Station Dashboard",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    stats = _compute_stats()
    return templates.TemplateResponse(request, "index.html", {
        "decisions": _decisions[-50:],
        "stats": stats,
    })


@app.get("/api/decisions")
async def get_decisions(limit: int = 50):
    return _decisions[-limit:]


@app.get("/api/stats")
async def get_stats():
    return _compute_stats()


@app.get("/api/current")
async def get_current():
    """Return image+location currently being analyzed, or empty {} if idle."""
    return dict(_current_analysis)


@app.get("/api/scenarios")
async def get_scenarios():
    """List available temporal-replay scenarios."""
    return {
        "scenarios": list_scenarios(),
        "active_key": _scenario_state.get("active_key"),
        "paused": _scenario_state.get("paused", True),
    }


@app.post("/api/scenarios/{key}")
async def set_scenario(key: str):
    """Activate a scenario, or 'off' to return to demo cycling."""
    if key == "off":
        _scenario_state["active_key"] = None
        _scenario_state["frame_index"] = 0
        _scenario_state["paused"] = False
        _scenario_state["generation"] = _scenario_state["generation"] + 1
        _current_analysis.clear()
        return {"status": "ok", "active_key": None}
    if key == "paused":
        _scenario_state["active_key"] = None
        _scenario_state["paused"] = True
        _scenario_state["generation"] = _scenario_state["generation"] + 1
        _decisions.clear()
        _current_analysis.clear()
        return {"status": "ok", "active_key": None}
    if key not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown scenario: {key}")
    scenario = SCENARIOS[key]
    first_frame = scenario.frames[0] if scenario.frames else None
    _scenario_state["active_key"] = key
    _scenario_state["frame_index"] = 0
    _scenario_state["paused"] = False
    _scenario_state["generation"] = _scenario_state["generation"] + 1
    _decisions.clear()
    _current_analysis.clear()
    if first_frame is not None:
        _current_analysis.update({
            "generation": _scenario_state["generation"],
            "location_name": f"{scenario.name} - {first_frame.label}",
            "position": {"lat": scenario.lat, "lon": scenario.lon},
            "frame_timestamp": first_frame.timestamp,
            "frame_label": first_frame.label,
            "partial_description": "Waiting for model slot…",
        })
    return {"status": "ok", "active_key": key}


@app.get("/api/position")
async def get_position():
    if not SIMSAT_URL:
        return {"lat": 0, "lon": 0, "alt": 0, "live": False}
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{SIMSAT_URL}/data/current/position", timeout=5)
            data = r.json()
            lon, lat, alt = data["lon-lat-alt"]
            live = not (lon == 0 and lat == 0 and alt == 0)
            return {"lat": lat, "lon": lon, "alt": alt, "live": live}
    except Exception:
        logger.exception("Failed to fetch satellite position")
        return {"lat": 0, "lon": 0, "alt": 0, "live": False}


@app.post("/api/decisions")
async def add_decision(decision: TriageDecision):
    _decisions.append(decision.model_dump(mode="json"))
    if len(_decisions) > MAX_DECISIONS:
        del _decisions[: len(_decisions) - MAX_DECISIONS]
    return {"status": "ok", "total": len(_decisions)}


def _compute_stats() -> dict:
    snapshot = list(_decisions)
    if not snapshot:
        return {
            "total_images": 0,
            "by_priority": {p.value: 0 for p in Priority},
            "savings_percent": 0,
            "critical_count": 0,
            "high_count": 0,
        }

    by_priority: dict[str, int] = {p.value: 0 for p in Priority}
    for d in snapshot:
        p = d.get("priority", "MEDIUM")
        by_priority[p] = by_priority.get(p, 0) + 1

    total = len(snapshot)
    # Naive: every image downlinked at full resolution (500KB image + 1KB metadata)
    naive_bytes = total * 501 * 1024
    smart_bytes = 0
    for d in snapshot:
        action = d.get("downlink_action", "TRANSMIT_SUMMARY_ONLY")
        if action == "TRANSMIT_IMAGE":
            smart_bytes += 501 * 1024
        elif action == "TRANSMIT_THUMBNAIL":
            smart_bytes += 51 * 1024
        else:
            smart_bytes += 1024

    savings = ((naive_bytes - smart_bytes) / naive_bytes * 100) if naive_bytes > 0 else 0

    return {
        "total_images": total,
        "by_priority": by_priority,
        "savings_percent": round(savings, 1),
        "critical_count": by_priority.get("CRITICAL", 0),
        "high_count": by_priority.get("HIGH", 0),
        "poll_interval": POLL_INTERVAL,
    }
