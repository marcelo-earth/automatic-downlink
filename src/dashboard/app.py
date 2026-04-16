"""Ground station dashboard — visualizes triage decisions from the satellite."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.triage.schemas import BandwidthStats, Priority, TriageDecision

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

# In-memory store — shared between dashboard routes and triage loop
_decisions: list[dict] = []

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
            )
        )
    else:
        logger.info("No SIMSAT_URL set — dashboard-only mode (POST /api/decisions to add data)")
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


@app.post("/api/decisions")
async def add_decision(decision: TriageDecision):
    _decisions.append(decision.model_dump(mode="json"))
    return {"status": "ok", "total": len(_decisions)}


def _compute_stats() -> dict:
    if not _decisions:
        return {
            "total_images": 0,
            "by_priority": {p.value: 0 for p in Priority},
            "savings_percent": 0,
            "critical_count": 0,
            "high_count": 0,
        }

    by_priority: dict[str, int] = {p.value: 0 for p in Priority}
    for d in _decisions:
        p = d.get("priority", "MEDIUM")
        by_priority[p] = by_priority.get(p, 0) + 1

    total = len(_decisions)
    # Naive: every image downlinked at full resolution (500KB image + 1KB metadata)
    naive_bytes = total * 501 * 1024
    smart_bytes = 0
    for d in _decisions:
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
    }
