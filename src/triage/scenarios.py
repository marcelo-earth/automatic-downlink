"""Temporal event scenarios - replay an event's evolution at a fixed location.

Each scenario is a sequence of (timestamp, label) frames. The triage loop
fetches Sentinel-2 imagery at each timestamp and runs inference, so judges
can see the model's reasoning shift as the same site changes over time:
e.g. clean baseline → active hazard → post-event debris → recovery.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioFrame:
    timestamp: str  # ISO 8601 datetime
    label: str  # human-readable phase, e.g. "Pre-event baseline"


@dataclass(frozen=True)
class Scenario:
    key: str
    name: str  # full location name shown in UI
    lat: float
    lon: float
    frames: tuple[ScenarioFrame, ...]
    peak_event: str = ""  # short context string shown in sidebar


SCENARIOS: dict[str, Scenario] = {
    # === Featured scenarios (validated baselines, expanded timelines) ===
    "eaton-fire": Scenario(
        key="eaton-fire",
        name="Eaton Fire, Los Angeles CA",
        lat=34.203483,
        lon=-118.069155,
        peak_event="Fire ignited Jan 7, 2025 near Altadena Drive/Midwick Drive - peak destruction Jan 8–13",
        frames=(
            ScenarioFrame("2024-04-10T18:00:00Z", "Quiet - Spring 2024"),
            ScenarioFrame("2024-08-15T18:00:00Z", "Quiet - Summer 2024"),
            ScenarioFrame("2024-11-15T18:00:00Z", "Pre-fire baseline (Nov 2024)"),
            ScenarioFrame("2024-12-20T18:00:00Z", "Pre-fire baseline (Dec 2024)"),
            ScenarioFrame("2025-01-08T18:00:00Z", "Day after ignition"),
            ScenarioFrame("2025-01-10T18:00:00Z", "Active fire - Day 3"),
            ScenarioFrame("2025-01-13T18:00:00Z", "Fire at peak"),
            ScenarioFrame("2025-01-16T18:00:00Z", "Fire slowing"),
            ScenarioFrame("2025-01-28T18:00:00Z", "Fresh burn scar"),
            ScenarioFrame("2025-02-10T18:00:00Z", "Burn scar settled"),
            ScenarioFrame("2025-03-15T18:00:00Z", "Early recovery"),
            ScenarioFrame("2025-05-01T18:00:00Z", "Spring regrowth"),
            ScenarioFrame("2025-06-15T18:00:00Z", "Late recovery"),
            ScenarioFrame("2025-10-01T18:00:00Z", "9 months later"),
        ),
    ),
    "lahaina-wildfire": Scenario(
        key="lahaina-wildfire",
        name="Lahaina wildfire, Maui HI",
        lat=20.871,
        lon=-156.6785,
        peak_event="Fire destroyed Lahaina town Aug 8–9, 2023 - 100+ fatalities, 2,700+ structures lost",
        frames=(
            ScenarioFrame("2022-10-01T20:00:00Z", "Quiet - Oct 2022"),
            ScenarioFrame("2023-03-15T20:00:00Z", "Quiet - Spring 2023"),
            ScenarioFrame("2023-06-01T20:00:00Z", "Pre-fire baseline (Jun 2023)"),
            ScenarioFrame("2023-07-15T20:00:00Z", "Pre-fire baseline (Jul 2023)"),
            ScenarioFrame("2023-08-09T20:00:00Z", "Day after fire"),
            ScenarioFrame("2023-08-25T20:00:00Z", "Post-fire damage"),
            ScenarioFrame("2023-10-01T20:00:00Z", "Burn scar stabilized"),
            ScenarioFrame("2024-02-15T20:00:00Z", "6 months later"),
            ScenarioFrame("2024-08-08T20:00:00Z", "1 year anniversary"),
            ScenarioFrame("2025-03-01T20:00:00Z", "1.5 years later"),
        ),
    ),
    "enga-landslide": Scenario(
        key="enga-landslide",
        name="Enga Province landslide, Papua New Guinea",
        lat=-5.37389,
        lon=143.38861,
        peak_event="Massive landslide May 24, 2024 near Yambali/Mulitaka - buried village, 2,000+ estimated casualties",
        frames=(
            ScenarioFrame("2023-07-01T00:00:00Z", "Quiet - Jul 2023"),
            ScenarioFrame("2023-11-15T00:00:00Z", "Quiet - Nov 2023"),
            ScenarioFrame("2024-03-01T00:00:00Z", "Pre-slide baseline (Mar 2024)"),
            ScenarioFrame("2024-04-15T00:00:00Z", "Pre-slide baseline (Apr 2024)"),
            ScenarioFrame("2024-05-30T00:00:00Z", "Days after slide"),
            ScenarioFrame("2024-07-01T00:00:00Z", "Debris field (1 month)"),
            ScenarioFrame("2024-09-15T00:00:00Z", "Stable scar (3 months)"),
            ScenarioFrame("2024-12-15T00:00:00Z", "6 months later"),
            ScenarioFrame("2025-04-15T00:00:00Z", "1 year later"),
        ),
    ),
    # === Peak checks (short calibration replays for demo debugging) ===
    "eaton-peak-check": Scenario(
        key="eaton-peak-check",
        name="Eaton Fire peak check, Los Angeles CA",
        lat=34.203483,
        lon=-118.069155,
        peak_event="Fire ignited Jan 7, 2025 near Altadena Drive/Midwick Drive - peak destruction Jan 8–13",
        frames=(
            ScenarioFrame("2025-01-08T18:00:00Z", "Day after ignition"),
            ScenarioFrame("2025-01-10T18:00:00Z", "Active fire - Day 3"),
            ScenarioFrame("2025-01-13T18:00:00Z", "Fire at peak"),
            ScenarioFrame("2025-01-16T18:00:00Z", "Fire slowing"),
        ),
    ),
    "lahaina-peak-check": Scenario(
        key="lahaina-peak-check",
        name="Lahaina wildfire peak check, Maui HI",
        lat=20.871,
        lon=-156.6785,
        peak_event="Fire destroyed Lahaina town Aug 8–9, 2023 - 100+ fatalities, 2,700+ structures lost",
        frames=(
            ScenarioFrame("2023-08-09T20:00:00Z", "Day after fire"),
            ScenarioFrame("2023-08-14T20:00:00Z", "Sentinel post-fire burn scar"),
            ScenarioFrame("2023-08-25T20:00:00Z", "Post-fire damage"),
        ),
    ),
    "enga-peak-check": Scenario(
        key="enga-peak-check",
        name="Enga landslide peak check, Papua New Guinea",
        lat=-5.37389,
        lon=143.38861,
        peak_event="Massive landslide May 24, 2024 near Yambali/Mulitaka - buried village, 2,000+ estimated casualties",
        frames=(
            ScenarioFrame("2024-05-25T00:00:00Z", "Day after landslide"),
            ScenarioFrame("2024-05-30T00:00:00Z", "Days after slide"),
            ScenarioFrame("2024-07-01T00:00:00Z", "Debris field (1 month)"),
        ),
    ),
    # === Additional scenarios (kept for reference, less reliable) ===
    "palisades-fire": Scenario(
        key="palisades-fire",
        name="Palisades Fire, Los Angeles CA",
        lat=34.07,
        lon=-118.55,
        frames=(
            ScenarioFrame("2024-12-10T18:00:00Z", "Pre-fire baseline"),
            ScenarioFrame("2025-01-10T18:00:00Z", "Fire active (peak)"),
            ScenarioFrame("2025-02-01T18:00:00Z", "Fresh burn scar"),
            ScenarioFrame("2025-05-01T18:00:00Z", "Recovery underway"),
        ),
    ),
    "valencia-flood": Scenario(
        key="valencia-flood",
        name="Valencia DANA flood, Spain",
        lat=39.47,
        lon=-0.38,
        frames=(
            ScenarioFrame("2024-10-01T10:00:00Z", "Pre-flood baseline"),
            ScenarioFrame("2024-11-01T10:00:00Z", "Flood at peak"),
            ScenarioFrame("2024-11-15T10:00:00Z", "Floodwater receding"),
            ScenarioFrame("2025-02-01T10:00:00Z", "Recovery"),
        ),
    ),
    "derna-flood": Scenario(
        key="derna-flood",
        name="Derna flash flood, Libya",
        lat=32.76,
        lon=22.64,
        frames=(
            ScenarioFrame("2023-08-15T09:00:00Z", "Pre-flood baseline"),
            ScenarioFrame("2023-09-15T09:00:00Z", "Immediate aftermath"),
            ScenarioFrame("2023-11-01T09:00:00Z", "Debris stable"),
            ScenarioFrame("2024-04-15T09:00:00Z", "Reconstruction"),
        ),
    ),
    "kelowna-wildfire": Scenario(
        key="kelowna-wildfire",
        name="Kelowna wildfire, BC Canada",
        lat=49.89,
        lon=-119.49,
        frames=(
            ScenarioFrame("2023-07-15T19:00:00Z", "Pre-fire baseline"),
            ScenarioFrame("2023-08-25T19:00:00Z", "Fire at peak"),
            ScenarioFrame("2023-10-15T19:00:00Z", "Burn scar"),
            ScenarioFrame("2024-04-15T19:00:00Z", "Spring regrowth"),
        ),
    ),
    "alexandroupoli-fire": Scenario(
        key="alexandroupoli-fire",
        name="Alexandroupoli wildfire, Greece",
        lat=41.10,
        lon=25.90,
        frames=(
            ScenarioFrame("2023-07-15T08:00:00Z", "Pre-fire baseline"),
            ScenarioFrame("2023-08-25T08:00:00Z", "Fire active"),
            ScenarioFrame("2023-10-15T08:00:00Z", "Burn scar"),
            ScenarioFrame("2024-04-15T08:00:00Z", "Recovery"),
        ),
    ),
}


FEATURED_KEYS = {"lahaina-wildfire", "lahaina-peak-check"}
DEMO_SAFE_KEYS = {"lahaina-wildfire", "lahaina-peak-check"}


def list_scenarios() -> list[dict]:
    """Return scenarios as a list of dicts suitable for JSON."""
    return [
        {
            "key": s.key,
            "name": s.name,
            "frame_count": len(s.frames),
            "lat": s.lat,
            "lon": s.lon,
            "featured": s.key in FEATURED_KEYS,
            "demo_safe": s.key in DEMO_SAFE_KEYS,
            "peak_event": s.peak_event,
        }
        for s in SCENARIOS.values()
    ]
