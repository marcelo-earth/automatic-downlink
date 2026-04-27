# Architecture

## System Overview

```
                                    SIMULATED SATELLITE (SimSat)
                    ┌─────────────────────────────────────────────────┐
                    │                                                 │
                    │   ┌──────────┐    ┌──────────────────────────┐  │
                    │   │ SimSat   │    │   Triage Engine          │  │
                    │   │ Orbit    │───>│                          │  │
                    │   │ Simulator│    │  ┌────────────────────┐  │  │
                    │   └──────────┘    │  │ LFM2.5-VL-450M    │  │  │
                    │                   │  │ (fine-tuned)       │  │  │
                    │   ┌──────────┐    │  └────────┬───────────┘  │  │
                    │   │ Sentinel │───>│           │              │  │
                    │   │ API      │    │  ┌────────▼───────────┐  │  │
                    │   └──────────┘    │  │ Priority Classifier│  │  │
                    │                   │  │ (prompt-based)     │  │  │
                    │   ┌──────────┐    │  └────────┬───────────┘  │  │
                    │   │ Mapbox   │───>│           │              │  │
                    │   │ API      │    └───────────┼──────────────┘  │
                    │   └──────────┘                │                 │
                    └──────────────────────────────┼─────────────────┘
                                                   │
                              ┌─────────────────────┤
                              │                     │
                    ┌─────────▼──────┐    ┌────────▼────────┐
                    │  Text Summary  │    │  Priority Queue  │
                    │  (ALL images)  │    │  (HIGH+ images)  │
                    │  ~1KB each     │    │  full resolution │
                    └─────────┬──────┘    └────────┬────────┘
                              │                     │
                              └──────────┬──────────┘
                                         │
                              SIMULATED DOWNLINK
                                         │
                              ┌──────────▼──────────┐
                              │    Ground Station   │
                              │    Dashboard        │
                              │                     │
                              │  - Triage feed      │
                              │  - Bandwidth stats  │
                              │  - Image viewer     │
                              └─────────────────────┘
```

## Components

### 1. SimSat Integration Layer
**Purpose:** Fetch satellite imagery from SimSat API
**Input:** Satellite position (lat, lon, alt, timestamp)
**Output:** Raw satellite images (Sentinel-2 multispectral, Mapbox RGB)
**Tech:** Python, requests, SimSat Docker container

Endpoints used:
- `GET /data/current/position` — satellite position
- `GET /data/current/image/sentinel` — current Sentinel-2 image
- `GET /data/current/image/mapbox` — current Mapbox image
- `GET /data/image/sentinel` — historical Sentinel-2 image

### 2. Triage Engine
**Purpose:** Analyze images and decide what to downlink
**Input:** Raw satellite image (PNG)
**Output:** JSON with description, priority, and reasoning

```json
{
  "image_id": "IMG_4821",
  "timestamp": "2026-04-15T12:34:56Z",
  "position": {"lat": -12.05, "lon": -77.04, "alt": 791.3},
  "description": "Urban area with visible flooding in eastern quadrant. Multiple roads submerged. Residential structures partially inundated.",
  "priority": "CRITICAL",
  "reasoning": "Active flooding affecting populated area. Time-sensitive for disaster response.",
  "categories": ["disaster", "flooding", "urban"],
  "downlink_recommendation": "TRANSMIT_IMAGE",
  "estimated_tokens": 47,
  "estimated_image_size_kb": 512
}
```

**Tech:** LFM2.5-VL-450M (fine-tuned), transformers library

#### Triage Levels

| Level | Criteria | Downlink Action |
|-------|----------|----------------|
| CRITICAL | Active disasters, immediate threats | Image + full metadata |
| HIGH | Significant changes, notable activity | Image + full metadata |
| MEDIUM | Moderate interest, routine changes | Thumbnail + metadata |
| LOW | Minimal interest but some content | Text summary only |
| SKIP | Clouds, empty ocean, no content | Text summary only |

### 3. Bandwidth Simulator
**Purpose:** Calculate and display bandwidth savings
**Input:** Triage decisions for a batch of images
**Output:** Statistics comparing naive vs smart downlink

Metrics:
- Total images captured
- Images by priority level
- Data transmitted (smart) vs data that would have been transmitted (naive)
- Bandwidth savings percentage
- Critical alerts latency

### 4. Ground Station Dashboard
**Purpose:** Visualize triage decisions for the demo
**Input:** Triage engine output stream
**Output:** Web UI showing real-time triage feed

Components:
- Live triage feed (scrolling list of decisions)
- Priority distribution chart
- Bandwidth savings gauge
- Image viewer for high-priority images
- Satellite position map

**Tech:** TBD (simple web UI — could be Streamlit, Next.js, or plain HTML)

### 5. Docker Packaging
**Purpose:** One-command setup for judges

```
docker compose up
```

Services:
- `simsat` — SimSat simulator (from their Docker setup)
- `triage` — Triage engine with LFM2.5-VL-450M
- `dashboard` — Ground station UI

## Data Flow

1. SimSat simulates satellite orbit, advancing position over time
2. At each position, Triage Engine fetches the current satellite image
3. Image is passed to LFM2.5-VL-450M with triage system prompt
4. Model generates description + priority classification
5. Result is sent to Ground Station Dashboard
6. Dashboard shows triage decisions and calculates bandwidth stats
7. Only HIGH+ priority images are "downlinked" (displayed in full)
8. SKIP/LOW images show only the text summary

## Model Pipeline

```
                    ┌──────────────┐
                    │ System Prompt│
                    │ (triage      │
                    │  instructions│
                    │  + priority  │
                    │  levels)     │
                    └──────┬───────┘
                           │
┌──────────┐    ┌──────────▼───────────┐    ┌──────────────────┐
│ Satellite│───>│  LFM2.5-VL-450M     │───>│ JSON Output      │
│ Image    │    │  (fine-tuned on      │    │ {description,    │
│ (PNG)    │    │   VRSBench satellite │    │  priority,       │
│          │    │   captioning)        │    │  reasoning,      │
└──────────┘    └──────────────────────┘    │  categories}     │
                                           └──────────────────┘
```

## Deployment Considerations (Satellite Target)

For actual OmniSat deployment (post-hackathon / if we win credits):

| Constraint | Value | How We Handle It |
|-----------|-------|-----------------|
| GPU | NVIDIA Orin 16GB | 450M model = ~1GB in bf16, fits easily |
| GPU time | 5 hours | At ~0.5s/image, that's ~36,000 images |
| Upload | 5MB | Model weights pre-loaded, only upload config |
| Download | 10MB | Text summaries (~1KB each) + few critical images |
| Storage | 1GB | Model + image buffer + results |
| Runtime | Docker | Already containerized |

## File Structure (Planned)

```
automatic-downlink/
  PRD.md
  RISKS.md
  CHANGELOG.md
  ARCHITECTURE.md
  CLAUDE.md
  README.md
  docker-compose.yml
  src/
    triage/
      engine.py          # Main triage pipeline
      model.py           # LFM2.5-VL-450M inference wrapper
      prompts.py         # System prompts for triage
      schemas.py         # Output JSON schemas
    simsat/
      client.py          # SimSat API client
    dashboard/
      app.py             # Ground station UI
  training/
    configs/
      satellite_triage.yaml  # leap-finetune config
    scripts/
      prepare_data.py        # Dataset preparation
  tests/
    test_triage.py
    test_simsat_client.py
  Dockerfile
  requirements.txt
```
