# automatic-downlink

> A Vision Language Model that runs on-board satellites to automatically decide what's worth downloading to Earth.

**AI in Space Hackathon** (Liquid AI x DPhi Space) | Liquid Track | April-May 2026

## The Problem

Satellites capture terabytes of imagery but can only transmit megabytes per ground station pass. Today, most data is either never analyzed or analyzed days late. Current on-board filtering is limited to narrow CNNs that do one thing (e.g., cloud detection).

## The Solution

A single LFM2.5-VL-450M vision-language model running on-board that:
- Describes every captured image in natural language
- Assigns priority levels (CRITICAL / HIGH / MEDIUM / LOW / SKIP)
- Only downlinks what matters — saving 50%+ bandwidth
- Can be re-tasked via prompt (disaster mode, surveillance mode, etc.) without uploading new weights

## Quick Start

```bash
# Clone with SimSat submodule
git clone --recursive https://github.com/marcelo-earth/automatic-downlink.git
cd automatic-downlink

# Set environment variables
cp .env.example .env  # then fill in MAPBOX_ACCESS_TOKEN, HF_TOKEN

# Run everything
docker compose up --build
```

Then open:
- **Ground Station Dashboard:** http://localhost:8080
- **SimSat Dashboard:** http://localhost:8000
- **SimSat API:** http://localhost:9005

The triage engine will start polling SimSat for satellite images and analyzing them automatically.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  SimSat (Satellite Simulator)                        │
│  ┌─────────────┐  ┌──────────────────────────────┐   │
│  │ Orbit Sim   │──│ Sentinel-2 / Mapbox Imagery  │   │
│  └─────────────┘  └──────────────────────────────┘   │
└──────────────────────────┬───────────────────────────┘
                           │ REST API (:9005)
┌──────────────────────────▼───────────────────────────┐
│  Triage Engine (On-board VLM)                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │ LFM2.5-VL-450M (fine-tuned on VRSBench)        │ │
│  │ → Describes image → Assigns priority → Decides  │ │
│  │   what to downlink (full image / thumbnail /    │ │
│  │   summary only / skip)                          │ │
│  └─────────────────────────────────────────────────┘ │
│  Prompt profiles: default | disaster | maritime      │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  Ground Station Dashboard (:8080)                    │
│  Live triage feed, bandwidth savings, priority stats │
└──────────────────────────────────────────────────────┘
```

## Fine-Tuning

We fine-tuned LFM2.5-VL-450M on 20,264 satellite image captions from [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench), converted to triage JSON format.

| | Base Model | Fine-Tuned |
|--|-----------|------------|
| Valid JSON output | 66% | **100%** |
| Correct priority | 33% | **100%** |
| Image-specific descriptions | 33% | **100%** |

**Model weights:** [marcelo-earth/LFM2.5-VL-450M-satellite-triage](https://huggingface.co/marcelo-earth/LFM2.5-VL-450M-satellite-triage)

Training details:
- **Method:** LoRA SFT (rank 16, alpha 32) via [leap-finetune](https://github.com/LiquidAI/leap-finetune)
- **Infrastructure:** Modal H100, 2 epochs, lr=1e-4
- **Data pipeline:** `training/scripts/prepare_triage_dataset.py`
- **Config:** `training/configs/triage_vlm_sft_modal.yaml`

## Prompt Steering (No Re-training Needed)

The same model supports multiple mission profiles via prompt engineering:

| Profile | Use Case | Priority Thresholds |
|---------|----------|-------------------|
| `default` | General monitoring | Standard 5-level triage |
| `disaster` | Natural disaster response | Lower threshold for CRITICAL/HIGH |
| `maritime` | Ocean surveillance | Focus on vessels, oil spills |

Set via environment variable: `TRIAGE_PROFILE=disaster`

## Project Structure

```
src/
├── triage/
│   ├── model.py        # VLM inference wrapper (MPS/CUDA/CPU)
│   ├── engine.py       # Image → VLM → JSON parsing → TriageDecision
│   ├── prompts.py      # System prompts (default, disaster, maritime)
│   ├── loop.py         # Polling loop (SimSat → triage → dashboard)
│   └── schemas.py      # Pydantic models
├── simsat/
│   └── client.py       # SimSat API client
└── dashboard/
    ├── app.py          # FastAPI dashboard + REST API
    └── templates/      # HTML dashboard
training/
├── scripts/prepare_triage_dataset.py  # VRSBench → triage JSONL
└── configs/triage_vlm_sft_modal.yaml  # Modal training config
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MAPBOX_ACCESS_TOKEN` | Yes | For SimSat satellite imagery |
| `HF_TOKEN` | Yes (build) | Hugging Face token for model download |
| `TRIAGE_PROFILE` | No | `default`, `disaster`, or `maritime` |
| `POLL_INTERVAL` | No | Seconds between triage runs (default: 30) |
| `MODEL_ID` | No | Override model (default: fine-tuned model) |

## License

MIT
