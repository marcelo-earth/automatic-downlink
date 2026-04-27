# automatic-downlink

> A hybrid onboard hazard-triage pipeline that decides which satellite scenes are worth downlinking to Earth.

**AI in Space Hackathon** (Liquid AI x DPhi Space) | Liquid Track | April-May 2026

## The Problem

Satellites capture far more imagery than they can transmit during a ground-station pass.
The operational problem is not just "understand the image." It is:

> Which scenes deserve scarce downlink bandwidth right now?

For this project, that question is framed as **hazard triage**.

## The Solution

`automatic-downlink` is a **hybrid cascade**, not a single magical model:

- deterministic prefilters reject obvious junk cheaply
- a compact VLM handles the harder scenes
- the system emits a lightweight triage decision for downlink allocation

Current priority semantics live in [`PRIORITY_POLICY.md`](PRIORITY_POLICY.md):

- `CRITICAL`: active hazard clearly visible
- `HIGH`: strong hazard evidence, visible aftermath, or elevated hazard risk
- `MEDIUM`: anomalous or informative, but not a confirmed hazard
- `LOW`: routine low-value scene
- `SKIP`: obscured, unusable, or mostly no-data

Important: `HIGH` is now **hazard-only**. A port, mine, or city does not become `HIGH`
unless the image shows a hazard-related reason to escalate it.

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
│  Triage Engine (On-board Cascade)                    │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Prefilter → VLM → Conservative decision layer   │ │
│  │ Rejects junk → describes scene → assigns        │ │
│  │ hazard priority → chooses downlink action        │ │
│  └─────────────────────────────────────────────────┘ │
│  Prompt profiles: default | disaster | maritime      │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  Ground Station Dashboard (:8080)                    │
│  Live triage feed, bandwidth savings, priority stats │
└──────────────────────────────────────────────────────┘
```

## Current Status

The repo still includes the earlier VRSBench-based fine-tuned model as a baseline:

- **Model weights:** [marcelo-earth/LFM2.5-VL-450M-satellite-triage](https://huggingface.co/marcelo-earth/LFM2.5-VL-450M-satellite-triage)
- **Historical labels:** [marcelo-earth/VRSBench-satellite-triage-labels](https://huggingface.co/datasets/marcelo-earth/VRSBench-satellite-triage-labels)

That baseline was useful for proving local inference and JSON generation, but the project
has now moved to **EXP 6**, which prioritizes:

- real-domain Sentinel-2 / SimSat evaluation first
- hazard-oriented priority policy
- cascade improvements before retraining
- targeted real-domain supervision for hazards

Benchmarking and experiment notes live under [`evals/`](evals) and
[`EXP_6.md`](EXP_6.md).

### Training direction after v5 post-mortem

A `v5` pass that mixed VRSBench with 17 oversampled hand-labeled hazard images
converged cleanly on training loss but scored `CRITICAL 0/3` and `HIGH 0/2` on
the frozen real-domain eval — the model learned VRSBench caption style instead
of hazard detection. Details in [`EXP_6.md`](EXP_6.md).

The next training pass drops VRSBench, moves to RGB + SWIR dual-image inputs,
uses frontier-model teacher labels on a programmatic grid of real Sentinel-2
captures, does full fine-tune instead of LoRA, and splits train/test
temporally to avoid Sentinel-2's 5-day revisit leaking into eval.

### Historical Training Reproduction

**Option A — Kaggle (free, recommended):**

1. Upload `training/notebooks/finetune_kaggle.ipynb` to [Kaggle](https://www.kaggle.com)
2. Enable GPU T4 x2 (Settings → Accelerator)
3. Add your `HF_TOKEN` in Secrets
4. Run all cells (~45 min)

**Option B — Modal (paid):**

```bash
cd training/leap-finetune
uv run leap-finetune ../configs/triage_vlm_sft_modal.yaml
```

### Historical Training Details

- **Method:** LoRA SFT (rank 8, alpha 16, dropout 0.1)
- **Labels:** Knowledge distillation over VRSBench-derived supervision
- **Data pipeline:** `training/scripts/prepare_triage_dataset.py --labels training/data/classified_captions.jsonl`
- **Eval script:** `training/scripts/evaluate_model.py`

## Prompt Steering

The current inference stack supports multiple mission profiles via prompt engineering:

| Profile | Use Case | Priority Thresholds |
|---------|----------|-------------------|
| `default` | General hazard triage | Standard hazard policy |
| `disaster` | Land hazard response | Lower threshold for visible wildfire/flood/landslide evidence |
| `maritime` | Coastal or ocean hazard triage | Focus on spills, coastal contamination, and maritime hazard context |

Set via environment variable: `TRIAGE_PROFILE=disaster`

## Hazard Scope

Current scope is intentionally narrow and demo-defensible:

- wildfire
- flood
- landslide
- oil spill, but only **under favorable conditions**

The capability analysis is documented in
[`DETECTION_CAPABILITIES.md`](DETECTION_CAPABILITIES.md).

## RGB + SWIR Direction

Sentinel-2 provides more than RGB. SimSat exposes companion bands such as `nir`,
`swir16`, and `swir22`, and the Liquid wildfire cookbook already demonstrates an
RGB + SWIR two-image pattern.

That means the next hazard-focused training pass does **not** need a new architecture.
The same VLM can consume multiple rendered views of the same tile, for example:

- RGB image
- SWIR composite of the same tile

This is especially relevant for wildfire and flood discrimination, and potentially
useful for hazard aftermath more broadly.

## Project Structure

```
src/
├── triage/
│   ├── model.py        # VLM inference wrapper (MPS/CUDA/CPU)
│   ├── engine.py       # Prefilter → VLM → JSON parsing → decision layer
│   ├── prompts.py      # Hazard-triage prompt profiles
│   ├── loop.py         # Polling loop (SimSat → triage → dashboard)
│   └── schemas.py      # Pydantic models
├── simsat/
│   └── client.py       # SimSat API client
└── dashboard/
    ├── app.py          # FastAPI dashboard + REST API
    └── templates/      # HTML dashboard
training/
├── scripts/prepare_triage_dataset.py  # VRSBench → triage JSONL
├── scripts/evaluate_model.py          # Stratified eval with metrics
├── notebooks/finetune_kaggle.ipynb    # Fine-tune on Kaggle (free GPU)
└── configs/                           # Modal training configs
evals/
├── sentinel_eval_v1.jsonl             # Frozen reviewed benchmark manifest
├── review_batches/                    # Human-reviewed additions / relabels
├── locations/                         # Reusable capture locations
└── results/                           # Reproducible benchmark reports
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
