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

We fine-tuned LFM2.5-VL-450M on 20,264 satellite image captions from [VRSBench](https://huggingface.co/datasets/xiang709/VRSBench), converted to triage JSON via knowledge distillation (Claude-generated labels replacing keyword heuristics).

| Metric | Base Model | Fine-Tuned (Exp 2) |
|--------|-----------|-------------------|
| Valid JSON output | 100% | **100%** |
| Priority accuracy | 2% | **50%** |
| Bandwidth savings | 0% | **~95%** |

The base model classifies everything as CRITICAL (useless). The fine-tuned model produces coherent triage JSON with specific descriptions and reasoning.

**Model weights:** [marcelo-earth/LFM2.5-VL-450M-satellite-triage](https://huggingface.co/marcelo-earth/LFM2.5-VL-450M-satellite-triage)
**Training labels:** [marcelo-earth/VRSBench-satellite-triage-labels](https://huggingface.co/datasets/marcelo-earth/VRSBench-satellite-triage-labels)
**Full metrics:** [METRICS.md](METRICS.md)

### Reproduce Training

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

### Training Details

- **Method:** LoRA SFT (rank 8, alpha 16, dropout 0.1)
- **Labels:** Knowledge distillation — Claude classified 20,264 VRSBench captions into priority/reasoning/categories
- **Data pipeline:** `training/scripts/prepare_triage_dataset.py --labels training/data/classified_captions.jsonl`
- **Eval script:** `training/scripts/evaluate_model.py`

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
├── scripts/evaluate_model.py          # Stratified eval with metrics
├── notebooks/finetune_kaggle.ipynb    # Fine-tune on Kaggle (free GPU)
└── configs/                           # Modal training configs
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
