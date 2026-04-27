# PRD: automatic-downlink

## Problem

Satellites in Low Earth Orbit capture terabytes of imagery daily but can only transmit megabytes per ground station pass. Today, downlink decisions are made with:
- Naive approaches: download everything (wasteful, slow)
- Rule-based filters: cloud percentage thresholds (misses everything else)
- Task-specific CNNs: one model per detection task (cloud, fire, ships) — rigid, not generalizable

Result: most captured data is never analyzed, or analyzed hours/days late. High-value images (disasters, anomalies) sit in a queue alongside empty ocean shots.

## Solution

A Vision Language Model (LFM2.5-VL-450M) running on-board the satellite that:

1. **Analyzes every captured image** and generates a natural language description
2. **Assigns a priority level** (CRITICAL / HIGH / MEDIUM / LOW / SKIP) with justification
3. **Optimizes downlink** — only transmits high-priority images + text summaries of everything else

```
Traditional:  Capture → Downlink ALL → Analyze on ground → Act (hours/days)
Ours:         Capture → Analyze on-board → Downlink ONLY what matters → Act (minutes)
```

## Why a VLM (Not Another CNN)

| Capability | CNN | VLM (Ours) |
|-----------|-----|------------|
| Cloud filtering | Yes | Yes |
| Disaster detection | Needs separate model | Same model |
| Anomaly detection | Needs separate model | Same model |
| Explain WHY an image matters | No | Yes (natural language) |
| Change priorities without new model | No | Yes (change the prompt) |
| Multi-criteria triage in one pass | No | Yes |

The key insight: **one general-purpose VLM replaces N task-specific models**, and it can be re-tasked via prompt without uploading new weights.

## Target Hardware

DPhi Space OmniSat satellite:
- NVIDIA Orin 16GB GPU
- 5 GPU hours per session
- 5MB upload / 10MB download limits
- 1GB in-space storage
- Docker runtime

LFM2.5-VL-450M is 450M parameters — fits comfortably in 16GB with room for data buffers.

## MVP Scope

### Must Have (for submission)
- Fine-tuned LFM2.5-VL-450M on satellite imagery captioning (VRSBench)
- Triage pipeline: image in → description + priority out
- SimSat integration: consumes DPhi API for satellite imagery
- Bandwidth savings calculation: show how much data is saved vs naive downlink
- Docker setup: `docker compose up` runs everything
- Demo video: end-to-end walkthrough explaining problem, architecture, solution

### Nice to Have (if time permits)
- Live dashboard showing triage decisions in real-time
- Comparison: base model vs fine-tuned model on satellite images
- GGUF/ONNX deployment showing it runs on edge hardware
- Multiple triage profiles via different system prompts (disaster mode, surveillance mode, environmental mode)

### Out of Scope
- Actual deployment to a satellite
- Real-time ground station communication protocol
- Multi-satellite coordination
- Training from scratch

## Hackathon Track

**Liquid Track** (LFM2-VL / LFM2.5-VL required)

## Judging Criteria Alignment

| Criteria | Weight | Our Approach |
|----------|--------|-------------|
| Use of Satellite Imagery | 10% | SimSat API as core data source (Sentinel-2 + Mapbox) |
| Innovation & Problem-Solution Fit | 35% | VLM triage is novel — nobody has done general-purpose prompt-steerable triage on-board. Clear product path (DPhi would want this as a feature) |
| Technical Implementation | 35% | Fine-tuned model + working pipeline + Docker packaging. Must run without debugging |
| Demo & Communication | 20% | Video explaining: the downlink bottleneck → why VLMs solve it → architecture → live demo |

## Success Metrics

- Judges can run it with a single `docker compose up`
- Fine-tuned model demonstrably better than base model on satellite imagery
- Triage decisions are sensible (disasters = CRITICAL, empty ocean = SKIP)
- Bandwidth savings > 50% compared to naive "download everything"
- Demo clearly articulates the problem and architecture

## Timeline

- **Week 1 (Apr 15-21):** Fine-tune model, validate inference on satellite images
- **Week 2 (Apr 22-28):** Build triage pipeline + SimSat integration
- **Week 3 (Apr 29-May 5):** Dashboard, Docker packaging, testing
- **Week 4 (May 5-8):** Demo video, polish, submit

## Team

Solo (Marcelo)

## References

- SimSat repo: https://github.com/DPhi-Space/SimSat
- LFM2.5-VL-450M: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
- Fine-tuning tutorial: https://docs.liquid.ai/examples/customize-models/satellite-vlm
- leap-finetune: https://github.com/Liquid4All/leap-finetune
- Hackathon page: https://lu.ma/ai-in-space
