# Project: automatic-downlink (AI in Space Hackathon)

## Context

This is a hackathon project for the Liquid AI x DPhi Space "AI in Space" hackathon (Apr 13 - May 8, 2026). **automatic-downlink**: a Vision Language Model (LFM2.5-VL-450M) that runs on-board a satellite, analyzes every captured image, and automatically decides what is worth downlinking to ground.

Track: Liquid Track (requires LFM2-VL or LFM2.5-VL models).

## Key Files

- `PRD.md` — product requirements and scope
- `RISKS.md` — risks and mitigations
- `ARCHITECTURE.md` — system design and component overview
- `CHANGELOG.md` — daily progress log
- `../hackathon-learning-and-context/` — research, SimSat repo, model docs, API reference

## Tech Stack

- **Model:** LFM2.5-VL-450M (Liquid AI) — fine-tuned on VRSBench satellite captioning
- **Fine-tuning:** leap-finetune + Modal (H100 serverless GPUs)
- **Inference:** transformers library (Python), GGUF for edge demo
- **Data:** SimSat API (Sentinel-2 + Mapbox satellite imagery)
- **Packaging:** Docker Compose

## Critical Rules

1. **App MUST run without debugging** — 35% of judging score. If judges can't run it with `docker compose up`, we're disqualified. Always keep the Docker setup working.
2. **Use SimSat API** as the data source — it's a judging requirement (10%).
3. **Fine-tuning is strongly encouraged** and rewarded. Document methodology, show measurable improvement, share weights and code.
4. **Demo quality matters** — 20% of score. Keep ARCHITECTURE.md updated, it drives the demo narrative.
5. **Update CHANGELOG.md** at the end of every work session.

## Conventions

- Python 3.11+
- Use `uv` for Python package management
- Type hints on all functions
- Keep Docker images small — judges need to download them
- Pin all dependency versions
- No secrets in the repo — use .env for API keys (MAPBOX_ACCESS_TOKEN, HF_TOKEN, MODAL_TOKEN)

## What NOT To Do

- Don't over-engineer. MVP first, polish later.
- Don't add features not in the PRD MVP scope.
- Don't break Docker setup for local dev convenience.
- Don't commit model weights to git — they go on HuggingFace or Modal volumes.
