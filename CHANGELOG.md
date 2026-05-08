# Changelog

## 2026-05-08

### Demo replay hardening and Sentinel no-data trimming

- Added production demo controls for finite replay scenarios, model-off default state, streaming ANALYZING rows, RGB+SWIR thumbnails, inference timing, and event context.
- Fixed scenario replay race conditions so stale in-flight results are discarded when switching events.
- Fixed scenario exhaustion so the model stops after the configured frame count instead of looping into more captures.
- Added capture date/time to feed cards and ordered scenario timelines newest-first for demo readability.
- Added a Sentinel no-data border trim before both dashboard rendering and VLM inference. This removes black edge-of-granule strips from RGB/SWIR crops without cropping real interior burn scars.
- Calibrated replay inputs with event-specific coordinates and added short peak-check scenarios for Eaton, Lahaina, and Enga.
- Added a semantic consistency layer so LOW cannot survive when the model's own text describes active fire, hotspots, flooding, or fresh landslide evidence. Replay event context is displayed in the dashboard only; it is not fed into model inference.
- Fixed a semantic-layer false positive where "no thermal hotspots" matched "hotspot", and removed prompt examples that caused repeated stock "dense urban/no hotspot" descriptions.

## 2026-05-06

### Docker smoke test passed, ARCHITECTURE.md rewritten for demo

#### Docker smoke test

- Cleared ghost container state (`docker system prune -f`) that was blocking `docker compose up`.
- All 3 services started successfully:
  - `simsat-dashboard` (:8000) - healthy
  - `simsat-api` (:9005) - up
  - `triage-dashboard` (:8080) - up
- LFM2.5-VL-450M v6d loaded on CPU in ~34 seconds (350 weight shards).
- First inference ran immediately: `IMG_0144FE30 → SKIP` (correct - cloudy image).
- `GET /` → HTTP 200. `GET /api/stats` → valid JSON `{"total_images":1,"SKIP":1,"savings_percent":99.8}`.
- **Smoke test: PASS.** `docker compose up` works end-to-end.

#### ARCHITECTURE.md rewritten

- Replaced stale "TBD web UI" content with accurate current-state description.
- Added: dual-image pipeline diagram, fine-tuning progression table (v6→v6d), hazard scope table with SWIR signatures, Docker service table, data flow diagram, OmniSat deployment constraints.
- ARCHITECTURE.md now serves as the demo narrative document for judges.

## 2026-05-05

### Scope change, v6c eval, v6d retrain, dual-image pipeline wired

#### Scope: oil spill removed

- Removed oil spill from hazard scope - no clean positive eval examples existed (both oil spill candidates were cloud-blown SKIP), and only 9 training samples.
- Updated `PRIORITY_POLICY.md`, `DETECTION_CAPABILITIES.md`, `src/triage/prompts.py`, `scripts/labeling_prompt.md`, `README.md`.
- Removed `MARITIME_MODE_SYSTEM_PROMPT` and `maritime` prompt profile entirely.
- Hazard scope is now: **wildfire, flood, landslide**.

#### v6c eval results

- Checkpoint: `LFM2.5-VL-450M-vlm_sft-exp6_train-…-20260430_053227` (e4s16)
- **Overall: 6/11 (55%)** - no improvement over v6b.
- **CRITICAL: 0/4** - all four Valencia flood tiles predicted MEDIUM.
- MEDIUM recall 1.0 - model collapsed to MEDIUM for every non-trivial scene.
- Root cause: 22 MEDIUM samples (38.6%) dominated training despite 5x CRITICAL upsampling. The 450M model defaulted to MEDIUM as its safe prediction.

#### v6d: CRITICAL-dominant rebalancing

- Built `training/data/exp6d_train.jsonl` via `training/scripts/build_exp6d_train.py`:
  - CRITICAL: 10 → 30 (3x upsample)
  - HIGH: 19 (unchanged)
  - LOW: 6 (unchanged)
  - MEDIUM: 22 → 6 (cut wildfire-MEDIUM which made up 13/22)
  - Total: 57 → 61 samples, CRITICAL share 17% → 49%
- Config: `training/configs/triage_vlm_sft_v6d_modal.yaml` (same hyperparams, 5 epochs)
- Kicked off on Modal H100, run timestamp `20260506_021457`.

#### Dual-image (RGB + SWIR) pipeline wired into inference

- `src/triage/model.py`: added `generate_dual(rgb, swir, …)` method alongside existing `generate()`.
- `src/triage/engine.py`: `analyze()` now accepts optional `swir_image`; uses `TRIAGE_DUAL_SYSTEM_PROMPT` when SWIR is present, single-image prompt otherwise.
- `src/triage/loop.py`: fetches SWIR companion (`["swir16","nir08","red"]`) alongside RGB for both live and demo modes.
- Previously the fine-tuned model (trained on dual images) was receiving only one image at inference - this mismatch is now fixed.

#### v6d eval results

- **CRITICAL recall: 3/4 (75%)** - first time model detects active hazard events.
- Overall accuracy: 4/11 (36%) - dropped because model now over-escalates flood-region tiles.
- MEDIUM recall: 1/6 - model is recall-first (appropriate for disaster triage).
- Per-sample: correctly identified 2 active Valencia floods (Nov 1) and 1 flood aftermath (Nov 10). Missed one aftermath tile. False positives on pre-flood and flood-region non-flooded tiles.

#### v6d pushed to HuggingFace

- `marcelo-earth/LFM2.5-VL-450M-satellite-triage-v6`
- Inference files only (1.03GB model.safetensors + configs) - optimizer states excluded.
- `Dockerfile` updated to download v6 model at build time.
- `src/triage/model.py` updated: default `MODEL_ID` → v6 repo, no pinned revision.

#### Dual-image inference wired

- `src/triage/model.py`: added `generate_dual()` for RGB+SWIR pair.
- `src/triage/engine.py`: `analyze()` accepts optional `swir_image`, routes to dual path.
- `src/triage/loop.py`: fetches SWIR companion alongside RGB for both live and demo modes.

#### HF token updated

- Old token (read-only, explaining why trackio never wrote data) replaced.
- Modal secret `huggingface-secret` updated.
- `.env` updated.

## 2026-04-30

### Exp 6c - CRITICAL class fix via upsampling + retrain

#### Problem diagnosed
- v6b eval: 6/11 (55%) overall accuracy. CRITICAL recall = 0/4 (model predicted MEDIUM for all four Valencia 2024 flood tiles).
- Root cause: only 2 CRITICAL examples in 49-sample training set (4.1%). Model learned to default to MEDIUM for uncertain scenes.

#### Fix: CRITICAL upsampling in `prepare_exp6_dataset.py`
- Added 5× repeat of CRITICAL training rows after the temporal split.
- Result: 10 CRITICAL out of 57 training samples (17.5%) - proportional to real-world hazard importance.
- Explored leap-finetune source (`/leap-finetune/src/`) to confirm no built-in class weighting or upsampling exists in the YAML config layer; dataset-level duplication is the correct approach.

#### Dataset regenerated + uploaded
- Re-ran `PYTHONPATH=. python3 training/scripts/prepare_exp6_dataset.py`
- Re-ran `training/scripts/upload_exp6_to_modal.py` (overwrites existing JSONL on volume, images unchanged)

#### v6c retrain launched
- Same config as v6b: `training/configs/triage_vlm_sft_v6_modal.yaml`
- Full fine-tune (no LoRA), H100, 5 epochs, LR 2e-5, `vision_encoder_lr_multiplier: 1.0`
- Tracking: Trackio at `marcelo-earth/automatic-downlink-trackio`

## 2026-04-22

### EXP 5/6 pivot - hazard-focused training data + v5 launch

#### Hazard event image capture
- Started SimSat locally (dummy MAPBOX_ACCESS_TOKEN; we only need Sentinel)
- Captured 17 real hazard events (wildfire x5, flood x5, landslide x4, oil_spill x3) from SimSat via AWS STAC, both RGB and SWIR views (34 images total)
- Events: Lahaina, Attica, Kelowna, Tenerife, Chile (wildfires); Rio Grande, Valencia, Derna, Pakistan Sindh, Thessaly (floods); Enga, Turkey Hatay, Brazil Sao Sebastiao, India Joshimath (landslides); Ventanilla, Tobago, Niger Delta (oil spills)
- Visual triage: 3 CRITICAL, 2 HIGH, 7 MEDIUM, 5 SKIP (clouds or no-data at the captured timestamps)

#### Eval manifest expansion
- Added 17 hazard samples to `evals/sentinel_eval_v1.jsonl` (28 → 45 samples)
- First eval set covering all five priority levels (previous set had no CRITICAL/HIGH)
- Baseline cascade eval (deterministic, decision layer ON): 34/45 match (75.6%)
- **CRITICAL recall: 0/3** - all three active-hazard CRITICAL samples were predicted MEDIUM. Model never emits CRITICAL.
- **HIGH recall: 1/2** - Lahaina wildfire correct, Tenerife downgraded to MEDIUM
- SKIP/LOW/MEDIUM on routine scenes: ~86% correct; prefilter catches 19/19 of the unambiguous SKIPs

#### Training data v5
- Rewrote system prompt (`src/triage/prompts.py`) to hazard-focused framing: explicit hazard scope (wildfire, flood, oil spill, landslide), reworded priority levels
- Built `training/data/train_v5_modal.jsonl` (2,700 samples):
  - 2,638 VRSBench samples with updated system prompt
  - 17 new hazard samples with full JSON labels
  - CRITICAL/HIGH oversampled 10x to balance (30 CRITICAL, 20 HIGH)
- Uploaded hazard images and v5 manifests to Modal volume `satellite-vlm`

#### v5 training launch
- Config: `training/configs/triage_vlm_sft_v5_modal.yaml`
- H100, 3 epochs, LoRA (rank 8, alpha 16), LR 1e-4, vision encoder 0.1x
- Tracking: Trackio at `marcelo-earth/automatic-downlink-trackio`

## 2026-04-17

### Fine-Tuning: Exp 1 - Model Working

#### Root cause analysis (MODEL_AUTORESEARCH.md)
- Diagnosed why Exp 0 model produced garbage: 48% of training data was VQA one-word answers ("green", "Yes", "Ships") and bounding box tokens - not captions
- Bug was in `_extract_caption()`: checked if `[caption]` was in GPT response (it never is - it's in the HUMAN prompt), so it extracted ALL GPT turns indiscriminately
- Learning rate 2e-5 was also too low for LoRA (should be 1e-4 to 5e-4)

#### Data fix
- Rewrote `_extract_caption()` to only take responses from `[caption]` tasks via new `_is_caption_task()`
- Added boilerplate stripping ("The image, sourced from GoogleEarth, ")
- Result: 20,264 clean caption-only samples (was 5,000 mixed garbage)
- Distribution: MEDIUM=16,896, LOW=1,945, SKIP=1,277, HIGH=142, CRITICAL=4

#### Training (Exp 1)
- Config: lr=1e-4, 3 epochs, batch 4, LoRA rank 16, H100 on Modal
- Only 2 of 3 epochs completed (Modal timeout), eval_loss 0.87→0.83
- Merged LoRA adapter with base model locally using peft

#### Results: dramatic improvement

| Metric | Base Model | Exp 0 (garbage data) | Exp 1 (clean data) |
|--------|-----------|---------------------|-------------------|
| Valid JSON | 66% | 0% | **100%** |
| Correct priority | 33% | 0% | **100%** |
| Unique descriptions | 33% | 0% | **100%** |
| Correct schema | 33% | 0% | **100%** |

- Uploaded to HuggingFace: `marcelo-earth/LFM2.5-VL-450M-satellite-triage`
- Added model card with usage, training details, and evaluation
- Docker image rebuilt with fine-tuned model - end-to-end tested, working
- Dashboard now shows **94.8% bandwidth savings** (was 0% with base model)

#### Other changes
- Expanded README with architecture diagram, quick start, fine-tuning results
- Updated Dockerfile to download both fine-tuned model and base model processor
- model.py now defaults to fine-tuned model, loads processor from base model

#### Status
- MVP is functional: SimSat → VLM triage → dashboard with bandwidth savings
- Fine-tuned model produces unique, image-specific descriptions with correct priorities

## 2026-04-15

### Research & Setup
- Researched DPhi Space platform, Liquid AI models, and hackathon rules
- Analyzed prior art: on-board satellite AI (PhiSat-1/2, OroraTech, RaVAEn, Planet Pelican-4), VLM captioning for remote sensing, edge AI in space
- Key finding: nobody has deployed a general-purpose VLM for on-board image triage. Current systems use narrow CNNs (cloud detection, fire detection) - one model per task
- Decided on idea: **Smart Downlink Triage** - VLM running on satellite that analyzes every image, describes it, and prioritizes what to downlink
- Track: **Liquid Track** (LFM2.5-VL-450M)

### Project Setup
- Created `automatic-downlink/` repo with documentation
- Created `hackathon-learning-and-context/` repo with:
  - SimSat cloned
  - Judging criteria document
  - Context files: DPhi Space, Liquid AI models, fine-tuning guide, SimSat API, hackathon rules, space compute rationale
- Verified LFM2.5-VL-450M inference code (simple: transformers + pillow)
- Verified fine-tuning pipeline (leap-finetune + Modal, $30 free credits)
- Created PRD, RISKS, ARCHITECTURE, CLAUDE.md

### Phase 1: Implementation Start

#### Completed
- [x] Project structure: `src/triage/`, `src/simsat/`, `src/dashboard/`, `training/`, `tests/`
- [x] Python setup: `pyproject.toml` with uv, deps for core + dashboard + training + dev
- [x] `.gitignore` (Python, model weights, data, IDE, Docker, Modal)
- [x] `.env.example` (HF_TOKEN, MAPBOX_ACCESS_TOKEN, MODAL tokens)
- [x] SimSat API client (`src/simsat/client.py`) - typed dataclasses for position, sentinel, mapbox responses. Health check. All 5 endpoints wrapped.
- [x] Model inference wrapper (`src/triage/model.py`) - loads LFM2.5-VL-450M, supports CUDA/MPS (Apple Silicon)/CPU. Chat template with system + user + image, generation params from model card.
- [x] Triage prompts (`src/triage/prompts.py`) - 3 prompt profiles: default, disaster mode, maritime mode. Prompt-steerable triage is a key differentiator.
- [x] Output schemas (`src/triage/schemas.py`) - Pydantic models: TriageDecision (priority, description, reasoning, categories, downlink action), BandwidthStats, Priority enum, DownlinkAction enum.
- [x] Triage engine (`src/triage/engine.py`) - orchestrates SimSat → VLM → JSON parsing → TriageDecision. Handles malformed JSON gracefully. Bandwidth savings calculator built in.
- [x] Docker setup (`Dockerfile`, `docker-compose.yml`) - 3 services: simsat-dashboard, simsat-api, triage. Single `docker compose up` runs everything on ports 8000/9005/8080.

#### Decisions made
- Docker: confirmed installed and running
- GPU: Mac (Apple Silicon) - model wrapper updated with MPS support + float16
- Mapbox: no token yet, marked as OPTIONAL. Sentinel-2 works without it.
- Project name: **automatic-downlink**

- [x] Renamed project from `project-unnamed` to `automatic-downlink`
- [x] Added SimSat as git submodule
- [x] SimSat Docker tested - API responds, simulation runs, position tracking works
- [x] Fetched 4 test images from SimSat API:
  - `sentinel_test1.png` - Lausanne, Switzerland (city + lake, 0% clouds)
  - `sentinel_amazon.png` - Amazon (97% clouds, nearly all white)
  - `sentinel_sahara.png` - Sahara desert (arid terrain, 18% clouds)
  - `sentinel_lima.png` - Lima, Peru (90% clouds, partial city)
  - Ocean test failed - Sentinel-2 doesn't cover open ocean (documented limitation)
- [x] Python 3.11 venv created (system Python 3.9 too old for transformers 5.x)
- [x] Base model inference tested (LFM2.5-VL-450M on MPS/Apple Silicon):
  - Model loads in ~2 seconds, inference ~10-15s per image
  - Correctly identifies: cloud cover, urban areas, arid terrain, vegetation
  - Problem: responds in free text, not structured JSON (expected without fine-tuning)
  - Problem: descriptions are generic, not satellite-specific
  - Conclusion: fine-tuning will improve JSON compliance and domain knowledge
- [x] Fixed: `torch_dtype` deprecation → `dtype`, added `torchvision` to deps, bumped transformers to `>=4.57.0`

#### Prompt Engineering
- [x] Tested 3 prompt variants: verbose, short+example, minimal+completion
- [x] Final prompt: few-shot with 5 diverse examples (urban, cloud, desert, fire, deforestation)
- [x] Added explicit cloud detection rule for bright/white images
- [x] Achieved 100% JSON parse rate on all test images
- [x] Categories now vary correctly (was always "urban" with single example)
- [x] Known limitation: Sahara→SKIP instead of LOW, Lausanne→HIGH with copied description. Fine-tuning expected to fix.

#### Dashboard
- [x] Built FastAPI dashboard (`src/dashboard/app.py`) - routes: GET / (HTML), GET/POST /api/decisions, GET /api/stats
- [x] Dark theme ground station UI (`src/dashboard/templates/index.html`) - priority badges, stats cards, priority distribution bars, triage feed, auto-refresh 5s
- [x] Fixed Starlette 1.0 TemplateResponse API change (request as first arg)
- [x] Fixed enum serialization - `model_dump(mode="json")` so priority renders as "CRITICAL" not "Priority.CRITICAL"
- [x] Tested with sample data: badges, stats, bandwidth savings all render correctly

#### Triage Loop (end-to-end pipeline)
- [x] Created `src/triage/loop.py` - async triage loop that polls SimSat, runs VLM, stores decisions
- [x] Integrated loop into dashboard via FastAPI lifespan (background task)
- [x] Live mode: uses SimSat simulation position + Sentinel-2 current endpoint
- [x] Demo mode: when simulation not running (pos 0,0,0), cycles through 10 interesting Earth locations using historical Sentinel-2 endpoint
- [x] Config via env vars: `SIMSAT_URL`, `TRIAGE_PROFILE` (default/disaster/maritime), `POLL_INTERVAL`
- [x] End-to-end tested: model loads → fetches Lausanne → triages as HIGH → decision appears in dashboard API and HTML
- [x] Fixed bandwidth savings calc (naive vs smart now consistent)
- [x] Known: base model copies few-shot example descriptions instead of truly describing image (fine-tuning will fix)

#### Fine-Tuning Pipeline
- [x] Cloned `leap-finetune` framework into `training/leap-finetune/`
- [x] Studied VLM SFT data format: messages with `[{type: "image", image: path}, {type: "text", text: prompt}]`
- [x] Created `training/scripts/prepare_triage_dataset.py` - downloads VRSBench (xiang709/VRSBench), converts captions to our triage JSON format with keyword-based priority assignment
- [x] Created `training/configs/triage_vlm_sft.yaml` - local training config (LFM2.5-VL-450M, LoRA, 3 epochs)
- [x] Created `training/configs/triage_vlm_sft_modal.yaml` - Modal H100 config (same + GPU/volume settings)
- [x] Tested data prep: 20 samples → 18 train / 2 eval JSONL, correct format verified
- [x] VRSBench dataset: `xiang709/VRSBench` on HuggingFace, 29K captioning samples, uses streaming to avoid pyarrow parse error on nested fields

#### Pending - needs Marcelo
- [ ] Create Modal account (`pip install modal && modal setup`)
- [ ] Run full data prep: `python training/scripts/prepare_triage_dataset.py --limit 5000`
- [ ] Run training: `cd training/leap-finetune && uv run leap-finetune ../configs/triage_vlm_sft_modal.yaml`
- [ ] (Optional) Create free Mapbox account for high-res imagery

#### Next session
- [ ] Create test suite for triage engine
- [ ] Polish dashboard (add location info, timestamp formatting)
- [ ] Demo video script
- [ ] Clean up Docker image for judge submission
