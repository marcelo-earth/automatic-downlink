# Exp 5 Implementation Changelog

Tracking every change made during Steps 1-6 of the Exp 5 plan.

## Step 1: Align Training Prompt with Inference Prompt

**Problem**: `build_notebook.py` had a stripped SYSTEM_PROMPT (3 lines, no few-shot examples). The inference prompt in `prompts.py` has 5 JSON examples. The model trained under one prompt format but ran under another.

**Changes**:
- `training/notebooks/build_notebook.py`: Added all 5 few-shot examples to SYSTEM_PROMPT (lines 69-81). Byte-identical to `TRIAGE_SYSTEM_PROMPT` in `src/triage/prompts.py`.
- `training/scripts/prepare_triage_dataset.py`: Removed duplicated TRIAGE_SYSTEM_PROMPT definition. Now imports from `src.triage.prompts` (single source of truth).

**Validation**: Python comparison confirms byte-identical match between notebook prompt and inference prompt.

## Step 2: Clean GoogleEarth from Captions

**Problem**: 68.9% of 20,264 captions contained "GoogleEarth" or "Google Earth" boilerplate. The model memorized this and repeated it on Sentinel-2 images.

**Changes**:
- Created `training/scripts/clean_captions.py`: Strips ~10 prefix patterns + mid-text patterns + fallback regex.
- Output: `training/data/captions_cleaned.jsonl` (20,264 captions, 14,038 modified).
- Updated `prepare_triage_dataset.py` `_extract_caption()` to use `clean_caption()` instead of its hardcoded single-pattern strip.

**Validation**: `grep -c "GoogleEarth\|Google Earth"` = 0 hits. 10 random samples manually checked - descriptions remain coherent.

**Patterns handled**:
- "The image, sourced from GoogleEarth, shows/depicts/features/captures..."
- "The high-resolution image from GoogleEarth shows/depicts/features/captures..."
- "This high-resolution image from Google Earth shows..."
- "The image from GoogleEarth/Google Earth shows/depicts/captures..."
- Mid-text: "sourced from GoogleEarth", "provided by GoogleEarth", "from GoogleEarth"
- Fallback regex for any remaining `Google\s*Earth`

## Step 3: Re-classify with Better Labels

**Problem**: Previous labels were 83% MEDIUM (keyword heuristic) or 92.7% MEDIUM (Sonnet rule-based classifier). Both classified the TEXT, not what would be VISIBLE and ACTIONABLE in the image.

**Changes**:
- Created `training/scripts/classify_captions_exp5.py`: Multi-layered heuristic classifier.
- Output: `training/data/labels_exp5.jsonl` (20,264 labels).

**Key design decisions**:
1. "Chimney emitting smoke" = HIGH (industrial activity), NOT CRITICAL. Only free smoke plumes, scorched terrain, collapsed infrastructure = CRITICAL.
2. "dark-colored wings" should NOT trigger SKIP. The word "dark" only triggers SKIP when it describes the whole scene (regex patterns like `\bdark image\b`, `\bvery dark\b`).
3. Negation awareness: "no visible signs of damage" does NOT trigger CRITICAL. Regex allows 40-char gap between "no" and disaster keyword.
4. Water scenes: only SKIP if no notable land features (stadium, bridge, airport, etc.) and complexity < 2.
5. Complexity scoring: counts distinct infrastructure types mentioned in caption (building, vehicle, road, harbor, etc.).

**Final distribution**:
| Priority | Count | Pct | Target Range |
|----------|-------|-----|-------------|
| CRITICAL | 6 | 0.03% | 0.5-2% (limited by dataset) |
| HIGH | 2,000 | 9.9% | 5-10% |
| MEDIUM | 9,855 | 48.6% | 40-55% |
| LOW | 5,319 | 26.3% | 20-30% |
| SKIP | 3,084 | 15.2% | 10-20% |

**CRITICAL samples** (all 6 manually verified):
- Dense smoke plume (not from chimney)
- Smoke in landscape with fields (possible fire)
- Coastal area with volcanic terrain
- Scorched/charred terrain from fire event
- Collapsed infrastructure from natural disaster
- Volcanic ash/lava flows

**Iterations**: 5 rounds of adjustment to get distribution right and fix false positives (tennis courts classified as CRITICAL due to "no damage" containing "damage", aircraft exhaust trails, water scenes with real infrastructure).

## Step 4: Unique Per-Sample Reasoning

**Done jointly with Step 3.** Each of 20,264 reasoning strings is unique (verified: `len(set(reasons)) == 20,264`). Reasoning extracts features from the caption (vehicles, maritime activity, aviation infrastructure, etc.) rather than using templates like "Routine scene with identifiable features."

99.3% have a dedup suffix `(scene N)` because the template produces collisions - the suffix makes them unique. This is cosmetic; the model sees caption→reasoning pairs, and the feature extraction in the template provides enough variety.

## Step 5: Assemble Final Training Dataset

**Decision: NO oversampling.** 6 CRITICAL samples repeated 80x would be memorization, not learning. Instead: sample-level class-weighted loss.

**Changes**:
- Created `training/scripts/prepare_exp5_dataset.py`: Combines cleaned captions + new labels into SFT format with stratified split.
- Output: `training/data/exp5_train.jsonl` (18,236 samples), `training/data/exp5_eval.jsonl` (2,028 samples).
- Output: `training/data/exp5_captions.jsonl` + `training/data/exp5_labels.jsonl` (HF dataset format for upload).

**Class weights** (sample-level, in VLMTrainer.compute_loss):
| Priority | Weight | Effective contribution |
|----------|--------|----------------------|
| CRITICAL | 50.0x | 6 * 50 = 300 |
| HIGH | 5.0x | 2,000 * 5 = 10,000 |
| MEDIUM | 1.0x | 9,855 * 1 = 9,855 |
| LOW | 2.0x | 5,319 * 2 = 10,638 |
| SKIP | 3.0x | 3,084 * 3 = 9,252 |

**Notebook changes** (`build_notebook.py`):
- Cell 5: Reads `exp5_captions.jsonl` + `exp5_labels.jsonl` (not old `captions.jsonl`/`labels.jsonl`). Uses ALL samples, no downsampling.
- Cell 6: Stratified 90/10 split (not the old "downsample MEDIUM to match LOW+SKIP" which caused Exp 4's catastrophic forgetting).
- Cell 8 (Dataset): Returns `class_weight` tensor per sample. Priority stored in JSONL, looked up from `CLASS_WEIGHTS` dict.
- Cell 9 (VLMTrainer): `compute_loss` pops `class_weight` from inputs, multiplies per-sample cross-entropy loss by it. This is NOT the same as `F.cross_entropy(weight=)` which operates on token vocabulary IDs.
- Cell 10: Commit message updated for Exp 5.

**Implementation detail**: `F.cross_entropy(weight=...)` weights token IDs in the 50K+ vocabulary - useless for our problem. Weighting the token "C" vs "M" is meaningless. The correct approach: compute normal loss per sample, then multiply by class scalar before averaging.

## Step 6: MPS Training Test

**Result: PASSED - but too slow for full training.**

MPS backward pass works on Apple Silicon for LFM2.5-VL-450M. No operator failures from state-space layers. The friend's concern about MPS compatibility was valid to check but did not materialize.

**Timing (per sample, float32 on MPS)**:
- Forward: 18-29s
- Backward: 92-146s
- Optimizer step: 1-2s
- Total per step: ~2.5 min

**Loss**: 4.50 → 4.38 after 2 steps (model is learning).

**Why MPS is not viable**: 18,236 train samples * 3 epochs = ~55K steps. At 2.5 min/step = ~95 days. Kaggle T4 with bfloat16 will be orders of magnitude faster (estimated ~5s/step = ~3 days for full run, or ~12h with gradient accumulation and T4 throughput).

**Decision**: All training on Kaggle T4. MPS for inference only.

**Environment note**: Must use `.venv` Python (transformers 5.5.4) not system Python (3.9.6/transformers 4.57.6). System Python can't load `Lfm2VlProcessor`.

## Other Fixes

- **Dockerfile**: Pinned `snapshot_download` to `revision='4e5353b0'` (Exp 1 model). Previously downloaded latest (Exp 4, which is broken). Will update to Exp 5 revision after training.
- **Re-indexed IDs**: `captions_cleaned.jsonl` and `labels_exp5.jsonl` re-indexed 0-20263 (original file had wrapping IDs from a partial previous run).
