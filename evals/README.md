# Eval Sets

This directory contains small, human-reviewable evaluation manifests for running
the current triage cascade on real or representative imagery.

## Format

Each manifest is JSONL. One line per image.

Required fields:

- `image_path`: path to the image, relative to the repo root
- `expected_priority`: one of `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `SKIP`
- `notes`: short human note explaining the label

Optional fields:

- `id`: stable identifier for the sample
- `source`: where the image came from (`test_images`, `simsat`, etc.)
- `ambiguous`: `true` when the label is debatable and should be treated with caution

## Current sets

- `sentinel_eval_v1.jsonl`
  Initial bootstrap set using the checked-in local test images. This is not a
  complete benchmark; it is a seed set for validating the cascade locally and
  making the evaluation workflow reproducible.

## Growing the benchmark

The recommended `EXP_6` workflow is:

1. Capture or collect candidate images.
2. Review them manually and assign `expected_priority` plus a short note.
3. Register the reviewed rows into a JSONL eval manifest.
4. Run `scripts/evaluate_current_cascade.py` on the updated manifest.

Useful scripts:

- `scripts/capture_eval_candidates.py`
  Stages unlabeled Sentinel candidate images from a local SimSat instance into
  `evals/candidates/`.

- `scripts/register_eval_samples.py`
  Validates reviewed rows and appends them into an evaluation manifest.

Related directories:

- `evals/candidates/`
  Raw candidate captures and capture metadata before review.

- `evals/review_batches/`
  Reviewed label batches that can be merged into the main manifest.

- `evals/results/`
  Human-readable notes on benchmark growth and evaluation outcomes.

- `evals/locations/`
  Reusable targeted location lists for capturing specific scene types.

Examples:

```bash
.venv/bin/python scripts/capture_eval_candidates.py --dry-run
.venv/bin/python scripts/capture_eval_candidates.py --limit 5

.venv/bin/python scripts/register_eval_samples.py \
  --manifest evals/sentinel_eval_v1.jsonl \
  --image-path test_images/sentinel_sahara.png \
  --expected-priority LOW \
  --notes "Routine desert tile with no obvious anomaly." \
  --id sentinel_sahara_manual \
  --source test_images

.venv/bin/python scripts/evaluate_current_cascade.py --offline
.venv/bin/python scripts/evaluate_current_cascade.py --offline --seed 42 --disable-decision-layer
.venv/bin/python scripts/evaluate_current_cascade.py --offline --seed 42
```
