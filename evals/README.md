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
