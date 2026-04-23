# Hazard Seed Capture

Date: `2026-04-22`

## Goal

Build the first reviewed hazard-focused seed slice for `EXP_6_HIGH` using
historical SimSat Sentinel-2 captures with paired `rgb` and `swir` views.

## Capture Command

```bash
python3 scripts/capture_eval_candidates.py \
  --locations-file evals/locations/hazard_high_seed_locations.jsonl \
  --view-preset rgb-swir \
  --limit 6
```

## Captured Groups

- `lahaina_wildfire_20230810T120000Z`
- `attica_wildfire_20230824T120000Z`
- `rio_grande_flood_20240508T120000Z`
- `valencia_flood_20241101T120000Z`
- `enga_landslide_20240527T120000Z`
- `ventanilla_spill_20220121T120000Z`

Each group produced:

- one `rgb` image
- one `swir` image

## Review Outcome

The first pass is intentionally mixed. That is useful because it reveals both
positive seeds and retargeting failures.

### Usable `HIGH` seeds

- `lahaina_wildfire_20230810T120000Z`
  Strong wildfire aftermath example. Burn scar is clear in RGB and remains
  supported by SWIR.
- `valencia_flood_20241101T120000Z`
  More ambiguous because of cloud cover, but still plausible as a reviewed flood
  aftermath `HIGH` seed.
- `attica_wildfire_20230824T120000Z`
  Kept as an ambiguous `HIGH` seed rather than a clean benchmark anchor.

### Hard negatives / failures

- `rio_grande_flood_20240508T120000Z`
  Informative urban scene, but not a defensible flood `HIGH` at this crop.
- `enga_landslide_20240527T120000Z`
  Failed due to a dominant no-data wedge.
- `ventanilla_spill_20220121T120000Z`
  Failed due to blown-out cloud / overexposure.

## Files Added

- Reviewed batch:
  [`evals/review_batches/2026-04-22_hazard_seed_reviewed.jsonl`](../review_batches/2026-04-22_hazard_seed_reviewed.jsonl)
- Seed manifest:
  [`evals/hazard_high_seed_v1.jsonl`](../hazard_high_seed_v1.jsonl)

## Baseline Evaluation on This Slice

Command:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --manifest evals/hazard_high_seed_v1.jsonl \
  --output-dir /tmp/automatic-downlink-hazard-seed-eval
```

Result:

- Samples: `6`
- Priority match: `3/6 = 50.0%`
- Prefilter hits: `2/6 = 33.3%`
- Expected distribution: `HIGH 3 / MEDIUM 1 / SKIP 2`
- Predicted distribution: `MEDIUM 4 / SKIP 2`

Interpretation:

- The current RGB-only cascade correctly skips the unusable landslide and spill captures.
- It keeps flattening the reviewed hazard `HIGH` seeds to `MEDIUM`.
- On this first slice, effective `HIGH` recall is `0/3`.
- That is directionally consistent with the broader `EXP_6_HIGH` diagnosis: the next
  useful work is targeted hazard supervision, not more `MEDIUM -> LOW` heuristics.

## Read

This is a **seed slice**, not yet a stable benchmark.

The right next move is:

1. add cleaner hazard positives, especially flood and landslide
2. keep oil spill, but only under favorable conditions
3. compare current RGB-only inference against this slice
4. later train against reviewed RGB + SWIR pairs, not just RGB
