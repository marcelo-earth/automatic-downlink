# Candidate Captures

This directory holds raw candidate Sentinel-2 captures collected from SimSat
before they are reviewed and promoted into an evaluation manifest.

Contents:

- `images/`
  Captured PNG tiles.
- `sentinel_candidates.jsonl`
  Capture metadata emitted by `scripts/capture_eval_candidates.py`.

Each candidate row may now belong to a multi-view group:

- `candidate_group_id`
  Stable id shared by related captures of the same tile and timestamp.
- `view_name`
  Name of the rendered view, e.g. `rgb` or `swir`.

Reviewed labels should not be written directly into `sentinel_candidates.jsonl`.
Instead, create a reviewed batch under `evals/review_batches/` and then append it
to an eval manifest with `scripts/register_eval_samples.py`.
