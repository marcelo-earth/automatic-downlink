# Candidate Captures

This directory holds raw candidate Sentinel-2 captures collected from SimSat
before they are reviewed and promoted into an evaluation manifest.

Contents:

- `images/`
  Captured PNG tiles.
- `sentinel_candidates.jsonl`
  Capture metadata emitted by `scripts/capture_eval_candidates.py`.

Reviewed labels should not be written directly into `sentinel_candidates.jsonl`.
Instead, create a reviewed batch under `evals/review_batches/` and then append it
to an eval manifest with `scripts/register_eval_samples.py`.
