# Reviewed Batches

This directory stores reviewed label batches before they are merged into a
shared eval manifest.

Recommended workflow:

1. Capture raw candidates into `evals/candidates/`.
2. Create a reviewed batch JSONL here with `expected_priority`, `notes`, and
   optional `ambiguous`.
3. Append that batch into `evals/sentinel_eval_v1.jsonl` using
   `scripts/register_eval_samples.py`.
4. Re-run `scripts/evaluate_current_cascade.py`.

Keeping reviewed batches separate makes it easy to audit when each group of
labels was added and what assumptions were used.
