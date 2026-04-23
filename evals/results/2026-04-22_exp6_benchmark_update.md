# EXP 6 Benchmark Update

Date: `2026-04-22`

## Historical Note

As of the hazard-policy update on `2026-04-22`, `HIGH` is reserved for hazard scenes
only. The four port / mine samples added in the earlier targeted `HIGH` slice were
useful diagnostics, but they are now treated as `MEDIUM` counterexamples rather than
valid `HIGH` targets.

Sections that discuss ports/mines as `HIGH` should therefore be read as **historical
pre-policy results**, not as the current benchmark definition.

## What Changed

- Added a first reviewed SimSat batch at
  [`evals/review_batches/2026-04-22_historical_demo_reviewed.jsonl`](../review_batches/2026-04-22_historical_demo_reviewed.jsonl)
- Added a second reviewed SimSat batch at
  [`evals/review_batches/2026-04-12_historical_demo_reviewed.jsonl`](../review_batches/2026-04-12_historical_demo_reviewed.jsonl)
- Expanded [`evals/sentinel_eval_v1.jsonl`](../sentinel_eval_v1.jsonl) from 4
  seed samples to 24 reviewed samples
- Added a policy relabel batch at
  [`evals/review_batches/2026-04-22_policy_relabel_nonhazard_high_to_medium.jsonl`](../review_batches/2026-04-22_policy_relabel_nonhazard_high_to_medium.jsonl)
- Added retry handling to
  [`scripts/capture_eval_candidates.py`](/Users/marcelo/Documents/GitHub/automatic-downlink/scripts/capture_eval_candidates.py:1)
- Extended
  [`scripts/evaluate_current_cascade.py`](/Users/marcelo/Documents/GitHub/automatic-downlink/scripts/evaluate_current_cascade.py:1)
  to compare generation presets, prompt modes, fixed seeds, and decision-layer toggles
- Added two bounded prefilter heuristics in
  [`src/triage/engine.py`](/Users/marcelo/Documents/GitHub/automatic-downlink/src/triage/engine.py:1)
  for mixed cloud-dominated frames and bright barren terrain
- Added a conservative post-VLM decision layer in
  [`src/triage/engine.py`](/Users/marcelo/Documents/GitHub/automatic-downlink/src/triage/engine.py:1)
  plus trace fields in
  [`src/triage/schemas.py`](/Users/marcelo/Documents/GitHub/automatic-downlink/src/triage/schemas.py:1)

## Benchmark Composition

- Total samples: `24`
- Expected priorities:
  - `SKIP`: `12`
  - `LOW`: `8`
  - `MEDIUM`: `4`
- Sources:
  - `4` checked-in seed images from `test_images/`
  - `20` reviewed SimSat Sentinel captures

Historical pre-policy `HIGH` expansion:

- Total samples: `28`
- Expected priorities:
  - `SKIP`: `12`
  - `LOW`: `8`
  - `MEDIUM`: `4`
  - `HIGH`: `4`

Current manifest after hazard-policy relabel:

- Total samples: `28`
- Expected priorities:
  - `SKIP`: `12`
  - `LOW`: `8`
  - `MEDIUM`: `8`
- Current valid `HIGH` hazard slice: pending rebuild

## Results

### A. Shipped Cascade on 14 Samples

- Samples: `14`
- Priority match: `9/14` = `64.3%`
- Prefilter hits: `7/14` = `50.0%`
- Predicted distribution: `SKIP 7`, `MEDIUM 7`

Interpretation:
- Prefilter handled obvious junk correctly.
- The model still collapsed heavily toward `MEDIUM` on non-trivial scenes.

### B. Shipped Cascade on 24 Samples

Command:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --output-dir /tmp/automatic-downlink-eval-24-baseline
```

- Samples: `24`
- Priority match: `17/24` = `70.8%`
- Prefilter hits: `12/24` = `50.0%`
- Avg latency: `3.15s`
- Predicted distribution: `SKIP 12`, `MEDIUM 10`, `LOW 2`

Interpretation:
- Baseline remains usable for obvious `SKIP`.
- The core weakness remains `LOW` vs `MEDIUM` separation.
- Several routine low-value terrain frames still drift upward to `MEDIUM`.

### C. Deterministic Generation + Simple Prompt on 24 Samples

Command:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --generation-preset deterministic \
  --prompt-mode simple \
  --output-dir /tmp/automatic-downlink-eval-24-deterministic-simple
```

- Samples: `24`
- Priority match: `15/24` = `62.5%`
- Prefilter hits: `12/24` = `50.0%`
- Avg latency: `1.37s`
- Predicted distribution: `SKIP 13`, `HIGH 7`, `MEDIUM 4`

Interpretation:
- This variant is faster.
- It is not better calibrated.
- Instead of collapsing toward `MEDIUM`, it over-escalates many routine `LOW`
  scenes to `HIGH`.

### D. Tuned Prefilter + Shipped Generation on 24 Samples

Command:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --output-dir /tmp/automatic-downlink-eval-24-prefilter-tuned
```

- Samples: `24`
- Priority match: `20/24` = `83.3%`
- Prefilter hits: `14/24` = `58.3%`
- Avg latency: `2.97s`
- Predicted distribution: `SKIP 13`, `MEDIUM 7`, `LOW 4`

Interpretation:
- This is the best result so far on the 24-sample benchmark.
- The added rules correctly absorbed a cloud-dominated rainforest frame and a
  bright overexposed desert frame before they reached the VLM.
- Remaining misses are still concentrated in `LOW` vs `MEDIUM`, not in obvious
  `SKIP`.

### E. Decision Layer Comparison on 24 Samples (`seed=42`)

Decision layer disabled:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --seed 42 \
  --disable-decision-layer \
  --output-dir /tmp/automatic-downlink-eval-24-no-decision-layer
```

- Priority match: `18/24` = `75.0%`
- Predicted distribution: `SKIP 13`, `MEDIUM 9`, `LOW 2`

Decision layer enabled:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --seed 42 \
  --output-dir /tmp/automatic-downlink-eval-24-with-decision-layer
```

- Priority match: `20/24` = `83.3%`
- Predicted distribution: `SKIP 13`, `MEDIUM 7`, `LOW 4`

Interpretation:
- With the same seed and the same tuned prefilter, the decision layer improved
  the benchmark by `+2` correct samples.
- The gains came from conservative `MEDIUM -> LOW` corrections on routine
  terrain/vegetation scenes.
- The layer did not change `SKIP`, `HIGH`, or `CRITICAL` behavior.

### F. Targeted `HIGH` Expansion on 28 Samples (`seed=42`)

Added reviewed `HIGH` scenes:

- `long_beach_port_20260412T120000Z`
- `jebel_ali_port_20260412T120000Z`
- `chuquicamata_mine_20260412T120000Z`
- `escondida_mine_20260412T120000Z`

Command:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --seed 42 \
  --output-dir /tmp/automatic-downlink-eval-28-with-high
```

- Samples: `28`
- Priority match: `20/28` = `71.4%`
- Predicted distribution: `SKIP 13`, `MEDIUM 10`, `LOW 5`
- `HIGH` recall on benchmark: `0/4`

Interpretation:
- The cascade still works reasonably in the `SKIP/LOW/MEDIUM` regime.
- The current model does not elevate any of the targeted `HIGH` scenes to
  `HIGH`.
- Ports are flattened to `MEDIUM`.
- One mine is flattened all the way to `LOW`.

Control check with decision layer disabled:

```bash
.venv/bin/python scripts/evaluate_current_cascade.py \
  --offline \
  --seed 42 \
  --disable-decision-layer \
  --output-dir /tmp/automatic-downlink-eval-28-no-decision-layer
```

- Priority match: `18/28` = `64.3%`
- `HIGH` recall on benchmark: `0/4`

Interpretation:
- The conservative decision layer is not the reason `HIGH` scenes are missed.
- The failure is upstream: the current VLM does not yet map these scenes into
  the `HIGH` class.

## Current Conclusion

The benchmark now supports a stronger statement than before:

- The deterministic prefilter is valuable and should stay.
- There is still low-hanging fruit in the prefilter layer.
- The current model is adequate for obvious `SKIP` cases.
- A conservative post-VLM decision layer improves `LOW` vs `MEDIUM`
  calibration without touching model weights.
- The current model is not reliably calibrated on non-trivial real-domain
  `LOW` / `MEDIUM` decisions.
- The current model also fails completely on the first targeted `HIGH`
  benchmark slice (`0/4` recall).
- Prompt simplification and deterministic decoding alone do not fix the issue.
- Small, benchmark-driven cascade improvements can move the total score
  materially without any retraining.

This does not force retraining immediately, but it does make the case for a
targeted next experiment stronger. The next likely useful comparison is:

1. shipped weights + shipped prompt
2. shipped weights + current decision layer on a benchmark with `HIGH` examples
3. targeted real-domain supervision for `HIGH` scenes
4. updated or re-labeled fine-tune on a small real-domain set

## Open Questions

- Are some of the `LOW` labels better treated as `MEDIUM` for the actual demo
  story, especially partially occluded urban strips?
- Should no-data wedges be handled by stronger deterministic rules so they never
  reach the VLM?
- Should partially missing urban tiles be forced to `SKIP`, `LOW`, or flagged as
  explicitly ambiguous in the product?
- Would a smaller schema plus class-constrained post-processing improve
  calibration without retraining?
- Does the decision layer still help once the benchmark includes `HIGH`
  examples, or does it only improve the low-value regime?
- Are ports/mines the right semantics for `HIGH` in the product, or should
  `HIGH` stay reserved for anomaly/change detection only?
- Is the right next step a small real-domain fine-tune, or simply a better
  decision layer on top of the current descriptions?
