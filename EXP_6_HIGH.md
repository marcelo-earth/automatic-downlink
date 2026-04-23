# Experiment 6 HIGH Plan

## Goal

Recover hazard-oriented `HIGH` recall on real-domain imagery without losing the gains
already achieved in `SKIP`, `LOW`, and `MEDIUM`.

This is a continuation of [`EXP_6.md`](/Users/marcelo/Documents/GitHub/automatic-downlink/EXP_6.md:1), not a separate experiment family.

The question is now:

> Can the current system learn to recognize visible hazard aftermath, probable hazard,
> and elevated hazard risk as `HIGH`, instead of flattening everything into
> `MEDIUM`?

---

## Policy Update

The product is now explicitly **hazard triage**.

That means:

- `CRITICAL` = active hazard clearly visible now
- `HIGH` = visible hazard aftermath, probable hazard, or strong hazard-linked risk
- `MEDIUM` = informative or anomalous scene without confirmed hazard

Important consequence:

- ports, mines, airports, cities, and industrial sites are **not** `HIGH` by default
- they are `MEDIUM` unless the image shows a hazard-related reason to escalate them

The source of truth is [`PRIORITY_POLICY.md`](PRIORITY_POLICY.md).

---

## Current Evidence

The earlier targeted `HIGH` slice used ports and mines as stand-ins for "important"
scenes. Under the current hazard policy, that slice is no longer a valid `HIGH`
benchmark.

Those scenes should now be treated as:

- useful counterexamples
- mostly `MEDIUM`
- evidence that "strategic-looking" is not enough

So the next version of `EXP_6_HIGH` needs a **new hazard-aligned `HIGH` slice**, not
another infrastructure slice.

---

## What This Experiment Is

`EXP_6_HIGH` is a **targeted real-domain hazard supervision experiment**.

It is not:

- a return to caption-derived labels as the main source of truth
- another generic infrastructure classifier
- an attempt to patch `HIGH` with upward heuristics

It is:

- a narrow attempt to teach the current model what hazard `HIGH` means in this
  product
- measured against a frozen real-domain benchmark

---

## Core Hypothesis

The current model can often describe satellite scenes in broadly plausible language,
but it does not yet map hazard-related visual evidence into the right priority policy.

So the hypothesis is:

> A small targeted fine-tune with real-domain hazard `HIGH` examples, matched to the
> current policy, will improve `HIGH` recall more effectively than further prompt
> tweaking or post-processing.

This is especially likely if the fine-tune uses richer Sentinel-2 views than RGB alone,
for example RGB + SWIR on the same tile.

---

## Constraints

This experiment must preserve the gains from `EXP_6`.

That means:

- do not break `SKIP` handling
- do not regress `LOW` / `MEDIUM` calibration badly
- do not remove the prefilter or conservative `MEDIUM -> LOW` decision layer
- evaluate on the existing benchmark before claiming success

Success is not "the model says `HIGH` more often."

Success is:

- hazard `HIGH` recall increases on reviewed examples
- overall benchmark quality remains defensible
- the system still behaves conservatively outside the hazard slice

---

## Scope

Focus `HIGH` on the current product definition:

- wildfire aftermath with clear burn scars or nearby risk context
- receding flood or obvious residual inundation effects
- landslide scars / unstable slopes with strong visual evidence
- oil spill or coastal contamination only under favorable conditions

Do **not** broaden `HIGH` yet to include everything that looks strategically important.

Keep the semantics narrow enough to train and measure.

---

## Dataset Strategy

### 1. Keep the current eval set frozen, but relabel policy mismatches

The existing reviewed manifest remains the regression guardrail for:

- `SKIP`
- `LOW`
- `MEDIUM`

Any scenes that were previously labeled `HIGH` for "strategic importance" should be
downgraded to `MEDIUM` or removed from the `HIGH` slice.

### 2. Build a new hazard `HIGH` set

Collect a compact real-domain set with emphasis on:

- wildfire aftermath
- flood aftermath
- landslide aftermath or unstable slopes
- oil spill / coastal contamination under favorable conditions

Target scale:

- `20-60` reviewed `HIGH` examples
- plus matched `MEDIUM` counterexamples

The counterexamples matter because the problem is discriminative:

- not "what does a port or coastline look like?"
- but "what deserves `HIGH` instead of `MEDIUM` under hazard policy?"

### 3. Include hard negatives

Counterexamples should include scenes that are visually salient but non-hazard:

- ports
- mines
- airports
- dense cities
- coastlines with no clear spill
- dark water without a defensible slick signal

### 4. Prefer RGB + companion views when useful

If the capture pipeline allows it, collect paired inputs such as:

- RGB composite
- SWIR composite
- optionally NIR-derived view

This follows the pattern already used in the wildfire cookbook and does not require a
new architecture.

---

## Labeling Policy

Each training example should include:

- image or image pair
- `description`
- `priority`
- short note on why that priority is correct

For `HIGH`, the note should express hazard semantics, not generic scene importance.

Good examples:

- "Clear burn scar and smoke-adjacent wildfire aftermath; hazard remains operationally relevant"
- "Floodwater has receded but saturated terrain and expanded water extent remain clearly visible"
- "Fresh landslide scar on vegetated slope with visible debris fan"
- "Dark coastal slick under favorable conditions with plausible contamination pattern"

Bad examples:

- "important infrastructure"
- "strategically valuable scene"
- "many objects visible"

---

## Training Plan

### Option A: Small targeted fine-tune on current model

Preferred first move.

Train the current model on:

- a small real-domain hazard `HIGH` set
- policy-aligned counterexamples
- the same output schema used at inference

### Option B: RGB + SWIR hazard pass

If RGB-only continues to blur hazard classes, the next pass should try:

- RGB input
- SWIR companion image of the same tile

This is especially attractive for:

- wildfire
- flood
- hazard aftermath discrimination

### Option C: Mixed refresh set

If pure `HIGH` tuning causes instability, mix:

- targeted `HIGH` examples
- a smaller number of `SKIP`, `LOW`, and `MEDIUM` anchor examples from the same
  deployment domain

This helps prevent class drift.

---

## Evaluation Plan

Every training run must be evaluated on:

1. the frozen general benchmark
2. the new hazard `HIGH` subset
3. prediction distribution drift

Track at minimum:

- overall accuracy on the benchmark
- hazard `HIGH` recall
- hazard `HIGH` precision, if sample count allows
- false escalation of non-hazard scenes
- whether `SKIP` or `LOW` regress badly

Key success criteria:

- hazard `HIGH` recall becomes meaningfully non-zero
- non-hazard infrastructure does not get promoted to `HIGH` without evidence
- overall benchmark quality stays defensible

Practical first success bar:

- at least `2-4` clearly correct hazard `HIGH` predictions on a reviewed subset
- no major collapse elsewhere

---

## What Not To Do

- Do not reuse "important infrastructure" as a shortcut for `HIGH`.
- Do not add aggressive upward post-processing heuristics.
- Do not treat oil spill as a strong capability outside favorable conditions.
- Do not expand `HIGH` semantics mid-experiment.
- Do not judge success by cherry-picked examples.

---

## Open Questions

- How many reviewed hazard `HIGH` examples are needed before the model separates them from routine `MEDIUM` scenes?
- Is RGB-only enough for the first hazard pass, or should RGB + SWIR be the default immediately?
- Which hazard family gives the cleanest first win: wildfire, flood, landslide, or oil spill?
- Should some "risk-only" scenes stay `HIGH`, or should they remain `MEDIUM` unless aftermath is visible?
- How much manual review is needed before oil-spill labels become reliable enough to train on?

---

## Recommended Immediate Next Steps

1. Freeze the current non-hazard benchmark after relabeling old policy mismatches.
2. Build a new reviewed hazard `HIGH` slice.
3. Start with the cleanest hazard family first, likely wildfire or flood.
4. Compare RGB-only against RGB + SWIR if capture/rendering is cheap enough.
5. Fine-tune only after the hazard slice exists and the labels are reviewed.
