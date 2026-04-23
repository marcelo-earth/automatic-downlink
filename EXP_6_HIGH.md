# Experiment 6 HIGH Plan

## Goal

Recover `HIGH` recall on real-domain imagery without losing the gains already achieved in `SKIP`, `LOW`, and `MEDIUM`.

This is a continuation of [`EXP_6.md`](/Users/marcelo/Documents/GitHub/automatic-downlink/EXP_6.md:1), not a separate experiment family.

The question is no longer:

> Can we build a real-domain benchmark and improve the cascade?

That has already been answered.

The new question is:

> Can the current system learn to recognize strategically important industrial / infrastructure scenes as `HIGH`, instead of flattening them to `MEDIUM` or `LOW`?

---

## Current Evidence

After the targeted `HIGH` benchmark expansion, the system now has:

- `28` reviewed eval samples
- expected distribution:
  - `SKIP`: `12`
  - `LOW`: `8`
  - `MEDIUM`: `4`
  - `HIGH`: `4`

The key result is:

- `HIGH recall = 0/4`

Specifically, the current model:

- describes major ports reasonably, but predicts `MEDIUM`
- describes large open-pit mines poorly or generically, sometimes predicting `LOW`

This matters because it changes the diagnosis:

- The main `LOW` / `MEDIUM` calibration problem can be improved by the cascade.
- The `HIGH` problem is upstream of the decision layer.
- The current VLM is not reliably mapping these scenes into the `HIGH` class.

That means more post-processing alone is unlikely to solve `HIGH`.

---

## What This Experiment Is

`EXP_6_HIGH` is a **targeted real-domain supervision experiment**.

It is not:

- a resurrection of `EXP_5`
- another broad caption-cleaning pass
- a generic “let’s retrain everything and hope” run

It is:

- a narrow attempt to teach the current model what `HIGH` should mean in this product
- using the real benchmark from `EXP_6` as the guardrail

---

## Core Hypothesis

The current model already has enough visual competence to recognize:

- ports
- ships
- industrial facilities
- open-pit mines

But it does **not** have a strong enough supervisory mapping from those visual patterns to the `HIGH` class in this product.

So the hypothesis is:

> A small targeted fine-tune with real-domain `HIGH` examples and matched policy labels will improve `HIGH` recall more effectively than further prompt tweaking or post-processing.

---

## Constraints

This experiment must preserve the gains from `EXP_6`.

That means:

- do not break `SKIP` handling
- do not regress `LOW` / `MEDIUM` calibration badly
- do not remove the prefilter or conservative decision layer
- evaluate on the existing benchmark before claiming success

Success is not “the model says HIGH more often.”

Success is:

- `HIGH` recall increases
- overall benchmark quality remains acceptable
- the system is still demo-defensible

---

## Scope

Focus `HIGH` on the current product definition:

- large industrial infrastructure
- major ports / maritime logistics hubs
- major mines / extraction sites
- concentrated strategic activity that is clearly more valuable than routine terrain

Do **not** broaden `HIGH` yet to include everything that might be important.

Keep the semantics narrow enough to train and measure.

---

## Dataset Strategy

### 1. Keep the current eval set frozen

Use the current `28`-sample benchmark as the primary regression guardrail.

Do not relabel it casually during training.

### 2. Build a small targeted training set for `HIGH`

Collect a compact real-domain set with emphasis on:

- ports
- shipping terminals
- airports if clearly visible and infrastructure-dense
- large mines / extraction facilities
- other major industrial zones only if they are visually unambiguous

Target scale:

- `20-60` reviewed `HIGH` examples
- plus a matched set of nearby `MEDIUM` or `LOW` counterexamples

The counterexamples matter because the problem is discriminative:

- not “what does a port look like?”
- but “what deserves `HIGH` instead of `MEDIUM`?”

### 3. Avoid proxy labels as the main source

Do not return to caption-derived priorities as the core label source.

Allowed:

- use captions or teacher models to speed up draft annotation

Not allowed:

- trust those draft labels without review for the key `HIGH` set

---

## Labeling Policy

Each training example should include:

- image
- `description`
- `priority`
- short note on why that priority is correct

For `HIGH`, the note should express product semantics, not abstract image semantics.

Examples:

- “Major container port with dense logistics infrastructure; high-value strategic scene”
- “Large open-pit mine and processing area; industrial monitoring target”

Avoid labels like:

- “interesting image”
- “complex scene”
- “many objects”

Those are not operational definitions.

---

## Training Plan

### Option A: Small targeted fine-tune on current model

Preferred first move.

Train the current model on:

- a small real-domain `HIGH`-focused set
- the existing schema used at inference

Possible recipe:

- keep the model base the same
- keep prompt format aligned with inference
- bias the small dataset toward the new `HIGH` distinction
- avoid trying to relearn everything

### Option B: Mixed refresh set

If pure `HIGH` tuning causes instability, mix:

- targeted `HIGH` examples
- a smaller number of `SKIP`, `LOW`, and `MEDIUM` anchor examples from the real benchmark domain

This helps prevent class drift.

---

## Evaluation Plan

Every training run must be evaluated on:

1. the full current benchmark
2. the `HIGH` subset alone
3. distribution shift in predictions

Track at minimum:

- overall accuracy on the benchmark
- `HIGH` recall
- `HIGH` precision, if sample count allows
- change in predicted distribution
- whether `SKIP` or `LOW` regress badly

Key success criteria:

- `HIGH recall` improves from `0/4` to something meaningfully non-zero
- overall benchmark quality stays defensible
- `SKIP` performance remains strong

Practical first success bar:

- at least `2/4` correct on current `HIGH` eval samples
- no major collapse elsewhere

---

## What Not To Do

- Do not rely on prompt changes alone and call that the solution.
- Do not add aggressive upward post-processing heuristics.
- Do not expand `HIGH` semantics mid-experiment.
- Do not discard the current benchmark if a training run performs poorly on it.
- Do not judge success by eyeballing a few cherry-picked examples.

---

## Open Questions

- Are ports and mines truly the right first-class `HIGH` semantics for the product, or are they only good stand-ins for “strategic scenes”?
- Should airports be included in the first targeted `HIGH` set, or are they too visually inconsistent in Sentinel tiles?
- How many reviewed `HIGH` examples are needed before the current model can separate them from routine `MEDIUM` scenes?
- Will a small targeted fine-tune be enough, or does the current model fundamentally need better domain adaptation?

---

## Recommended Next Step

The next practical step is:

1. curate `20-60` real-domain `HIGH` examples plus counterexamples
2. package them into a small training set aligned to the current inference schema
3. run a targeted fine-tune
4. evaluate against the frozen `EXP_6` benchmark

This keeps the work inside the `EXP_6` logic:

- real benchmark first
- measurable hypothesis
- targeted intervention
- regression check before claiming progress
