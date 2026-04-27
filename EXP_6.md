# Experiment 6 Plan

## Goal

Build a **credible onboard hazard-triage system** for satellite imagery, not just a model that emits plausible JSON.

This is the parent experiment.

- `EXP_6`: real-domain benchmark, cascade design, prefilter tuning, and non-training improvements
- `EXP_6_HIGH`: targeted continuation focused on recovering hazard `HIGH` recall on real-domain wildfire / flood / landslide / spill scenes

The target system should:

1. Reject obvious junk cheaply before invoking the VLM
2. Produce stable, valid JSON on real Sentinel-2 / SimSat imagery
3. Assign priorities that are useful for bandwidth allocation, not just linguistically plausible
4. Be defensible in a demo as an **onboard triage pipeline**, not as a single magical model

---

## Reframed Problem

The problem is **not** "how do we caption satellite images?"

The problem is:

> Given limited downlink bandwidth, how do we decide which captured images deserve full transmission because they show a hazard, likely hazard, or visible hazard aftermath?

This matters because several previous experiments optimized for proxies:

- VRSBench caption quality
- JSON validity
- prompt-following behavior
- class balance in synthetic labels

Those are useful subproblems, but they are not the core objective.

The real objective is **hazard-triage quality on deployment-domain imagery**.

---

## What Exp 5 Got Right

Exp 5 identified several real problems correctly:

1. **The domain gap is real.**
   VRSBench is not Sentinel-2 / SimSat.

2. **Training/inference prompt alignment matters.**
   If the model is trained under one prompt distribution and evaluated under another, results become noisy and misleading.

3. **Caption-derived labels are noisy.**
   Priority should reflect what is visible and operationally useful, not just what a caption mentions.

4. **Blindly downsampling for balance is dangerous.**
   Throwing away most of the data caused catastrophic degradation.

These conclusions should be preserved.

---

## What Exp 5 Still Got Wrong

### 1. It still treated caption-based supervision as the main path

Even after recognizing that priority does not really come from captions, Exp 5 still centered the plan on:

- cleaning captions
- reclassifying captions
- rebalancing caption-derived labels
- training on those labels at scale

That can improve the model as a **caption-conditioned triage imitator**, but it does not directly solve the actual task.

### 2. Real-domain evaluation came too late

Exp 5 put real Sentinel-2 evaluation near the end of the plan.

That is backwards.

If the deployment domain is Sentinel-2 / SimSat, then a small real-domain benchmark must exist at the start and be used throughout.

### 3. The model was still asked to do too much

The current schema asks the model to:

- describe
- prioritize
- reason
- categorize

For a 450M VLM under domain shift, this is a lot.

Some of these fields are core to the task. Some are presentation sugar.

### 4. The architecture remained too model-centric

The actual system should be a **cascade**:

- deterministic rejectors
- optional lightweight classifier
- VLM for ambiguous / meaningful cases

Exp 5 still mainly framed progress as "make the VLM better."

### 5. Distribution shaping risk remained high

Target class distributions are useful as sanity checks, but dangerous as objectives.

If we force the data to "look reasonable" statistically, we may produce a nice training set that does not reflect real onboard triage.

---

## Core Hypothesis for Exp 6

The best path is **not**:

`image -> VLM -> all decisions`

It is:

`image -> cheap rejectors -> maybe lightweight classifier -> VLM for the hard cases`

This is the main architectural change in Exp 6.

---

## Exp 6 System Design

### Stage A: Deterministic prefilter

Before the VLM sees anything, run a cheap first-pass filter that catches obvious low-value frames.

Examples:

- heavy cloud cover
- underexposed / overexposed images
- extremely low-contrast / featureless tiles
- near-empty ocean or uniform terrain
- obviously corrupted or unusable frames

Output of the prefilter:

- `SKIP`
- `LOW`
- `UNCERTAIN`

Only `UNCERTAIN` proceeds to the next stage by default.

Important principle:

The prefilter should handle only **obvious** cases.
It should not attempt to solve the whole problem.

### Stage B: Optional lightweight gate

If needed, insert a tiny learned component between the deterministic prefilter and the VLM.

Examples:

- binary classifier: `junk vs non-junk`
- ternary classifier: `skip / low / needs-vlm`
- simple feature-based model using image statistics

This stage exists only if deterministic rules are too brittle.

### Stage C: VLM for hard cases

The VLM handles what cheap filtering cannot safely decide:

- meaningful scene description
- downlink priority for non-trivial images
- explanation suitable for operator review / demo

At this stage the VLM should ideally output a **minimal task-aligned schema**:

```json
{
  "description": "...",
  "priority": "CRITICAL | HIGH | MEDIUM | LOW | SKIP"
}
```

Optional fields like `reasoning` and `categories` should be treated as secondary.

They can be:

- added later
- generated by a second pass
- derived heuristically
- included only for demo/UI purposes

---

## New Training Philosophy

### Principle 1: Train on the real task as early as possible

The first benchmark should not be VRSBench-only.

It should be a small, manually reviewed set of real Sentinel-2 / SimSat images.

### Principle 2: Separate "representation learning" from "deployment triage"

VRSBench is still useful, but only as a supporting dataset:

- good for learning satellite-ish visual grounding and description style
- not sufficient as the main supervision source for onboard triage decisions

### Principle 3: Prefer a small, correct real-domain set over a huge proxy set

50-200 well-labeled real-domain images are more strategically valuable than 20K noisy proxy labels if the deployment problem is truly different.

### Principle 4: Use the VLM where language helps

The VLM is valuable because it can:

- describe what it sees
- justify decisions
- support prompt steering

It should not be wasted on obvious cloud rejection or image quality failure cases.

---

## Exp 6 Plan

### Step 0: Define the operational decision policy

Before any more training:

- define exactly what `CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `SKIP` mean operationally
- define what gets:
  - full image
  - thumbnail
  - text summary only
- define whether uncertainty should map to `MEDIUM`, `LOW`, or a distinct fallback path

Deliverable:

- one short policy file with examples

Current source of truth:

- [`PRIORITY_POLICY.md`](PRIORITY_POLICY.md)

Success criterion:

- priorities correspond to downlink actions, not just semantic labels

### Step 1: Build the prefilter baseline

Implement and tune deterministic rules on SimSat imagery.

Start simple:

- cloud ratio
- brightness / darkness
- contrast / variance
- simple uniformity / texture checks

Measure:

- what fraction of frames get filtered before the VLM
- false reject rate on obviously useful frames

Deliverable:

- a baseline prefilter with logged decisions and thresholds

Success criterion:

- prefilter rejects a meaningful chunk of obvious junk without obviously destroying recall

### Step 2: Create a real-domain eval set first

Build a benchmark of roughly 30-100 Sentinel-2 / SimSat images with reviewed labels.

Each sample should include:

- image
- expected priority
- short justification
- notes if ambiguous

The set should include:

- cloudy frames
- dark or poor-quality frames
- urban scenes
- agriculture
- forest / vegetation
- water / maritime
- barren terrain
- hazard positives across the current scope when available:
  - wildfire
  - flood
  - landslide
  - oil spill under favorable conditions
- non-hazard counterexamples that might look "important" but should still be `MEDIUM`

Deliverable:

- a frozen `sentinel_eval_v1`

Success criterion:

- the team can manually inspect and agree the labels are defensible under the hazard policy

### Step 3: Reduce the schema for training

Train first on:

```json
{
  "description": "...",
  "priority": "..."
}
```

Do not require the model to learn `reasoning` and `categories` yet unless there is clear evidence it handles them without regressions.

Deliverable:

- simplified SFT format

Success criterion:

- training targets are shorter, clearer, and more task-aligned

### Step 4: Realign training and inference prompts

Keep one prompt source of truth.

Training and inference must share:

- same schema
- same wording for priority policy
- same examples if examples are used

Success criterion:

- byte-identical prompt assets, or a clearly versioned shared prompt module

### Step 5: Reposition VRSBench as auxiliary, not authoritative

Use VRSBench for:

- satellite visual familiarity
- description generation style
- broad scene understanding

Do **not** treat it as the main source of trustworthy priority labels.

Possible uses:

- caption-only warmup
- auxiliary description fine-tuning
- multi-stage training before a real-domain triage pass

Success criterion:

- the training narrative explicitly distinguishes "satellite understanding" from "downlink triage"

### Step 6: Add a small real-domain supervised set

Collect and label a small set of SimSat / Sentinel-2 images specifically for triage.

Label source options:

- careful manual labeling
- frontier model labeling with human review
- mixed manual + teacher labeling

This set should become the main source of supervision for `priority`.

If Sentinel-2 companion views help, the real-domain set may include:

- RGB view
- SWIR or NIR companion view of the same tile

Success criterion:

- the model is directly trained on examples from its deployment domain

### Step 7: Run small pipeline checks before any full training run

Before a large run:

- train on 10 examples
- then 50
- then 100

Check:

- valid JSON rate
- schema compliance
- whether outputs correspond to the actual input image
- whether priorities vary sensibly

Success criterion:

- no large run happens until small-run outputs are sane

### Step 8: Full run only after passing real-domain checks

The first full run should happen only once:

- prefilter exists
- real-domain eval exists
- prompt is aligned
- simplified schema works
- small-run checks are stable

This full run may combine:

- VRSBench auxiliary supervision
- small real-domain triage supervision
- class-aware weighting or careful augmentation if needed

Success criterion:

- improvement on `sentinel_eval_v1`, not just on VRSBench

### Step 9: Reintroduce reasoning only if it helps

Once `description + priority` works reliably, test whether adding:

- `reasoning`
- `categories`

improves the product enough to justify the added complexity.

If not, keep them outside the main model target.

Success criterion:

- no regression in real-domain priority quality

---

## Metrics for Exp 6

### Primary metrics

These are the metrics that actually matter:

1. **Real-domain valid JSON rate**
2. **Real-domain priority accuracy**
3. **Real-domain false reject rate**
   - useful image incorrectly filtered as `SKIP` / `LOW`
4. **Bandwidth reduction under the actual cascade**
   - not from class collapse, but from system behavior

### Secondary metrics

- VRSBench description quality
- prompt compliance
- inference latency
- category quality
- reasoning quality

### Anti-metrics

Metrics that should not drive decisions on their own:

- eval loss on VRSBench only
- class distribution aesthetics
- JSON validity on synthetic prompts alone

---

## Concrete Success Criteria

Exp 6 is successful if we achieve all of the following:

1. The pipeline can be described honestly as:
   **prefilter + VLM triage**
2. The prefilter removes a meaningful amount of obvious junk
3. The VLM works on a frozen real-domain Sentinel-2 eval set
4. Priority predictions are meaningfully better than trivial baselines
5. Reported bandwidth savings come from the full system, not from class collapse

---

## Strong Recommendations (High Confidence)

These are the things I am confident about:

1. **A real-domain eval set must come first, not last.**
2. **A deterministic prefilter is the right first stage.**
3. **`description + priority` is a better first schema than the full 4-field JSON.**
4. **Caption-derived priority labels are too weak to remain the primary supervision source.**
5. **The demo should frame the system as a hybrid pipeline, not a single all-powerful VLM.**
6. **RGB + companion SWIR/NIR views are worth testing for hazard discrimination.**

---

## Open Questions

These are important uncertainties, and the right answer is empirical, not ideological.

### Q1. How far can a 450M VLM generalize after prefiltering?

Unknown.

It may be enough once the easiest junk is removed and the output schema is simplified.
It may still be too small for robust transfer from VRSBench-like imagery to Sentinel-2.

If it fails, a 1.6B model may be the right next step before more data engineering.

### Q2. How much real-domain data is actually needed?

Unknown.

It might be that:

- 50 reviewed SimSat samples are enough to correct the priority head
- or it might take 200-500

This should be measured rather than assumed.

### Q3. Should VRSBench remain in the loop at all?

Probably yes, but maybe not in the way previous experiments used it.

It may work best as:

- auxiliary description pretraining
- warmup
- or a separate stage before a real-domain triage pass

But there is also a chance it adds more confusion than value once the domain mismatch becomes dominant.

### Q4. How aggressive should the prefilter be?

Unknown.

A conservative prefilter preserves recall but saves less compute.
A more aggressive prefilter saves more but risks discarding useful images.

This is a threshold-tuning problem, not a conceptual one.

### Q5. Should `reasoning` be generated by the same model pass?

Unclear.

For product polish, reasoning is useful.
For task performance, it may be unnecessary load.

It may be better produced:

- in a second pass
- heuristically from priority + scene type
- or only in the UI layer for demo purposes

### Q6. Is a lightweight learned gate worth adding?

Maybe.

If deterministic rules catch enough obvious junk, the extra complexity is not worth it.
If the rules prove too brittle, a tiny classifier may be justified.

### Q7. What is the right fallback on invalid model output?

Current behavior maps invalid JSON to low-value treatment in spirit, but this may be operationally unsafe.

Possible better options:

- retry once with a stricter prompt
- fall back to thumbnail transmission
- mark as `UNCERTAIN`
- log for operator review

This should be explicitly decided.

---

## Recommended Narrative for the Demo

The strongest honest story is:

> We built an onboard triage pipeline for satellite imagery.
> Obvious junk is rejected cheaply with deterministic filters.
> Ambiguous or meaningful frames are analyzed by a compact VLM.
> The result is a lightweight decision payload that helps the satellite spend bandwidth on what matters.

This is a stronger story than:

> One small model replaces all filtering and all triage.

Because it is:

- more realistic
- more defensible
- closer to how a production system would actually work

---

## Bottom Line

Exp 6 shifts the project from:

**"How do we make a VLM emit triage JSON?"**

to:

**"How do we build a believable onboard bandwidth-allocation system where the VLM is used where it adds the most value?"**

That is the version of the project most likely to both:

- work better technically
- and land better with judges

---

## Retrospective: v5 attempt (2026-04-22)

v5 tried to fix the hazard gap by mixing the existing VRSBench supervision with 17 hand-labeled hazard samples (10x oversampled to 50 copies). Training converged cleanly (eval loss 2.24 → 1.03), but on the frozen real-domain eval the v5 model scored **CRITICAL 0/3 and HIGH 0/2** — actually worse than the unchanged baseline on `HIGH`. The predicted distribution had zero `CRITICAL` and zero `HIGH` across all 45 samples.

Root cause was structural, not a tuning bug:

1. **Data imbalance.** 2638 VRSBench + 50 hazard copies = 1.9% hazard signal. The model learned to write VRSBench-style captions ("The image sourced from GoogleEarth features...") and default to `MEDIUM` / `LOW`. On the Attica wildfire it wrote "a small town and a large parking lot" — it never saw the burn scar.
2. **RGB-only input.** Active hazards (burn scars, flood water, stressed vegetation) are strongest in SWIR. Training on RGB-only asks the model to detect hazards from the weakest available band.
3. **LoRA on vision tower.** LoRA only adapts the language head. The vision encoder, which actually needs to learn that SWIR composites carry hazard signal, was effectively frozen.
4. **Oversampling vs. diversity.** 10x copies of 17 images teaches memorization of 17 scenes, not generalization across hazard types and conditions.
5. **Decision layer is defensive only.** The cascade can downgrade `MEDIUM → LOW` but cannot escalate `MEDIUM → HIGH`. If the VLM never emits `HIGH` or `CRITICAL`, nothing downstream can recover the hazard.

## Revised direction for the next training pass

Based on the v5 post-mortem, the next training pass follows a structurally different approach:

1. **Drop VRSBench entirely** for this model. Pure real-domain supervision.
2. **Capture pairs, not single images.** Every training sample is a co-registered `(RGB, SWIR)` pair for the same tile. SWIR is the primary hazard signal.
3. **Programmatic grid sampling.** Locations × timestamps × spatial offsets, chosen so the same regions contribute both hazard-present and non-hazard frames — giving real class balance instead of forced oversampling.
4. **Frontier-model teacher labels.** Each `(RGB, SWIR)` pair is labeled by a frontier model against the hazard priority policy. No hand-labeling at scale.
5. **Full fine-tune, not LoRA.** The vision tower needs to actually learn what SWIR composites look like. LoRA cannot retrain an encoder.
6. **Temporal train/test split.** Older 80% of timestamps go to train, newest 20% to eval. Prevents near-duplicates from Sentinel-2's ~5-day revisit from leaking across the split.
7. **Weighted hazard scope.** Wildfire and flood get the most samples (strongest SWIR signatures). Landslides and oil spills get smaller representation — the system still covers them in the demo but the model does its strongest work where the physics cooperates.

Target for the first pass: ~200 labeled `(RGB, SWIR)` pairs spread across all four hazards, with enough non-hazard frames from the same regions that the model learns the contrast — not just the positive class.

Success criterion stays the same as the rest of Exp 6: improvement on the frozen real-domain eval set, not training loss. Specifically, measurable `CRITICAL` and `HIGH` recall instead of zero.
