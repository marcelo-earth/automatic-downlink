# Experiment 5 Plan

## Goal

A fine-tuned LFM2.5-VL-450M that:
1. Produces valid JSON on Sentinel-2 images (not just VRSBench crops)
2. Assigns varied priorities (not 83% MEDIUM)
3. Describes what it sees without saying "GoogleEarth"
4. Uses the exact same prompt in training and inference

## Post-Mortem: What Went Wrong in Exp 0–4

### Exp 0 — Garbage data in, garbage out
- **Bug**: `_extract_caption()` grabbed ALL GPT responses (VQA one-word answers, bounding boxes), not just captions. 48% of training data was noise.
- **lr too low**: 2e-5 for LoRA; should be 1e-4 to 5e-4.
- **Lesson**: Always inspect raw training samples before launching a run.

### Exp 1 — Right format, wrong priorities
- Fixed the data bug → 20,264 clean captions.
- 2/3 epochs completed (Modal timeout).
- Produced valid JSON, correct schema. First time the system actually worked end-to-end.
- **Root problem**: Labels were assigned by keyword matching on the caption text, not by what the image shows. This produced 83% MEDIUM because most satellite captions mention "building" or "road."
- **Hidden problem**: Captions say "GoogleEarth" literally ("The image, sourced from GoogleEarth, shows...") — 61% of all captions. The model memorized this and repeats it at inference, even on Sentinel-2 imagery.
- **Hidden problem**: Training prompt had NO few-shot examples. Inference prompt has 5 examples. The model never saw the few-shot format during training.

### Exp 4 — Killed the model trying to fix it
- Tried to fix priority imbalance by downsampling MEDIUM to match LOW+SKIP count.
- Went from 20,264 samples → 2,940 samples.
- Kaggle T4 instead of H100.
- **Result**: Complete gibberish output. Model can't even form words.
- **Root cause**: 2,940 samples is catastrophically too few for a 450M parameter VLM. The model "forgot" how to generate text. This is catastrophic forgetting — the LoRA update was large enough relative to the tiny dataset that it overwrote the base model's text generation capability.
- **Lesson**: Never throw away data to balance. Oversample the minority class instead, or use class weights.

### Cross-cutting mistakes (all experiments)
1. **Never evaluated on the real domain.** We always tested on VRSBench images (same distribution as training). First Sentinel-2 test was in production. There was no Sentinel-2 eval set.
2. **Domain gap was known but never addressed.** VRSBench = high-res Google Earth crops, RGB, perfect nadir, no atmosphere. Sentinel-2 = lower resolution, atmospheric effects, clouds, multi-spectral. The model never saw anything resembling its actual input during training.
3. **Prompt mismatch between training and inference.** Training prompt is 3 lines of instructions. Inference prompt adds 5 JSON examples. The model learned to generate under one distribution and was asked to generate under another.
4. **Labels don't come from the image.** Both the keyword heuristic and the "sophisticated" classifier in `classify_captions.py` classify the TEXT of the caption, not the image content. A photo of a cloud-covered city gets labelled MEDIUM because the caption mentions "buildings." The priority should reflect what's visible, not what's described.
5. **Reasoning field is templated garbage.** Every MEDIUM sample gets "Routine scene with identifiable features — standard downlink" or similar. The model memorized these templates instead of learning to reason.

## Exp 5 Design: Incremental, Validated Steps

Philosophy: **test with 10 before training with 1,000 before deploying with 20,000.** Each step has an explicit validation gate before proceeding to the next.

### Step 1: Align the prompt (no training needed)

Make training prompt = inference prompt. One single source of truth.

**Action**: The system prompt used during training will be the exact `TRIAGE_SYSTEM_PROMPT` from `src/triage/prompts.py` (including the 5 few-shot examples). The user prompt will be the exact `TRIAGE_USER_PROMPT`. The Kaggle notebook and `prepare_triage_dataset.py` will import or copy this verbatim — no separate hardcoded strings.

**Validation**: Diff the training prompt vs inference prompt. They must be byte-identical.

### Step 2: Clean the captions (20,264 samples)

Remove all "GoogleEarth" boilerplate from captions before they become training targets.

Transforms:
- "The image, sourced from GoogleEarth, shows..." → "Shows..."
- "The high-resolution image from GoogleEarth depicts..." → "Depicts..."  
- "This high-resolution image from Google Earth shows..." → "Shows..."
- "sourced from GoogleEarth" → removed
- "from GoogleEarth" → removed
- "from Google Earth" → removed
- Capitalize first letter after stripping.

**Validation**: Grep the cleaned output for "GoogleEarth" and "Google Earth" — must be 0 hits. Spot-check 20 random samples to ensure descriptions still make sense.

### Step 3: Re-classify with better labels

The current labels are broken (92.7% MEDIUM from the Sonnet classifier, or 83% MEDIUM from keyword heuristic). The challenge: we're classifying TEXT, but priority should come from the IMAGE. Since we can't look at 20K images, we accept that text-based classification is a proxy — but we need a much better proxy.

**Approach**: Use Claude with a carefully designed prompt that:
- Considers what would be VISIBLE in the image, not just what's described
- Has explicit rules: "If the caption mentions clouds/haze covering the scene → SKIP regardless of what else is mentioned"
- Provides 10+ examples per priority level
- Processes in batches of 50 with distribution tracking to catch drift

**Target distribution** (based on what real satellite imagery looks like):
- CRITICAL: 0.5–2% (disasters are rare)
- HIGH: 5–10%
- MEDIUM: 40–55%
- LOW: 20–30%
- SKIP: 10–20%

If the resulting distribution is outside these ranges, the prompt needs tuning. This is NOT a hard constraint on the classifier — it's a sanity check that the classifier is working reasonably.

**Validation gate before training**:
- Distribution within target ranges
- Manual review of 10 random samples per priority level (50 total)
- Check that SKIP samples actually mention clouds/haze/darkness
- Check that CRITICAL/HIGH samples mention real anomalies, not routine things
- Check that LOW samples are genuinely featureless

### Step 4: Better reasoning (not templates)

Instead of "Routine scene with identifiable features", the reasoning should be specific to the caption. We can generate this as part of Step 3 — ask Claude to write a 1-sentence reasoning that explains why this specific image gets this specific priority.

**Validation**: No two reasoning strings should be identical. Grep for duplicates.

### Step 5: Prepare the training data

- Use ALL 20,264 samples (no downsampling)
- Oversample CRITICAL and HIGH to ~500 samples each (duplicate with slight prompt variations?)
- This gives roughly: ~500 CRITICAL, ~500 HIGH, ~10K MEDIUM, ~5K LOW, ~3K SKIP = ~19K total
- Apply cleaned captions from Step 2
- Apply new labels from Step 3
- Apply aligned prompt from Step 1
- 90/10 train/eval split, stratified by priority

**Validation**: Check distribution of train and eval sets match. Check 5 random samples from train set parse as valid JSON. Run the model's tokenizer on 10 samples to verify they encode correctly.

### Step 6: Train with 10 samples first

Before burning GPU hours, verify the pipeline:
- Take 10 samples (2 per priority class)
- Run 1 epoch of LoRA fine-tuning locally or on Kaggle
- Check: does it converge? Does the loss go down? Does the output look reasonable?
- This catches data format bugs, prompt encoding issues, and tokenization problems

**Validation**: Loss decreases. Model produces parseable JSON on train samples. No gibberish.

### Step 7: Train with 100 samples

Scale up to 100 samples (20 per class):
- 3 epochs, lr=1e-4, LoRA r=8
- Evaluate on a held-out 10 samples
- Compare output quality vs Step 6

**Validation**: Lower eval loss. Better priority distribution on eval. Still produces valid JSON.

### Step 8: Full training (20K samples)

Full training run on Kaggle T4 or Modal H100:
- All 20,264 samples (with oversampling of rare classes)
- 3 epochs, lr=1e-4, LoRA r=8, alpha=16
- Same hyperparams as Exp 1 (which worked), NOT Exp 4

**Validation**: Eval loss < 0.8. Valid JSON on 100% of eval samples. Priority distribution across eval predictions roughly matches the target.

### Step 9: Evaluate on Sentinel-2 (the real test)

This is the test that we never ran before:
- Capture 20+ images from SimSat (Sentinel-2 endpoint) across diverse locations
- Run inference on each
- Check: valid JSON? Reasonable descriptions? Correct priorities?
- This directly measures the domain gap

**Validation**: >80% valid JSON on Sentinel-2 images. Descriptions should not say "GoogleEarth." Priority should be reasonable (cloud images → SKIP, city → MEDIUM, etc.)

### Step 10: (If Step 9 fails) Bridge the domain gap

If the model still produces garbage on Sentinel-2 after Steps 1-8:
- Capture 50-100 Sentinel-2 images via SimSat
- Label them with Claude Vision (which can describe Sentinel-2 correctly)
- Add to training set as a small fine-tuning set
- Re-train (probably just 1 more epoch on the Sentinel-2 data)

This is a contingency plan — we may not need it if the caption cleaning and prompt alignment fix enough of the gap.

## Open Questions (things we don't know yet)

1. **Will cleaning "GoogleEarth" from captions break anything?** The model learned image→text associations that include "GoogleEarth." Removing it changes the target distribution. The model might struggle to generate descriptions without the anchor phrase. We'll know after Step 6.

   **Answer: Almost certainly not.** "GoogleEarth" is a source attribution, not a visual feature. The model didn't learn to associate pixels with that phrase — it learned that the phrase appears in most targets, so it repeats it. Removing it just removes a crutch. Confidence: very high.

2. **Is 450M params enough for domain transfer?** The model might fundamentally not generalize from Google Earth crops to Sentinel-2 imagery. Larger models (7B+) handle domain gaps better. The pre-filter helps, but the VLM needs to work on the images that pass the filter. We'll know after Step 9.

   **Answer: Unknown, but there are options.** 450M is tight for generalization across domains this different (high-res RGB crops vs. lower-res atmospheric Sentinel-2). But the constrained output space (structured JSON, finite priority levels) helps — we're not asking for open-ended reasoning. The few-shot examples in the prompt also act as a strong prior. Note: 450M is not the only option. LFM2.5-VL also comes in **1.6B** (and the older LFM2-VL goes up to 3B). All are valid for the Liquid Track. If Step 9 fails with 450M, try 1.6B before investing in Step 10 bridge data — the extra parameters might solve the domain gap for free while still being realistic for NVIDIA Orin 16GB. Confidence: low — depends on model-specific behavior we can only test empirically.

3. **Does oversampling CRITICAL/HIGH help or hurt?** We only have 4 real CRITICAL captions and ~142 HIGH. Oversampling to 500 means repeating each CRITICAL sample ~125x. The model might memorize them instead of learning the pattern. Alternative: augment by paraphrasing the captions slightly for each duplicate.

   **Answer: Naive oversampling (copy-paste) will cause memorization.** 4 samples repeated 125x is not a training signal, it's rote memorization. Two better approaches:
   - **Paraphrase augmentation**: Have Claude rephrase each caption 5-10 ways. Gets to ~40-50 unique-ish samples, then oversample those to ~500. Much better signal diversity.
   - **Sample-level loss weighting**: Multiply the loss for each training sample by a scalar based on its priority class (e.g., 50x for CRITICAL, 10x for HIGH, 1x for MEDIUM). **Important correction**: `F.cross_entropy(weight=...)` operates at the **token vocabulary level** (weighting 50K+ token IDs), NOT at the sample level. Weighting the token "C" differently from "M" is meaningless for this problem. The correct approach is to compute the normal loss per sample in `VLMTrainer.compute_loss`, then multiply by the class-specific scalar before averaging. This is a different code change than just passing `weight=` — must implement correctly in Step 6.
   
   Confidence: very high — this is well-established ML practice. The token-vs-sample weight distinction is a subtle but critical implementation detail.

4. **Will the few-shot examples in the prompt confuse the model?** The 5 examples in the system prompt show the exact JSON schema. If the training data also has the prompt + examples, the model sees examples-then-response. This could help (model learns the format) or hurt (model copies the examples instead of generating). We'll know after Step 6.

   **Answer: They'll most likely help.** This effectively teaches the model in-context format compliance during training, which is exactly what we want. The risk of copying examples verbatim is real but low — the model sees the same examples paired with a *different* image each time, so it learns "produce something like this format" not "repeat this exact text." Step 6 (10 samples) will catch it immediately if it happens. Confidence: reasonably high, though untested specifically with LFM2.5-VL architecture.

5. **Should we use the pre-filter to exclude cloud/dark images from training?** If we already catch these with pixel analysis, we might not need to teach the VLM about them. Training only on images that pass the pre-filter would let the VLM focus on the harder cases. Counter-argument: the VLM should be the fallback if the pre-filter misses something.

   **Answer: Keep them in.** The pre-filter uses pixel heuristics, which will have false negatives (thin clouds, twilight, partially obscured scenes). The VLM should know what clouds look like so it can SKIP them when the pre-filter lets one through. It's defense in depth — cheap insurance with near-zero cost to the other classes. Confidence: very high.

6. **What's the right eval metric?** Val_loss on VRSBench told us nothing about Sentinel-2 performance. We need a metric on the actual deployment domain. Options: JSON parse rate + priority accuracy on our Sentinel-2 eval set. But "priority accuracy" requires human-labeled ground truth for Sentinel-2 images.

   **Answer: Two-tier evaluation.**
   - **Automated (run on every checkpoint):** JSON parse rate + schema compliance. This is the "did we break something" metric.
   - **Manual (run once after Step 8, once after Step 9):** Priority accuracy on ~30 hand-labeled Sentinel-2 images. We independently assign priorities, then compare model output. The set should cover diverse scenes (5 clouds, 5 cities, 5 vegetation, etc.).
   
   Don't overthink it — for a hackathon, "judges look at the dashboard and the priorities make sense" is the real test. The 30-image Sentinel-2 set is mostly to give us confidence before the demo. Confidence: reasonably high on the structure, the specific number (30) is a rough estimate.

## What To Do With The Docker Problem

Independent of Exp 5, the Docker demo needs to work for judges:
- Pre-seed the dashboard with 5-10 example decisions (with images) so it's not empty on startup
- Add a loading indicator when VLM inference is in progress
- Auto-start the SimSat simulation (POST start command from the triage loop)
- Ensure the Dockerfile downloads the correct model revision

These are separate tasks from the training pipeline and can be done in parallel.
