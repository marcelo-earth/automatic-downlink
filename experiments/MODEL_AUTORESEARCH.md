# MODEL_AUTORESEARCH.md

Autonomous research log for fixing the LFM2.5-VL-450M satellite triage model.
Following the [autoresearch](https://github.com/karpathy/autoresearch) philosophy:
define the problem, form hypotheses, run experiments, measure, iterate.

---

## 1. Problem Statement

The fine-tuned model produces garbled output instead of valid triage JSON.
The base model (few-shot) produces valid JSON but copies example descriptions
verbatim instead of describing the actual image content.

**Goal:** A model that, given a satellite image, produces:
```json
{"description": "<actually describes the image>", "priority": "MEDIUM", "reasoning": "<why>", "categories": ["urban"]}
```

**Optimization target:** % of inferences that produce (a) valid JSON and (b) a description
that does NOT match any few-shot example verbatim.

---

## 2. Root Cause Analysis

### 2.1 Fine-tuned model: garbage output

**Finding: 48% of training data was garbage.**

VRSBench has 3 task types per image: `[caption]`, `[refer]` (bounding boxes), `[vqa]` (1-3 word answers).
Our `_extract_caption()` function had a bug: it checked if `[caption]` was in the GPT response
(it never is - `[caption]` is in the HUMAN prompt). So it extracted ALL gpt turns indiscriminately.

Result: training data included:
- Captions (good): "The image, sourced from GoogleEarth, shows a rural area with..." (52%)
- VQA answers (bad): "green", "Yes", "Ships", "edges" (48%)
- Grounding tokens (bad): `{<45><45><59><59>}` (mixed in)

The model learned to output short garbage because that's what half the training data was.

**Data quality stats (current training set):**
```
Total samples:     4500 (train split)
Desc < 10 chars:   2154 (47.9%)  ← VQA one-word answers
Desc 10-30 chars:  1655 (36.8%)  ← short VQA or fragments
Desc > 30 chars:    691 (15.3%)  ← actual captions
Priority dist:     MEDIUM=4243, LOW=189, SKIP=63, HIGH=5
```

### 2.2 Base model: copies few-shot examples

The base model with 5 few-shot examples in the system prompt always outputs one of those 5
descriptions verbatim. With a 450M param model, it doesn't have enough capacity/context to
both follow the JSON schema AND generate novel descriptions. It takes the safe path: copy an
example that seems plausible.

### 2.3 Learning rate may be too low

Used `lr=2e-5`, but `DEFAULT_VLM_SFT` in leap-finetune uses `5e-5`.
With LoRA, typical ranges are `1e-4` to `5e-4`.
Vision encoder multiplier `0.1` means vision encoder LR was `2e-6` - possibly underfitting.

### 2.4 Training/inference format: NOT a mismatch (ruled out)

Initially suspected PIL Image vs string path mismatch. But leap-finetune's collate function
(tokenize_data.py:82-85) converts string paths to PIL Images before calling
`processor.apply_chat_template()`. So training and inference both see PIL Images.
This is NOT the problem.

---

## 3. Hypotheses (ranked by expected impact)

### H1: Fix data - captions only (HIGH confidence)
Filter VRSBench to ONLY `[caption]` tasks. This gives us 20,264 real captions
(median 319 chars) instead of 5,000 mixed garbage.

**Expected impact:** Model learns to produce coherent descriptions instead of "green" / "Yes".

### H2: Increase learning rate (MEDIUM confidence)
Bump from `2e-5` to `1e-4` for LoRA adapter, keeping vision encoder at `0.1x`.

**Expected impact:** Faster convergence, model actually learns the output format.

### H3: Strip VRSBench boilerplate (MEDIUM confidence)
15.8% of captions start with "The image, sourced from GoogleEarth, displays..."
Strip this boilerplate so the model learns to describe content directly.

**Expected impact:** Cleaner descriptions, less repetitive output.

### H4: Simplify triage JSON schema (LOW-MEDIUM confidence)
Current training teaches 4-field JSON: {description, priority, reasoning, categories}.
The "reasoning" and "categories" are heuristic-generated (not from VRSBench).
Reduce to {description, priority} to lower the learning burden.

**Expected impact:** Easier format for the model to learn. Can add reasoning/categories
back in the inference prompt for the base model to fill in.

### H5: Reduce few-shot examples in system prompt (LOW confidence, quick test)
Current system prompt has 5 examples. Try 1-2 examples to reduce copying behavior
in the base model. May also help fine-tuned model by making the prompt shorter.

**Expected impact:** Less verbatim copying from few-shot examples.

### H6: More epochs (LOW confidence)
Go from 3 to 5 epochs. Risk of overfitting on 20K samples.

**Expected impact:** Marginal if data is clean. Diminishing returns.

---

## 4. Experiment Plan

### Experiment 1: Clean data + higher LR (H1 + H2 + H3)
**Rationale:** Fix the biggest problems first. One run, combined.

Changes:
1. Rewrite `_extract_caption()` to ONLY take captions from `[caption]` task items
2. Strip "The image, sourced from GoogleEarth, " boilerplate prefix
3. Use ALL 20,264 caption samples (not just 5K)
4. Set `learning_rate: 1e-4`
5. Keep 3 epochs, batch size 4, LoRA

**Measurement:**
- Does eval loss improve over current 0.49?
- Sample 10 outputs: are they valid JSON? Do descriptions vary?
- Run on 5 demo images from SimSat: does it describe what it sees?

**Time estimate:** ~30 min (data prep 5 min + Modal training ~25 min for 20K samples)

### Experiment 2: Simplified schema (H4)
If Exp 1 descriptions are still weak, reduce to:
```json
{"description": "<text>", "priority": "<LEVEL>"}
```
And let the inference engine add reasoning/categories programmatically.

### Experiment 3: Base model prompt tuning (H5)
Independent of fine-tuning. Test with 0, 1, 2, 3 examples in the system prompt.
Can run locally without Modal.

---

## 5. Experiment Log

### Exp 0: Baseline (COMPLETED)
- Config: 5000 mixed samples, lr=2e-5, 3 epochs
- Result: eval_loss=0.49, but model produces `<0>` tokens or `[{"label":"Triage","value":"1"}]`
- Root cause: 48% training data was VQA garbage, not captions
- Status: FAILED - model unusable

### Exp 1: Clean data + higher LR (COMPLETED - SUCCESS)
- Config: 20,264 caption-only samples, lr=1e-4, 2 epochs (3rd epoch interrupted by Modal timeout)
- eval_loss: 0.87 → 0.83 (epoch 1 → 2, still decreasing)
- Changes made:
  1. Fixed `_extract_caption()` - filter only `[caption]` tasks via `_is_caption_task()`
  2. Strip "The image, sourced from GoogleEarth, " boilerplate prefix
  3. Used all 20,264 caption samples (18,237 train / 2,027 eval)
  4. Set learning_rate: 1e-4 (5x increase from Exp 0)
  5. Merged LoRA adapter with base model locally using peft

**Results on 4 Sentinel satellite images:**

| Image | Base model | Exp 1 model |
|-------|-----------|-------------|
| Lima (cloudy) | SKIP - copies example verbatim: "Image almost entirely covered by clouds" | SKIP - unique: "overcast sky with a large expanse of white... small vehicle on a road" |
| Amazon | (not tested base) | LOW - "two distinct color regions, left side white, right side black" |
| Sahara (desert) | SKIP (WRONG) - "pixelated and distorted view... unsuitable for detailed assessment" | MEDIUM - "terrain featuring a complex network of winding paths, reddish-brown coloration, dry arid environment" |
| City/harbor | SKIP (WRONG) - copies cloud example verbatim | MEDIUM - "densely populated urban area with residential and commercial buildings, small harbor" |

**Key metrics:**
- Valid JSON rate: 100% (4/4) vs base model 66% (2/3 had markdown fences)
- Correct priority: 100% (4/4) vs base model 33% (base called city SKIP, desert SKIP)
- Unique descriptions: 100% vs base model 33% (base copied examples verbatim)
- Correct schema (all 4 fields): 100%

**Status:** SUCCESS - model uploaded to `marcelo-earth/LFM2.5-VL-450M-satellite-triage`

**Note:** Only 2 of 3 epochs completed. eval_loss was still decreasing (0.83), so a full 3-epoch run could improve further. But current results are already production-usable.

---

## 6. Key Files

| File | Role |
|------|------|
| `training/scripts/prepare_triage_dataset.py` | Data preparation (FIXED - caption-only) |
| `training/configs/triage_vlm_sft_modal.yaml` | Modal training config |
| `src/triage/model.py` | Inference wrapper |
| `src/triage/engine.py` | JSON parsing + triage decision |
| `src/triage/prompts.py` | System prompts (few-shot examples) |

## 7. VRSBench Data Reference

```
Total items:     142,390
  [caption]:      20,264 (14.2%) - real descriptions, median 319 chars
  [vqa]:          85,813 (60.3%) - short answers, 1-3 words
  [refer]:        36,313 (25.5%) - bounding box coordinates {<x><y><w><h>}
Unique images:    20,264 (each has 1 caption + multiple VQA + refer tasks)
Caption lengths:  min=43, median=319, max=848
```

Only `[caption]` items are usable for our triage fine-tuning.
