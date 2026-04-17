# Caption Classification Task

## Context

We are building **automatic-downlink** — a Vision Language Model (LFM2.5-VL-450M, 450M parameters) that runs on-board a satellite, analyzes every captured image, and decides what to downlink to ground. It's for the AI in Space Hackathon (Liquid AI x DPhi Space).

The model needs to produce triage decisions like this:

```json
{
  "description": "Residential area with buildings damaged after earthquake, debris on roads",
  "priority": "CRITICAL",
  "reasoning": "Earthquake damage with structural collapse — immediate alert for disaster response",
  "categories": ["disaster", "urban", "infrastructure"]
}
```

## The Problem

We have 20,264 satellite image captions from VRSBench (written by humans who looked at the actual images). But VRSBench only provides the **description** — it does NOT provide priority, reasoning, or categories.

In our first attempt, we assigned these labels using dumb keyword matching (`if "fire" in caption → CRITICAL`). This is bad because:
- "Buildings damaged by earthquake" gets MEDIUM because it contains "building"
- The reasoning is always a fixed template, not actual reasoning
- The model learns to replicate keyword rules instead of learning to think

## Your Task

Read each caption and assign **priority**, **reasoning**, and **categories** based on your understanding of what the image content means for a satellite downlink triage system.

### Input

File: `training/data/captions_to_classify.jsonl`

Each line is:
```json
{"id": 0, "image": "P0966_0001.png", "caption": "The high-resolution image from GoogleEarth depicts...", "source": "train"}
```

### Output

File: `training/data/classified_captions.jsonl`

Each line must be:
```json
{"id": 0, "priority": "MEDIUM", "reasoning": "Harbor infrastructure with no anomalies — routine monitoring", "categories": ["infrastructure", "water"]}
```

The `id` field must match the input so we can join them later.

### Priority Levels

Assign based on **how urgent it is to downlink this image to ground**:

- **CRITICAL**: Active disasters, fires, floods, explosions, volcanic activity, oil spills, structural collapse, anything requiring immediate ground response. These are rare.
- **HIGH**: Suspicious or unusual activity — deforestation in progress, unauthorized construction, unusual vessel patterns, refugee camps, environmental damage that isn't an active emergency. Worth prioritizing.
- **MEDIUM**: Routine scenes with identifiable content — cities, farms, ports, roads, normal shipping. Useful data but not urgent.
- **LOW**: Low-information scenes — empty desert, barren terrain, snow/ice with nothing notable, sparse grassland. Exists but not worth bandwidth.
- **SKIP**: No usable data — heavy cloud cover, fog, haze, empty ocean, image artifacts, overexposed/dark images. Don't waste bandwidth.

### Reasoning

Write 1 sentence explaining WHY this priority. Be specific to the content. Examples:

- Good: "Active smoke plumes over forested area suggest ongoing wildfire — requires immediate ground alert"
- Good: "Standard agricultural area with regular field patterns — routine monitoring, no anomalies"
- Bad: "Routine scene with identifiable features" (too generic, could apply to anything)
- Bad: "Immediate threat detected" (doesn't say what the threat is)

### Categories

Pick 1-4 from this list:

`urban`, `infrastructure`, `vegetation`, `water`, `terrain`, `disaster`, `environmental_change`, `cloud_cover`, `vehicles`, `agriculture`, `industrial`, `military`, `maritime`

### Important Notes

- Be realistic about the distribution. Most satellite images are mundane (MEDIUM/LOW). CRITICAL should be rare (<1%). Don't over-classify things as HIGH/CRITICAL.
- Cloud/haze/fog descriptions should be SKIP regardless of what might be underneath.
- "Sparse vegetation" or "barren terrain" = LOW, not MEDIUM.
- A harbor with ships = MEDIUM. A harbor with an oil spill = CRITICAL. Context matters.
- Descriptions mentioning "no distinguishable objects" or "no notable features" lean toward LOW/SKIP.

## How to Execute

This file has 20,264 captions. Process them in batches to avoid losing work.

### Recommended approach

Spawn a Sonnet agent (cheaper model, sufficient for this task) to process batches:

```
model: sonnet
```

1. Read 100 captions from `training/data/captions_to_classify.jsonl` (lines N to N+100)
2. Classify each one
3. Append results to `training/data/classified_captions.jsonl`
4. Repeat until done

Track progress in `training/data/classification_progress.txt` — just write the last processed ID so you can resume if interrupted.

### Verification

After completing all 20,264, verify:
- Line count of `classified_captions.jsonl` == 20,264
- Every ID from 0 to 20,263 is present
- Priority distribution is roughly: MEDIUM ~50-60%, LOW ~15-20%, SKIP ~10-15%, HIGH ~5-10%, CRITICAL <1%
- Spot check 10 random entries for quality

## After Classification

Once this task is done, the main project will:
1. Update `prepare_triage_dataset.py` to use these labels instead of keyword matching
2. Retrain the model on Modal with the new labels
3. Evaluate improvement

This is a knowledge distillation step — a large model (you) teaching a small model (450M params) how to reason about satellite image triage.
