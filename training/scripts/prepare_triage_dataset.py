"""Prepare VRSBench captioning data for satellite triage fine-tuning.

Downloads the VRSBench captioning split and converts it to the leap-finetune
VLM SFT format with our triage JSON output format.

Usage:
    # Local (small subset for testing)
    python training/scripts/prepare_triage_dataset.py --limit 500

    # Full dataset on Modal (recommended)
    python training/scripts/prepare_triage_dataset.py --modal
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Our triage system prompt (same as src/triage/prompts.py)
TRIAGE_SYSTEM_PROMPT = """\
You are a satellite image triage system. Analyze the image and respond ONLY with a JSON object. No other text.

Priority: CRITICAL (disasters, fires, floods), HIGH (deforestation, unusual activity, anomalies), MEDIUM (routine urban, agriculture), LOW (featureless desert, barren terrain), SKIP (heavy clouds >80%, empty ocean, image artifacts).

If the image is mostly white/bright with no ground features visible, it is cloud-covered — mark SKIP.

Examples:
{"description": "Dense urban area with buildings and road network along a coastline", "priority": "MEDIUM", "reasoning": "Routine urban scene, no anomalies detected", "categories": ["urban", "infrastructure"]}
{"description": "Image almost entirely covered by clouds, no ground features visible", "priority": "SKIP", "reasoning": "Cloud cover exceeds 80%, no usable data", "categories": ["cloud_cover"]}
{"description": "Arid desert terrain with sand dunes and dry riverbeds", "priority": "LOW", "reasoning": "Featureless barren landscape with no activity", "categories": ["terrain", "desert"]}
{"description": "Active wildfire with visible smoke plumes spreading over forested area", "priority": "CRITICAL", "reasoning": "Active fire threatening forested region, immediate alert needed", "categories": ["disaster", "fire", "vegetation"]}
{"description": "Fresh clearing in dense forest with exposed soil and new access road", "priority": "HIGH", "reasoning": "Possible deforestation activity with new road construction", "categories": ["deforestation", "vegetation", "environmental_change"]}\
"""

TRIAGE_USER_PROMPT = "Triage this satellite image. Respond with JSON only."

# Keyword-based heuristic for assigning priority to VRSBench captions
# (VRSBench doesn't have priority labels, so we infer them from content)
PRIORITY_KEYWORDS = {
    "CRITICAL": [
        "fire", "wildfire", "burning", "flood", "flooding", "tsunami",
        "earthquake", "collapsed", "explosion", "smoke plume", "lava",
        "volcanic", "hurricane", "cyclone", "tornado",
    ],
    "HIGH": [
        "deforestation", "clearing", "logging", "illegal", "oil spill",
        "pollution", "construction site", "excavation", "unusual",
        "military", "refugee", "displacement", "erosion",
    ],
    "MEDIUM": [
        "urban", "city", "building", "road", "highway", "bridge", "airport",
        "harbor", "port", "agricultural", "farm", "crop", "residential",
        "industrial", "commercial", "parking", "stadium", "school",
        "vehicle", "car", "truck", "ship", "boat", "train",
    ],
    "LOW": [
        "desert", "barren", "empty", "sand", "arid", "sparse",
        "grassland", "plain", "snow", "ice", "glacier",
        "mountain", "rocky", "wilderness",
    ],
    "SKIP": [
        "cloud", "cloudy", "overcast", "haze", "fog", "dark",
        "ocean", "sea", "water only", "no feature",
    ],
}

# Categories mapping from keywords
CATEGORY_KEYWORDS = {
    "urban": ["urban", "city", "building", "residential", "commercial"],
    "infrastructure": ["road", "highway", "bridge", "airport", "harbor", "port", "railway"],
    "vegetation": ["forest", "tree", "vegetation", "crop", "agricultural", "farm", "green"],
    "water": ["river", "lake", "ocean", "sea", "water", "coast", "shoreline"],
    "terrain": ["desert", "mountain", "hill", "sand", "rock", "barren"],
    "disaster": ["fire", "flood", "earthquake", "collapsed", "smoke", "damage"],
    "environmental_change": ["deforestation", "clearing", "erosion", "construction"],
    "cloud_cover": ["cloud", "overcast", "haze"],
    "vehicles": ["vehicle", "car", "truck", "ship", "boat", "airplane", "train"],
}


def assign_priority(caption: str) -> str:
    """Assign triage priority based on caption content."""
    caption_lower = caption.lower()
    for priority in ["CRITICAL", "HIGH", "SKIP", "LOW", "MEDIUM"]:
        for keyword in PRIORITY_KEYWORDS[priority]:
            if keyword in caption_lower:
                return priority
    return "MEDIUM"  # default


def assign_categories(caption: str) -> list[str]:
    """Assign categories based on caption content."""
    caption_lower = caption.lower()
    cats = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in caption_lower for kw in keywords):
            cats.append(category)
    return cats if cats else ["general"]


def assign_reasoning(caption: str, priority: str) -> str:
    """Generate a reasoning string for the triage decision."""
    reasons = {
        "CRITICAL": "Immediate threat detected — requires urgent ground station alert",
        "HIGH": "Significant activity or change detected — warrants priority downlink",
        "MEDIUM": "Routine scene with identifiable features — standard downlink",
        "LOW": "Low-information scene — thumbnail or summary sufficient",
        "SKIP": "No usable data in this capture — skip downlink",
    }
    return reasons.get(priority, "Standard triage assessment")


def caption_to_triage_json(caption: str) -> str:
    """Convert a VRSBench caption to our triage JSON format."""
    priority = assign_priority(caption)
    categories = assign_categories(caption)
    reasoning = assign_reasoning(caption, priority)

    triage = {
        "description": caption,
        "priority": priority,
        "reasoning": reasoning,
        "categories": categories,
    }
    return json.dumps(triage)


def convert_to_vlm_sft(caption: str, image_path: str) -> dict:
    """Convert a single VRSBench sample to leap-finetune VLM SFT format."""
    triage_json = caption_to_triage_json(caption)

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": TRIAGE_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": TRIAGE_USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": triage_json}],
            },
        ]
    }


def prepare_local(limit: int | None = None, output_dir: str = "training/data") -> None:
    """Prepare dataset locally using HuggingFace datasets streaming."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Loading VRSBench dataset via streaming...")
    ds = load_dataset("xiang709/VRSBench", split="train", streaming=True)

    images_dir = out / "images"
    images_dir.mkdir(exist_ok=True)

    train_path = out / "train.jsonl"
    eval_path = out / "eval.jsonl"

    samples = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break

        caption = row.get("caption", "")
        image_filename = row.get("image", "")  # filename like "00002_0000.png"

        if not caption:
            continue

        # VRSBench stores images as filenames — they're extracted from zip by HF.
        # The image data is available via the HF cache. We save locally.
        # For streaming, we use the filename as image_path (for Modal, images will be on volume).
        img_filename = f"vrsbench_{i:06d}.jpg"
        img_path = images_dir / img_filename

        # If dataset provides PIL image directly (some configs do), save it
        # Otherwise, just reference the filename for Modal processing
        if not img_path.exists():
            # Mark as placeholder — actual images will be fetched in Modal
            img_path_str = image_filename
        else:
            img_path_str = str(img_path.resolve())

        # Convert to VLM SFT format
        sample = convert_to_vlm_sft(caption, img_path_str)
        samples.append(sample)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d samples", i + 1)

    # Shuffle and split
    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    eval_samples = samples[split_idx:]

    # Write JSONL
    with open(train_path, "w") as f:
        for s in train_samples:
            f.write(json.dumps(s) + "\n")

    with open(eval_path, "w") as f:
        for s in eval_samples:
            f.write(json.dumps(s) + "\n")

    logger.info("Written %d train, %d eval samples", len(train_samples), len(eval_samples))
    logger.info("Images: %s", images_dir)
    logger.info("Train JSONL: %s", train_path)
    logger.info("Eval JSONL: %s", eval_path)

    # Print priority distribution
    from collections import Counter
    priorities = Counter()
    for s in samples:
        assistant_msg = s["messages"][2]["content"][0]["text"]
        parsed = json.loads(assistant_msg)
        priorities[parsed["priority"]] += 1
    logger.info("Priority distribution: %s", dict(priorities.most_common()))


def main():
    parser = argparse.ArgumentParser(description="Prepare VRSBench for triage fine-tuning")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--output", default="training/data", help="Output directory")
    parser.add_argument("--modal", action="store_true", help="Run on Modal (not yet implemented)")
    args = parser.parse_args()

    if args.modal:
        logger.error("Modal mode not yet implemented. Use --limit for local testing first.")
        sys.exit(1)

    prepare_local(limit=args.limit, output_dir=args.output)


if __name__ == "__main__":
    main()
