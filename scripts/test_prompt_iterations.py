"""Iterate on triage prompts quickly — loads model once, tests multiple prompts."""

import json
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.triage.model import TriageModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"

# Attempt: much shorter, more direct, with a concrete example
PROMPT_V2_SYSTEM = """\
You are a satellite image triage system. Analyze the image and respond ONLY with a JSON object.

Example response:
{"description": "Dense urban area with buildings and roads visible", "priority": "MEDIUM", "reasoning": "Routine urban scene, no anomalies", "categories": ["urban"]}

Priority levels: CRITICAL (disasters, fires, floods), HIGH (deforestation, unusual activity), MEDIUM (routine urban/agriculture), LOW (featureless terrain), SKIP (clouds >80%, empty ocean).\
"""

PROMPT_V2_USER = "Triage this satellite image. Respond with JSON only."

# Attempt: even more minimal
PROMPT_V3_SYSTEM = """\
Satellite image triage system. Output JSON only, no other text.
Format: {"description": "...", "priority": "CRITICAL|HIGH|MEDIUM|LOW|SKIP", "reasoning": "...", "categories": ["..."]}"""

PROMPT_V3_USER = '{"description":'


def test_prompt(model, image, system_prompt, user_prompt, label):
    print(f"\n{'='*60}")
    print(f"Prompt: {label}")
    print(f"{'='*60}")
    response = model.generate(image=image, system_prompt=system_prompt, user_prompt=user_prompt)
    print(f"Response:\n{response}")

    # Check if it's valid JSON
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(cleaned[start:end])
            print(f"\nParsed JSON: {json.dumps(parsed, indent=2)}")
            return True
    except json.JSONDecodeError:
        pass
    print("\nFailed to parse as JSON")
    return False


def main():
    model = TriageModel()
    model.load()

    # Use Sahara and Lausanne as test images (clearest content)
    test_files = ["sentinel_sahara.png", "sentinel_test1.png", "sentinel_amazon.png"]
    images = {}
    for name in test_files:
        path = TEST_IMAGES_DIR / name
        if path.exists():
            images[name] = Image.open(path).convert("RGB")

    if not images:
        print("No test images found")
        return

    # Test one image per prompt variant to save time
    img_name = "sentinel_sahara.png"
    image = images[img_name]
    print(f"Testing with: {img_name}")

    test_prompt(model, image, PROMPT_V2_SYSTEM, PROMPT_V2_USER, "V2: Short + example")
    test_prompt(model, image, PROMPT_V3_SYSTEM, PROMPT_V3_USER, "V3: Minimal + completion")

    # If V2 or V3 works, test on all images with the winning prompt
    print(f"\n\n{'#'*60}")
    print("Testing winning prompt on all images...")
    print(f"{'#'*60}")

    for name, img in images.items():
        test_prompt(model, img, PROMPT_V2_SYSTEM, PROMPT_V2_USER, f"V2 on {name}")


if __name__ == "__main__":
    main()
