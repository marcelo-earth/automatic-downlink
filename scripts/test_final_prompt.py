"""Test the updated triage prompt on all satellite images."""

import json
import logging
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.triage.model import TriageModel
from src.triage.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"


def main():
    model = TriageModel()
    model.load()

    images = sorted(TEST_IMAGES_DIR.glob("*.png"))
    if not images:
        print("No test images found")
        return

    results = []
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")

        image = Image.open(img_path).convert("RGB")
        response = model.generate(
            image=image,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_prompt=TRIAGE_USER_PROMPT,
        )
        print(f"Raw: {response}")

        # Parse
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start:end])
                print(f"Parsed: {json.dumps(parsed, indent=2)}")
                results.append((img_path.name, parsed))
                continue
            except json.JSONDecodeError:
                pass
        print("FAILED TO PARSE")
        results.append((img_path.name, None))

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, parsed in results:
        if parsed:
            print(f"{name:30s} | {parsed.get('priority', '?'):8s} | {parsed.get('categories', [])}")
        else:
            print(f"{name:30s} | PARSE FAILED")


if __name__ == "__main__":
    main()
