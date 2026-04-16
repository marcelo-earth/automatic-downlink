"""Quick test: run the base LFM2.5-VL-450M on satellite images to see baseline quality."""

import logging
import sys
from pathlib import Path

from PIL import Image

# Add project root to path
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
        print(f"No test images found in {TEST_IMAGES_DIR}")
        return

    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Image: {img_path.name}")
        print(f"{'='*60}")

        image = Image.open(img_path).convert("RGB")
        print(f"Size: {image.size}")

        response = model.generate(
            image=image,
            system_prompt=TRIAGE_SYSTEM_PROMPT,
            user_prompt=TRIAGE_USER_PROMPT,
        )

        print(f"Response:\n{response}")


if __name__ == "__main__":
    main()
