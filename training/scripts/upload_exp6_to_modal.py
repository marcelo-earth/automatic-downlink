"""Upload Exp 6 training data and hazard grid images to the Modal volume."""

from __future__ import annotations

import glob
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUME_NAME = "satellite-vlm"

JSONL_FILES = [
    "training/data/exp6_train.jsonl",
    "training/data/exp6_eval.jsonl",
]
IMAGES_DIR = REPO_ROOT / "evals/candidates/hazard_grid_v1"
# Images land at evals/candidates/hazard_grid_v1/<name> inside the volume,
# matching the image_root="/satellite-vlm" + relative path in the JSONL.
IMAGES_REMOTE_DIR = "evals/candidates/hazard_grid_v1"


def main() -> None:
    vol = modal.Volume.from_name(VOLUME_NAME)

    with vol.batch_upload(force=True) as batch:
        # JSONL datasets
        for local_rel in JSONL_FILES:
            local_path = REPO_ROOT / local_rel
            if not local_path.exists():
                print(f"  MISSING: {local_rel} — run prepare_exp6_dataset.py first")
                continue
            remote_path = f"data/{local_path.name}"
            batch.put_file(str(local_path), remote_path)
            size_mb = local_path.stat().st_size / 1e6
            print(f"  Queued {local_rel} ({size_mb:.1f} MB) -> {remote_path}")

        # PNG images (RGB + SWIR)
        pngs = sorted(glob.glob(str(IMAGES_DIR / "*.png")))
        print(f"\n  Queuing {len(pngs)} PNG images...")
        for img_path in pngs:
            name = Path(img_path).name
            batch.put_file(img_path, f"{IMAGES_REMOTE_DIR}/{name}")

    print(f"Done — {len(pngs)} images + JSONL files uploaded to Modal volume.")


if __name__ == "__main__":
    main()
