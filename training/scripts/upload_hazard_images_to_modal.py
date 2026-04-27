"""Upload hazard event images and v5 training data to the Modal volume."""

from __future__ import annotations

import glob
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUME_NAME = "satellite-vlm"
# Volume is mounted at /satellite-vlm/ in the container, so paths inside the
# volume are relative. Files at "data/foo" become "/satellite-vlm/data/foo"
# inside the container.
MODAL_DATA_DIR = "data"


def main() -> None:
    vol = modal.Volume.from_name(VOLUME_NAME)

    hazard_images = sorted(glob.glob(
        str(REPO_ROOT / "evals" / "candidates" / "images" / "*__rgb.png")
    ))
    print(f"Found {len(hazard_images)} RGB hazard images to upload")

    with vol.batch_upload() as batch:
        for img_path in hazard_images:
            rel = Path(img_path).relative_to(REPO_ROOT)
            remote_path = f"{MODAL_DATA_DIR}/{rel}"
            batch.put_file(img_path, remote_path)
            print(f"  Queued {rel}")

        data_files = [
            "training/data/train_v5_modal.jsonl",
            "training/data/eval_v5_modal.jsonl",
        ]
        for data_file in data_files:
            local_path = str(REPO_ROOT / data_file)
            remote_path = f"{MODAL_DATA_DIR}/{Path(data_file).name}"
            batch.put_file(local_path, remote_path)
            print(f"  Queued {data_file} -> {remote_path}")

    print("Done — all files uploaded to Modal volume.")


if __name__ == "__main__":
    main()
