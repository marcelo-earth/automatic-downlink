"""Upload Exp 6 training and eval data to the Modal volume."""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VOLUME_NAME = "satellite-vlm"

FILES = [
    "training/data/exp6_train.jsonl",
    "training/data/exp6_eval.jsonl",
]


def main() -> None:
    vol = modal.Volume.from_name(VOLUME_NAME)

    with vol.batch_upload() as batch:
        for local_rel in FILES:
            local_path = REPO_ROOT / local_rel
            if not local_path.exists():
                print(f"  MISSING: {local_rel} — run prepare_exp6_dataset.py first")
                continue
            remote_path = f"data/{local_path.name}"
            batch.put_file(str(local_path), remote_path)
            size_mb = local_path.stat().st_size / 1e6
            print(f"  Queued {local_rel} ({size_mb:.1f} MB) -> {remote_path}")

    print("Done — exp6 data uploaded to Modal volume.")


if __name__ == "__main__":
    main()
