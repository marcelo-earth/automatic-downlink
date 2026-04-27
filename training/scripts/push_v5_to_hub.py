"""Push the v5 merged model from the Modal volume to HuggingFace Hub."""

from __future__ import annotations

import modal

VOLUME_NAME = "satellite-vlm"
MERGED_MODEL_DIR = (
    "LFM2.5-VL-450M-vlm_sft-train_v5_m-all-lr0.0001-w0p2-lora_a-20260423_045700/"
    "LFM2.5-VL-450M-vlm_sft-train_v5_m-all-lr0.0001-w0p2-lora_m-20260423_045700"
)
HF_REPO_ID = "marcelo-earth/LFM2.5-VL-450M-satellite-triage-v5"

app = modal.App("push-v5-to-hub")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.25",
)
vol = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def push() -> str:
    from huggingface_hub import HfApi, create_repo

    local_path = f"/vol/{MERGED_MODEL_DIR}"
    print(f"Source: {local_path}")

    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=False)

    api = HfApi()
    api.upload_folder(
        folder_path=local_path,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="v5 hazard-focused fine-tune (3 epochs, LoRA rank 8)",
    )
    return f"https://huggingface.co/{HF_REPO_ID}"


@app.local_entrypoint()
def main() -> None:
    url = push.remote()
    print(f"Pushed to: {url}")
