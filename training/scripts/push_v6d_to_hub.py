"""Push the v6d checkpoint from the Modal volume to HuggingFace Hub.

Run after v6d training completes. The run directory name encodes the timestamp
-update _RUN and _CKPT if the actual directory names differ.

Usage (from training/leap-finetune/):
    PATH="$(pwd)/.venv/bin:$PATH" .venv/bin/python ../../scripts/push_v6d_to_hub.py
"""

from __future__ import annotations

import modal

VOLUME_NAME = "satellite-vlm"
_RUN = "LFM2.5-VL-450M-vlm_sft-exp6d_train-all-lr2em05-w0p2-no_lora-20260506_021457"
_CKPT = f"{_RUN.split('/')[-1].replace('exp6d_train', 'exp6d_train')}-e5s25-20260506_021457"
HF_REPO_ID = "marcelo-earth/LFM2.5-VL-450M-satellite-triage-v6"

app = modal.App("push-v6d-to-hub")
image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub>=0.25")
vol = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    image=image,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def push() -> str:
    import os
    from huggingface_hub import HfApi, create_repo

    # Find the latest checkpoint in the run directory
    run_path = f"/vol/{_RUN}"
    if not os.path.isdir(run_path):
        # Try to find any exp6d run
        candidates = [d for d in os.listdir("/vol") if "exp6d" in d or "20260506" in d]
        if not candidates:
            raise FileNotFoundError(f"Run dir not found: {run_path}. Available: {os.listdir('/vol')}")
        run_path = f"/vol/{sorted(candidates)[-1]}"
        print(f"Using run: {run_path}")

    # Use 'latest' symlink if present, else pick the last epoch checkpoint
    latest = os.path.join(run_path, "latest")
    if os.path.isdir(latest):
        source = latest
    else:
        checkpoints = sorted([
            d for d in os.listdir(run_path)
            if os.path.isdir(os.path.join(run_path, d)) and d != "ray_logs"
        ])
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints in {run_path}")
        source = os.path.join(run_path, checkpoints[-1])

    print(f"Pushing from: {source}")

    # Only upload inference-time files - skip optimizer states and training artifacts
    INFERENCE_PATTERNS = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model-*.safetensors",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    )
    IGNORE_PATTERNS = [
        "*optim_states*",
        "*model_states*",
        "rng_state*",
        "*.pt",
        "trainer_state*",
        "training_args*",
    ]

    import glob
    files_to_upload = []
    for pattern in INFERENCE_PATTERNS:
        files_to_upload.extend(glob.glob(os.path.join(source, pattern)))
    files_to_upload = [f for f in files_to_upload if os.path.isfile(f)]

    if not files_to_upload:
        raise FileNotFoundError(f"No inference files found in {source}. Contents: {os.listdir(source)}")

    print(f"Uploading {len(files_to_upload)} inference files:")
    for f in files_to_upload:
        size_mb = os.path.getsize(f) / 1e6
        print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=False)
    api = HfApi()

    for filepath in files_to_upload:
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=os.path.basename(filepath),
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"v6d: upload {os.path.basename(filepath)}",
        )
        print(f"  ✓ {os.path.basename(filepath)}")

    return f"https://huggingface.co/{HF_REPO_ID}"


@app.local_entrypoint()
def main() -> None:
    url = push.remote()
    print(f"Pushed to: {url}")
