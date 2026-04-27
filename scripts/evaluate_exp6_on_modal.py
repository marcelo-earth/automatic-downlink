"""Evaluate the v6 fine-tuned checkpoint on the temporal hold-out set (Modal).

Loads the checkpoint from the satellite-vlm volume, runs dual-image (RGB + SWIR)
inference on exp6_eval.jsonl, and prints a per-class precision/recall report.

Usage:
    python3 scripts/evaluate_exp6_on_modal.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import modal

VOLUME_NAME = "satellite-vlm"
CHECKPOINT_DIR = (
    "/satellite-vlm/LFM2.5-VL-450M-vlm_sft-exp6_train-all-lr2em05-w0p2-no_lora-20260427_024215"
    "/checkpoint-14"
)
EVAL_JSONL = "/satellite-vlm/data/exp6_eval.jsonl"
IMAGE_ROOT = "/satellite-vlm"
BASE_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

SYSTEM_PROMPT = """\
You are an onboard satellite hazard triage system. You receive two images of the same scene:
1. RGB composite (natural color)
2. SWIR composite (swir16, nir08, red) — active fire appears bright red/orange, burn scars appear dark brown/black, floodwater appears dark blue, stressed vegetation appears orange/yellow, healthy vegetation appears bright green, urban areas appear magenta/pink

Analyze both images together and respond ONLY with a JSON object. No other text.

Hazard scope: wildfire, flood, oil spill, landslide.

Priority:
- CRITICAL: active hazard clearly visible (fire, flooding, large spill, fresh landslide)
- HIGH: visible hazard aftermath, probable hazard, or elevated hazard risk
- MEDIUM: informative or anomalous scene but no confirmed hazard
- LOW: routine low-value terrain, vegetation, or barren landscape
- SKIP: heavy clouds, no-data wedges, empty/obscured image, image artifacts\
"""
USER_PROMPT = "Triage this satellite image pair (RGB then SWIR). Respond with JSON only."

GENERATION_KWARGS = {
    "max_new_tokens": 256,
    "temperature": 0.1,
    "do_sample": True,
    "repetition_penalty": 1.05,
}

vol = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==5.2.0",
        "accelerate>=0.26.0",
        "pillow",
        "huggingface_hub",
    )
)

app = modal.App("automatic-downlink-eval-exp6")


@app.function(
    image=image,
    gpu="H100:1",
    volumes={"/satellite-vlm": vol},
    timeout=1800,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_eval() -> dict:
    import torch
    from PIL import Image as PILImage
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print(f"Loading processor from {BASE_MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    print(f"Loading model from checkpoint {CHECKPOINT_DIR}...")
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT_DIR,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    print("Model loaded.")

    samples = []
    with open(EVAL_JSONL) as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Eval samples: {len(samples)}")

    results = []
    for i, sample in enumerate(samples):
        msgs = sample["messages"]
        # Ground truth is in assistant message
        gt_text = msgs[2]["content"][0]["text"]
        gt = json.loads(gt_text)
        expected = gt["priority"]

        # Load images from user message
        user_content = msgs[1]["content"]
        rgb_path = Path(IMAGE_ROOT) / user_content[0]["image"]
        swir_path = Path(IMAGE_ROOT) / user_content[1]["image"]

        rgb_img = PILImage.open(rgb_path).convert("RGB")
        swir_img = PILImage.open(swir_path).convert("RGB")

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": rgb_img},
                {"type": "image", "image": swir_img},
                {"type": "text", "text": USER_PROMPT},
            ]},
        ]

        inputs = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **GENERATION_KWARGS)

        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        raw = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Parse priority from response
        predicted = "PARSE_ERROR"
        try:
            parsed = json.loads(raw)
            predicted = parsed.get("priority", "PARSE_ERROR")
        except json.JSONDecodeError:
            m = re.search(r'"priority"\s*:\s*"([A-Z]+)"', raw)
            if m:
                predicted = m.group(1)

        match = predicted == expected
        print(f"[{i+1}/{len(samples)}] {rgb_path.stem[:40]} | expected={expected} predicted={predicted} {'✓' if match else '✗'}")
        print(f"  raw: {raw[:120]}")

        results.append({
            "id": rgb_path.stem.replace("__rgb", ""),
            "expected": expected,
            "predicted": predicted,
            "match": match,
            "raw": raw,
        })

    # Per-class precision / recall
    classes = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "SKIP"]
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    for r in results:
        e, p = r["expected"], r["predicted"]
        if e == p:
            tp[e] += 1
        else:
            fn[e] += 1
            fp[p] += 1

    print("\n" + "=" * 60)
    print("EXP 6 EVAL RESULTS")
    print("=" * 60)
    total = len(results)
    correct = sum(r["match"] for r in results)
    print(f"Overall accuracy: {correct}/{total} ({100*correct/total:.0f}%)\n")
    print(f"{'Class':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'Prec':>6} {'Rec':>6}")
    print("-" * 40)
    for cls in classes:
        t = tp[cls]
        f_p = fp[cls]
        f_n = fn[cls]
        prec = t / (t + f_p) if (t + f_p) > 0 else 0.0
        rec = t / (t + f_n) if (t + f_n) > 0 else 0.0
        print(f"{cls:<10} {t:>4} {f_p:>4} {f_n:>4} {prec:>6.2f} {rec:>6.2f}")
    print("=" * 60)

    return {"results": results, "accuracy": correct / total}


@app.local_entrypoint()
def main() -> None:
    result = run_eval.remote()
    print(f"\nFinal accuracy: {result['accuracy']*100:.1f}%")
