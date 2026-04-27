#!/usr/bin/env python3
"""Step 6: Quick MPS training test — 1 epoch on 2 samples.

Tests whether LFM2.5-VL-450M backward pass works on Apple Silicon MPS.
If this fails, we need Kaggle T4 for all training.
Expected runtime: ~2 minutes.
"""

import json
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.triage.prompts import TRIAGE_SYSTEM_PROMPT, TRIAGE_USER_PROMPT

BASE_MODEL = "LiquidAI/LFM2.5-VL-450M"
DEVICE = "mps"
DTYPE = torch.float32  # MPS doesn't support bfloat16 for backward


def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    # Load model
    print("Loading base model...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=DTYPE)
    model = model.to(DEVICE)
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Apply LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "in_proj", "linear_1", "linear_2"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.train()

    # Build 2 dummy samples from the training data
    print("\nPreparing 2 training samples...")
    with open("training/data/exp5_train.jsonl") as f:
        lines = [next(f) for _ in range(2)]

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for step, line in enumerate(lines):
        sample = json.loads(line)
        messages = sample["messages"]
        priority = sample.get("priority", "MEDIUM")

        # Replace image path with a dummy image (we don't have VRSBench images locally)
        from PIL import Image
        dummy_image = Image.new("RGB", (224, 224), color=(100, 150, 100))

        proc_messages = []
        for msg in messages:
            new_content = []
            for c in msg["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    new_content.append({"type": "image", "image": dummy_image})
                else:
                    new_content.append(c)
            proc_messages.append({"role": msg["role"], "content": new_content})

        inputs = processor.apply_chat_template(
            proc_messages,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )
        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Move to device
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        labels = labels.to(DEVICE)

        # Forward pass
        print(f"\nStep {step}: forward pass...", end=" ", flush=True)
        t1 = time.time()
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"{time.time() - t1:.1f}s")

        # Loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().unsqueeze(0)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Class weight
        weights = {"CRITICAL": 50.0, "HIGH": 5.0, "MEDIUM": 1.0, "LOW": 2.0, "SKIP": 3.0}
        weighted_loss = loss * weights.get(priority, 1.0)
        print(f"  Loss: {loss.item():.4f}, weighted ({priority}): {weighted_loss.item():.4f}")

        # Backward pass — THIS IS THE TEST
        print(f"  Backward pass...", end=" ", flush=True)
        t2 = time.time()
        optimizer.zero_grad()
        weighted_loss.backward()
        print(f"{time.time() - t2:.1f}s")

        # Optimizer step
        print(f"  Optimizer step...", end=" ", flush=True)
        t3 = time.time()
        optimizer.step()
        print(f"{time.time() - t3:.1f}s")

        print(f"  PASS — step {step} complete")

    print("\n" + "=" * 60)
    print("MPS TRAINING TEST: PASSED")
    print("Backward pass works on Apple Silicon for LFM2.5-VL-450M")
    print("=" * 60)


if __name__ == "__main__":
    main()
