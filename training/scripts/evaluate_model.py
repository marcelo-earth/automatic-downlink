"""Evaluate a fine-tuned model on the eval sample set.

Usage:
    python training/scripts/evaluate_model.py --model ../models/exp2-merged --tag exp2
    python training/scripts/evaluate_model.py --model LiquidAI/LFM2.5-VL-450M --tag base
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

SYSTEM_PROMPT = """\
You are a satellite image triage system. Analyze the image and respond ONLY with a JSON object. No other text.

Priority: CRITICAL (disasters, fires, floods), HIGH (deforestation, unusual activity, anomalies), MEDIUM (routine urban, agriculture), LOW (featureless desert, barren terrain), SKIP (heavy clouds >80%, empty ocean, image artifacts).

If the image is mostly white/bright with no ground features visible, it is cloud-covered — mark SKIP.

Examples:
{"description": "Dense urban area with buildings and road network along a coastline", "priority": "MEDIUM", "reasoning": "Routine urban scene, no anomalies detected", "categories": ["urban", "infrastructure"]}
{"description": "Active wildfire with visible smoke plumes spreading over forested area", "priority": "CRITICAL", "reasoning": "Active fire threatening forested region, immediate alert needed", "categories": ["disaster", "fire", "vegetation"]}"""

USER_PROMPT = "Triage this satellite image. Respond with JSON only."


def load_model(model_path: str):
    print(f"Loading model: {model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    model.eval()
    print("Model loaded.")
    return model, processor


def run_inference(model, processor, image: Image.Image) -> tuple[str, float]:
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": USER_PROMPT},
        ]},
    ]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True,
        return_tensors="pt", return_dict=True, tokenize=True,
    )
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    elapsed = time.time() - t0
    response = processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response, elapsed


def parse_triage(text: str) -> dict | None:
    try:
        obj = json.loads(text)
        if "priority" in obj and "description" in obj:
            return obj
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        obj = json.loads(text[start:end])
        if "priority" in obj:
            return obj
    except (ValueError, json.JSONDecodeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval-data", default="training/data/eval_v2.jsonl")
    parser.add_argument("--sample-indices", default="training/data/eval_sample_indices.json")
    parser.add_argument("--output-dir", default="training/data/eval_results")
    args = parser.parse_args()

    with open(args.sample_indices) as f:
        indices = json.load(f)
    with open(args.eval_data) as f:
        all_lines = [json.loads(l) for l in f]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.tag}_results.jsonl"

    # Resume: skip already-processed indices
    done_indices = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                done_indices.add(r["idx"])
        print(f"Resuming: {len(done_indices)} already done, {len(indices) - len(done_indices)} remaining")

    remaining = [i for i in indices if i not in done_indices]
    if not remaining:
        print("All samples already evaluated.")
        indices = indices  # still need full set for summary
    else:
        model, processor = load_model(args.model)

    results = []
    # Load existing results for summary
    if done_indices:
        with open(results_path) as f:
            results = [json.loads(l) for l in f]

    for progress_i, idx in enumerate(remaining):
        item = all_lines[idx]
        gt_text = item["messages"][2]["content"][0]["text"]
        gt = json.loads(gt_text)

        img_path = None
        for msg in item["messages"]:
            for c in msg["content"]:
                if isinstance(c, dict) and c.get("type") == "image":
                    img_path = c["image"]

        image = Image.open(img_path).convert("RGB")
        response, elapsed = run_inference(model, processor, image)
        parsed = parse_triage(response)

        result = {
            "idx": idx,
            "gt_priority": gt["priority"],
            "pred_raw": response,
            "pred_parsed": parsed,
            "pred_priority": parsed["priority"] if parsed else None,
            "valid_json": parsed is not None,
            "time_s": round(elapsed, 2),
        }
        results.append(result)

        status = "OK" if parsed else "FAIL"
        match = "MATCH" if parsed and parsed["priority"] == gt["priority"] else "MISS"
        print(f"[{progress_i+1}/{len(indices)}] {status} {match} gt={gt['priority']} pred={result['pred_priority']} ({elapsed:.1f}s)")

        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    valid = sum(1 for r in results if r["valid_json"])
    matches = sum(1 for r in results if r["pred_priority"] == r["gt_priority"])
    total = len(results)
    avg_time = sum(r["time_s"] for r in results) / total

    print(f"\n{'='*50}")
    print(f"Model: {args.tag}")
    print(f"Valid JSON: {valid}/{total} ({100*valid/total:.1f}%)")
    print(f"Priority match: {matches}/{total} ({100*matches/total:.1f}%)")
    print(f"Avg inference time: {avg_time:.1f}s")

    from collections import Counter
    pred_dist = Counter(r["pred_priority"] for r in results if r["valid_json"])
    print(f"Predicted distribution: {dict(pred_dist.most_common())}")

    summary = {
        "tag": args.tag,
        "model": args.model,
        "total": total,
        "valid_json": valid,
        "valid_json_pct": round(100 * valid / total, 1),
        "priority_match": matches,
        "priority_match_pct": round(100 * matches / total, 1),
        "avg_time_s": round(avg_time, 1),
        "pred_distribution": dict(pred_dist.most_common()),
    }
    with open(out_dir / f"{args.tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
