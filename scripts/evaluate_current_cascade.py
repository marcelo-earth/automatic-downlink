"""Evaluate the current triage cascade locally on a JSONL eval manifest.

Runs the shipped cascade as-is:
- deterministic prefilter from `TriageEngine`
- current configured VLM from `TriageModel`

No retraining, no prompt rewriting, no SimSat dependency.

Examples:
    .venv/bin/python scripts/evaluate_current_cascade.py
    .venv/bin/python scripts/evaluate_current_cascade.py --manifest evals/sentinel_eval_v1.jsonl
    .venv/bin/python scripts/evaluate_current_cascade.py --offline --output-dir evals/results
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import src.triage.model as triage_model_module
import src.triage.prompts as triage_prompts_module
from src.triage.engine import TriageEngine
from src.triage.schemas import Priority

DEFAULT_MANIFEST = REPO_ROOT / "evals" / "sentinel_eval_v1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "evals" / "results"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
SIMPLE_SYSTEM_PROMPT = """\
You are an onboard satellite hazard triage system. Analyze the image and respond ONLY with valid JSON.

Hazard scope: wildfire, flood, oil spill, landslide.

Return this schema and nothing else:
{"description": "<1 sentence>", "priority": "<CRITICAL|HIGH|MEDIUM|LOW|SKIP>"}

Priority guide:
- SKIP: heavy clouds, no-data wedges, empty/obscured image
- LOW: routine barren terrain, vegetation, or other low-information scene
- MEDIUM: informative or anomalous scene but no confirmed hazard
- HIGH: visible hazard aftermath, probable hazard, or elevated hazard risk
- CRITICAL: active hazard such as wildfire, flood, landslide, or a clear large spill
"""
SIMPLE_USER_PROMPT = "Triage this satellite image. Respond with JSON only using description and priority."
GENERATION_PRESETS = {
    "shipped": dict(triage_model_module.GENERATION_KWARGS),
    "deterministic": {
        "max_new_tokens": 192,
        "temperature": 0.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
    },
}


def _snapshot_dir(repo_id: str, revision: str | None = None) -> Path:
    namespace, name = repo_id.split("/", 1)
    repo_root = HF_CACHE_DIR / f"models--{namespace}--{name}"
    root = repo_root / "snapshots"
    if revision:
        direct_candidate = root / revision
        if direct_candidate.is_dir():
            return direct_candidate

        refs_dir = repo_root / "refs"
        ref_candidate = refs_dir / revision
        if ref_candidate.is_file():
            resolved = ref_candidate.read_text(encoding="utf-8").strip()
            candidate = root / resolved
            if candidate.is_dir():
                return candidate

        prefix_matches = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(revision)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No cached snapshot found for {repo_id} in {root}")
    return dirs[-1]


def resolve_model_source(model_id: str, revision: str | None, offline: bool) -> str:
    if not offline:
        return model_id
    return str(_snapshot_dir(model_id, revision))


def configure_generation(preset: str) -> dict[str, Any]:
    if preset not in GENERATION_PRESETS:
        raise ValueError(f"Unknown generation preset: {preset}")
    generation_kwargs = dict(GENERATION_PRESETS[preset])
    triage_model_module.GENERATION_KWARGS = generation_kwargs
    return generation_kwargs


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_prompts(prompt_mode: str) -> tuple[str, str]:
    if prompt_mode == "shipped":
        return triage_prompts_module.TRIAGE_SYSTEM_PROMPT, triage_prompts_module.TRIAGE_USER_PROMPT
    if prompt_mode == "simple":
        triage_prompts_module.TRIAGE_SYSTEM_PROMPT = SIMPLE_SYSTEM_PROMPT
        triage_prompts_module.TRIAGE_USER_PROMPT = SIMPLE_USER_PROMPT
        triage_prompts_module.PROMPT_PROFILES["default"] = SIMPLE_SYSTEM_PROMPT
        return SIMPLE_SYSTEM_PROMPT, SIMPLE_USER_PROMPT
    raise ValueError(f"Unknown prompt mode: {prompt_mode}")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if "image_path" not in row or "expected_priority" not in row:
                raise ValueError(f"{path}:{line_no} missing required fields")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in manifest: {path}")
    return rows


def evaluate_sample(engine: TriageEngine, sample: dict[str, Any]) -> dict[str, Any]:
    image_path = REPO_ROOT / sample["image_path"]
    image = Image.open(image_path).convert("RGB")

    prefilter_result = engine._prefilter(image)
    t0 = time.time()
    decision = engine.analyze(
        image=image,
        timestamp="2026-04-22T00:00:00Z",
        position={"lat": 0.0, "lon": 0.0, "alt": 0.0},
        source=str(sample.get("source", "local")),
        image_id=str(sample.get("id", image_path.stem)),
    )
    elapsed = round(time.time() - t0, 2)

    expected_priority = Priority(sample["expected_priority"])
    result = {
        "id": sample.get("id", image_path.stem),
        "image_path": sample["image_path"],
        "expected_priority": expected_priority.value,
        "predicted_priority": decision.priority.value,
        "priority_match": decision.priority == expected_priority,
        "prefilter_hit": prefilter_result is not None,
        "ambiguous": bool(sample.get("ambiguous", False)),
        "notes": sample.get("notes", ""),
        "elapsed_s": elapsed,
        "decision": decision.model_dump(mode="json"),
    }
    if prefilter_result is not None:
        result["prefilter_result"] = prefilter_result
    return result


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matches = sum(1 for r in results if r["priority_match"])
    prefilter_hits = sum(1 for r in results if r["prefilter_hit"])
    expected_dist = Counter(r["expected_priority"] for r in results)
    predicted_dist = Counter(r["predicted_priority"] for r in results)
    return {
        "total_samples": total,
        "priority_match_count": matches,
        "priority_match_pct": round(100 * matches / total, 1),
        "prefilter_hits": prefilter_hits,
        "prefilter_hit_pct": round(100 * prefilter_hits / total, 1),
        "avg_latency_s": round(sum(r["elapsed_s"] for r in results) / total, 2),
        "expected_distribution": dict(expected_dist),
        "predicted_distribution": dict(predicted_dist),
    }


def render_report(
    manifest_path: Path,
    model_source: str,
    processor_source: str,
    generation_preset: str,
    generation_kwargs: dict[str, Any],
    prompt_mode: str,
    seed: int,
    decision_layer_enabled: bool,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
) -> str:
    lines = [
        "# Current Cascade Evaluation",
        "",
        f"**Manifest:** {manifest_path.relative_to(REPO_ROOT)}  ",
        f"**Model source:** {model_source}  ",
        f"**Processor source:** {processor_source}  ",
        f"**Generation preset:** {generation_preset} `{generation_kwargs}`  ",
        f"**Prompt mode:** {prompt_mode}  ",
        f"**Seed:** {seed}  ",
        f"**Decision layer:** {'enabled' if decision_layer_enabled else 'disabled'}",
        "",
        "## Summary",
        "",
        f"- Samples: {summary['total_samples']}",
        f"- Priority match: {summary['priority_match_count']}/{summary['total_samples']} ({summary['priority_match_pct']}%)",
        f"- Prefilter hits: {summary['prefilter_hits']}/{summary['total_samples']} ({summary['prefilter_hit_pct']}%)",
        f"- Avg latency: {summary['avg_latency_s']}s",
        f"- Expected distribution: {summary['expected_distribution']}",
        f"- Predicted distribution: {summary['predicted_distribution']}",
        "",
        "## Per-sample results",
        "",
        "| id | expected | predicted | match | prefilter | latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    for row in results:
        lines.append(
            f"| {row['id']} | {row['expected_priority']} | {row['predicted_priority']} | "
            f"{'yes' if row['priority_match'] else 'no'} | "
            f"{'yes' if row['prefilter_hit'] else 'no'} | {row['elapsed_s']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current triage cascade locally.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"Path to eval manifest JSONL (default: {DEFAULT_MANIFEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write results (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--model-id",
        default=triage_model_module.MODEL_ID,
        help="Model repo id or local path. Defaults to the current configured model.",
    )
    parser.add_argument(
        "--model-revision",
        default=triage_model_module.MODEL_REVISION,
        help="Model revision to use when resolving a cached snapshot offline.",
    )
    parser.add_argument(
        "--processor-id",
        default=triage_model_module.BASE_MODEL_ID,
        help="Processor repo id or local path. Defaults to the current configured processor.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Resolve model and processor from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--generation-preset",
        choices=tuple(GENERATION_PRESETS),
        default="shipped",
        help="Generation config to use for evaluation.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("shipped", "simple"),
        default="shipped",
        help="Prompt schema to use for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--disable-decision-layer",
        action="store_true",
        help="Bypass the post-VLM decision layer and use raw model priorities.",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    samples = load_manifest(manifest_path)

    model_source = resolve_model_source(args.model_id, args.model_revision, args.offline)
    processor_source = resolve_model_source(args.processor_id, None, args.offline)
    configure_seed(args.seed)
    generation_kwargs = configure_generation(args.generation_preset)
    configure_prompts(args.prompt_mode)

    triage_model_module.BASE_MODEL_ID = processor_source
    model = triage_model_module.TriageModel(model_id=model_source)
    model.load()
    engine = TriageEngine(
        model=model,
        profile="default",
        use_decision_layer=not args.disable_decision_layer,
    )

    results = [evaluate_sample(engine, sample) for sample in samples]
    summary = build_summary(results)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest_path.stem
    report_path = output_dir / f"{stem}_report.md"
    results_path = output_dir / f"{stem}_results.json"
    summary_path = output_dir / f"{stem}_summary.json"

    report = render_report(
        manifest_path,
        model_source,
        processor_source,
        args.generation_preset,
        generation_kwargs,
        args.prompt_mode,
        args.seed,
        not args.disable_decision_layer,
        summary,
        results,
    )
    report_path.write_text(report, encoding="utf-8")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(report, end="")
    print(f"Wrote {report_path}")
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
