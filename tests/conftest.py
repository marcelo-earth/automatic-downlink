"""Shared fixtures: a fake model so engine tests never load real weights."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from src.triage.engine import TriageEngine


class FakeModel:
    """Stands in for TriageModel and returns a canned raw output."""

    def __init__(self, raw_output: str = "") -> None:
        self.raw_output = raw_output
        self.calls: list[str] = []

    def generate(self, image, system_prompt, user_prompt, on_token=None) -> str:
        self.calls.append("generate")
        return self.raw_output

    def generate_dual(self, rgb_image, swir_image, system_prompt, user_prompt, on_token=None) -> str:
        self.calls.append("generate_dual")
        return self.raw_output


def model_json(priority: str, description: str = "", reasoning: str = "", categories=None) -> str:
    return json.dumps(
        {
            "description": description,
            "priority": priority,
            "reasoning": reasoning,
            "categories": categories or [],
        }
    )


def noisy_image(seed: int = 0, size: int = 128) -> Image.Image:
    """Mid-brightness, high-variance image that passes every prefilter rule."""
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(20, 200, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def fake_model() -> FakeModel:
    return FakeModel()


@pytest.fixture
def engine(fake_model: FakeModel) -> TriageEngine:
    return TriageEngine(model=fake_model)
