"""LFM2.5-VL-450M inference wrapper with MPS (Apple Silicon) support."""

from __future__ import annotations

import logging
import os
import platform
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "marcelo-earth/LFM2.5-VL-450M-satellite-triage")

# Recommended generation params from model card
GENERATION_KWARGS = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "do_sample": True,
    "repetition_penalty": 1.05,
}


def _detect_device() -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and platform.processor() == "arm":
        return "mps"
    return "cpu"


def _dtype_for_device(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class TriageModel:
    """Wrapper for LFM2.5-VL-450M vision-language model."""

    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or _detect_device()
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load model and processor into memory."""
        logger.info("Loading model %s on %s...", self.model_id, self.device)

        dtype = _dtype_for_device(self.device)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            device_map=self.device if self.device == "cuda" else None,
            dtype=dtype,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Model loaded on %s (%s).", self.device, dtype)

    def generate(self, image: Image.Image, system_prompt: str, user_prompt: str) -> str:
        """Run inference on a single image with system + user prompt.

        Args:
            image: PIL Image to analyze.
            system_prompt: System instructions (triage rules).
            user_prompt: User query about the image.

        Returns:
            Model's text response.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device)

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **GENERATION_KWARGS)

        # Decode only the generated tokens (skip the input)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.processor.decode(generated_ids, skip_special_tokens=True)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None
