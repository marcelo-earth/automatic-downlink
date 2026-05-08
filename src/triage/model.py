"""LFM2.5-VL-450M inference wrapper with MPS (Apple Silicon) support."""

from __future__ import annotations

import logging
import os
import platform
from threading import Thread
from typing import TYPE_CHECKING, Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, TextIteratorStreamer

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

MODEL_ID = os.environ.get("MODEL_ID", "marcelo-earth/LFM2.5-VL-450M-satellite-triage-v6")
MODEL_REVISION = os.environ.get("MODEL_REVISION", None)  # use latest commit
BASE_MODEL_ID = "LiquidAI/LFM2.5-VL-450M"

# Recommended generation params from model card
GENERATION_KWARGS = {
    "max_new_tokens": 256,
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
        kwargs = dict(device_map=self.device if self.device == "cuda" else None, dtype=dtype)
        if MODEL_REVISION:
            kwargs["revision"] = MODEL_REVISION
        self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
        logger.info("Model loaded on %s (%s).", self.device, dtype)

    def generate(
        self,
        image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Run inference on a single image with system + user prompt."""
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
        return self._run_conversation(conversation, on_token=on_token)

    def generate_dual(
        self,
        rgb_image: Image.Image,
        swir_image: Image.Image,
        system_prompt: str,
        user_prompt: str,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Run inference on an RGB + SWIR image pair (dual-image model input)."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rgb_image},
                    {"type": "image", "image": swir_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        return self._run_conversation(conversation, on_token=on_token)

    def _run_conversation(
        self,
        conversation: list[dict],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Shared inference path for single- and dual-image conversations."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        ).to(self.model.device if self.device == "cuda" else self.device)

        if on_token is None:
            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **GENERATION_KWARGS)
            generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            return self.processor.decode(generated_ids, skip_special_tokens=True)

        # Streaming path: run generate() in a background thread and forward
        # tokens to on_token as they arrive.
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        gen_kwargs = {**inputs, **GENERATION_KWARGS, "streamer": streamer}

        def _run() -> None:
            with torch.inference_mode():
                self.model.generate(**gen_kwargs)

        thread = Thread(target=_run, daemon=True)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            try:
                on_token(accumulated)
            except Exception:
                logger.exception("on_token callback failed; continuing generation")
        thread.join()
        return accumulated

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None
