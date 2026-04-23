"""Triage engine: connects SimSat imagery to the VLM for prioritization."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

import numpy as np

from src.triage.model import TriageModel
from src.triage.prompts import PROMPT_PROFILES, TRIAGE_USER_PROMPT
from src.triage.schemas import (
    BandwidthStats,
    DownlinkAction,
    Priority,
    TriageDecision,
    PRIORITY_TO_ACTION,
)

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Approximate image sizes for bandwidth calculation
IMAGE_SIZE_BYTES = 500 * 1024  # ~500KB per satellite image
THUMBNAIL_SIZE_BYTES = 50 * 1024  # ~50KB compressed thumbnail
SUMMARY_SIZE_BYTES = 1024  # ~1KB text summary
LOW_VALUE_KEYWORDS = {
    "arid",
    "barren",
    "canopy",
    "cloud",
    "cloudy",
    "desert",
    "dune",
    "dry riverbed",
    "erosion",
    "foliage",
    "forest",
    "geological",
    "geology",
    "rainforest",
    "ridge",
    "sand",
    "scrub",
    "terrain",
    "vegetation",
}
STRUCTURED_SCENE_KEYWORDS = {
    "airport",
    "building",
    "city",
    "coast",
    "coastal",
    "harbor",
    "highway",
    "industrial",
    "infrastructure",
    "neighborhood",
    "port",
    "rail",
    "residential",
    "road",
    "settlement",
    "shore",
    "shoreline",
    "urban",
}


class TriageEngine:
    """Analyzes satellite images and produces triage decisions."""

    def __init__(
        self,
        model: TriageModel,
        profile: str = "default",
        use_decision_layer: bool = True,
    ):
        self.model = model
        self.profile = profile
        self.use_decision_layer = use_decision_layer
        self.decisions: list[TriageDecision] = []

    @property
    def system_prompt(self) -> str:
        return PROMPT_PROFILES.get(self.profile, PROMPT_PROFILES["default"])

    def _image_signals(self, image: Image.Image) -> dict[str, float]:
        """Compute cheap image signals used by the prefilter and decision layer."""
        arr = np.array(image.resize((128, 128)).convert("RGB"), dtype=np.float32)
        mean_rgb = arr.mean(axis=(0, 1))
        green_frac = (
            ((arr[:, :, 1] > arr[:, :, 0] + 8) & (arr[:, :, 1] > arr[:, :, 2] + 8)).mean()
        )
        return {
            "brightness": float(mean_rgb.mean()),
            "std_rgb": float(arr.std()),
            "white_frac": float(((arr > 220).all(axis=2)).mean()),
            "near_white_frac": float(((arr > 245).all(axis=2)).mean()),
            "dark_frac": float(((arr < 30).all(axis=2)).mean()),
            "near_black_frac": float(((arr < 8).all(axis=2)).mean()),
            "low_sat_frac": float(((arr.max(axis=2) - arr.min(axis=2)) < 12).mean()),
            "green_frac": float(green_frac),
        }

    def _prefilter(self, image: Image.Image, signals: dict[str, float] | None = None) -> dict | None:
        """Fast pixel-level check for cloud cover, darkness, or featureless terrain.

        Returns a triage dict if the image can be classified without the VLM,
        or None if the VLM should handle it.
        """
        signals = signals or self._image_signals(image)
        brightness = signals["brightness"]
        std_rgb = signals["std_rgb"]
        white_frac = signals["white_frac"]
        near_white_frac = signals["near_white_frac"]
        dark_frac = signals["dark_frac"]
        low_sat_frac = signals["low_sat_frac"]

        # Heavy cloud cover: mostly white/bright pixels
        if white_frac > 0.6 or (brightness > 200 and std_rgb < 30):
            logger.info("Pre-filter: cloud cover (white=%.0f%%, bright=%.0f)", white_frac * 100, brightness)
            return {
                "description": "Image dominated by cloud cover — no ground features visible.",
                "priority": "SKIP",
                "reasoning": "Heavy cloud cover detected by pixel analysis — VLM skipped to save compute.",
                "categories": ["cloud_cover"],
            }

        # Mixed cloud fields can evade the pure-white rule while still hiding the scene.
        if brightness > 125 and white_frac > 0.2 and low_sat_frac > 0.3 and dark_frac < 0.1:
            logger.info(
                "Pre-filter: mixed cloud cover (white=%.0f%%, low_sat=%.0f%%, bright=%.0f)",
                white_frac * 100,
                low_sat_frac * 100,
                brightness,
            )
            return {
                "description": "Cloud and haze dominate the frame, leaving too little clear ground detail.",
                "priority": "SKIP",
                "reasoning": "Mixed bright cloud cover detected by pixel analysis — frame skipped before VLM inference.",
                "categories": ["cloud_cover"],
            }

        # Very dark / night image
        if dark_frac > 0.7 or brightness < 25:
            logger.info("Pre-filter: dark image (dark=%.0f%%, bright=%.0f)", dark_frac * 100, brightness)
            return {
                "description": "Image is too dark to extract useful information.",
                "priority": "SKIP",
                "reasoning": "Dark/underexposed image detected by pixel analysis — VLM skipped.",
                "categories": [],
            }

        # Very bright, low-detail barren scenes are typically overexposed desert or salt flats.
        if brightness > 210 and std_rgb < 45 and white_frac < 0.1 and near_white_frac < 0.05:
            logger.info(
                "Pre-filter: bright barren terrain (bright=%.0f, std=%.1f, white=%.0f%%)",
                brightness,
                std_rgb,
                white_frac * 100,
            )
            return {
                "description": "Bright barren terrain with limited actionable detail.",
                "priority": "LOW",
                "reasoning": "Overexposed low-information terrain detected by pixel analysis — minimal downlink value.",
                "categories": ["terrain"],
            }

        # Featureless terrain: low contrast, mid-brightness (desert, ocean, ice)
        if std_rgb < 18 and 40 < brightness < 190:
            if low_sat_frac > 0.75:
                logger.info("Pre-filter: featureless (std=%.1f, bright=%.0f)", std_rgb, brightness)
                return {
                    "description": "Featureless terrain with minimal visual variation.",
                    "priority": "LOW",
                    "reasoning": "Low-contrast featureless scene detected by pixel analysis — minimal information value.",
                    "categories": ["terrain"],
                }

        return None

    def _apply_decision_layer(
        self,
        parsed: dict,
        signals: dict[str, float],
    ) -> tuple[Priority, str | None]:
        """Conservative post-VLM correction layer.

        This layer only downgrades MEDIUM -> LOW for scenes that look routine
        and low-information according to both the generated description and
        cheap pixel statistics.
        """
        base_priority = Priority(parsed.get("priority", "MEDIUM"))
        if base_priority != Priority.MEDIUM:
            return base_priority, None

        description = str(parsed.get("description", "")).lower()
        reasoning = str(parsed.get("reasoning", "")).lower()
        categories = [str(c).lower() for c in parsed.get("categories", [])]
        text_blob = " ".join([description, reasoning, " ".join(categories)])

        if any(keyword in text_blob for keyword in STRUCTURED_SCENE_KEYWORDS):
            return base_priority, None

        brightness = signals["brightness"]
        std_rgb = signals["std_rgb"]
        white_frac = signals["white_frac"]
        green_frac = signals["green_frac"]
        low_sat_frac = signals["low_sat_frac"]

        terrain_like = any(keyword in text_blob for keyword in LOW_VALUE_KEYWORDS)
        bright_barren = (
            terrain_like
            and brightness > 145
            and std_rgb < 70
            and white_frac < 0.18
        )
        cloudy_vegetation = (
            terrain_like
            and green_frac > 0.18
            and white_frac > 0.05
            and low_sat_frac > 0.2
            and brightness < 120
        )

        if bright_barren:
            return (
                Priority.LOW,
                "Decision layer downgraded MEDIUM to LOW: barren terrain description plus bright low-information image signals.",
            )

        if cloudy_vegetation:
            return (
                Priority.LOW,
                "Decision layer downgraded MEDIUM to LOW: routine vegetation/cloud scene with no structured activity.",
            )

        return base_priority, None

    def analyze(
        self,
        image: Image.Image,
        timestamp: str,
        position: dict[str, float],
        source: str = "sentinel",
        image_id: str | None = None,
    ) -> TriageDecision:
        """Analyze a single satellite image and produce a triage decision.

        Args:
            image: PIL Image from SimSat.
            timestamp: ISO-8601 capture timestamp.
            position: dict with lat, lon, alt.
            source: "sentinel" or "mapbox".
            image_id: Optional identifier. Auto-generated if not provided.

        Returns:
            TriageDecision with priority, description, and downlink action.
        """
        image_id = image_id or f"IMG_{uuid.uuid4().hex[:8].upper()}"
        signals = self._image_signals(image)

        # Fast pixel check — skip VLM for obvious cloud/dark/empty scenes
        prefilter_result = self._prefilter(image, signals=signals)
        override_reason = None
        if prefilter_result is not None:
            parsed = prefilter_result
            base_priority = Priority(parsed.get("priority", "MEDIUM"))
            final_priority = base_priority
        else:
            raw_output = self.model.generate(
                image=image,
                system_prompt=self.system_prompt,
                user_prompt=TRIAGE_USER_PROMPT,
            )
            parsed = self._parse_model_output(raw_output)
            base_priority = Priority(parsed.get("priority", "MEDIUM"))
            if self.use_decision_layer:
                final_priority, override_reason = self._apply_decision_layer(parsed, signals)
            else:
                final_priority = base_priority

        priority = final_priority
        decision = TriageDecision(
            image_id=image_id,
            timestamp=timestamp,
            position=position,
            description=parsed.get("description", "Unable to analyze image."),
            priority=priority,
            base_priority=base_priority,
            final_priority=final_priority,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
            override_reason=override_reason,
            categories=parsed.get("categories", []),
            downlink_action=PRIORITY_TO_ACTION[priority],
            source=source,
        )

        self.decisions.append(decision)
        logger.info(
            "Triage: %s | %s | %s | %s",
            decision.image_id,
            decision.priority.value,
            decision.downlink_action.value,
            decision.description[:80],
        )
        return decision

    def _parse_model_output(self, raw: str) -> dict:
        """Parse the model's JSON output, handling common formatting issues."""
        cleaned = raw.strip()

        # Strip markdown code fences if present
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the output
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse model output as JSON: %s", cleaned[:200])
        return {
            "description": cleaned[:200] if cleaned else "Model output could not be parsed.",
            "priority": "LOW",
            "reasoning": "Model output was not valid JSON — treating as low-value to conserve bandwidth.",
            "categories": [],
        }

    def get_bandwidth_stats(self) -> BandwidthStats:
        """Calculate bandwidth savings from triage decisions so far."""
        if not self.decisions:
            return BandwidthStats(
                total_images=0,
                by_priority={p.value: 0 for p in Priority},
                naive_bytes=0,
                smart_bytes=0,
                savings_percent=0.0,
                critical_count=0,
                high_count=0,
            )

        by_priority: dict[str, int] = {p.value: 0 for p in Priority}
        for d in self.decisions:
            by_priority[d.priority.value] += 1

        total = len(self.decisions)
        naive_bytes = total * IMAGE_SIZE_BYTES

        smart_bytes = 0
        for d in self.decisions:
            if d.downlink_action == DownlinkAction.TRANSMIT_IMAGE:
                smart_bytes += IMAGE_SIZE_BYTES + SUMMARY_SIZE_BYTES
            elif d.downlink_action == DownlinkAction.TRANSMIT_THUMBNAIL:
                smart_bytes += THUMBNAIL_SIZE_BYTES + SUMMARY_SIZE_BYTES
            else:
                smart_bytes += SUMMARY_SIZE_BYTES

        savings = ((naive_bytes - smart_bytes) / naive_bytes * 100) if naive_bytes > 0 else 0.0

        return BandwidthStats(
            total_images=total,
            by_priority=by_priority,
            naive_bytes=naive_bytes,
            smart_bytes=smart_bytes,
            savings_percent=round(savings, 1),
            critical_count=by_priority.get("CRITICAL", 0),
            high_count=by_priority.get("HIGH", 0),
        )

    def reset(self) -> None:
        """Clear all stored decisions."""
        self.decisions.clear()
