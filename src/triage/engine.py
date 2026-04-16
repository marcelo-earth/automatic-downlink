"""Triage engine: connects SimSat imagery to the VLM for prioritization."""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING

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


class TriageEngine:
    """Analyzes satellite images and produces triage decisions."""

    def __init__(self, model: TriageModel, profile: str = "default"):
        self.model = model
        self.profile = profile
        self.decisions: list[TriageDecision] = []

    @property
    def system_prompt(self) -> str:
        return PROMPT_PROFILES.get(self.profile, PROMPT_PROFILES["default"])

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

        raw_output = self.model.generate(
            image=image,
            system_prompt=self.system_prompt,
            user_prompt=TRIAGE_USER_PROMPT,
        )

        parsed = self._parse_model_output(raw_output)

        priority = Priority(parsed.get("priority", "MEDIUM"))
        decision = TriageDecision(
            image_id=image_id,
            timestamp=timestamp,
            position=position,
            description=parsed.get("description", "Unable to analyze image."),
            priority=priority,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
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
            "priority": "MEDIUM",
            "reasoning": "Defaulting to MEDIUM — model output was not valid JSON.",
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
