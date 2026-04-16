"""Output schemas for triage decisions."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    SKIP = "SKIP"


class DownlinkAction(str, Enum):
    TRANSMIT_IMAGE = "TRANSMIT_IMAGE"
    TRANSMIT_THUMBNAIL = "TRANSMIT_THUMBNAIL"
    TRANSMIT_SUMMARY_ONLY = "TRANSMIT_SUMMARY_ONLY"


PRIORITY_TO_ACTION: dict[Priority, DownlinkAction] = {
    Priority.CRITICAL: DownlinkAction.TRANSMIT_IMAGE,
    Priority.HIGH: DownlinkAction.TRANSMIT_IMAGE,
    Priority.MEDIUM: DownlinkAction.TRANSMIT_THUMBNAIL,
    Priority.LOW: DownlinkAction.TRANSMIT_SUMMARY_ONLY,
    Priority.SKIP: DownlinkAction.TRANSMIT_SUMMARY_ONLY,
}


class TriageDecision(BaseModel):
    image_id: str
    timestamp: str
    position: dict[str, float] = Field(description="lat, lon, alt of capture")
    description: str = Field(description="Natural language description of the image content")
    priority: Priority
    reasoning: str = Field(description="Why this priority was assigned")
    categories: list[str] = Field(default_factory=list, description="Content categories detected")
    downlink_action: DownlinkAction
    source: str = Field(description="sentinel or mapbox")


class BandwidthStats(BaseModel):
    total_images: int
    by_priority: dict[str, int]
    naive_bytes: int = Field(description="Total bytes if all images were downlinked")
    smart_bytes: int = Field(description="Bytes with triage filtering")
    savings_percent: float
    critical_count: int
    high_count: int
