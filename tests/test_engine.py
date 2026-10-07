"""Tests for TriageEngine: parsing, prefilter, correction layers, bandwidth stats."""

from __future__ import annotations

from PIL import Image

from src.triage.engine import IMAGE_SIZE_BYTES, SUMMARY_SIZE_BYTES, TriageEngine
from src.triage.schemas import DownlinkAction, Priority
from tests.conftest import FakeModel, model_json, noisy_image

POSITION = {"lat": 0.0, "lon": 0.0, "alt": 500.0}


def analyze(engine: TriageEngine, image: Image.Image | None = None, **kwargs):
    return engine.analyze(
        image=image or noisy_image(),
        timestamp="2026-10-06T00:00:00Z",
        position=POSITION,
        **kwargs,
    )


# --- _parse_model_output ---


def test_parse_plain_json(engine: TriageEngine) -> None:
    parsed = engine._parse_model_output(model_json("HIGH", "burn scar"))
    assert parsed["priority"] == "HIGH"


def test_parse_strips_markdown_fences(engine: TriageEngine) -> None:
    raw = "```json\n" + model_json("CRITICAL") + "\n```"
    assert engine._parse_model_output(raw)["priority"] == "CRITICAL"


def test_parse_extracts_json_from_surrounding_text(engine: TriageEngine) -> None:
    raw = "Sure, here is the triage: " + model_json("MEDIUM") + " Hope it helps."
    assert engine._parse_model_output(raw)["priority"] == "MEDIUM"


def test_parse_garbage_falls_back_to_low(engine: TriageEngine) -> None:
    parsed = engine._parse_model_output("the model rambled with no json")
    assert parsed["priority"] == "LOW"
    assert parsed["description"].startswith("the model rambled")


def test_parse_empty_output(engine: TriageEngine) -> None:
    parsed = engine._parse_model_output("")
    assert parsed["priority"] == "LOW"
    assert parsed["description"] == "Model output could not be parsed."


# --- _prefilter ---


def test_prefilter_white_image_is_cloud_skip(engine: TriageEngine) -> None:
    result = engine._prefilter(Image.new("RGB", (64, 64), (250, 250, 250)))
    assert result is not None
    assert result["priority"] == "SKIP"
    assert result["categories"] == ["cloud_cover"]


def test_prefilter_black_image_is_dark_skip(engine: TriageEngine) -> None:
    result = engine._prefilter(Image.new("RGB", (64, 64), (5, 5, 5)))
    assert result is not None
    assert result["priority"] == "SKIP"
    assert "dark" in result["description"].lower()


def test_prefilter_flat_gray_is_featureless_low(engine: TriageEngine) -> None:
    result = engine._prefilter(Image.new("RGB", (64, 64), (120, 120, 120)))
    assert result is not None
    assert result["priority"] == "LOW"


def test_prefilter_textured_image_goes_to_vlm(engine: TriageEngine) -> None:
    assert engine._prefilter(noisy_image()) is None


def test_analyze_prefilter_skips_model_call(fake_model: FakeModel, engine: TriageEngine) -> None:
    decision = analyze(engine, Image.new("RGB", (64, 64), (250, 250, 250)))
    assert decision.priority == Priority.SKIP
    assert fake_model.calls == []


# --- _semantic_priority_floor ---


def test_semantic_floor_raises_low_with_active_fire(engine: TriageEngine) -> None:
    parsed = {"description": "Active fire front with smoke plume", "priority": "LOW"}
    priority, reason = engine._semantic_priority_floor(parsed, Priority.LOW)
    assert priority == Priority.CRITICAL
    assert reason is not None


def test_semantic_floor_raises_low_with_burn_scar(engine: TriageEngine) -> None:
    parsed = {"description": "Large burn scar east of the town", "priority": "LOW"}
    priority, _ = engine._semantic_priority_floor(parsed, Priority.LOW)
    assert priority == Priority.HIGH


def test_semantic_floor_ignores_negated_hotspot(engine: TriageEngine) -> None:
    # Regression: "no thermal hotspots" used to match "thermal hotspot".
    parsed = {"description": "Dense urban area, no thermal hotspots visible", "priority": "LOW"}
    priority, reason = engine._semantic_priority_floor(parsed, Priority.LOW)
    assert priority == Priority.LOW
    assert reason is None


def test_semantic_floor_never_touches_skip(engine: TriageEngine) -> None:
    parsed = {"description": "active fire", "priority": "SKIP"}
    priority, _ = engine._semantic_priority_floor(parsed, Priority.SKIP)
    assert priority == Priority.SKIP


def test_semantic_floor_ignores_no_data_frames(engine: TriageEngine) -> None:
    parsed = {"description": "Mostly no-data, possible active fire at edge", "priority": "LOW"}
    priority, _ = engine._semantic_priority_floor(parsed, Priority.LOW)
    assert priority == Priority.LOW


# --- _apply_decision_layer ---

BRIGHT_BARREN_SIGNALS = {
    "brightness": 160.0,
    "std_rgb": 40.0,
    "white_frac": 0.05,
    "green_frac": 0.0,
    "low_sat_frac": 0.1,
}


def test_decision_layer_downgrades_barren_medium(engine: TriageEngine) -> None:
    parsed = {"description": "Arid desert terrain with dunes", "priority": "MEDIUM"}
    priority, reason = engine._apply_decision_layer(parsed, BRIGHT_BARREN_SIGNALS)
    assert priority == Priority.LOW
    assert reason is not None


def test_decision_layer_keeps_structured_scene(engine: TriageEngine) -> None:
    parsed = {"description": "Desert terrain next to an airport", "priority": "MEDIUM"}
    priority, reason = engine._apply_decision_layer(parsed, BRIGHT_BARREN_SIGNALS)
    assert priority == Priority.MEDIUM
    assert reason is None


def test_decision_layer_only_touches_medium(engine: TriageEngine) -> None:
    parsed = {"description": "Arid desert terrain", "priority": "HIGH"}
    priority, _ = engine._apply_decision_layer(parsed, BRIGHT_BARREN_SIGNALS)
    assert priority == Priority.HIGH


# --- analyze end to end (fake model) ---


def test_analyze_semantic_floor_overrides_model_low() -> None:
    model = FakeModel(model_json("LOW", "Bright red/orange hotspot along the ridge"))
    decision = analyze(TriageEngine(model=model))
    assert decision.base_priority == Priority.LOW
    assert decision.priority == Priority.CRITICAL
    assert decision.downlink_action == DownlinkAction.TRANSMIT_IMAGE
    assert decision.override_reason is not None


def test_analyze_uses_dual_prompt_when_swir_given() -> None:
    model = FakeModel(model_json("MEDIUM", "Coastal city"))
    analyze(TriageEngine(model=model), swir_image=noisy_image(seed=1))
    assert model.calls == ["generate_dual"]


def test_analyze_single_image_uses_generate() -> None:
    model = FakeModel(model_json("MEDIUM", "Coastal city"))
    analyze(TriageEngine(model=model))
    assert model.calls == ["generate"]


def test_analyze_streams_partial_description() -> None:
    class StreamingModel(FakeModel):
        def generate(self, image, system_prompt, user_prompt, on_token=None) -> str:
            on_token('{"description": "Flooded fields')
            return model_json("HIGH", "Flooded fields")

    partials: list[str] = []
    analyze(TriageEngine(model=StreamingModel()), on_partial=partials.append)
    assert partials == ["Flooded fields"]


# --- bandwidth stats ---


def test_bandwidth_stats_empty(engine: TriageEngine) -> None:
    stats = engine.get_bandwidth_stats()
    assert stats.total_images == 0
    assert stats.savings_percent == 0.0


def test_bandwidth_stats_counts_and_bytes() -> None:
    model = FakeModel(model_json("CRITICAL", "Active flooding across the delta"))
    engine = TriageEngine(model=model)
    analyze(engine)
    analyze(engine, Image.new("RGB", (64, 64), (250, 250, 250)))  # prefilter SKIP

    stats = engine.get_bandwidth_stats()
    assert stats.total_images == 2
    assert stats.critical_count == 1
    assert stats.by_priority["SKIP"] == 1
    assert stats.naive_bytes == 2 * IMAGE_SIZE_BYTES
    assert stats.smart_bytes == IMAGE_SIZE_BYTES + 2 * SUMMARY_SIZE_BYTES


def test_reset_clears_decisions(engine: TriageEngine) -> None:
    analyze(engine, Image.new("RGB", (64, 64), (250, 250, 250)))
    engine.reset()
    assert engine.get_bandwidth_stats().total_images == 0
