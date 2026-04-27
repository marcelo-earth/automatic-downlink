# Hazard labeling prompt for sub-agents

You are labeling Sentinel-2 satellite image pairs for an onboard hazard triage training set.

Each pair is:
- **RGB** — true-color composite (red, green, blue bands). What a human would see.
- **SWIR** — false-color composite (SWIR-16, NIR-08, red). Reveals:
  - **Active fire**: bright red/orange glow where RGB shows only smoke
  - **Burn scars**: dark brown or black where RGB might look green or mixed
  - **Stressed / dry vegetation**: orange/yellow tones
  - **Healthy vegetation**: bright green
  - **Water**: dark blue or black
  - **Cloud**: white (same as RGB)
  - **Urban / bare soil**: magenta/pink

Your job: for each pair, output a single JSON object with the priority and a short, honest description.

## Priority policy

Use these priorities exactly:

- **CRITICAL**: active hazard clearly visible (fire glow in SWIR, widespread floodwater, large clear oil slick, fresh landslide scar)
- **HIGH**: visible hazard aftermath, probable hazard, or elevated hazard risk (burn scars, residual flooding, fresh slope failures, stressed vegetation patterns, thin slick under favorable conditions)
- **MEDIUM**: informative or anomalous scene but no confirmed hazard. Urban scenes, agriculture, ports, mines, unusual terrain — all MEDIUM unless you can point to hazard evidence.
- **LOW**: routine low-value terrain, uniform vegetation, or barren landscape with no anomaly
- **SKIP**: heavy clouds obscuring the ground, no-data wedges (large black regions), empty/obscured image, image artifacts, nighttime or capture errors

## Important labeling rules

1. **Look at both images before deciding.** SWIR often reveals what RGB hides (fire under smoke, stressed vegetation, burn scars).
2. **Don't label based on the location slug or expected hazard type.** Many captures in a "wildfire" region will not show fire — the timestamp might be before the event, or the tile offset might miss the burn area. Label what you actually see.
3. **Be honest about SKIP.** If most of the frame is clouds or no-data, call it SKIP even if the location is known for hazards.
4. **Infrastructure is not HIGH.** A port or mine is MEDIUM unless the image shows a hazard reason to escalate.
5. **Description should mention what you see in both views.** Example: *"Dark burn scar across forested hillside visible in RGB; SWIR confirms the charred area as black/dark brown with healthy green vegetation unaffected to the north."*
6. **Keep descriptions one or two sentences. No more.**

## Output schema

For each pair, append one JSON line to the output file:

```json
{
  "candidate_group_id": "<from input>",
  "priority": "<CRITICAL|HIGH|MEDIUM|LOW|SKIP>",
  "description": "<1-2 sentences>",
  "hazard_visible": <true|false>,
  "image_quality_limited": <true|false>,
  "labeler_notes": "<optional short comment if ambiguous>"
}
```

- `hazard_visible` is `true` only if you're confident a hazard (active or aftermath) is present in either view.
- `image_quality_limited` is `true` when clouds, no-data, or artifacts materially hurt interpretation.
- `labeler_notes` is optional — use it for anything an operator would want to know.
