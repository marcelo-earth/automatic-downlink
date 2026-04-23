# Priority Policy

This file defines the current product semantics for onboard **hazard triage**.

The system is not trying to rank "important-looking" infrastructure in the abstract.
It is trying to decide which scenes deserve downlink because they show a hazard, likely
hazard, or hazard aftermath.

## Priority Semantics

| Priority | Meaning | Default downlink action |
|---|---|---|
| `CRITICAL` | Active hazard clearly visible in the current image | Full image immediately |
| `HIGH` | Strong hazard evidence, visible aftermath, or elevated hazard risk that still deserves full review | Full image, ahead of routine scenes |
| `MEDIUM` | Informative or anomalous scene, but no confirmed hazard | Thumbnail + summary by default; full image only if bandwidth allows |
| `LOW` | Routine low-value scene with no visible hazard signal | Summary only or backlog thumbnail |
| `SKIP` | Unusable, obscured, mostly no-data, or otherwise not worth downlink | No downlink |

## Hazard-Oriented Examples

### `CRITICAL`

- Active wildfire with visible smoke plume
- Active flooding or clear inundation over land
- Large clear oil slick on water under favorable viewing conditions
- Fresh landslide with obvious debris and slope failure

### `HIGH`

- Burn scar or wildfire aftermath with clear hazard relevance
- Receding flood or saturated floodplain with visible residual impact
- Coastal contamination or ambiguous slick under favorable conditions
- Fresh landslide scar or unstable slope with strong visual evidence

### `MEDIUM`

- Dense port, mine, airport, or city with no hazard evidence
- Unusual-looking terrain or coastline without confirmed hazard
- Informative maritime or urban scene that may merit human review but does not show a hazard

### `LOW`

- Routine barren terrain
- Routine vegetation or agricultural scene
- Clear but unremarkable coastline, desert, or inland terrain

### `SKIP`

- Heavy cloud cover
- Washed-out or underexposed image
- Mostly no-data wedge
- Corrupted or nearly blank frame

## Important Non-Rule

`HIGH` does **not** mean generic strategic infrastructure.

A major port or mine is `MEDIUM` unless the image shows a hazard, visible damage,
contamination, or some other hazard-linked reason to escalate.

## Current Scope

The current hazard scope is:

- Wildfire
- Flood
- Oil spill, but only as "detectable under favorable conditions"
- Landslide

RGB is the current baseline view, but Sentinel-2 companion views such as SWIR and NIR
are considered in-scope for improving hazard discrimination.
