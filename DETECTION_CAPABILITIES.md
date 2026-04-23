# Detection Capabilities

## Data Source: Sentinel-2

SimSat serves real Sentinel-2 imagery from the AWS Element84 STAC archive. This gives us:

- any coordinate on the planet
- any date from ~2015 to present
- 10m/pixel resolution
- 5km tiles by default
- revisit every ~5 days at the equator
- available bands including RGB, NIR, SWIR, red edge, coastal, and more

The important point is not just that Sentinel-2 has RGB. It also gives us extra bands
that can be rendered into companion views. That means the current VLM can ingest:

- RGB alone
- RGB + SWIR
- RGB + another hazard-relevant companion view

without changing the model architecture.

## Priority Mapping

This project now uses a hazard-only interpretation of the top classes:

- `CRITICAL`: active hazard clearly visible in the current image
- `HIGH`: visible hazard aftermath, probable hazard, or elevated hazard risk
- `MEDIUM`: anomalous or informative scene, but no confirmed hazard
- `LOW`: routine low-value scene
- `SKIP`: unusable / obscured / mostly no-data frame

See [`PRIORITY_POLICY.md`](PRIORITY_POLICY.md) for the full policy.

## Hazard Scope

The current onboard triage scope targets four hazard families:

1. wildfire
2. flood
3. oil spill, but only under favorable conditions
4. landslide

## What We Detect

### 1. Wildfire

**Visual signature:** smoke plumes, active fire context, scorched terrain, and burn scars.

**Why it works:** wildfire evidence is often large-scale and high-contrast. Burn scars
persist after the active flames are gone, which is useful for `HIGH` hazard aftermath.

**Priority mapping:** active wildfire with visible smoke = `CRITICAL`. Burn scar,
clear aftermath, or nearby elevated wildfire risk = `HIGH`.

**Band note:** wildfire is a strong candidate for RGB + SWIR because burned terrain and
post-fire structure separate more clearly outside pure RGB.

### 2. Flood

**Visual signature:** water covering land, expanded river boundaries, submerged fields,
or floodwater where it normally does not belong.

**Why it works:** flood extent is often spatially large and visually distinct. The
aftermath can persist long enough for Sentinel-2 revisits.

**Priority mapping:** active inundation = `CRITICAL`. Receding flood, saturated
terrain, or strong residual flood evidence = `HIGH`.

**Band note:** flood is another strong candidate for RGB + SWIR or NIR companion views,
because water / land separation improves outside RGB.

### 3. Oil Spill

**Visual signature:** dark slick or contamination pattern on water or along the coast.

**Why it is weaker:** in single-pass RGB, a slick can be confused with shadow,
sunglint effects, sediment, dark water, or other natural surface patterns.

**Claim we can defend:** oil spill is **detectable under favorable conditions**, not
yet a strong universal capability claim.

**Priority mapping:** large clear slick under favorable conditions = `CRITICAL`.
Residual sheen or coastal contamination pattern under favorable conditions = `HIGH`.

**Band note:** extra bands may help somewhat, but this remains the most ambiguous of the
current hazard targets.

### 4. Landslide

**Visual signature:** exposed soil scar on vegetated slope, debris fan, disturbed slope
geometry, or fresh slope failure context.

**Why it works:** fresh landslides can create strong contrast against surrounding
vegetation and persist well beyond the triggering event.

**Priority mapping:** fresh landslide with clear debris / active failure evidence =
`CRITICAL`. Fresh scar or unstable slope with strong visual evidence = `HIGH`.

**Ambiguity note:** old scars and exposed earth can look similar, so temporal ambiguity
remains unless the visual evidence is strong.

## Why These Four

| Criterion | Wildfire | Flood | Oil Spill | Landslide |
|-----------|----------|-------|-----------|-----------|
| Detectable in RGB alone? | Often | Often | Sometimes | Often when fresh |
| Improved by extra bands? | Yes | Yes | Maybe | Sometimes |
| Persists long enough for revisit? | Yes | Yes | Often | Yes |
| Has public event examples? | Yes | Yes | Yes | Yes |
| Hazard domain | Land | Land | Maritime | Land |

Together they cover fire, water, sea, and earth while remaining narrow enough to defend
in a hackathon demo.

## Training Direction

For each hazard family:

1. identify known events with public coordinates and dates
2. capture tiles from SimSat / Sentinel-2
3. render RGB and, where useful, SWIR/NIR companion views
4. label with a frontier model plus human review
5. fine-tune the onboard VLM on the reviewed real-domain set

This keeps the labels tied to actual Sentinel-2 imagery instead of caption-only proxy
supervision.

## What We Do Not Detect

These are out of scope or weakly supported right now:

- **pre-eruptive volcanic activity** without thermal / SAR context
- **generic temporal change detection** such as deforestation or urban growth from a single image
- **sub-pixel phenomena** such as individual vehicles or small boats
- **atmospheric composition** such as gas leaks or chemical plumes
- **reliable oil-spill detection under all conditions**
