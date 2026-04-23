# Targeted Capture Locations

This directory stores reusable location lists for targeted eval-set expansion.

- `high_value_locations.jsonl`
  Historical candidate locations that produced non-routine infrastructure or
  industrial scenes. Under the current hazard-triage policy, these are useful
  mostly as `MEDIUM` counterexamples, not as `HIGH` hazard targets.

- `hazard_high_seed_locations.jsonl`
  Seed event locations with timestamps for building the next reviewed `HIGH`
  hazard slice. These are capture targets, not trusted labels.

Format:

```json
{"location_slug":"example_slug","location_name":"Example Name","lat":0.0,"lon":0.0}
```

Extended fields are allowed for targeted capture:

```json
{
  "location_slug":"example_slug",
  "location_name":"Example Name",
  "lat":0.0,
  "lon":0.0,
  "timestamp":"2024-01-01T12:00:00Z",
  "hazard_type":"wildfire",
  "event_name":"Example Event"
}
```
