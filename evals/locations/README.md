# Targeted Capture Locations

This directory stores reusable location lists for targeted eval-set expansion.

- `high_value_locations.jsonl`
  Historical candidate locations that produced non-routine infrastructure or
  industrial scenes. Under the current hazard-triage policy, these are useful
  mostly as `MEDIUM` counterexamples, not as `HIGH` hazard targets.

Format:

```json
{"location_slug":"example_slug","location_name":"Example Name","lat":0.0,"lon":0.0}
```
