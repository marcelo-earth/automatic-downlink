# Current Cascade Evaluation

**Manifest:** evals/sentinel_eval_v1.jsonl  
**Model source:** marcelo-earth/LFM2.5-VL-450M-satellite-triage-v5  
**Processor source:** LiquidAI/LFM2.5-VL-450M  
**Generation preset:** deterministic `{'max_new_tokens': 192, 'temperature': 0.0, 'do_sample': False, 'repetition_penalty': 1.0}`  
**Prompt mode:** shipped  
**Seed:** 42  
**Decision layer:** enabled

## Summary

- Samples: 45
- Priority match: 35/45 (77.8%)
- Prefilter hits: 19/45 (42.2%)
- Avg latency: 1.45s
- Expected distribution: {'SKIP': 17, 'LOW': 8, 'MEDIUM': 15, 'CRITICAL': 3, 'HIGH': 2}
- Predicted distribution: {'SKIP': 20, 'LOW': 9, 'MEDIUM': 16}

## Per-sample results

| id | expected | predicted | match | prefilter | latency (s) |
|---|---|---|---|---|---|
| sentinel_amazon | SKIP | SKIP | yes | yes | 0.01 |
| sentinel_lima | SKIP | SKIP | yes | yes | 0.0 |
| sentinel_sahara | LOW | LOW | yes | no | 5.54 |
| sentinel_test1 | MEDIUM | MEDIUM | yes | no | 3.14 |
| amazon_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| borneo_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| cape_town_20260422T234447Z | LOW | MEDIUM | no | no | 2.06 |
| greenland_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| lausanne_20260422T234447Z | MEDIUM | MEDIUM | yes | no | 2.33 |
| lima_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| nile_delta_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| outback_20260422T234447Z | LOW | LOW | yes | no | 2.28 |
| sahara_20260422T234447Z | LOW | LOW | yes | yes | 0.0 |
| tokyo_20260422T234447Z | SKIP | SKIP | yes | yes | 0.0 |
| amazon_20260412T120000Z | LOW | LOW | yes | no | 2.22 |
| borneo_20260412T120000Z | SKIP | SKIP | yes | yes | 0.0 |
| cape_town_20260412T120000Z | LOW | SKIP | no | yes | 0.0 |
| greenland_20260412T120000Z | SKIP | SKIP | yes | yes | 0.0 |
| lausanne_20260412T120000Z | MEDIUM | MEDIUM | yes | no | 2.26 |
| lima_20260412T120000Z | SKIP | SKIP | yes | yes | 0.0 |
| nile_delta_20260412T120000Z | SKIP | SKIP | yes | yes | 0.0 |
| outback_20260412T120000Z | LOW | LOW | yes | no | 2.03 |
| sahara_20260412T120000Z | LOW | LOW | yes | no | 2.03 |
| tokyo_20260412T120000Z | MEDIUM | MEDIUM | yes | no | 2.08 |
| long_beach_port_20260412T120000Z | MEDIUM | MEDIUM | yes | no | 2.01 |
| chuquicamata_mine_20260412T120000Z | MEDIUM | LOW | no | no | 2.27 |
| jebel_ali_port_20260412T120000Z | MEDIUM | MEDIUM | yes | no | 2.57 |
| escondida_mine_20260412T120000Z | MEDIUM | LOW | no | no | 2.2 |
| attica_wildfire_rgb | CRITICAL | MEDIUM | no | no | 2.44 |
| derna_flood_rgb | CRITICAL | MEDIUM | no | no | 2.57 |
| valencia_flood_rgb | CRITICAL | SKIP | no | no | 2.66 |
| lahaina_wildfire_rgb | HIGH | MEDIUM | no | no | 2.97 |
| tenerife_wildfire_rgb | HIGH | LOW | no | no | 2.16 |
| kelowna_wildfire_rgb | MEDIUM | MEDIUM | yes | no | 2.17 |
| chile_wildfire_rgb | MEDIUM | MEDIUM | yes | no | 2.82 |
| pakistan_flood_sindh_rgb | MEDIUM | MEDIUM | yes | no | 2.13 |
| tobago_spill_rgb | MEDIUM | SKIP | no | no | 2.66 |
| turkey_hatay_rgb | MEDIUM | MEDIUM | yes | no | 2.41 |
| brazil_sao_sebastiao_rgb | MEDIUM | MEDIUM | yes | no | 2.88 |
| rio_grande_flood_rgb | MEDIUM | MEDIUM | yes | no | 2.53 |
| ventanilla_spill_rgb | SKIP | SKIP | yes | yes | 0.0 |
| niger_delta_spill_rgb | SKIP | SKIP | yes | yes | 0.0 |
| enga_landslide_rgb | SKIP | SKIP | yes | yes | 0.0 |
| thessaly_flood_rgb | SKIP | SKIP | yes | yes | 0.0 |
| india_joshimath_rgb | SKIP | SKIP | yes | yes | 0.0 |
