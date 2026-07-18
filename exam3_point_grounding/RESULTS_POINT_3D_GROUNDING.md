# Point-Supervised 3D Referent Grounding Results

Prediction CSV: `exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/predictions.csv`

## Overall

- samples: 4000
- valid output rate: 1.0000
- nearest-anchor set F1: 0.4326
- nearest-anchor exact rate: 0.0185
- primary Margin-F1@1.0: 0.2503
- Margin-F1@0.5: 0.1134
- Margin-F1@2.0: 0.4105
- mean scene-normalized error: 0.1569
- mean Euclidean error in world units: 7.6540
- invalid reason counts: {}

## Per Scene

| scene | samples | anchor-set F1 | Margin-F1@1.0 | mean scene-normalized error |
|---|---:|---:|---:|---:|
| scene1 | 800 | 0.4545 | 0.2448 | 0.1594 |
| scene2 | 800 | 0.2652 | 0.0474 | 0.2067 |
| scene3 | 800 | 0.4888 | 0.3337 | 0.1216 |
| scene4_room1 | 200 | 0.5473 | 0.3799 | 0.1493 |
| scene4_room2 | 200 | 0.3299 | 0.0976 | 0.3491 |
| scene4_room3 | 200 | 0.6471 | 0.3627 | 0.1359 |
| scene4_room4 | 200 | 0.4286 | 0.2603 | 0.1624 |
| scene5 | 800 | 0.4860 | 0.3521 | 0.1046 |

## Notes

Malformed model outputs are treated as empty prediction sets in all end-to-end metrics. Candidate anchors are used only by the evaluator after inference to map predicted points to nearest anchors.
