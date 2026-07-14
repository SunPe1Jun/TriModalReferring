# Point-Supervised 3D Referent Grounding Results

Prediction CSV: `exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/predictions.csv`

## Overall

- samples: 3971
- valid output rate: 1.0000
- nearest-anchor set F1: 0.4257
- nearest-anchor exact rate: 0.0169
- primary Margin-F1@1.0: 0.2492
- Margin-F1@0.5: 0.1139
- Margin-F1@2.0: 0.4107
- mean scene-normalized error: 0.1590
- mean Euclidean error in world units: 7.7695
- invalid reason counts: {}

## Per Scene

| scene | samples | anchor-set F1 | Margin-F1@1.0 | mean scene-normalized error |
|---|---:|---:|---:|---:|
| scene1 | 800 | 0.4545 | 0.2448 | 0.1594 |
| scene2 | 791 | 0.2372 | 0.0467 | 0.2171 |
| scene3 | 800 | 0.4888 | 0.3337 | 0.1216 |
| scene4_room1 | 186 | 0.4937 | 0.3275 | 0.1586 |
| scene4_room2 | 200 | 0.3299 | 0.0976 | 0.3491 |
| scene4_room3 | 199 | 0.6479 | 0.3642 | 0.1360 |
| scene4_room4 | 199 | 0.4302 | 0.2615 | 0.1627 |
| scene5 | 796 | 0.4878 | 0.3532 | 0.1046 |

## Notes

Malformed model outputs are treated as empty prediction sets in all end-to-end metrics. Candidate anchors are used only by the evaluator after inference to map predicted points to nearest anchors.
