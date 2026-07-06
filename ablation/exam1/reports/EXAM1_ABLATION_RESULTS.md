# Experiment 1 Modality Ablation Summary

Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`. Candidate anchors and evaluator are unchanged.

| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 0.7660 |  | 0.7714 | 0.3205 | 0.5326 |  |
| language_anchors_only | 4000 | 0.6280 | -0.1380 | 0.6324 | 0.2553 | 0.4896 | -0.0430 |
| no_gaze | 4000 | 0.7170 | -0.0490 | 0.7221 | 0.2840 | 0.4890 | -0.0436 |
| no_hand | 4000 | 0.7665 | 0.0005 | 0.7719 | 0.3180 | 0.5326 | 0.0000 |
| no_visual | 4000 | 0.7302 | -0.0358 | 0.7354 | 0.3147 | 0.5425 | 0.0099 |

## Notes

- `no_gaze` hides structured gaze fields and gaze-derived sparse timeline proposals. If the source video contains a green gaze marker, the prompt tells the model to ignore it but the pixels are not edited.
- `no_visual` is the clean visual-removal control because the model receives only a blank placeholder image.
- `language_anchors_only` keeps the candidate anchor interface because removing anchors would change the closed-set task definition.
