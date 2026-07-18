# Experiment 1 Modality Ablation Summary

Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`. Candidate anchors and evaluator are unchanged.

| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 0.7798 |  | 0.7798 | 0.3287 | 0.5399 |  |
| language_anchors_only | 4000 | 0.6425 | -0.1373 | 0.6425 | 0.2622 | 0.4976 | -0.0422 |
| no_gaze | 4000 | 0.7315 | -0.0483 | 0.7315 | 0.2915 | 0.4968 | -0.0430 |
| no_gaze_hand | 4000 | 0.7288 | -0.0510 | 0.7288 | 0.2863 | 0.4956 | -0.0443 |
| no_hand | 4000 | 0.7800 | 0.0002 | 0.7800 | 0.3250 | 0.5399 | -0.0000 |
| no_visual | 4000 | 0.7440 | -0.0358 | 0.7440 | 0.3215 | 0.5498 | 0.0100 |

## Notes

- `no_gaze` hides structured gaze fields and gaze-derived sparse timeline proposals. If the source video contains a green gaze marker, the prompt tells the model to ignore it but the pixels are not edited.
- `no_visual` is the clean visual-removal control because the model receives only a blank placeholder image.
- `language_anchors_only` keeps the candidate anchor interface because removing anchors would change the closed-set task definition.
