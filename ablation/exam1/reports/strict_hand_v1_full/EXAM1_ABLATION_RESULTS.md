# Experiment 1 Modality Ablation Summary

Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`. Candidate anchors and evaluator are unchanged.

| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 0.7798 |  | 0.7798 | 0.3287 | 0.5399 |  |
| no_hand_strict | 4000 | 0.7790 | -0.0008 | 0.7790 | 0.3267 | 0.5400 | 0.0001 |

## Notes

- `no_gaze` hides structured gaze fields and gaze-derived sparse timeline proposals. If the source video contains a green gaze marker, the prompt tells the model to ignore it but the pixels are not edited.
- `no_visual` is the clean visual-removal control because the model receives only a blank placeholder image.
- `language_anchors_only` keeps the candidate anchor interface because removing anchors would change the closed-set task definition.
