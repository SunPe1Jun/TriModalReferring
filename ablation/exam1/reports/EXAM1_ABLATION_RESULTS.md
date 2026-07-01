# Experiment 1 Modality Ablation Summary

Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`. Candidate anchors and evaluator are unchanged.

| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 0.7660 |  | 0.7714 | 0.3205 | 0.5326 |  |
| no_gaze | 2 | 0.5000 | -0.2660 | 0.5000 | 0.5000 | 0.5000 | -0.0326 |
| no_hand | 2 | 0.5000 | -0.2660 | 0.5000 | 0.5000 | 0.6667 | 0.1341 |
| no_visual | 4000 | 0.7302 | -0.0358 | 0.7354 | 0.3147 | 0.5425 | 0.0099 |

## Notes

- `no_gaze` hides structured gaze fields and gaze-derived sparse timeline proposals. If the source video contains a green gaze marker, the prompt tells the model to ignore it but the pixels are not edited.
- `no_visual` is the clean visual-removal control because the model receives only a blank placeholder image.
- `language_anchors_only` keeps the candidate anchor interface because removing anchors would change the closed-set task definition.
