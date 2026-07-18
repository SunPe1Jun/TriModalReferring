# Experiment 2 Modality Ablation Summary

Baseline is reused from `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`. Manifest construction and evaluator are unchanged unless a variant explicitly changes the number of panels.

| Variant | Events | Predictions | Time F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 | Mean Point Dist@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 8695 | 0.7348 | 0.2497 | 0.2067 |  | 57.8782 |
| full_panels_no_crop | 4000 | 8637 | 0.7003 | 0.2245 | 0.1767 | -0.0300 | 58.6555 |
| instruction_only_prompt | 4000 | 7582 | 0.6583 | 0.2294 | 0.1832 | -0.0235 | 57.6385 |
| no_gaze | 4000 | 8632 | 0.6880 | 0.2030 | 0.1554 | -0.0512 | 59.0650 |
| no_gaze_text_prior | 4000 | 8617 | 0.6890 | 0.2205 | 0.1730 | -0.0337 | 58.1662 |

## Notes

- `full_panels_no_crop` removes the gaze-centered crop path but keeps full visual panels.
- `no_gaze_text_prior` removes gaze-specific prompt wording and uses full panels, but it does not edit any visible gaze marker.
- `no_gaze` additionally masks the projected green gaze marker in copied panel images before inference.
- The current experiment-2 manifest has no explicit hand summary field, so hand contribution is not claimed from this workflow.
