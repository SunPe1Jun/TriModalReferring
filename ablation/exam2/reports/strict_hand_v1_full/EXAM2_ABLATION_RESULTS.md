# Experiment 2 Modality Ablation Summary

Baseline is reused from `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`. Manifest construction and evaluator are unchanged unless a variant explicitly changes the number of panels.

| Variant | Events | Predictions | Time F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 | Mean Point Dist@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 8695 | 0.7348 | 0.2497 | 0.2067 |  | 57.8782 |
| no_hand_strict | 4000 | 8833 | 0.6935 | 0.1350 | 0.0853 | -0.1214 | 61.9473 |

## Notes

- `full_panels_no_crop` removes the gaze-centered crop path but keeps full visual panels.
- `no_gaze_text_prior` removes gaze-specific prompt wording and uses full panels, but it does not edit any visible gaze marker.
- `no_gaze` additionally masks the projected green gaze marker in copied panel images before inference.
- The current experiment-2 manifest has no explicit hand summary field, so hand contribution is not claimed from this workflow.
