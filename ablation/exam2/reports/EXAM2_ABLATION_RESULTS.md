# Experiment 2 Modality Ablation Summary

Baseline is reused from `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`. Manifest construction and evaluator are unchanged unless a variant explicitly changes the number of panels.

| Variant | Events | Predictions | Time F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 | Mean Point Dist@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_baseline | 4000 | 8635 | 0.7333 | 0.2470 | 0.2038 |  | 57.9929 |
| full_panels_no_crop | 4000 | 8637 | 0.7010 | 0.2236 | 0.1759 | -0.0280 | 58.7615 |
| instruction_only_prompt | 4000 | 7582 | 0.6549 | 0.2262 | 0.1799 | -0.0240 | 57.5738 |
| no_gaze | 4000 | 8632 | 0.6886 | 0.2027 | 0.1551 | -0.0488 | 59.0221 |
| no_gaze_text_prior | 4000 | 8617 | 0.6896 | 0.2199 | 0.1725 | -0.0313 | 58.2964 |

## Notes

- `full_panels_no_crop` removes the gaze-centered crop path but keeps full visual panels.
- `no_gaze_text_prior` removes gaze-specific prompt wording and uses full panels, but it does not edit any visible gaze marker.
- `no_gaze` additionally masks the projected green gaze marker in copied panel images before inference.
- The current experiment-2 manifest has no explicit hand summary field, so hand contribution is not claimed from this workflow.
