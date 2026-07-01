# VR-TriRef Modality Ablation Results

This report covers ablations for the two established experiments only.
Experiment 3 is excluded from this round.

## Experiment 1

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

## Experiment 2

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

## Reproducibility

- Experiment 1 runner: `ablation/exam1/run_exam1_ablation.sh`
- Experiment 2 runner: `ablation/exam2/run_exam2_ablation.sh`
- Parallel smoke: `ablation/run_parallel_smoke.sh`
- Parallel full run: `ablation/run_parallel_full.sh`
