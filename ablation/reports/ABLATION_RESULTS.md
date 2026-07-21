# VR-TriRef Modality Ablation Results

This report covers the completed descriptive ablations for all three experiments.

## Experiment 1

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
- The experiment-1 table uses the 4000-sample full summary in `ablation/exam1/reports/exam1_ablation_summary.csv`.

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

## Experiment 3

Experiment 3 uses the frozen v9 candidate-free measured point-hypothesis protocol. All variants share the same 4,000 samples, selected evidence frames, checkpoint, parser, greedy decoding, and evaluator; only model-visible fields are masked after frame selection.

| Variant | Valid | Anchor F1 | Delta F1 | Exact | Margin-F1@1.0 | Margin-F1@2.0 | Scene-normalized error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 4000 | 0.4326 | 0.0000 | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| no_visual | 4000 | 0.4341 | +0.0014 | 0.0180 | 0.2518 | 0.4120 | 0.1560 |
| no_gaze | 4000 | 0.0673 | -0.3653 | 0.0013 | 0.0055 | 0.0263 | 0.3771 |
| no_hand | 4000 | 0.4353 | +0.0027 | 0.0187 | 0.2527 | 0.4134 | 0.1558 |
| no_gaze_hand | 3999 | 0.0542 | -0.3784 | 0.0000 | 0.0000 | 0.0037 | 0.5376 |
| no_instruction | 4000 | 0.4332 | +0.0006 | 0.0177 | 0.2510 | 0.4111 | 0.1565 |

The result is dominated by exposed copyable gaze hypotheses: removing gaze causes a large degradation, whereas removing visual input, hand telemetry, or event instruction barely changes predictions. Since the target-free selector itself used gaze/hand availability, these are descriptive post-selection input ablations rather than strict causal modality interventions.

- Experiment 3 runner: `ablation/exam3/run_parallel_full.sh`
- Unified report: `ablation/exam3/reports/full_v3/EXPERIMENT3_QWEN30B_ABLATION.md`
- Compact evidence: `paper_experiment_evidence/ablation/experiment3_qwen30b/`
