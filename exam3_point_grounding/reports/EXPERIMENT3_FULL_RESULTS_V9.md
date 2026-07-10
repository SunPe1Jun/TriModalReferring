# Experiment 3 Full Results: Candidate-Free Point-Supervised 3D Grounding

Run: `qwen3vl30b_point_grounding_v9_full`  
Date: 2026-07-10  
Samples: 3971 evaluable referential interactions  
Model: local Qwen3-VL-30B-A3B-Instruct  
Output root: `exam3_point_grounding/outputs_full_v9_20260709`

## Scope

This report summarizes the full v9 run for Experiment 3. The task is candidate-free point-supervised 3D grounding: the model does not receive candidate anchors or ground-truth anchor coordinates, and predicted 3D points are evaluated against hidden scene anchors after inference.

The v9 prompt should be interpreted as a measured point-hypothesis diagnostic. It exposes gaze and hand ray endpoints as behavioral point hypotheses, with gaze as the primary copyable cue and hand as fallback. Therefore deterministic cue baselines are reported beside Qwen.

## Overall Results

| method | N | valid | anchor P | anchor R | anchor F1 | exact | M-F1@0.5 | M-F1@1.0 | M-F1@2.0 | mean scene-norm err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3vl30b_v9 | 3971 | 1.0000 | 0.4157 | 0.4362 | 0.4257 | 0.0169 | 0.1139 | 0.2492 | 0.4107 | 0.1590 |
| gaze_copy | 3971 | 1.0000 | 0.4175 | 0.4342 | 0.4257 | 0.0408 | 0.1163 | 0.2535 | 0.4196 | 0.1541 |
| hand_copy | 3971 | 1.0000 | 0.1035 | 0.0854 | 0.0936 | 0.0008 | 0.0013 | 0.0072 | 0.0335 | 0.8322 |
| gaze_hand_fusion | 3971 | 1.0000 | 0.3677 | 0.4422 | 0.4015 | 0.0083 | 0.1054 | 0.2297 | 0.3843 | 0.2235 |

## Qwen Per-Scene Results

| scene | N | anchor F1 | M-F1@1.0 | M-F1@2.0 | mean scene-norm err |
| --- | ---: | ---: | ---: | ---: | ---: |
| scene1 | 800 | 0.4545 | 0.2448 | 0.3957 | 0.1594 |
| scene2 | 791 | 0.2372 | 0.0467 | 0.1699 | 0.2171 |
| scene3 | 800 | 0.4888 | 0.3337 | 0.5541 | 0.1216 |
| scene4_room1 | 186 | 0.4937 | 0.3275 | 0.5186 | 0.1586 |
| scene4_room2 | 200 | 0.3299 | 0.0976 | 0.2595 | 0.3491 |
| scene4_room3 | 199 | 0.6479 | 0.3642 | 0.5369 | 0.1360 |
| scene4_room4 | 199 | 0.4302 | 0.2615 | 0.4335 | 0.1627 |
| scene5 | 796 | 0.4878 | 0.3532 | 0.4924 | 0.1046 |

## Single-Target vs Multi-Target

| method | partition | N | anchor F1 | M-F1@1.0 | M-F1@2.0 | mean scene-norm err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3vl30b_v9 | single_target | 973 | 0.4218 | 0.2184 | 0.3569 | 0.0998 |
| qwen3vl30b_v9 | multi_target | 2998 | 0.4266 | 0.2562 | 0.4229 | 0.1673 |
| gaze_copy | single_target | 973 | 0.4254 | 0.2294 | 0.3818 | 0.1004 |
| gaze_copy | multi_target | 2998 | 0.4257 | 0.2587 | 0.4277 | 0.1618 |
| hand_copy | single_target | 973 | 0.0561 | 0.0029 | 0.0163 | 0.9647 |
| hand_copy | multi_target | 2998 | 0.1011 | 0.0081 | 0.0373 | 0.8132 |
| gaze_hand_fusion | single_target | 973 | 0.3701 | 0.1978 | 0.3299 | 0.0998 |
| gaze_hand_fusion | multi_target | 2998 | 0.4084 | 0.2371 | 0.3970 | 0.2401 |

## Main Observations

- Qwen produced valid JSON point predictions for all 3971 samples; invalid count is 0.
- Qwen reached anchor-set F1 0.4257, Margin-F1@1.0 0.2492, and Margin-F1@2.0 0.4107.
- The deterministic gaze-copy baseline is essentially tied with Qwen on anchor-set F1 (0.4257) and slightly stronger on margin-normalized metrics (Margin-F1@1.0 0.2535, Margin-F1@2.0 0.4196).
- Hand-only copying is weak (anchor-set F1 0.0936), indicating that hand endpoints alone are not a reliable substitute for gaze in this protocol.
- Gaze-hand fusion underperforms gaze-copy (anchor-set F1 0.4015), suggesting that naive fusion can introduce noisy hand endpoints.
- Scene-level variation is substantial: Qwen is strongest on `scene4_room3` (anchor-set F1 0.6479) and weakest on `scene2` (anchor-set F1 0.2372).

## Interpretation Boundary

These results support Experiment 3 as a candidate-free measured point-hypothesis diagnostic, not as evidence that the model performs unconstrained 3D object reconstruction. Since the prompt asks the model to prefer measured gaze hypotheses, Qwen's result should be interpreted together with the gaze-copy baseline. The most defensible claim is that gaze-derived 3D point hypotheses provide a useful target-free signal for referent grounding, while the current VLM prompt mainly serializes/selects these hypotheses rather than independently reconstructing object centers.

## Files

Committed compact result files:

- `exam3_point_grounding/reports/EXPERIMENT3_FULL_RESULTS_V9.md`
- `exam3_point_grounding/reports/experiment3_full_results_v9_summary.json`
- `exam3_point_grounding/reports/experiment3_full_results_v9_overall.csv`
- `exam3_point_grounding/reports/experiment3_full_results_v9_per_scene.csv`
- `exam3_point_grounding/reports/experiment3_full_results_v9_partitions.csv`

Large or verbose files intentionally not committed:

- raw model outputs under `outputs_full_v9_20260709/qwen3vl30b/raw/`
- per-sample `evaluation_detail.csv`
- full prediction CSVs
- run logs
