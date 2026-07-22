# Experiment 3: Candidate-Free Point-Supervised 3D Grounding

## Purpose

Test whether language, target-free evidence frames, and measured VR telemetry can produce useful 3D referent point hypotheses when candidate anchor ids and coordinates are hidden from the model. The exact task name is **candidate-free measured point-hypothesis diagnostic**.

## Protocol

Up to three chronological evidence frames are selected inside each event without using the target anchor. The model receives language, images, camera pose, gaze and hand telemetry, robust scene bounds, and copyable measured gaze/hand ray endpoints. It emits a variable-size JSON list of Unity world-coordinate points.

The evaluator loads hidden scene anchors only after inference. Each point is mapped to its nearest anchor for set evaluation and matched against GT points for margin-normalized and scene-normalized errors. Invalid outputs become empty sets and remain in the denominator. This is not unconstrained 3D reconstruction, 3D box grounding, or 3D IoU evaluation.

The repaired manifest has 4,000 evaluable interactions: scene1, scene2, scene3, and scene5 each contain 800; each of the four scene4 rooms contains 200.

## Metrics

- Nearest-anchor set micro precision, recall, F1, and exact rate.
- Margin-F1@0.5/@1.0/@2.0, normalized by local distractor spacing.
- Matched Euclidean error and robust scene-normalized error.
- Valid-output rate and single-target/multi-target partitions.

## Full Results

| Model | Valid | Anchor P | Anchor R | Anchor F1 | Exact | M-F1@1.0 | M-F1@2.0 | Mean scene-norm error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B v9 | 4,000/4,000 | 0.4256 | 0.4399 | 0.4326 | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| Qwen3-VL-8B | 4,000/4,000 | 0.4262 | 0.4166 | 0.4213 | 0.0225 | 0.2440 | 0.4004 | 0.1550 |
| InternVL3-38B | 4,000/4,000 | 0.4337 | 0.3594 | 0.3931 | 0.0480 | 0.2310 | 0.3909 | 0.1615 |

The deterministic gaze-copy baseline reaches anchor F1 0.4325 and Margin-F1@2.0 0.4193, nearly tying Qwen3-VL-30B on anchor F1 and exceeding it on Margin-F1. This baseline is necessary for interpreting the task.

## Qwen3-VL-30B Input Ablation

Removing model-visible gaze cues reduces anchor F1 from 0.4326 to 0.0673 and Margin-F1@1.0 from 0.2503 to 0.0055. In contrast, `no_visual`, `no_hand`, and `no_instruction` remain within 0.003 anchor F1 of the full condition. This indicates that the frozen v9 prompt is dominated by exposed copyable gaze hypotheses. Since evidence frames were selected before masking with a selector that used gaze/hand availability, these are descriptive post-selection input ablations rather than strict causal modality estimates.

The five variants each contain 4,000 samples. One `no_gaze_hand` output is invalid and remains an empty prediction in the denominator. Detailed results and compact evidence are under `ablation/exam3/reports/full_v3/` and `paper_experiment_evidence/ablation/experiment3_qwen30b/`.

A stricter hand-input control removes all structured hand fields and masks projected hand regions in every frozen panel. It obtains anchor F1 0.4351 and Margin-F1@1.0 0.2525, versus 0.4326 and 0.2503 for Full; 96.92% of outputs are identical. Thus the current v9 experiment provides no evidence of an independent hand contribution once copyable gaze hypotheses are exposed. This control remains post-selection because the frozen selector used hand availability.

## Completion Evidence

- Qwen3-VL-8B merged evaluation: `qwen8/outputs/exam3_qwen3vl8b_point_grounding_merged_20260713/eval/evaluation_summary.json`
- InternVL3-38B merged evaluation: `internvl/outputs/exam3_internvl38b_point_grounding_merged_20260714/eval/evaluation_summary.json`
- Unified report: `paper_experiment_evidence/EXPERIMENT3_FULL_RESULTS_V9.md`
- Task definition: `exam3_point_grounding/EXPERIMENT3_TASK_DEFINITION.md`

## Reproduction Entry Points

- Qwen3-VL-30B: `exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh`
- Qwen3-VL-8B: `qwen8/run_exam3_qwen3vl8b_point_grounding.sh`
- InternVL3-38B: `internvl/run_exam3_internvl38b_point_grounding.sh`
- Shared manifest/evaluator: `exam3_point_grounding/build_point_grounding_manifest.py` and `exam3_point_grounding/evaluate_point_grounding.py`

## Interpretation Boundary

The v9 protocol is candidate-free relative to scene anchors, but it deliberately exposes measured gaze/hand endpoints and asks the model to select or copy them. The defensible claim is that measured behavioral point hypotheses provide target-free 3D grounding signal. Do not claim unconstrained 3D reconstruction, object extents, boxes, or 3D IoU.

The older camera-centered angular diagnostic under `exam3/` is archived and is not the final Experiment 3 result.

The deterministic gaze-copy baseline is reported because the v9 prompt exposes measured gaze hypotheses that the model can copy. In the final run it ties Qwen3-VL-30B on anchor-set F1 while slightly exceeding it on Margin-F1. See `paper_experiment_evidence/EXPERIMENT3_FULL_RESULTS_V9.md`.
