# Experiment 3: Candidate-Free Point-Supervised 3D Grounding

## Purpose

Test whether language, target-free evidence frames, and measured VR telemetry can produce useful 3D referent point hypotheses when candidate anchor ids and coordinates are hidden from the model.

## Protocol

Up to three chronological evidence frames are selected inside each event without using the target anchor. The model receives language, images, camera pose, gaze and hand telemetry, robust scene bounds, and copyable measured gaze/hand ray endpoints. It emits a variable-size JSON list of Unity world-coordinate points.

The evaluator loads hidden scene anchors only after inference. Each point is mapped to its nearest anchor for set evaluation and matched against GT points for margin-normalized and scene-normalized errors. Invalid outputs become empty sets and remain in the denominator.

The manifest has 3,971 evaluable interactions: scene1 800, scene2 791, scene3 800, scene4 rooms 784, and scene5 796.

## Metrics

- Nearest-anchor set micro precision, recall, F1, and exact rate.
- Margin-F1@0.5/@1.0/@2.0, normalized by local distractor spacing.
- Matched Euclidean error and robust scene-normalized error.
- Valid-output rate and single-target/multi-target partitions.

## Full Results

| Model | Valid | Anchor P | Anchor R | Anchor F1 | Exact | M-F1@1.0 | M-F1@2.0 | Mean scene-norm error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B v9 | 3,971/3,971 | 0.4157 | 0.4362 | 0.4257 | 0.0169 | 0.2492 | 0.4107 | 0.1590 |
| Qwen3-VL-8B | 3,971/3,971 | 0.4163 | 0.4132 | 0.4147 | 0.0222 | 0.2435 | 0.4009 | 0.1573 |
| InternVL3-38B | 3,971/3,971 | 0.4228 | 0.3557 | 0.3863 | 0.0468 | 0.2302 | 0.3910 | 0.1642 |

The deterministic gaze-copy baseline reaches anchor F1 0.4257 and Margin-F1@2.0 0.4196, essentially tying or slightly exceeding Qwen3-VL-30B. This baseline is necessary for interpreting the task.

## Completion Evidence

- Qwen3-VL-8B merged evaluation: `qwen8/outputs/exam3_qwen3vl8b_point_grounding_merged_20260713/eval/evaluation_summary.json`
- InternVL3-38B merged evaluation: `internvl/outputs/exam3_internvl38b_point_grounding_merged_20260714/eval/evaluation_summary.json`
- Qwen3-VL-30B report: `exam3_point_grounding/reports/EXPERIMENT3_FULL_RESULTS_V9.md`
- Task definition: `exam3_point_grounding/EXPERIMENT3_TASK_DEFINITION.md`

## Reproduction Entry Points

- Qwen3-VL-30B: `exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh`
- Qwen3-VL-8B: `qwen8/run_exam3_qwen3vl8b_point_grounding.sh`
- InternVL3-38B: `internvl/run_exam3_internvl38b_point_grounding.sh`
- Shared manifest/evaluator: `exam3_point_grounding/build_point_grounding_manifest.py` and `exam3_point_grounding/evaluate_point_grounding.py`

## Interpretation Boundary

The v9 protocol is candidate-free relative to scene anchors, but it deliberately exposes measured gaze/hand endpoints and asks the model to select or copy them. The defensible claim is that measured behavioral point hypotheses provide target-free 3D grounding signal. Do not claim unconstrained 3D reconstruction, object extents, boxes, or 3D IoU.

The older camera-centered angular diagnostic under `exam3/` is archived and is not the final Experiment 3 result.
