# Experiment 1: Closed-Set 3D Anchor Selection

## Purpose

Measure whether a VLM can identify the intended scene referent from language, egocentric visual evidence, gaze, hand, camera/world telemetry, and a closed inventory of scene-level 3D anchors.

## Protocol

Each referential interaction is serialized with its multimodal evidence and candidate anchor inventory. The model returns predicted anchor ids as JSON. The shared parser normalizes the response, and the evaluator compares the predicted anchor set with the mapped GT set. Invalid outputs remain in the all-sample denominator.

The repaired corpus contains 4,000 events, all with mapped/evaluable GT anchors. All reported set metrics therefore use the same 4,000-event denominator.

## Metrics

- Hit-All: at least one correct anchor, divided by all 4,000 events.
- Hit-Mapped: at least one correct anchor among mapped/evaluable events.
- Exact: exact predicted/GT set equality among mapped/evaluable events.
- Micro precision, recall, and F1 over anchor-set TP/FP/FN.
- Valid-output rate, reported separately from task accuracy.

## Main Result

The audited Qwen3-VL-30B full baseline reaches Hit-All/Hit-Mapped 0.7798, exact 0.3287, macro set F1 0.5657, and micro set F1 0.5399. The audit reconstructs these values from per-sample match-evaluation CSVs.

Supplemental full baselines:

| Model | Valid | Hit-All | Exact | Micro P | Micro R | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 3,872/4,000 | 0.6733 | 0.2377 | 0.5743 | 0.3947 | 0.4679 |
| InternVL3-38B | 3,982/4,000 | 0.7215 | 0.2592 | 0.5392 | 0.4479 | 0.4893 |

## Reproduction and Sources

- Main workflow: `scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh`
- Qwen3-VL-8B wrapper: `qwen8/run_exam1_qwen3vl8b_baseline.sh`
- InternVL3-38B wrapper: `internvl/run_exam1_internvl38b_baseline.sh`
- Audited results: `paper_experiment_evidence/model_results.csv` and `paper_experiment_evidence/predictions/exp1/`
- Supplemental results: per-partition JSON files under `qwen8/outputs/exam1_qwen3vl8b_baseline/eval/` and `internvl/outputs/exam1_internvl3_38b_baseline/eval/`

## Interpretation

This experiment supports claims about referent selection when a candidate inventory is available. It does not establish candidate-free localization.
