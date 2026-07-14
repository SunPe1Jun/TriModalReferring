# Experiment 1: Closed-Set 3D Anchor Selection

## Purpose

Measure whether a VLM can identify the intended scene referent from language, egocentric visual evidence, gaze, hand, camera/world telemetry, and a closed inventory of scene-level 3D anchors.

## Protocol

Each referential interaction is serialized with its multimodal evidence and candidate anchor inventory. The model returns predicted anchor ids as JSON. The shared parser normalizes the response, and the evaluator compares the predicted anchor set with the mapped GT set. Invalid outputs remain in the all-sample denominator.

The full corpus contains 4,000 events. Of these, 3,972 have mapped/evaluable GT anchors. Hit-All retains all 4,000 events; mapped-only and exact-set values use the evaluator's mapped set as defined in the source summaries.

## Metrics

- Hit-All: at least one correct anchor, divided by all 4,000 events.
- Hit-Mapped: at least one correct anchor among mapped/evaluable events.
- Exact: exact predicted/GT set equality among mapped/evaluable events.
- Micro precision, recall, and F1 over anchor-set TP/FP/FN.
- Valid-output rate, reported separately from task accuracy.

## Main Result

The audited Qwen3-VL-30B full baseline reaches Hit-All 0.7660, Hit-Mapped 0.7714, exact 0.3205, and micro set F1 0.5326. The audit reconstructs these values from per-sample match-evaluation CSVs.

Supplemental full baselines:

| Model | Valid | Hit-All | Exact | Micro P | Micro R | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 3,872/4,000 | 0.6625 | 0.2329 | 0.5610 | 0.3931 | 0.4623 |
| InternVL3-38B | 3,982/4,000 | 0.7093 | 0.2533 | 0.5263 | 0.4457 | 0.4827 |

## Reproduction and Sources

- Main workflow: `scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh`
- Qwen3-VL-8B wrapper: `qwen8/run_exam1_qwen3vl8b_baseline.sh`
- InternVL3-38B wrapper: `internvl/run_exam1_internvl38b_baseline.sh`
- Audited main result: `analysis_outputs/ablation_audit/anchor_ablation_summary.csv`
- Supplemental results: per-partition JSON files under `qwen8/outputs/exam1_qwen3vl8b_baseline/eval/` and `internvl/outputs/exam1_internvl3_38b_baseline/eval/`

## Interpretation

This experiment supports claims about referent selection when a candidate inventory is available. It does not establish candidate-free localization.
