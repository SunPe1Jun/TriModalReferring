# Experiment 2: Projected-2D Point Diagnostic

## Purpose

Decompose referent grounding into two questions: whether the model selects the correct temporal evidence panels and whether its predicted image-plane points fall near GT anchor projections.

## Protocol

GT 3D anchors are projected into egocentric evidence panels. The model consumes paired visual evidence and multimodal context, then returns panel/point predictions in JSON. The evaluator retains missing or invalid records as empty predictions rather than removing them.

The full corpus contains 4,000 events and 8,519 GT referents. The same manifest and GT panels/points are shared across the three final model runs.

## Metrics

- Temporal precision/recall/F1 for evidence-panel selection.
- Point@K precision/recall/F1 for localization within K pixels, independent of temporal correctness.
- Joint@K precision/recall/F1 requiring both the correct panel and a point within K pixels.
- Matched point distance in pixels.

The standard summaries report K in {50, 100, 150, 200}; the audit emphasizes 50, 100, and 200.

## Main Result

The audited Qwen3-VL-30B full baseline reaches temporal F1 0.7348, Point@100 F1 0.2497, Joint@100 F1 0.2067, and Joint@200 F1 0.3674. Two outputs remain invalid after three same-configuration retries and are retained as empty predictions.

Supplemental full baselines:

| Model | Temporal F1 | Point@100 F1 | Joint@100 F1 | Joint@200 F1 |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 0.7461 | 0.2350 | 0.2013 | 0.3669 |
| InternVL3-38B | 0.7330 | 0.1228 | 0.0977 | 0.2701 |

## Reproduction and Sources

- Main workflow: `exam2/run_qwen3vl_30b_2d_full.sh`
- Qwen3-VL-8B wrapper: `qwen8/run_exam2_qwen3vl8b_baseline.sh`
- InternVL3-38B wrapper: `internvl/run_exam2_internvl38b_baseline.sh`
- Audited results: `paper_experiment_evidence/model_results.csv` and `paper_experiment_evidence/predictions/exp2/`
- Qwen3-VL-8B result: `qwen8/outputs/exam2_qwen3vl8b_baseline_2d_point_hybrid_v10/eval/2d_eval_summary.json`
- InternVL3-38B result: `internvl/outputs/exam2_internvl3_38b_baseline/eval/2d_eval_summary.json`

## Interpretation

Temporal and point metrics expose different failure modes. Joint@K is the end-to-end diagnostic. The experiment evaluates projected points in selected images; it should not be described as direct 3D grounding.
