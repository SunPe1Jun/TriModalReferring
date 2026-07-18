# VR-TriRef Full Experiment Flow

This directory is the compact entry point for the completed experiments. It records the purpose, protocol, measured results, reproduction entry points, and interpretation boundary of each experiment without duplicating large raw predictions.

## Evidence Bundle

The auditable paper bundle is `paper_experiment_evidence/`. It contains nine unified sample-level exports, explicit denominator audits, evaluator copies, manifests, run provenance, and machine-readable validation. The bundle is generated from the final run directories without rerunning inference. Values in the tables below are table-ready legacy-compatible summaries; the evidence bundle also records invalid, missing, and parse-failure rows separately.

## Experiment Map

| Experiment | Purpose | Model output | Main evaluation | Completed scale |
| --- | --- | --- | --- | ---: |
| Experiment 1 | Test closed-set referent identification from multimodal evidence | Scene anchor ids | Hit-All, mapped-only hit, exact set, set F1 | 4,000 events |
| Experiment 2 | Separate temporal evidence selection from image-plane localization | Evidence panel ids and 2D points | Temporal F1, Point@K F1, Joint@K F1 | 4,000 events |
| Experiment 3 | Test candidate-free 3D referent point hypotheses | One or more Unity world-coordinate points | Nearest-anchor set F1, Margin-F1, scene-normalized error | 4,000 events |

The experiments are complementary. Metrics must not be compared numerically across experiments because their output spaces, supervision, and denominators differ.

## Headline Results

### Main Qwen3-VL-30B runs

| Experiment | Key measured result |
| --- | --- |
| Experiment 1 | Hit-All 0.7798; exact 0.3287; micro set F1 0.5399 |
| Experiment 2 | Temporal F1 0.7348; Point@100 F1 0.2497; Joint@100 F1 0.2067 |
| Experiment 3 | Anchor-set F1 0.4326; Margin-F1@1.0 0.2503; Margin-F1@2.0 0.4105 |

### Supplemental-model completion

Qwen3-VL-8B and InternVL3-38B full runs are complete for all three experiments. Experiment 3 produced 4,000/4,000 valid outputs for both models after the completion shards were merged and evaluated.

| Model | Exp. 1 Hit-All | Exp. 2 Temporal F1 | Exp. 2 Point@100 F1 | Exp. 3 anchor F1 | Exp. 3 M-F1@1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 0.6733 | 0.7461 | 0.2350 | 0.4213 | 0.2440 |
| InternVL3-38B | 0.7215 | 0.7330 | 0.1228 | 0.3931 | 0.2310 |

Exact source paths and additional metrics are recorded in `results/model_results.csv`.

## Recommended Reading Order

1. `experiment1_closed_set_anchor.md`
2. `experiment2_projected_2d.md`
3. `experiment3_candidate_free_3d.md`
4. `ablation_audit.md`
5. `results/model_results.csv` and `results/ablation_results.csv`

## Claim Boundaries

- Experiment 1 is closed-set anchor selection; candidate anchors are model inputs.
- Experiment 2 is a projected-2D diagnostic, not native 3D localization.
- Experiment 3 hides anchor candidates from the model, but its v9 prompt exposes measured gaze/hand endpoints. It is therefore a measured point-hypothesis diagnostic, not unconstrained 3D reconstruction or 3D box grounding.
- The completed ablations do not meet the strict single-modality definition in the audit specification. They are descriptive hybrid/input/preprocessing/prompt ablations, and no strict paired-bootstrap significance claim is available.
- Experiment 3 is a candidate-free measured point-hypothesis diagnostic. It is not unconstrained 3D reconstruction, 3D box grounding, or 3D IoU evaluation. The gaze-copy baseline is reported with the model results.
- All values in this directory come from unified evaluator outputs. Existing valid model outputs were preserved; only newly evaluable or invalid records were inferred with the original run configuration, and no model output was manually edited.
