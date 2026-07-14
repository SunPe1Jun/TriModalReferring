# VR-TriRef Full Experiment Flow

This directory is the compact entry point for the completed experiments. It records the purpose, protocol, measured results, reproduction entry points, and interpretation boundary of each experiment without duplicating large raw predictions.

## Experiment Map

| Experiment | Purpose | Model output | Main evaluation | Completed scale |
| --- | --- | --- | --- | ---: |
| Experiment 1 | Test closed-set referent identification from multimodal evidence | Scene anchor ids | Hit-All, mapped-only hit, exact set, set F1 | 4,000 events |
| Experiment 2 | Separate temporal evidence selection from image-plane localization | Evidence panel ids and 2D points | Temporal F1, Point@K F1, Joint@K F1 | 4,000 events |
| Experiment 3 | Test candidate-free 3D referent point hypotheses | One or more Unity world-coordinate points | Nearest-anchor set F1, Margin-F1, scene-normalized error | 3,971 evaluable events |

The experiments are complementary. Metrics must not be compared numerically across experiments because their output spaces, supervision, and denominators differ.

## Headline Results

### Main Qwen3-VL-30B runs

| Experiment | Key measured result |
| --- | --- |
| Experiment 1 | Hit-All 0.7660; exact 0.3205; micro set F1 0.5326 |
| Experiment 2 | Temporal F1 0.7333; Point@100 F1 0.2470; Joint@100 F1 0.2038 |
| Experiment 3 | Anchor-set F1 0.4257; Margin-F1@1.0 0.2492; Margin-F1@2.0 0.4107 |

### Supplemental-model completion

Qwen3-VL-8B and InternVL3-38B full runs are complete for all three experiments. Experiment 3 produced 3,971/3,971 valid outputs for both models after the two-shard runs were merged and evaluated.

| Model | Exp. 1 Hit-All | Exp. 2 Temporal F1 | Exp. 2 Point@100 F1 | Exp. 3 anchor F1 | Exp. 3 M-F1@1.0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-8B | 0.6625 | 0.7456 | 0.2341 | 0.4147 | 0.2435 |
| InternVL3-38B | 0.7093 | 0.7320 | 0.1232 | 0.3863 | 0.2302 |

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
- The completed ablations do not meet the strict single-modality definition in the audit specification. They are useful hybrid/input/preprocessing/prompt ablations, and no strict paired-bootstrap significance claim is available.
- All values in this directory come from existing evaluator outputs. No model output was edited and no experiment was rerun to create this summary.
