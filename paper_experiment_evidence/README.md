# VR-TriRef Paper Experiment Evidence

This bundle is a compact, read-only export of the final VR-TriRef experiments at commit `46db9cc`. It is intended for paper tables, denominator audits, and reproducibility review. No model inference was rerun and no raw prediction was edited.

## Experiments

- **Experiment 1: closed-set 3D anchor selection.** The model receives a candidate anchor inventory and returns anchor IDs. Hit-All uses all 4,000 rows; Hit-Mapped, exact set, and macro set metrics use the 3,972 mapped rows. Micro TP/FP/FN are retained over all rows so predictions on unmapped rows remain visible.
- **Experiment 2: projected-2D point diagnostic.** The model selects temporal panels and image-plane points. Temporal, Point@50/100/150/200, and Joint@50/100/150/200 are corpus metrics over all 4,000 rows. Missing records and parse failures remain empty predictions.
- **Experiment 3: candidate-free measured point-hypothesis diagnostic.** The model emits measured 3D point hypotheses without candidate IDs. Nearest-anchor association is performed only by the evaluator; reported metrics include anchor-set P/R/F1, exact set, Margin-F1, scene-normalized error, and single/multi partitions. This is not unconstrained reconstruction, box grounding, or 3D IoU.

## Final Runs

The final sources are the complete baseline directories named in `run_provenance/run_provenance.csv` and the source paths encoded in each prediction file. All three models use the same unified IDs and GT hashes within each experiment. Chat templates, image/video adapters, and checkpoints are recorded as run-specific provenance; these implementation differences are not silently treated as identical.

Qwen3-VL-8B Exp.1 has a legacy result with Hit-All `0.60075` under `data/match_eval_qwen3vl8b` and a final mention-first run with Hit-All `0.6625` under `qwen8/outputs/exam1_qwen3vl8b_baseline`. The final run uses 8 video frames, `mention_first`, and max 1,536 new tokens; the legacy run uses 16 frames, standard prompt strategy, and a different prompt. The difference is therefore attributable to documented run configuration, not an inferred model improvement.

## Ablations and Limits

The supplied ablations are descriptive hybrid/input/preprocessing/prompt ablations, not strict single-modality causal ablations. The bootstrap CSV contains headers but no bootstrap samples, so this bundle makes no significance, p-value, or confidence-interval claim.

## Files

`predictions/` contains nine compact sample-level CSVs; `denominator_audit/` contains explicit unmapped, missing, and Exp.3-excluded IDs; `manifests/` and `anchor_tables/` preserve compact GT context; `evaluators/` and `prompts_and_configs/` preserve the relevant code/prompt; `validation/` contains machine-readable checks; `model_results.csv` is the table-ready summary.
