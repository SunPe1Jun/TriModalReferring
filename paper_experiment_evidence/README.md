# VR-TriRef Paper Experiment Evidence

The core nine-model evidence bundle was rebuilt from the final VR-TriRef evaluator outputs at source commit `ad190b747f73a8c613148cd6813a3a98e43c8818`. The strict Experiment 3 hand-input extension uses implementation commit `2bab7fe8f80cf3d8981fa2bb8239f0c612b33f15`. Existing model outputs were not edited; all metrics use the recorded evaluator for their experiment.

## Experiments and denominators

- **Experiment 1: closed-set 3D anchor selection.** All 4,000 interactions now have fully mapped GT. Hit-All, Hit-Mapped, exact set, macro set F1, and micro P/R/F1 therefore share the 4,000-row denominator.
- **Experiment 2: projected-2D point diagnostic.** Temporal, Point@50/100/150/200, and Joint@50/100/150/200 are corpus-level TP/FP/FN metrics over 4,000 interactions. Missing predictions remain empty predictions; the final runs have zero missing records. Qwen3-VL-30B retains 2 deterministic parse failures after three same-configuration attempts, listed in `denominator_audit/invalid_output_ids.csv`.
- **Experiment 3: candidate-free measured point-hypothesis diagnostic.** All 4,000 interactions are evaluated. Models emit measured world-coordinate point hypotheses without candidate IDs or hidden GT in the model-facing manifest. Nearest-anchor association occurs only in evaluation. This is not unconstrained 3D reconstruction, 3D box grounding, or 3D IoU.

## Fairness and provenance

Within each experiment, all three models have identical `scene::row_index` sample hashes and GT hashes. The semantic input protocol, prompt objective, evaluator, and greedy decoding policy are shared. Model-specific chat/vision adapters remain necessary; InternVL Exp.3 uses two evidence images and 8-bit loading with 256 output tokens, while Qwen uses up to three images and 512 tokens. These differences are explicit in `run_provenance/run_provenance.csv`.

Qwen3-VL-8B Exp.1's paper run is `qwen8/outputs/exam1_qwen3vl8b_baseline`, reevaluated on the repaired 4,000-row GT. Its earlier `0.6625` value used the same final raw run before GT completion; `data/match_eval_qwen3vl8b` with Hit-All `0.60075` is a separate legacy prompt/input run. Neither value should replace the rebuilt result in `model_results.csv`.

## Ablations and limits

The Exp.1/Exp.2 ablations are descriptive hybrid/input/preprocessing/prompt ablations, not strict single-modality causal ablations. Experiment 3 contains five descriptive controls under `ablation/experiment3_qwen30b/` and a strict model-input hand control under `ablation/experiment3_qwen30b_strict_hand/`. The strict control removes hand fields and masks visible projected hand regions, but its frozen frame selector used hand signals before masking; it is therefore not an end-to-end causal pipeline intervention. No bootstrap samples exist beyond the header-only file, so no p-value, significance, or confidence interval is reported.

The explicit location/region taxonomy contains 37 canonical anchors and identifies 1372 interactions. The old 1,461 count is not reproducible because no committed taxonomy supported it; it must not be cited.
