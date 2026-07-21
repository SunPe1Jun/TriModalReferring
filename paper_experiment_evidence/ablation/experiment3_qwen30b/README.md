# Experiment 3 Qwen3-VL-30B Input Ablation Evidence

This directory contains compact evidence for five descriptive input ablations of the frozen v9 **candidate-free measured point-hypothesis diagnostic**. Each CSV contains 4,000 unified `scene::row_index` samples and merges GT, parsed point hypotheses, and evaluator detail. Model prompts and raw response text are intentionally excluded.

| Variant | Valid | Anchor F1 | Exact | Margin-F1@1.0 | Margin-F1@2.0 | Mean scene-normalized error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full baseline | 4,000 | 0.4326 | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| no_visual | 4,000 | 0.4341 | 0.0180 | 0.2518 | 0.4120 | 0.1560 |
| no_gaze | 4,000 | 0.0673 | 0.0013 | 0.0055 | 0.0263 | 0.3771 |
| no_hand | 4,000 | 0.4353 | 0.0187 | 0.2527 | 0.4134 | 0.1558 |
| no_gaze_hand | 3,999 | 0.0542 | 0.0000 | 0.0000 | 0.0037 | 0.5376 |
| no_instruction | 4,000 | 0.4332 | 0.0177 | 0.2510 | 0.4111 | 0.1565 |

The frozen target-free frame selector used gaze/hand availability before these fields were masked. Moreover, the v9 prompt explicitly instructs the model to copy measured gaze hypotheses by default. These results therefore quantify dependence on model-visible inputs under the frozen protocol; they are not strict causal single-modality ablations.

`no_gaze_hand` has one invalid output (`scene4_room1::32`, `point_entry_0_wrong_dimension`). It remains an empty prediction in the 4,000-sample denominator. No model output was manually repaired.

See `run_provenance.csv`, `compact_evidence_validation.json`, and `invalid_outputs.csv` for configuration, hashes, and denominator checks. The complete statistical report is `ablation/exam3/reports/full_v3/EXPERIMENT3_QWEN30B_ABLATION.md`.
