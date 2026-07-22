# Experiment 3 Strict Hand-Input Ablation Evidence

This directory is the final compact evidence bundle for the Qwen3-VL-30B `no_hand_strict` condition. It uses the same 4,000 frozen interactions and selected panels as the Full run, removes structured hand fields from the prompt, and masks projected tracked hand regions in every panel before model inference.

| Condition | Valid | Anchor F1 | Exact | Margin-F1@1.0 | Margin-F1@2.0 | Mean scene-normalized error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 4,000 | 0.4326 | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| no_hand_strict | 4,000 | 0.4351 | 0.0187 | 0.2525 | 0.4132 | 0.1558 |

The strict condition produced 4,000/4,000 valid outputs. Its output is identical to Full on 3,877/4,000 samples (96.92%). The small change is consistent with the v9 protocol being dominated by explicit copyable gaze hypotheses.

## Mask audit

- 11,001 panels across 4,000 samples
- 9,323 panels with in-frame tracked hand joints and a generated mask
- 1,678 panels with tracked hand projections fully off-screen
- 0 panels without a tracked-hand status
- mean mask fraction: 0.1138777
- maximum frame-to-telemetry time error: 0 seconds
- mask: expanded projected-joint bbox with neutral RGB fill `[127,127,127]`, `hand_mask_v1`

The selector and panel list remain frozen; therefore this is a strict model-input hand ablation after frame selection, not a strict intervention on the hand-aware selector itself.

The prediction CSV contains GT, parsed hypotheses and evaluator detail only. Raw model JSON, prompts, original videos and model weights are not part of this compact bundle.
