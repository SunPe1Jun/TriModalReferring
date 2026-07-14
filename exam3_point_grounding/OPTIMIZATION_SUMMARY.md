# Experiment 3 Optimization Summary

Date: 2026-07-09

## Smoke Setup

- Scene: `scene1`
- Samples: first 10 evaluable rows
- Model: local `Qwen3-VL-30B-A3B-Instruct`
- Output root pattern: `exam3_point_grounding/outputs_smoke10_scene1_v*_20260709`
- Evaluator unchanged across versions; parser remains strict on exactly three finite coordinates per point.

## Results

| version | valid rate | invalid | anchor P | anchor R | anchor F1 | margin F1@2 | mean scene-norm err |
|---|---:|---:|---:|---:|---:|---:|---:|
| v4 | 1.0000 | 0 | 0.1111 | 0.1000 | 0.1053 | 0.0000 | 0.2427 |
| v5 | 1.0000 | 0 | 0.2000 | 0.2000 | 0.2000 | 0.0952 | 0.2780 |
| v6 | 1.0000 | 0 | 0.2000 | 0.2000 | 0.2000 | 0.0952 | 0.2170 |
| v7 | 1.0000 | 0 | 0.2143 | 0.3000 | 0.2500 | 0.1333 | 0.2337 |
| v8 | 1.0000 | 0 | 0.1429 | 0.2000 | 0.1667 | 0.1538 | 0.1577 |
| v9 | 1.0000 | 0 | 0.3478 | 0.8000 | 0.4848 | 0.3500 | 0.0731 |
| gaze-copy baseline | 1.0000 | 0 | 0.4000 | 0.8000 | 0.5333 | 0.3784 | 0.0731 |

## Interpretation

- v4 fixed the major invalid-output problem by removing confusing flat vectors from the model-facing prompt, but localization remained poor.
- v5-v8 explored progressively stronger constraints against free-form coordinate invention and toward measured ray-endpoint hypotheses.
- v9 is the best Qwen prompt among these smoke runs: it uses ID-based measured gaze hypotheses and reaches anchor F1 0.4848 with 100% valid outputs.
- Deterministic gaze-copy remains slightly stronger on this smoke subset: anchor F1 0.5333 and margin F1@2 0.3784.
- This suggests the current replacement Experiment 3 is methodologically cleaner as a measured point-hypothesis diagnostic than as unconstrained VLM 3D coordinate reconstruction.

## Current Recommendation

- Do not run a full Qwen v9 evaluation as the final Experiment 3 headline without first confirming the intended claim.
- If the paper claim is “VLM can recover 3D points candidate-free,” the smoke evidence is still weak.
- If the paper claim is “measured gaze/hand point hypotheses provide a target-free 3D diagnostic, and VLM can serialize/select these hypotheses,” v9 is a reasonable prompt to freeze and full-run.
- Keep reporting deterministic cue baselines beside Qwen; they are stronger and important for interpreting the diagnostic.
