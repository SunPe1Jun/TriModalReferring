# Experiment 3 Qwen3-VL-30B Input Ablation

## Scope

This is a descriptive model-input ablation of the frozen v9 candidate-free measured point-hypothesis diagnostic. All variants reuse the same evidence-frame selection, GT manifest, model checkpoint, parser, greedy decoding (`do_sample=false`, `max_new_tokens=512`), and evaluator. The current evaluation denominator is 4000 samples. Only model-visible input fields are masked.

Because the frozen target-free frame selector itself used gaze/hand availability and stability, `no_gaze`, `no_hand`, and `no_gaze_hand` do not constitute strict causal single-modality ablations. They remove those fields after panel selection and must be reported as controlled input ablations.

## Frozen Baseline

- samples: 4000
- nearest-anchor set F1: 0.4326
- exact set: 0.0185
- Margin-F1@0.5/1.0/2.0: 0.1134 / 0.2503 / 0.4105
- mean scene-normalized error: 0.1569

## Overall Results

| variant | N | valid | anchor P | anchor R | anchor F1 | exact | M-F1@0.5 | M-F1@1.0 | M-F1@2.0 | scene norm err | same outputs as full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 4000 | 4000 | 0.4256 | 0.4399 | 0.4326 | 0.0185 | 0.1134 | 0.2503 | 0.4105 | 0.1569 | n/a |
| gaze_copy_reference | 4000 | 4000 | 0.4273 | 0.4379 | 0.4325 | 0.0435 | 0.1156 | 0.2547 | 0.4193 | 0.1519 | n/a |
| no_visual | 4000 | 4000 | 0.4261 | 0.4423 | 0.4341 | 0.0180 | 0.1144 | 0.2518 | 0.4120 | 0.1560 | 0.9840 |
| no_gaze | 4000 | 4000 | 0.1190 | 0.0469 | 0.0673 | 0.0013 | 0.0008 | 0.0055 | 0.0263 | 0.3771 | 0.0000 |
| no_hand | 4000 | 4000 | 0.4287 | 0.4421 | 0.4353 | 0.0187 | 0.1148 | 0.2527 | 0.4134 | 0.1558 | 0.9695 |
| no_gaze_hand | 4000 | 3999 | 0.0934 | 0.0382 | 0.0542 | 0.0000 | 0.0000 | 0.0000 | 0.0037 | 0.5376 | 0.0000 |
| no_instruction | 4000 | 4000 | 0.4251 | 0.4416 | 0.4332 | 0.0177 | 0.1137 | 0.2510 | 0.4111 | 0.1565 | 0.9865 |

`gaze_copy_reference` is the frozen deterministic gaze-copy baseline and is included because the v9 prompt explicitly exposes copyable measured gaze hypotheses.

## Target-Count Partitions

| variant | partition | N | anchor F1 | M-F1@1.0 | M-F1@2.0 | scene norm err |
|---|---|---:|---:|---:|---:|---:|
| full | single_target | 930 | 0.4361 | 0.2240 | 0.3654 | 0.0913 |
| full | multi_target | 3070 | 0.4320 | 0.2558 | 0.4200 | 0.1655 |
| gaze_copy_reference | single_target | 930 | 0.4391 | 0.2358 | 0.3912 | 0.0922 |
| gaze_copy_reference | multi_target | 3070 | 0.4313 | 0.2584 | 0.4249 | 0.1599 |
| no_visual | single_target | 930 | 0.4370 | 0.2249 | 0.3662 | 0.0909 |
| no_visual | multi_target | 3070 | 0.4335 | 0.2575 | 0.4216 | 0.1645 |
| no_gaze | single_target | 930 | 0.0435 | 0.0033 | 0.0132 | 0.3609 |
| no_gaze | multi_target | 3070 | 0.0705 | 0.0058 | 0.0282 | 0.3786 |
| no_hand | single_target | 930 | 0.4427 | 0.2274 | 0.3703 | 0.0909 |
| no_hand | multi_target | 3070 | 0.4339 | 0.2579 | 0.4223 | 0.1643 |
| no_gaze_hand | single_target | 930 | 0.0506 | 0.0000 | 0.0020 | 0.6230 |
| no_gaze_hand | multi_target | 3070 | 0.0548 | 0.0000 | 0.0040 | 0.5275 |
| no_instruction | single_target | 930 | 0.4358 | 0.2244 | 0.3653 | 0.0909 |
| no_instruction | multi_target | 3070 | 0.4327 | 0.2566 | 0.4207 | 0.1651 |

## Variant Definitions

- `no_visual`: sends no image tensors and removes image paths from the prompt; language, camera, gaze, and hand telemetry remain.
- `no_gaze`: removes gaze coordinates, validity, directions, copyable hypotheses, gaze-derived distances, and selection metadata; images, language, camera, and hand remain.
- `no_hand`: removes hand state, coordinates, directions, copyable hypotheses, hand-derived distances, and selection metadata; images, language, camera, and gaze remain.
- `no_gaze_hand`: removes both behavioral cue families and their derived metadata; images, language, and camera remain.
- `no_instruction`: removes instruction and utterance values while retaining the task instruction, images, camera, gaze, and hand.

## Interpretation Boundary

The v9 task prompt defaults to copying distinct measured gaze hypotheses and uses hand only as fallback. Therefore limited changes under `no_visual`, `no_hand`, or `no_instruction` are evidence that the frozen protocol is dominated by exposed gaze point hypotheses, not evidence that those modalities are generally unnecessary for referential grounding. Conversely, degradation under `no_gaze` measures dependence on model-visible gaze hypotheses under this protocol.

No bootstrap significance test is included. These results must not be described as unconstrained 3D reconstruction, 3D box grounding, or strict single-modality causal attribution.

## Validation

Every variant passed exact sample-set, unique-key, variant-label, and raw prompt-mask validation. Invalid model outputs remain empty predictions in the 4000-sample denominator. Machine-readable files in this directory contain overall/single/multi metrics and the full input audit.

The only invalid output is `scene4_room1::32` under `no_gaze_hand`: the model returned six values per point (`point_entry_0_wrong_dimension`). It is retained as an empty prediction in the 4,000-sample denominator and was not manually repaired.

Compact sample-level exports under `paper_experiment_evidence/ablation/experiment3_qwen30b/` merge GT, parsed point hypotheses, and evaluator detail without including model prompts or raw response text. Their hashes and run settings are recorded in `compact_evidence_validation.json` and `run_provenance.csv`.
