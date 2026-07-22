# Experiment 3 Qwen3-VL-30B Input Ablation

## Scope

This is a descriptive model-input ablation of the frozen v9 candidate-free measured point-hypothesis diagnostic. All variants reuse the same evidence-frame selection, GT manifest, model checkpoint, parser, greedy decoding (`do_sample=false`, `max_new_tokens=512`), and evaluator. The current evaluation denominator is 4000 samples. Only model-visible input fields are masked.

Because the frozen target-free frame selector itself used gaze/hand availability and stability, `no_gaze`, `no_hand`, and `no_hand_strict` do not constitute strict causal pipeline interventions. `no_hand_strict` is a strict model-input hand ablation: it removes hand fields and masks projected hand regions in every input panel after panel selection.

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
| no_hand_strict | 4000 | 4000 | 0.4286 | 0.4418 | 0.4351 | 0.0187 | 0.1146 | 0.2525 | 0.4132 | 0.1558 | 0.9692 |

`gaze_copy_reference` is the frozen deterministic gaze-copy baseline and is included because the v9 prompt explicitly exposes copyable measured gaze hypotheses.

## Target-Count Partitions

| variant | partition | N | anchor F1 | M-F1@1.0 | M-F1@2.0 | scene norm err |
|---|---|---:|---:|---:|---:|---:|
| full | single_target | 930 | 0.4361 | 0.2240 | 0.3654 | 0.0913 |
| full | multi_target | 3070 | 0.4320 | 0.2558 | 0.4200 | 0.1655 |
| gaze_copy_reference | single_target | 930 | 0.4391 | 0.2358 | 0.3912 | 0.0922 |
| gaze_copy_reference | multi_target | 3070 | 0.4313 | 0.2584 | 0.4249 | 0.1599 |
| no_hand_strict | single_target | 930 | 0.4427 | 0.2274 | 0.3703 | 0.0909 |
| no_hand_strict | multi_target | 3070 | 0.4337 | 0.2577 | 0.4221 | 0.1643 |

## Variant Definitions

- `no_visual`: sends no image tensors and removes image paths from the prompt; language, camera, gaze, and hand telemetry remain.
- `no_gaze`: removes gaze coordinates, validity, directions, copyable hypotheses, gaze-derived distances, and selection metadata; images, language, camera, and hand remain.
- `no_hand`: removes hand state, coordinates, directions, copyable hypotheses, hand-derived distances, and selection metadata; images, language, camera, and gaze remain.
- `no_hand_strict`: removes the same hand fields as `no_hand` and replaces every selected panel with a hand-masked image generated from projected tracked hand joints; panel count and panel selection are unchanged.
- `no_gaze_hand`: removes both behavioral cue families and their derived metadata; images, language, and camera remain.
- `no_instruction`: removes instruction and utterance values while retaining the task instruction, images, camera, gaze, and hand.

## Interpretation Boundary

The v9 task prompt defaults to copying distinct measured gaze hypotheses and uses hand only as fallback. Therefore limited changes under `no_visual`, `no_hand`, or `no_instruction` are evidence that the frozen protocol is dominated by exposed gaze point hypotheses, not evidence that those modalities are generally unnecessary for referential grounding. Conversely, degradation under `no_gaze` measures dependence on model-visible gaze hypotheses under this protocol.

No bootstrap significance test is included. These results must not be described as unconstrained 3D reconstruction, 3D box grounding, or strict single-modality causal attribution.

## Validation

Every variant passed exact sample-set, unique-key, variant-label, and raw prompt/mask validation. Invalid model outputs remain empty predictions in the 4000-sample denominator. Machine-readable files in this directory contain overall/single/multi metrics and the full input audit.

Invalid outputs: none. They are retained as empty predictions and were not manually repaired.

## Hand Mask Audit

The strict run used 11,001 panels from 4,000 samples. 9,323 panels had in-frame tracked hand joints and received a neutral-fill mask; 1,678 tracked-hand projections were fully off-screen. The mean mask fraction was 0.1138777 and the maximum panel-to-telemetry time error was 0 seconds. The path-free audit is `paper_experiment_evidence/ablation/experiment3_qwen30b_strict_hand/hand_mask_audit.csv`.

Compact sample-level exports under `paper_experiment_evidence/ablation/experiment3_qwen30b_strict_hand/` merge GT, parsed point hypotheses, and evaluator detail without including model prompts or raw response text. Their hashes and run settings are recorded in `compact_evidence_validation.json` and `run_provenance.csv`.
