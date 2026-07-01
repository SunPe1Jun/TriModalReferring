# Ablation Plan

## Scope

This ablation round covers only:

1. Closed-set 3D anchor selection.
2. Projected-2D point diagnostic.

Camera-centered 3D directional point diagnostic is not included.

## Experiment 1

The candidate anchor list and evaluator remain fixed. Model-visible evidence is hidden through `ABLATE_MODALITIES`, which is propagated into the existing local Qwen3-VL single-event 3D runner.

Core variants:

- `no_visual`
- `no_gaze`
- `no_hand`
- `no_gaze_hand`
- `language_anchors_only`
- optional `no_structured_geometry`

Primary metrics:

- overall any-hit accuracy
- mapped-only accuracy
- exact-set accuracy
- micro precision / recall / F1
- per-scene summaries

## Experiment 2

The manifest, parser, and evaluator remain fixed except for variants that explicitly change panel count. The current experiment-2 interface does not carry explicit hand summaries, so no primary hand ablation is claimed here.

Core variants:

- `full_panels_no_crop`
- `no_gaze_text_prior`
- `no_gaze`
- `instruction_only_prompt`
- optional `single_panel`
- optional `blank_visual`

Primary metrics:

- time F1
- point@100 F1
- joint@100 F1
- point@50 / joint@50 as stricter diagnostics
- mean matched distance at 100 px

## Running Policy

Run smoke tests first, then full variants after prompt/parser sanity checks.

Two A100s can be used by binding experiment 1 to GPU 0 and experiment 2 to GPU 1 through `CUDA_VISIBLE_DEVICES`.
