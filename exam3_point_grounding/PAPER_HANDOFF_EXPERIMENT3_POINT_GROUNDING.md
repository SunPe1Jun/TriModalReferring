# Paper Handoff: Experiment 3 Point-Supervised 3D Grounding

## Current Status

The old camera-centered angular diagnostic in `exam3/` is archived and should not be used as the final Experiment 3 result.

The replacement experiment lives in `exam3_point_grounding/` and implements candidate-free point-supervised 3D referent grounding.

## Paper Claim Boundary

Use:

- point-supervised 3D referent grounding;
- 3D anchor-point grounding;
- candidate-free 3D point grounding.

Do not use:

- 3D box grounding;
- 3D IoU;
- object-extent localization;
- object-size-aware grounding;
- ScanRefer-style box evaluation.

## Method Summary

The model sees instruction text, up to three target-free evidence frames, and measured camera/gaze/hand telemetry in Unity world coordinates. It predicts a variable-size set of 3D points. Candidate anchors are hidden from the model and are used only by the evaluator after inference.

## Metrics

- Candidate-free nearest-anchor set precision/recall/F1/exact.
- Nearest-distractor Margin-F1@0.5, @1.0, @2.0.
- Robust scene-normalized matched point distance.
- Valid-output rate and invalid-output reason counts.

## Required Before Paper Edits

1. Generate `DATA_AUDIT.md` by running the manifest builder.
2. Inspect at least 20 prompts for leakage.
3. Run cue baselines.
4. Run a 10-sample-per-scene Qwen smoke test.
5. Freeze prompt and thresholds.
6. Run full Qwen inference.
7. Use only measured `evaluation_summary.json` values in the paper.

No numeric result should be copied from the archived angular diagnostic into the new Experiment 3 section.
