# Strict Hand-Visual Ablation

This document defines the strict hand ablation used for Experiments 1 and 2.
The previous `no_hand` variants are retained as descriptive prompt/telemetry
ablations and are not relabeled as strict visual ablations.

## Definition

For each frozen visual input, tracked left/right hand joints from the original
multimodal telemetry are projected into the camera image using the camera
position, camera rotation, camera FOV, and the image dimensions. The projected
joint extent is expanded by a depth-aware margin and filled with neutral gray.
Only the resulting masked image is passed to the multimodal processor. The
original videos and baseline outputs are never modified.

The mask is an expanded projected-joint bounding box. It is intentionally
conservative so that rendered hand pixels and nearby gesture cues are removed;
off-screen or unprojectable tracked hands are recorded in the audit rather than
silently treated as visible evidence.

## Experiment 1

Experiment 1 keeps the original video input protocol and the original video
frame sampling parameters (`max_video_frames=16`, deterministic decoding). The
sampled frames are loaded in memory, masked one by one, and then supplied as
the video input to Qwen3-VL. No re-encoded replacement video is used.

Video timestamps are mapped to telemetry sample timestamps using the frozen
Exp. 2 manifest's `video_time_offset_seconds` for the same source video. The
nearest telemetry sample is selected for each sampled frame. Each prediction
JSON records the offset source and a per-frame mask audit.

Run variant: `no_hand_strict`

## Experiment 2

Experiment 2 reuses the original frozen `manifest_all.csv`, panel selection,
panel ordering, panel dimensions, prompt mode, decoding parameters, and
coordinate mapping. Each source panel is masked using the telemetry sample at
its existing `json_sample_time`. The masked panel is created before any optional
gaze crop or paired-panel composition. Thus full-panel geometry and normalized
prediction coordinates remain unchanged.

Run variant: `no_hand_strict`, with `ABLATE_MODALITIES=hand_visual`

Each per-event JSON records the masked panel path, mask status, mask fraction,
and nearest telemetry sample time.

## Prompt and Evaluation Controls

Experiment 1 removes hand summaries, hand-ray coordinates, hand-derived
timeline counts, and hand-related source options from the strict prompt. The
strict visual masking is an additional input transformation; it does not
change the candidate anchor list, evaluator, parser, decoding mode, or model
checkpoint. Experiment 2 has no independent hand telemetry field in its
baseline prompt, so its strict variant changes the visual input only and keeps
the prompt identical to the baseline.

Both variants use the existing experiment-specific parser and evaluator. The
strict variant is evaluated separately from the old descriptive variants.

## Smoke Validation

The Exp. 1 smoke run used 10 scene1 rows. All 10 predictions completed without
errors; every extracted frame had `status=masked`, and no hand coordinate or
hand summary field appeared in the prompt text. The Exp. 2 smoke run used 10
scene1 rows and 3 panels per row. All 30 panels were masked and all 10 model
outputs parsed successfully. These smoke numbers are process checks only and
must not be used as full-dataset paper results.

Full-run output roots:

- `ablation/exam1/outputs_strict_hand_v1_full/no_hand_strict/`
- `ablation/exam2/outputs_strict_hand_v1_full/no_hand_strict/`

## Full-run qualification

The full runs completed, but the v1 mask did not pass semantic visual
qualification. A manual comparison of the recorded source and model-input
panels found visible rendered hands outside the projected rectangles in
multiple samples, while some other rectangles covered most of the scene. The
full-run metrics are preserved as an engineering diagnostic, not as a
paper-ready strict causal modality result. See
`paper_experiment_evidence/ablation/STRICT_HAND_ABLATION_AUDIT.md` for the
sample IDs, mask fractions, and delivery files.
