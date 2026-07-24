# Strict Hand-Input Ablation Audit

## Status

The three Qwen3-VL-30B `no_hand_strict` runs completed and their technical
artifacts are internally consistent. The strict **model-input intervention is
feasible**: hand telemetry is removed from the model-facing input and the
actual image/video tensors are replaced by hand-masked inputs before
processing. The semantic visual audit is **not pixel-perfect**: the current
`hand_mask_v1` projected-joint masks do not remove every visible rendered hand
in the inspected panels.

This distinction is intentional. The run verifies the requested strict input
ablation method and its end-to-end execution, while recording the limitation
that the current telemetry-to-image projection is not a validated hand
segmentation. It should not be described as an end-to-end causal intervention
on panel selection or as proof of pixel-perfect visual removal.

## Technical Checks Passed

- Exp.1: 4,000 sample predictions, 60,988 sampled video frames, and a mask
  audit for every prediction. The strict runner applies `mask_pil_image` to
  each decoded frame before constructing the video tensor. Structured hand
  telemetry fields were absent from the strict prompt scan.
- Exp.2: 4,000 sample predictions and 11,834 panel audits. Every recorded
  model-input image path points to the saved `hand_masked` directory, and all
  recorded files exist.
- Exp.3: 4,000 valid predictions. Its existing compact evidence audit reports
  11,001 selected panels and 9,323 panels with in-frame projected tracked
  joints.
- All three sample sets contain the same 4,000 `scene::row_index` identifiers;
  no raw `event_id` is used as the global key.
- Existing raw model outputs were not edited.

Machine-readable checks are in
`strict_hand_validation.json`. The complete sample-level exports are under
`experiment1_qwen30b_strict_hand/`,
`experiment2_qwen30b_strict_hand/`, and the existing
`experiment3_qwen30b_strict_hand/` directory.

## Semantic Visual Audit

The committed contact sheet is
`../qualitative_cases/strict_hand_mask_examples.jpg`; it compares the source
panel with the actual image passed to the model. The following observations
were made from the recorded v1 model-input images:

| Sample | Mask fraction | Observation |
| --- | ---: | --- |
| `scene5::366`, `P1` | 0.0079 | A visible lower-centre hand remains outside the gray mask. |
| `scene3::40`, `P3` | 0.1444 | A large left-side hand remains partly outside the central mask. |
| `scene3::126`, `P2` | 0.2152 | The projected rectangle covers the centre, while visible hand pixels remain near the lower/left edge. |
| `scene1::278`, `P2` | 0.3838 | The central rectangle does not cover the complete lower-left hand. |
| `scene3::65`, `P3` | 0.8922 | The mask is very broad; it removes much of the scene and is not an isolated hand intervention. |

The quantitative audit also shows the collateral-occlusion risk. Exp.1 has
3,331/60,988 frames with more than 25% of pixels masked and 204 above 50%.
Exp.2 has 358/11,834 panels above 25% and 48 above 50%.

## Interpretation

The v1 implementation is an image-level hand-input occlusion attempt, not a
validated hand segmentation. Its expanded rectangular boxes are derived from
world-to-camera projection of telemetry joints. The audit indicates that the
rendered hand and telemetry projection are not perfectly aligned for all
scene/time samples; an off-screen or poorly aligned tracked hand can remain
visible, while the conservative rectangle can remove unrelated scene content.

Consequently:

1. The method and full-run numbers may be reported as a **strict model-input
   hand ablation feasibility result**.
2. The paper should state that the mask is a projected-joint rectangular
   occlusion and that residual visible-hand pixels and collateral occlusion
   are limitations; it should not claim complete pixel-level hand removal.
3. A stronger pixel-complete rerun would require a validated image-space hand detector or a
   calibrated hand-render projection. It should first pass a manually checked
   smoke set, then rerun all three experiments with a new mask version and
   preserve this v1 run unchanged.
