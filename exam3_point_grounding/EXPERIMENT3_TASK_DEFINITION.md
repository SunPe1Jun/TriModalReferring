# Experiment 3 Task Definition: Candidate-Free Point-Supervised 3D Referent Grounding

This document describes the current Experiment 3 protocol for the VR-TriRef paper. It is intended for the paper task-definition and method sections. Numeric results should be taken only from the generated evaluation summaries, not from this document.

## Position in the Paper

Experiments 1 and 2 evaluate referent grounding with progressively different supervision and output spaces:

- Experiment 1, closed-set 3D anchor selection: the model chooses from scene-level candidate anchors.
- Experiment 2, projected-2D point diagnostic: ground-truth 3D anchors are projected onto egocentric evidence panels, and the model is evaluated on 2D point localization and temporal evidence selection.
- Experiment 3, candidate-free point-supervised 3D grounding: candidate anchors are hidden from the model. The model receives language, egocentric evidence frames, and measured VR telemetry, then outputs one or more 3D world-coordinate points.

The current Experiment 3 should be framed as a target-free measured point-hypothesis diagnostic, not as unconstrained 3D reconstruction from images. The model is not asked to recover a full 3D object extent or a 3D box. It predicts 3D point hypotheses in the released Unity world coordinate system, and candidate anchors are used only after inference by the evaluator.

## Task Input

Each sample corresponds to one referential interaction event from the existing scene API CSVs:

- `data/{scene}_api_input.csv`
- raw multimodal JSON referenced by each API row
- first-person video referenced by each API row
- scene anchor table `data/{scene}_anchor_table.tsv`, used only for evaluation and broad scene-scale auditing
- match-eval CSVs under `data/match_eval_qwen3vl30b_mention_first_v3/`, used to build evaluator-side ground truth

For a sample `i`, the model-facing input is:

- instruction text and optional utterance text
- up to three chronological egocentric evidence frames
- camera pose telemetry for each evidence frame
- gaze and hand ray telemetry for each evidence frame
- robust scene coordinate bounds, used only as broad scale and sanity context

The model-facing prompt explicitly excludes:

- candidate anchor ids
- candidate anchor coordinates
- ground-truth anchor ids
- ground-truth anchor coordinates
- target-description fields that directly reveal the evaluator mapping
- projected-2D ground-truth fields from Experiment 2

## Evidence Frame Selection

Evidence frames are selected without using the ground-truth target anchor. The selector operates inside the event interval `[t_start, t_end]` from the API row.

The implementation samples candidate times every `0.5` seconds, using the nearest multimodal telemetry sample. Each candidate receives a target-free score from measured cue availability and local stability:

```text
score =
  1.5 + 0.8 * gaze_stability    if gaze is valid
+ 1.1 + 0.7 * hand_stability    if hand ray hit is valid
+ 0.3 + 0.4 * camera_stability  if camera pose is valid
```

Gaze and hand stability are measured over a local `0.3` second window with a `4.0` world-unit scale. Camera-position stability uses a `0.35` world-unit scale. The selector first tries to include a strong gaze-valid frame and a strong hand-valid frame, then fills remaining slots by score. Selected frames must be separated by at least `0.85` seconds. The final evidence panels are sorted chronologically and named `P1`, `P2`, and `P3`.

If no positive cue candidate is found, the selector falls back to frames near 25%, 50%, and 75% of the event interval. Frames are extracted from the video with an automatically estimated video-time offset from metadata and video filename timestamps.

## Coordinate System and Scale

All reported 3D coordinates use the released Unity world-coordinate convention:

```text
point = [x_world, y_world, z_world]
```

The implementation assumes that anchor coordinates, camera positions, gaze hits, and hand hits are already expressed in the same Unity world coordinate system. No additional coordinate transform is introduced.

The prompt provides the model with:

- `camera_position_world`
- camera basis vectors derived from `cameraRotation`
- `camera_forward_world_unit_vector_do_not_copy`
- `camera_right_world_unit_vector_do_not_copy`
- `camera_up_world_unit_vector_do_not_copy`
- camera FOV
- scene robust 5%-95% coordinate ranges
- robust scene diagonal in world units
- camera-to-gaze, camera-to-hand, and gaze-to-hand distances

The camera basis vectors and ray directions are marked as `do_not_copy`. They are provided to indicate orientation and scale, not as valid target points.

Unity world units are not independently calibrated as meters. The paper should therefore use "world units" unless a separate calibration is introduced later.

## Measured Point Hypotheses

For each selected panel, the manifest serializes two types of measured 3D point hypotheses:

```text
primary_copyable_gaze_point_hypotheses:
  P1_GAZE: [x, y, z]
  P2_GAZE: [x, y, z]
  P3_GAZE: [x, y, z]

secondary_copyable_hand_point_hypotheses:
  P1_HAND: [x, y, z]
  P2_HAND: [x, y, z]
  P3_HAND: [x, y, z]
```

The v9 prompt instructs the model to treat gaze points as the primary measured point hypotheses and hand points as a fallback when gaze is unavailable. This keeps the task candidate-free with respect to scene anchors while avoiding unsupported free-form coordinate invention.

Important interpretation boundary:

- Gaze and hand ray endpoints are behavioral pointing cues, not ground truth.
- Ray endpoints can land on floors, walls, background surfaces, or points behind the referent.
- The diagnostic evaluates whether these measured cue points can support 3D referent grounding when candidate anchors are hidden.
- The experiment should not be described as predicting 3D bounding boxes or object extents.

## Model Output

The model must return strict JSON only:

```json
{
  "points_3d": [
    {
      "referent": "P1_GAZE",
      "point": [1.0, 2.0, 3.0],
      "confidence": 0.7
    }
  ]
}
```

Rules:

- `points_3d` is a list and may contain zero or more predicted referent points.
- Each `point` must contain exactly three finite numeric values.
- The model must not output direction vectors, 2D points, anchor ids, camera basis vectors, ray origins, quaternions, boxes, or concatenated coordinates.
- The model may output multiple points for multi-referent interactions.
- If the evidence is insufficient, the model may return an empty `points_3d` list.

## Parser and Invalid Outputs

The parser extracts the first JSON object from the model response and validates the `points_3d` schema. An output is marked invalid if:

- no JSON object can be extracted
- `points_3d` is missing or is not a list
- any point entry is not an object
- any point is missing, non-numeric, non-finite, or not exactly length three
- confidence is present but not finite

Invalid outputs are saved with the raw model text and invalid reason. For end-to-end evaluation, invalid outputs are represented as an empty predicted point set. They are not removed from the denominator; the evaluator reports valid-output count, invalid-output count, and valid-output rate alongside all-sample metrics.

## Ground Truth

Ground truth is built from the existing match-eval mappings. For each event, the evaluator obtains a set of valid ground-truth anchors:

```text
G_i = {g_i1, g_i2, ..., g_iM}
```

where each `g_ij` is the 3D point from the scene anchor table for a mapped referent. Multi-answer events keep all mapped GT anchors. Samples without a valid GT anchor mapping are excluded from the evaluable model-facing manifest and counted in the manifest status summary.

Candidate anchors are not shown to the model. They are loaded only by the evaluator after inference.

## Evaluation

Let the model output a predicted point set:

```text
P_i = {p_i1, p_i2, ..., p_iN}
```

The evaluator reports several complementary metrics.

### Nearest-Anchor Set Metrics

Each predicted point is mapped to the nearest scene anchor:

```text
a(p) = argmin_a ||p - x_a||_2
```

The predicted anchor set is compared with the GT anchor set. The evaluator computes micro precision, recall, F1, exact-set accuracy, cardinality error, and duplicate nearest-anchor rate.

This metric answers: after hiding candidate anchors from the model, do the predicted 3D points land closest to the intended anchors?

### Margin-Normalized F1

For each GT anchor `g`, the evaluator computes a local ambiguity margin:

```text
margin(g) = 0.5 * min_{a not in GT} ||g - x_a||_2
```

Predicted points are matched to GT anchors with a minimum-cost assignment using normalized errors:

```text
e(p, g) = ||p - g||_2 / margin(g)
```

For thresholds `tau in {0.5, 1.0, 2.0}`, a matched prediction is a true positive if `e(p, g) <= tau`. Unmatched predictions are false positives and unmatched GT anchors are false negatives. The evaluator reports `Margin-F1@0.5`, `Margin-F1@1.0`, and `Margin-F1@2.0`.

This metric adapts the tolerance to local anchor density. A point must be closer to the GT anchor than to nearby distractors to receive credit.

### Scene-Normalized Point Error

The evaluator also reports matched Euclidean distance normalized by the robust scene diagonal:

```text
scene_norm_error(p, g) = ||p - g||_2 / robust_scene_diagonal
```

The robust scene diagonal is computed from the 5%-95% coordinate ranges of the scene anchor table. This metric provides a scale-comparable continuous localization error across scenes.

### Partitions

The evaluator writes overall, per-scene, single-target, and multi-target summaries. Main output files include:

- `evaluation_detail.csv`
- `evaluation_summary.csv`
- `evaluation_summary.json`
- `RESULTS_POINT_3D_GROUNDING.md`

## Baselines

The experiment includes deterministic cue-copy baselines:

- `gaze_copy`: copies distinct selected gaze-hit points.
- `hand_copy`: copies distinct selected hand-hit points.
- `gaze_hand_fusion`: averages nearby gaze and hand hits when they agree, then deduplicates.

These baselines are important because the v9 model prompt intentionally exposes measured point hypotheses. The paper should report them beside the VLM result to distinguish model reasoning from the strength of raw behavioral telemetry.

## Reproducibility Entry Points

Main script:

```bash
bash exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh
```

Key implementation files:

- `exam3_point_grounding/build_point_grounding_manifest.py`
- `exam3_point_grounding/select_target_free_evidence.py`
- `exam3_point_grounding/prompts/qwen3vl_point_grounding.md`
- `exam3_point_grounding/run_qwen3vl_point_grounding.py`
- `exam3_point_grounding/point_parser.py`
- `exam3_point_grounding/run_cue_baselines.py`
- `exam3_point_grounding/evaluate_point_grounding.py`

The full v9 run uses:

```text
OUTPUT_ROOT=exam3_point_grounding/outputs_full_v9_20260709
QWEN_OUTPUT_DIR=exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b
CUDA_VISIBLE_DEVICES=0
RUN_MANIFEST=1
RUN_BASELINES=1
RUN_QWEN=1
RUN_EVAL=1
OVERWRITE_FRAMES=0
OVERWRITE_INFERENCE=1
```

## Suggested Paper Wording

We define a candidate-free 3D point-supervised referent grounding diagnostic. Given a referential interaction, the model observes the instruction, a small set of target-free egocentric evidence frames, and synchronized camera, gaze, and hand telemetry in a shared Unity world coordinate system. Unlike the closed-set anchor-selection experiment, no candidate anchor list or ground-truth anchor coordinate is exposed to the model. The model returns a variable-size set of 3D world-coordinate points. After inference, each predicted point is evaluated against hidden scene anchors by nearest-anchor set matching, local-margin-normalized F1, and scene-normalized point error. Invalid JSON or malformed point outputs are retained in the denominator and treated as empty predictions.

## Current Limitations

- The task is a measured point-hypothesis diagnostic, not a proof of unconstrained 3D reconstruction.
- Unity world units are not independently verified as meters.
- Released supervision provides static scene-level anchor points, not time-varying object centers.
- Gaze and hand endpoints are noisy behavioral cues and may not coincide with object centers.
- The prompt is intentionally constrained to reduce invalid outputs and coordinate hallucination; this should be stated when interpreting VLM performance.
