# Paper Handoff: Experiment 3 Camera-Centered 3D Directional Diagnostic

This document is intended for a local Codex or writing assistant that will help revise the VR-TriRef paper. It explains what experiment 3 does, how it reuses the existing code/data pipeline, what should be reported, and what should not be overclaimed.

## One-Sentence Purpose

Experiment 3 evaluates whether a multimodal LLM can predict the correct camera-centered 3D referring direction, rather than the exact 3D depth, by comparing the direction from the selected camera position to the model's predicted 3D point against the direction from the same camera position to the GT anchor.

## Relationship To The Other Experiments

The paper has three related diagnostics:

1. Closed-set 3D anchor selection:
   - The model chooses a referent from scene-level candidate 3D anchors.
   - This is a discrete object/anchor selection task.

2. Projected-2D point diagnostic:
   - GT 3D anchors are projected into first-person image panels.
   - This evaluates temporal evidence selection, point@K, and joint@K on selected evidence panels.

3. Camera-centered 3D directional point diagnostic:
   - The model outputs a free-form 3D world point, `point_3d = [x, y, z]`.
   - Evaluation ignores exact depth and compares only the direction from the selected camera pose to the predicted point against the direction to the GT anchor.
   - This is the final experiment documented here.

The main reason for experiment 3 is that asking an LLM to recover exact metric 3D depth is fragile. A prediction can be directionally correct but have a poor depth estimate. The camera-centered angular metric isolates whether the model points in the correct 3D direction.

## Final Version Used

Use V3 as the final experiment 3 version.

- Prompt: `exam3/prompts/camera_centered_3d_directional_prompt_v3.md`
- Runner wrapper: `exam3/run_qwen3vl_30b_3d_directional.sh`
- Inference script: `exam3/run_qwen3vl_3d_directional.py`
- Evaluation script: `exam3/evaluate_3d_directional.py`
- Full output root: `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- Result report: `exam3/RESULTS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
- Statistics summary: `exam3/STATS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
- Iteration log: `exam3/ITERATION_LOG_3D_DIRECTIONAL.md`

V4 was considered because V3 has some wrong-dimension JSON outputs, but the user decided not to run V4. The paper should report V3 honestly, including invalid outputs.

## Data Interface

The experiment reuses the experiment 2 v10 manifest:

- `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv`

Each sample is one referential interaction group keyed by `(scene, row_index)`.

The final V3 full run uses the evaluable subset:

- manifest event groups: 4000
- skipped groups without valid GT anchors: 29
- evaluated samples: 3971

The 29 skipped groups are excluded before inference because they do not contain valid GT anchor coordinates and therefore cannot be scored by the angular metric. This is not the same as dropping invalid model outputs. Invalid model outputs inside the 3971 evaluated samples remain in the all-sample denominator.

## Inputs Given To The Model

For each evaluated sample, the prompt and model input contain:

- one selected evidence frame/panel image;
- the referential instruction text;
- event-level gaze summary;
- event-level hand summary;
- selected evidence sample gaze/camera hit cues;
- camera world position and rotation for the evidence frame;
- scene-level candidate anchor table in world coordinates.

Candidate anchors come from:

- `data/*_anchor_table.tsv`

GT anchors come from valid mapped anchor coordinates serialized in the experiment 2 manifest.

## Evidence Panel Selection

Experiment 3 reuses the projected-2D diagnostic evidence panel selection. The selected evidence frame is not chosen using GT identity or model output.

The final rule is `highest_score`:

1. Keep extracted panels whose frame files exist.
2. Choose the panel with the largest `panel_selection_score`.
3. Break ties by panel index and frame time.

The camera position `c_i` used for evaluation is the camera world position from the multimodal sample nearest to the selected panel's `json_sample_time`.

Important paper wording:

- It is safe to say the experiment uses a selected first-person evidence frame/panel from the interaction.
- Do not claim the final V3 run feeds a full continuous video into the model. The current implementation uses one selected evidence image/panel per sample.

## Prompt Design

The V3 prompt requires strict JSON:

```json
{
  "point_3d": [x, y, z],
  "reason": "short reason"
}
```

The prompt explicitly tells the model:

- all coordinates are scene/world coordinates;
- candidate anchors, camera pose, gaze/hand cues, and `point_3d` use the same coordinate system;
- prefer outputting the exact coordinate of one candidate anchor matching the referent;
- for parts, surfaces, edges, handles, tabletops, bases, sides, and doorways, output the parent object candidate anchor;
- do not output gaze points, hand ray hits, camera hit points, the camera position, homogeneous coordinates, direction vectors, two-point extents, or bounding boxes;
- keep the reason short and do not produce free-form explanatory text.

V3 was introduced after V2 still produced one six-number edge/surface output on a targeted smoke set. V3 strengthened the prompt against edge extents and part/surface outputs.

## Parser And Invalid Handling

The parser is in `exam3/run_qwen3vl_3d_directional.py`.

Parsing rules:

- extract a JSON object from the model response;
- read `point_3d`;
- accept either `[x, y, z]` or an object with `x`, `y`, `z`;
- require exactly three finite numeric values;
- reject missing values, NaN, Inf, non-numeric values, and wrong dimensions.

Invalid outputs are saved with:

- raw model output;
- parsed JSON if any;
- `parse_ok=False`;
- `invalid_reason`.

Invalid predictions are not manually modified and are not removed from the all-sample denominator. In V3 full, all invalid outputs are:

- `point_3d_wrong_dimension`: 110

## Evaluation Formula

For sample `i`:

- selected evidence camera world position: `c_i`
- GT anchor world coordinate: `g_i`
- model predicted world point: `p_hat_i`

The prediction direction and GT direction are:

```text
u_i = normalize(p_hat_i - c_i)
v_i = normalize(g_i - c_i)
```

The angular error is:

```text
theta_i = arccos(clip(u_i dot v_i, -1, 1))
```

Report `theta_i` in degrees.

Equivalent interpretation: project the predicted point onto the sphere centered at `c_i` with radius `||g_i - c_i||`, then compare its angular displacement from the GT anchor direction. This evaluates 3D direction, not precise depth.

If a sample has multiple valid GT anchors, the evaluator computes the angular error to every valid GT anchor and uses the minimum angle for that sample. It also retains the matched GT anchor id in the detail CSV.

## Metrics

Report:

- total evaluated samples;
- valid prediction count;
- invalid count;
- valid rate;
- mean angular error over valid predictions;
- median angular error over valid predictions;
- angular accuracy at 5, 10, 15, 20, and 30 degrees;
- per-scene/per-partition results.

The main paper-facing tolerance range is 5 to 20 degrees. Treat `@30` as a loose supplementary reference only, not as the main success criterion.

Mean and median angular error are valid-only because invalid outputs have no defined angle. Accuracy metrics have both all-sample and valid-only variants; the paper should prefer all-sample accuracy when discussing end-to-end model behavior.

## Final V3 Results

Overall V3 full evaluable subset:

| metric | value |
|---|---:|
| evaluated samples | 3971 |
| valid predictions | 3861 |
| invalid predictions | 110 |
| valid rate | 97.23% |
| mean angular error, valid-only | 14.53 deg |
| median angular error, valid-only | 8.97 deg |
| accuracy @5 deg, all samples | 43.24% |
| accuracy @10 deg, all samples | 49.86% |
| accuracy @15 deg, all samples | 58.78% |
| accuracy @20 deg, all samples | 68.55% |
| accuracy @30 deg, all samples | 79.65% |

Per-scene V3 full:

| partition | samples | valid | valid rate | mean deg | median deg | @20 all |
|---|---:|---:|---:|---:|---:|---:|
| scene1 | 800 | 798 | 99.75% | 13.86 | 10.44 | 71.25% |
| scene2 | 791 | 779 | 98.48% | 18.64 | 15.59 | 59.67% |
| scene3 | 800 | 764 | 95.50% | 21.92 | 19.42 | 48.75% |
| scene4_room1 | 186 | 169 | 90.86% | 12.91 | 0.00 | 75.27% |
| scene4_room2 | 200 | 177 | 88.50% | 2.04 | 0.00 | 85.50% |
| scene4_room3 | 199 | 180 | 90.45% | 4.30 | 0.00 | 83.92% |
| scene4_room4 | 199 | 198 | 99.50% | 2.47 | 0.00 | 96.98% |
| scene5 | 796 | 796 | 100.00% | 12.50 | 7.46 | 77.76% |

## Failure Modes

1. Schema failures:
   - 110 outputs have the wrong `point_3d` dimension.
   - These are counted as invalid and remain in the denominator.
   - Do not silently truncate four-number or six-number outputs in the paper.

2. Scene-dependent difficulty:
   - Scene3 is the hardest partition, with median 19.42 deg and @20 all 48.75%.
   - Scene4 rooms often have exact-anchor median 0 deg but lower valid rate because of schema failures.

3. Wrong referent direction:
   - Some large valid angular errors indicate the model chose a different plausible object/anchor direction.
   - Worst examples include `scene2 row_5`, `scene3 row_383`, `scene3 row_370`, and `scene4_room4 row_188`.

4. Multi-referent ambiguity:
   - The metric evaluates one predicted point.
   - For multiple GT anchors, it gives credit for the closest valid GT direction.
   - It does not evaluate complete set recovery.

## Suggested Paper Wording

Possible concise methods wording:

> To evaluate whether the model recovers the 3D direction of a referent rather than exact metric depth, we introduce a camera-centered directional point diagnostic. For each referential interaction, we reuse the evidence panel selected by the projected-2D diagnostic and obtain the corresponding camera world position. The model is prompted with the instruction, the selected first-person image, gaze and hand summaries, camera pose, and candidate scene anchors, and must output a JSON object containing a single 3D world point. We compare the direction from the selected camera position to the predicted point with the direction from the same camera position to the GT anchor, reporting angular error and accuracy at 5, 10, 15, and 20 degrees. Invalid JSON or non-3D outputs remain in the all-sample denominator.

Possible concise results wording:

> On the 3971-sample evaluable subset, Qwen3-VL-30B achieves a valid-only median angular error of 8.97 degrees and all-sample angular accuracies of 43.24%, 49.86%, 58.78%, and 68.55% at 5, 10, 15, and 20 degrees, respectively. These results indicate that the model often recovers the correct camera-centered referential direction even though exact 3D depth is not directly evaluated. The model produces 110 invalid wrong-dimension outputs, which are retained in the denominator and reported as a schema-compliance limitation.

Suggested caution:

> We do not interpret the 30-degree threshold as high-confidence grounding; it is reported only as a loose reference. The main diagnostic range is 5 to 20 degrees.

## What Not To Claim

- Do not claim V3 uses full continuous video input. It uses one selected evidence frame/panel per interaction.
- Do not claim invalid outputs were fixed, filtered, or manually cleaned.
- Do not use `@30` as the main success result.
- Do not describe the metric as evaluating exact depth or full 3D reconstruction.
- Do not claim multi-object instructions are fully recovered; the metric uses the nearest valid GT direction for one predicted point.

## Reproducibility Pointers

Run command pattern:

```bash
PYTHON_BIN=/workspace/usr3/miniconda3/envs/trimodal/bin/python \
MODEL_TAG=qwen3vl30b_3d_directional_v3_full_evaluable \
OUTPUT_ROOT=/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable \
PROMPT_TEMPLATE=/workspace/usr3/TriModal-Referring/exam3/prompts/camera_centered_3d_directional_prompt_v3.md \
LIMIT= \
OVERWRITE=1 \
CONTINUE_ON_ERROR=1 \
SKIP_MISSING_GT_ANCHORS=1 \
bash exam3/run_qwen3vl_30b_3d_directional.sh
```

Evaluation output files:

- `eval/3d_directional_eval_summary.json`
- `eval/3d_directional_eval_detail.csv`
- `eval/3d_directional_eval_by_scene.csv`
- `report.md`

The generated output directories and logs are intentionally gitignored; the committed paper-facing summaries are:

- `exam3/RESULTS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
- `exam3/STATS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
- `exam3/PAPER_HANDOFF_EXPERIMENT3_3D_DIRECTIONAL.md`
