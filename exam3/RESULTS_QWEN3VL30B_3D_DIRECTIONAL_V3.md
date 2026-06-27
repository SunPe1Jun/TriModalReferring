# Camera-Centered 3D Directional Point Diagnostic - V3 Full Results

## Run Identity

- Model: Qwen3-VL-30B-A3B-Instruct
- Prompt: `exam3/prompts/camera_centered_3d_directional_prompt_v3.md`
- Runner: `exam3/run_qwen3vl_30b_3d_directional.sh`
- Prediction script: `exam3/run_qwen3vl_3d_directional.py`
- Evaluation script: `exam3/evaluate_3d_directional.py`
- Output root: `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- Full-run log: `exam3/qwen3vl30b_3d_directional_v3_full_evaluable.log`
- Evaluation summary: `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/eval/3d_directional_eval_summary.json`
- Evaluation detail: `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/eval/3d_directional_eval_detail.csv`
- Per-scene summary: `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/eval/3d_directional_eval_by_scene.csv`

## Data Inputs

The run reuses the experiment 2 v10 manifest:

- `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv`

Each sample is one referential interaction group keyed by `(scene, row_index)`. V3 was run on the evaluable subset with `--skip_missing_gt_anchors`, which selected 3971 samples after skipping 29 groups without valid GT anchor coordinates. This is a data-interface exclusion for samples that cannot be scored by the directional metric; invalid model outputs within the 3971 evaluated samples are not removed from the denominator.

Candidate anchors are loaded from the scene anchor tables under `data/*_anchor_table.tsv`. GT anchors are the valid mapped anchor coordinates serialized in the exam2 manifest.

## Evidence Panel And Camera Rule

Each sample uses one evidence panel from the projected-2D diagnostic manifest. The selection rule is `highest_score`:

1. Keep extracted panels whose frame files exist.
2. Choose the panel with the largest `panel_selection_score`.
3. Break ties by panel index and frame time.

The camera position `c_i` is the camera world position from the multimodal sample nearest to the selected panel's `json_sample_time`. The evidence rule reuses experiment 2's diagnostic panel selection and does not use GT anchor identity or model output.

## Prompt And Output

The prompt requires strict JSON:

```json
{
  "point_3d": [x, y, z],
  "reason": "short reason"
}
```

The prompt explicitly states that all coordinates are scene/world coordinates and that candidate anchors, camera pose, gaze/hand cues, and `point_3d` share the same coordinate system. It tells the model to prefer a candidate anchor coordinate, to map parts/surfaces/edges to the parent object anchor, and to avoid gaze points, hand hits, homogeneous coordinates, direction vectors, bounding boxes, or two-point edge extents.

## Evaluation Formula

For sample `i`:

- camera position: `c_i`
- GT anchor coordinate: `g_i`
- predicted 3D point: `p_hat_i`

The evaluated directions are:

```text
u_i = normalize(p_hat_i - c_i)
v_i = normalize(g_i - c_i)
theta_i = arccos(clip(u_i dot v_i, -1, 1))
```

`theta_i` is reported in degrees. If a sample has multiple valid GT anchors, the evaluator computes the angular error to every valid GT anchor and uses the minimum angle, retaining the matched GT anchor id in the detail CSV.

Invalid model outputs are kept in the all-sample denominator. Mean and median angular error are valid-only because invalid outputs have no defined angle.

## Overall Results

V3 full evaluable subset:

| metric | value |
|---|---:|
| total samples | 3971 |
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

The main acceptance range for the paper-facing directional diagnostic is 5 to 20 degrees. Under that criterion, V3 is usable: all-sample `@20` is 68.55%, and the valid-only median angular error is 8.97 degrees. `@30` is retained only as a loose reference metric.

The original engineering valid-rate gate of 98% is not met because 110 outputs have the wrong `point_3d` dimension. Per the final experiment decision on 2026-06-27, no V4 iteration is run; V3 is adopted and the schema failures are reported as a limitation rather than manually repaired.

## Per-Scene Results

| partition | samples | valid | valid rate | mean deg | median deg | @5 all | @10 all | @15 all | @20 all | @30 all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| scene1 | 800 | 798 | 99.75% | 13.86 | 10.44 | 44.50% | 48.50% | 66.38% | 71.25% | 79.75% |
| scene2 | 791 | 779 | 98.48% | 18.64 | 15.59 | 24.15% | 37.93% | 48.80% | 59.67% | 75.73% |
| scene3 | 800 | 764 | 95.50% | 21.92 | 19.42 | 22.88% | 28.25% | 35.62% | 48.75% | 69.88% |
| scene4_room1 | 186 | 169 | 90.86% | 12.91 | 0.00 | 58.60% | 60.75% | 66.13% | 75.27% | 77.96% |
| scene4_room2 | 200 | 177 | 88.50% | 2.04 | 0.00 | 84.50% | 85.50% | 85.50% | 85.50% | 85.50% |
| scene4_room3 | 199 | 180 | 90.45% | 4.30 | 0.00 | 79.40% | 80.90% | 82.41% | 83.92% | 86.43% |
| scene4_room4 | 199 | 198 | 99.50% | 2.47 | 0.00 | 93.47% | 93.97% | 95.98% | 96.98% | 97.49% |
| scene5 | 796 | 796 | 100.00% | 12.50 | 7.46 | 45.85% | 54.52% | 60.68% | 77.76% | 86.06% |

## Main Failure Modes

- Schema failures: all 110 invalid outputs are `point_3d_wrong_dimension`, typically an array with four or more values instead of exactly three finite numbers. These are counted as invalid and remain in the all-sample denominator.
- Scene-dependent directional errors: scene3 is the hardest partition by both median angle and `@20`, while scene4 rooms often have near-zero median error but lower valid rates due to wrong-dimension outputs.
- Wrong referent direction: the largest valid angular errors are cases where the model points toward a different plausible candidate anchor. Examples include `scene2 row_5`, `scene3 row_383`, `scene3 row_370`, and `scene4_room4 row_188`.
- Multi-referent instructions: the evaluator uses the nearest valid GT direction for a single predicted point, so the metric does not measure complete recovery of all mentioned objects.

## Current Limitations

- The diagnostic uses one selected evidence panel rather than a continuous video input to the model.
- GT is anchor-based and can be coarse for parts, surfaces, grouped objects, and large objects.
- The metric intentionally evaluates direction from the selected camera frame, not precise depth recovery.
- Invalid outputs are not manually corrected. The V3 result therefore reflects both directional grounding quality and remaining JSON/schema compliance errors.

## Final Decision

V3 is adopted as the final camera-centered 3D directional point diagnostic result. Further V4 iteration is stopped by decision on 2026-06-27 after the V3 full run produced acceptable directional metrics under the 5 to 20 degree evaluation range.
