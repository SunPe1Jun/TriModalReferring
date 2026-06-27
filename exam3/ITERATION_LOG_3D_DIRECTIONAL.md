# Camera-Centered 3D Directional Diagnostic Iteration Log

This log records monitoring, failure analysis, and every implementation or prompt iteration for experiment 3. It is meant to be auditable: no model outputs are manually edited, and invalid outputs remain in the all-sample denominator.

## Acceptance Criteria

Current engineering thresholds for deciding whether the diagnostic is good enough to keep:

- valid prediction rate >= 98%
- median angular error <= 15 degrees
- angular accuracy @20 degrees, all-sample denominator >= 60%

These thresholds are not paper claims. They are working criteria for deciding whether to iterate.

## Baseline V1 Full Run

- output root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- run log: `/workspace/usr3/TriModal-Referring/exam3/qwen3vl30b_3d_directional_v1_full.log`
- prompt template: `/workspace/usr3/TriModal-Referring/exam3/prompts/camera_centered_3d_directional_prompt.md`
- evidence panel rule: `highest_score`, reusing the exam2 v10 panel with the highest `panel_selection_score`
- model: `/workspace/usr3/Qwen3-VL-30B-A3B-Instruct`
- launch mode: `setsid` background process so SSH disconnects do not terminate inference
- status at log creation: running
- progress at log creation: 3412 raw outputs. The original monitor denominator used manifest rows; this was later corrected to unique `(scene,row_index)` inference samples

Observed issue before full-run completion:

- `scene5_row_223` parsed invalid with `point_3d_wrong_dimension`.
- Raw output had four numbers in `point_3d`: `[2.48, -0.24, 16.0, 0.0]`.
- This confirms the parser is enforcing the schema correctly. It also suggests a potential prompt-only iteration if invalid rate is material.

Policy:

- Do not modify the current V1 full-run prompt or outputs while the baseline is running.
- When V1 finishes, evaluate with `exam3/evaluate_3d_directional.py`.
- If any acceptance criterion fails, inspect failure modes and run a new smoke test before launching any new full run.

### Monitor Implementation Note - 2026-06-26

- The first monitor version counted raw manifest CSV rows as the expected total.
- This was incorrect because experiment 3 groups the manifest by `(scene,row_index)` before inference.
- `exam3/monitor_3d_directional_run.py` was updated to count unique `(scene,row_index)` groups.
- Existing model outputs were not changed.

### Monitor Update - 2026-06-26 21:29:52 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3494 / 25887 (13.50%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_322 parse_ok=True`

### Monitor Update - 2026-06-26 21:30:32 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3502 / 25887 (13.53%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_330 parse_ok=True`

### Monitor Update - 2026-06-26 21:40:33 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3628 / 25887 (14.01%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_456 parse_ok=True`

### Monitor Update - 2026-06-26 21:50:33 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3756 / 25887 (14.51%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_584 parse_ok=True`

### Monitor Update - 2026-06-26 21:52:46 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3784 / 4000 (94.60%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_612 parse_ok=True`

### Monitor Update - 2026-06-26 21:57:01 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3838 / 4000 (95.95%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_666 parse_ok=True`

### Monitor Update - 2026-06-26 22:02:01 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3901 / 4000 (97.52%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_729 parse_ok=True`

### Monitor Update - 2026-06-26 22:07:02 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3964 / 4000 (99.10%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_792 parse_ok=True`

### Monitor Update - 2026-06-26 22:12:02 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- raw outputs: 3971 / 4000 (99.28%)
- inference_process_running: False
- latest run log line: `Wrote report: /workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full/report.md`

## Baseline Full-Run Result - 2026-06-26 22:12:02 HKT
- total_samples: 4000
- valid_prediction_count: 3900
- invalid_count: 100
- valid_rate: 97.50%
- mean_angular_error_deg_valid_only: 18.44
- median_angular_error_deg_valid_only: 14.10
- acc@5_all: 35.33%
- acc@10_all: 41.55%
- acc@15_all: 51.78%
- acc@30_all: 75.55%
- invalid_reason_counts: `{"point_3d_wrong_dimension": 69, "runtime_error": 29, "no_json_object": 2}`
- thresholds: `{"max_median_deg": 15.0, "min_acc30_all": 0.7, "min_valid_rate": 0.98}`
- decision: NEEDS ITERATION: valid_rate 97.50% (target >= 98.00%)
- summary_json: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full/eval/3d_directional_eval_summary.json`
- report_md: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full/report.md`

## Iteration V2 Plan - 2026-06-26

Baseline V1 failed the engineering valid-rate criterion: 3900/4000 valid predictions (97.50%), with median 14.10 deg and acc@30 all 75.55%. Failure audit:

- 69 invalids were `point_3d_wrong_dimension`, mostly model outputs such as `[x, y, z, 0.0]` or six-number arrays.
- 29 invalids were runner `runtime_error` from manifest groups with no GT anchors and no extracted evidence panel. A manifest audit found 3971 `gt_panel` groups and 29 `no_gt_no_panel` groups.
- The no-GT groups cannot be evaluated by the camera-centered angular formula without fabricating GT. V2 therefore adds an explicit evaluable-subset option instead of treating them as model failures.

V2 changes:

- Add `--skip_missing_gt_anchors` to `exam3/run_qwen3vl_3d_directional.py` for the evaluable diagnostic subset.
- Add `exam3/prompts/camera_centered_3d_directional_prompt_v2.md`. It explicitly says to prefer copying one candidate anchor coordinate, forbids `[x,y,z,0]` and six-number arrays, and forbids using gaze/hand/camera-hit points as the answer unless they coincide with the intended candidate anchor.

Smoke-test requirement:

- Run a targeted cross-scene V2 smoke test including scenes with previous dimension failures and high angular errors.
- Compare V2 smoke metrics against V1 where possible.
- Continue iterating if V2 still has parse/schema failures or clearly worse angular metrics.

## Acceptance Criterion Update - 2026-06-26

The main angular-accuracy criterion was tightened from @30 deg to @20 deg after user feedback. Rationale:

- 30 deg is useful as a broad, low-confidence diagnostic tolerance.
- 5-20 deg is a more credible range for acceptable directional grounding.
- Future iteration decisions use @20 deg all-sample accuracy as the primary angular-accuracy gate, while @30 deg remains reported as a supplementary loose metric.

Updated engineering gates:

- valid prediction rate >= 98%
- median angular error <= 15 degrees
- angular accuracy @20 degrees, all-sample denominator >= 60%

## V1 Re-Evaluation With @20 - 2026-06-26

V1 full run was re-evaluated with thresholds `5,10,15,20,30` after tightening the main angular criterion. Outputs:

- summary: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full/eval_thresholds_5_10_15_20_30/3d_directional_eval_summary.json`
- report: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full/report_thresholds_5_10_15_20_30.md`

Overall V1 with @20:

- valid_rate: 97.50%
- median angular error valid-only: 14.10 deg
- acc@15 all: 51.78%
- acc@20 all: 61.42%
- acc@30 all, loose reference only: 75.55%

Decision under updated gates: V1 still needs iteration because valid_rate is below 98%, even though median and acc@20 pass the current engineering gates.

## Iteration V3 Plan - 2026-06-26

V2 targeted smoke improved the schema failure pattern but still produced one six-number `point_3d` for `scene4_room4:123` (`Inspect the dining table's right edge and wipe the tabletop.`). The model appeared to encode an edge/surface extent as two 3D points.

V3 prompt-only change:

- Add `exam3/prompts/camera_centered_3d_directional_prompt_v3.md`.
- Explicitly map parts/surfaces/edges/doorways/bases/tabletops/handles/sides to the parent object candidate anchor.
- Explicitly forbid bounding boxes, edge extents, and two-point six-number outputs.

Next action: rerun the same targeted smoke sample set with V3 and thresholds `5,10,15,20,30`.

## V2/V3 Targeted Smoke Results - 2026-06-27

Targeted smoke set: 24 requested sample keys, with 2 no-GT groups skipped by `--skip_missing_gt_anchors`, leaving 22 evaluable samples. The set intentionally includes V1 dimension failures and high-angular-error scene4/scene5 samples.

V2 targeted smoke:

- output root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v2_targeted_smoke`
- valid_rate: 95.45% (21/22)
- invalid_reason_counts: `{"point_3d_wrong_dimension": 1}`
- median angular error valid-only: 8.84 deg
- acc@20 all: 0.00% (V2 was evaluated before @20 became the default, so this key may be absent)
- acc@30 all loose: 81.82%

V3 targeted smoke:

- output root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_targeted_smoke`
- valid_rate: 100.00% (22/22)
- invalid_reason_counts: `{}`
- median angular error valid-only: 0.00 deg
- acc@20 all: 86.36%
- acc@30 all loose: 95.45%

Decision: V3 passes the targeted schema smoke test (0 invalids), fixes the V2 edge/surface six-number failure, and passes the updated @20 gate on this targeted subset. Launch a V3 full evaluable-subset run with `--skip_missing_gt_anchors`; do not overwrite V1 outputs.

### Monitor Update - 2026-06-27 00:35:19 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 26 / 4000 (0.65%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_25 parse_ok=True`

### Monitor Update - 2026-06-27 00:40:19 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 101 / 4000 (2.53%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_100 parse_ok=True`

### Monitor Update - 2026-06-27 00:45:20 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 177 / 4000 (4.42%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_176 parse_ok=True`

### Monitor Update - 2026-06-27 00:50:20 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 252 / 4000 (6.30%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_251 parse_ok=True`

### Monitor Update - 2026-06-27 00:55:21 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 328 / 4000 (8.20%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_327 parse_ok=True`

### Monitor Update - 2026-06-27 01:00:21 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 403 / 4000 (10.08%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_402 parse_ok=True`

### Monitor Update - 2026-06-27 01:05:21 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 478 / 4000 (11.95%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_477 parse_ok=True`

### Monitor Update - 2026-06-27 01:10:22 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 552 / 4000 (13.80%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_551 parse_ok=True`

### Monitor Update - 2026-06-27 01:15:22 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 627 / 4000 (15.68%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_626 parse_ok=True`

### Monitor Update - 2026-06-27 01:20:23 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 702 / 4000 (17.55%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_701 parse_ok=True`

### Monitor Update - 2026-06-27 01:25:23 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 777 / 4000 (19.43%)
- inference_process_running: True
- latest run log line: `[ok] scene1 row_776 parse_ok=True`

### Monitor Update - 2026-06-27 01:30:23 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 853 / 4000 (21.32%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_52 parse_ok=True`

### Monitor Update - 2026-06-27 01:35:24 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 928 / 4000 (23.20%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_127 parse_ok=True`

### Monitor Update - 2026-06-27 01:40:24 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1004 / 4000 (25.10%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_203 parse_ok=True`

### Monitor Update - 2026-06-27 01:45:25 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1080 / 4000 (27.00%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_281 parse_ok=True`

### Monitor Update - 2026-06-27 01:50:25 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1156 / 4000 (28.90%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_358 parse_ok=True`

### Monitor Update - 2026-06-27 01:55:25 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1232 / 4000 (30.80%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_436 parse_ok=True`

### Monitor Update - 2026-06-27 02:00:26 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1308 / 4000 (32.70%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_513 parse_ok=True`

### Monitor Update - 2026-06-27 02:05:26 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1384 / 4000 (34.60%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_590 parse_ok=True`

### Monitor Update - 2026-06-27 02:10:26 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1460 / 4000 (36.50%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_666 parse_ok=True`

### Monitor Update - 2026-06-27 02:15:27 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1535 / 4000 (38.38%)
- inference_process_running: True
- latest run log line: `[ok] scene2 row_741 parse_ok=True`

### Monitor Update - 2026-06-27 02:20:27 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1610 / 4000 (40.25%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_18 parse_ok=True`

### Monitor Update - 2026-06-27 02:25:28 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1688 / 4000 (42.20%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_96 parse_ok=True`

### Monitor Update - 2026-06-27 02:30:28 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1765 / 4000 (44.12%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_173 parse_ok=True`

### Monitor Update - 2026-06-27 02:35:28 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1842 / 4000 (46.05%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_250 parse_ok=True`

### Monitor Update - 2026-06-27 02:40:29 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1919 / 4000 (47.98%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_327 parse_ok=True`

### Monitor Update - 2026-06-27 02:45:29 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 1997 / 4000 (49.93%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_405 parse_ok=False`

### Monitor Update - 2026-06-27 02:50:29 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2074 / 4000 (51.85%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_482 parse_ok=True`

### Monitor Update - 2026-06-27 02:55:30 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2152 / 4000 (53.80%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_560 parse_ok=True`

### Monitor Update - 2026-06-27 03:00:30 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2228 / 4000 (55.70%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_636 parse_ok=True`

### Monitor Update - 2026-06-27 03:05:31 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2305 / 4000 (57.63%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_713 parse_ok=True`

### Monitor Update - 2026-06-27 03:10:31 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2382 / 4000 (59.55%)
- inference_process_running: True
- latest run log line: `[ok] scene3 row_790 parse_ok=True`

### Monitor Update - 2026-06-27 03:15:31 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2458 / 4000 (61.45%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room1 row_75 parse_ok=True`

### Monitor Update - 2026-06-27 03:20:32 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2534 / 4000 (63.35%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room1 row_156 parse_ok=True`

### Monitor Update - 2026-06-27 03:25:32 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2608 / 4000 (65.20%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room2 row_30 parse_ok=True`

### Monitor Update - 2026-06-27 03:30:33 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2684 / 4000 (67.10%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room2 row_106 parse_ok=True`

### Monitor Update - 2026-06-27 03:35:33 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2759 / 4000 (68.97%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room2 row_181 parse_ok=True`

### Monitor Update - 2026-06-27 03:40:33 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2835 / 4000 (70.88%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room3 row_58 parse_ok=True`

### Monitor Update - 2026-06-27 03:45:34 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2911 / 4000 (72.78%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room3 row_134 parse_ok=False`

### Monitor Update - 2026-06-27 03:50:34 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 2987 / 4000 (74.67%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room4 row_10 parse_ok=True`

### Monitor Update - 2026-06-27 03:55:34 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3066 / 4000 (76.65%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room4 row_90 parse_ok=True`

### Monitor Update - 2026-06-27 04:00:35 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3144 / 4000 (78.60%)
- inference_process_running: True
- latest run log line: `[ok] scene4_room4 row_168 parse_ok=True`

### Monitor Update - 2026-06-27 04:05:35 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3221 / 4000 (80.53%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_47 parse_ok=True`

### Monitor Update - 2026-06-27 04:10:36 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3296 / 4000 (82.40%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_123 parse_ok=True`

### Monitor Update - 2026-06-27 04:15:36 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3371 / 4000 (84.28%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_199 parse_ok=True`

### Monitor Update - 2026-06-27 04:20:36 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3448 / 4000 (86.20%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_276 parse_ok=True`

### Monitor Update - 2026-06-27 04:25:37 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3523 / 4000 (88.08%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_351 parse_ok=True`

### Monitor Update - 2026-06-27 04:30:37 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3600 / 4000 (90.00%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_428 parse_ok=True`

### Monitor Update - 2026-06-27 04:35:37 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3677 / 4000 (91.92%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_505 parse_ok=True`

### Monitor Update - 2026-06-27 04:40:38 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3753 / 4000 (93.83%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_581 parse_ok=True`

### Monitor Update - 2026-06-27 04:45:38 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3829 / 4000 (95.73%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_657 parse_ok=True`

### Monitor Update - 2026-06-27 04:50:39 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3905 / 4000 (97.62%)
- inference_process_running: True
- latest run log line: `[ok] scene5 row_733 parse_ok=True`

### Monitor Update - 2026-06-27 04:55:39 HKT
- output_root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable`
- raw outputs: 3971 / 4000 (99.28%)
- inference_process_running: False
- latest run log line: `Wrote report: /workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/report.md`

## Baseline Full-Run Result - 2026-06-27 04:55:39 HKT
- total_samples: 3971
- valid_prediction_count: 3861
- invalid_count: 110
- valid_rate: 97.23%
- mean_angular_error_deg_valid_only: 14.53
- median_angular_error_deg_valid_only: 8.97
- acc@5_all: 43.24%
- acc@10_all: 49.86%
- acc@15_all: 58.78%
- acc@20_all: 68.55%
- acc@30_all: 79.65%
- invalid_reason_counts: `{"point_3d_wrong_dimension": 110}`
- thresholds: `{"max_median_deg": 15.0, "min_acc20_all": 0.6, "min_valid_rate": 0.98}`
- decision: NEEDS ITERATION: valid_rate 97.23% (target >= 98.00%)
- summary_json: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/eval/3d_directional_eval_summary.json`
- report_md: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/report.md`

## Final V3 Adoption Decision - 2026-06-27

User decision: do not run V4. Generate the V3 report/statistics documents and stop iteration after the full V3 result is documented.

Rationale:

- V3 full run completed on the 3971-sample evaluable subset.
- The primary paper-facing angular range is 5-20 deg, with @30 retained only as a loose reference.
- V3 directional metrics are acceptable for the camera-centered 3D directional diagnostic:
  - median angular error valid-only: 8.97 deg
  - acc@15 all: 58.78%
  - acc@20 all: 68.55%
  - acc@30 all, loose reference only: 79.65%
- V3 does not meet the earlier 98% engineering valid-rate gate:
  - valid_rate: 97.23%
  - invalid_count: 110
  - invalid_reason_counts: `{"point_3d_wrong_dimension": 110}`
- The remaining schema failures are reported as a limitation instead of being manually repaired or triggering another full run.

Final documentation:

- report: `exam3/RESULTS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
- statistics: `exam3/STATS_QWEN3VL30B_3D_DIRECTIONAL_V3.md`
