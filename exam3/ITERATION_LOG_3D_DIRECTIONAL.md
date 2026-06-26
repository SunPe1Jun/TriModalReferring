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
