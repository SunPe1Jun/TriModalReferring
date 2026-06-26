# Camera-Centered 3D Directional Diagnostic Iteration Log

This log records monitoring, failure analysis, and every implementation or prompt iteration for experiment 3. It is meant to be auditable: no model outputs are manually edited, and invalid outputs remain in the all-sample denominator.

## Acceptance Criteria

Current engineering thresholds for deciding whether the diagnostic is good enough to keep:

- valid prediction rate >= 98%
- median angular error <= 15 degrees
- angular accuracy @30 degrees, all-sample denominator >= 70%

These thresholds are not paper claims. They are working criteria for deciding whether to iterate.

## Baseline V1 Full Run

- output root: `/workspace/usr3/TriModal-Referring/exam3/outputs_qwen3vl30b_3d_directional_v1_full`
- run log: `/workspace/usr3/TriModal-Referring/exam3/qwen3vl30b_3d_directional_v1_full.log`
- prompt template: `/workspace/usr3/TriModal-Referring/exam3/prompts/camera_centered_3d_directional_prompt.md`
- evidence panel rule: `highest_score`, reusing the exam2 v10 panel with the highest `panel_selection_score`
- model: `/workspace/usr3/Qwen3-VL-30B-A3B-Instruct`
- launch mode: `setsid` background process so SSH disconnects do not terminate inference
- status at log creation: running
- progress at log creation: 3412 raw outputs out of 25887 manifest samples

Observed issue before full-run completion:

- `scene5_row_223` parsed invalid with `point_3d_wrong_dimension`.
- Raw output had four numbers in `point_3d`: `[2.48, -0.24, 16.0, 0.0]`.
- This confirms the parser is enforcing the schema correctly. It also suggests a potential prompt-only iteration if invalid rate is material.

Policy:

- Do not modify the current V1 full-run prompt or outputs while the baseline is running.
- When V1 finishes, evaluate with `exam3/evaluate_3d_directional.py`.
- If any acceptance criterion fails, inspect failure modes and run a new smoke test before launching any new full run.

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
