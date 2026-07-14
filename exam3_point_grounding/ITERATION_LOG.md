# Experiment 3 Point-Grounding Iteration Log

## 2026-07-09 v5 Prompt/Telemetry Optimization

- Previous v4 smoke (`outputs_smoke10_scene1_v4_20260709`) fixed parser compliance: 10/10 valid, 0 invalid, but Qwen still underperformed gaze-copy (`anchor_set_f1_micro=0.1053`, `margin_f1_at_2_0=0.0`).
- Failure pattern: model often copied a single gaze/hand ray endpoint, especially distant floor/background hits with y around -9, instead of estimating the intended referent point from image + temporal evidence.
- v5 changes: restore camera/gaze/hand direction context in the prompt, but format coordinates as `x=..., y=..., z=...`; mark direction vectors as `do_not_copy`; rename hit points as `ray_endpoint`; add guidance that ray endpoints are cues, not object centers; allow up to three distinct hypotheses for one ambiguous referent.
- No evaluator change and no GT/candidate-anchor leakage change.

## 2026-07-09 v6 Bounds/Fallback Optimization

- v5 smoke (`outputs_smoke10_scene1_v5_20260709`) kept 10/10 valid and improved over v4 (`anchor_set_f1_micro=0.2000`, `margin_f1_at_2_0=0.0952`), but still underperformed gaze-copy (`anchor_set_f1_micro=0.5333`).
- Failure pattern: the model avoided some ray endpoints but invented generic near-camera coordinates such as `[0, -0.5, 15]`, often outside scene robust bounds.
- v6 changes: stronger scene-bounds sanity check; explicitly reject arbitrary canonical coordinates; when object-center estimation is uncertain, prefer distinct measured gaze-ray endpoint hypotheses, with hand endpoints secondary.
- No evaluator change and no GT/candidate-anchor leakage change.

## 2026-07-09 v7 Measured-Hypothesis Optimization

- v6 smoke (`outputs_smoke10_scene1_v6_20260709`) kept 10/10 valid and improved mean distance over v5, but anchor-set F1 stayed at 0.2000 and generic coordinates such as `[0, -0.5, 15]` remained.
- Failure pattern: free-form coordinate generation is not reliable without candidate anchors or object extents; measured gaze endpoints remain a much stronger target-free proxy on the same 10 samples (`gaze_copy anchor_set_f1_micro=0.5333`).
- v7 changes: expose only gaze/hand ray endpoints as `copyable_measured_point_hypotheses`; instruct Qwen to output 1-3 distinct measured hypotheses as separate JSON entries when object-center estimation is uncertain; still no candidate anchors, GT, target descriptions, or evaluator fields in the model prompt.
- No evaluator/parser change and no manual output repair.

## 2026-07-09 v8 Gaze-Primary Hypothesis Optimization

- v7 smoke (`outputs_smoke10_scene1_v7_20260709`) kept 10/10 valid and improved over v6 (`anchor_set_f1_micro=0.2500`, `margin_f1_at_2_0=0.1333`), but still underperformed the deterministic gaze-copy baseline (`anchor_set_f1_micro=0.5333`).
- Failure pattern: Qwen sometimes copied hand endpoints when gaze endpoints were more predictive, and sometimes still produced free-form near-camera points.
- v8 changes: split measured hypotheses into primary gaze and secondary hand groups; instruct Qwen to default to distinct non-null gaze endpoint hypotheses, using hand only when gaze is missing, hand-action is explicit, or hand agrees with gaze.
- No evaluator/parser change and no GT/candidate-anchor leakage change.

## 2026-07-09 v9 ID-Based Gaze-Hypothesis Optimization

- v8 smoke (`outputs_smoke10_scene1_v8_20260709`) kept 10/10 valid and improved distance metrics, but anchor-set F1 dropped to 0.1667 while gaze-copy remained 0.5333.
- Failure pattern: Qwen still mixed selective copying with occasional free-form coordinate generation, and did not consistently output all useful gaze hypotheses.
- v9 changes: give each primary gaze and secondary hand endpoint an explicit hypothesis id (`P1_GAZE`, `P1_HAND`, etc.); instruct Qwen to output all distinct non-null primary gaze hypotheses by default, copying coordinates exactly, and only fall back to hand if no gaze is available.
- This keeps the task candidate-free and target-free, but intentionally narrows it to measured point-hypothesis copying rather than free 3D reconstruction.

## 2026-07-09 Summary After v9

- Best Qwen smoke version so far: v9, with `anchor_set_f1_micro=0.4848`, `margin_f1_at_2_0=0.3500`, and 10/10 valid outputs.
- Same-subset deterministic gaze-copy remains stronger: `anchor_set_f1_micro=0.5333`, `margin_f1_at_2_0=0.3784`.
- Decision: do not start full run automatically; freeze v9 only if the experiment is reframed as measured point-hypothesis selection/copying rather than unconstrained 3D reconstruction.
- Detailed table written to `exam3_point_grounding/OPTIMIZATION_SUMMARY.md`.
