# Result Validation

All checks below were performed from sample-level detail files. No original result was overwritten.

## Structural checks

| Setting | Variant | Rows | Unique IDs | Duplicate IDs | Missing vs full | Valid outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| anchor_selection | full_baseline | 4000 | 4000 | 0 | 0 | 3970 |
| anchor_selection | language_anchors_only | 4000 | 4000 | 0 | 0 | 3947 |
| anchor_selection | no_gaze | 4000 | 4000 | 0 | 0 | 3991 |
| anchor_selection | no_hand | 4000 | 4000 | 0 | 0 | 3980 |
| anchor_selection | no_visual | 4000 | 4000 | 0 | 0 | 3906 |
| anchor_selection | no_gaze_hand | 4000 | 4000 | 0 | 0 | 3985 |
| projected_2d | full_baseline | 4000 | 4000 | 0 | 0 | 3964 |
| projected_2d | full_panels_no_crop | 4000 | 4000 | 0 | 0 | 3971 |
| projected_2d | instruction_only_prompt | 4000 | 4000 | 0 | 0 | 3964 |
| projected_2d | no_gaze | 4000 | 4000 | 0 | 0 | 3971 |
| projected_2d | no_gaze_text_prior | 4000 | 4000 | 0 | 0 | 3971 |

## Summary reproduction

Anchor unified values are recomputed from `*_match_eval.csv`. Hit-All, Hit-Mapped, Exact, and micro F1 reproduce the count-based legacy summaries. Unified macro Set-F1 intentionally differs from the legacy `macro_f1` because zero-overlap rows are restored as F1=0.

| Anchor variant | Recomputed Hit-All | Recomputed Hit-Mapped | Recomputed Exact | Recomputed macro F1 | Recomputed micro F1 | Source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_baseline | 0.7660000000 | 0.7713997986 | 0.3204934542 | 0.5589033770 | 0.5325880114 | per-scene match_eval CSVs |
| language_anchors_only | 0.6280000000 | 0.6324269889 | 0.2552870091 | 0.4676476394 | 0.4896293323 | per-scene match_eval CSVs |
| no_gaze | 0.7170000000 | 0.7220543807 | 0.2839879154 | 0.5104609431 | 0.4889867841 | per-scene match_eval CSVs |
| no_hand | 0.7665000000 | 0.7719033233 | 0.3179758308 | 0.5590530413 | 0.5326282899 | per-scene match_eval CSVs |
| no_visual | 0.7302500000 | 0.7353977845 | 0.3147029204 | 0.5471510852 | 0.5424643454 | per-scene match_eval CSVs |
| no_gaze_hand | 0.7140000000 | 0.7190332326 | 0.2789526687 | 0.5075815008 | 0.4877232810 | per-scene match_eval CSVs |

Projected-2D corpus metrics are recomputed by summing the existing per-event TP/FP/FN fields. Mean/median matched distance is independently reconstructed from the unchanged evaluator's Point@100 matching function.

| Projected variant | Temporal F1 | Point@100 F1 | Joint@100 F1 | Original summary discrepancy | Source |
| --- | ---: | ---: | ---: | --- | --- |
| full_baseline | 0.7332743884 | 0.2469790746 | 0.2038314176 | 0.000e+00 | 2d_eval_detail.csv and summary JSON |
| full_panels_no_crop | 0.7010078387 | 0.2236105381 | 0.1758708080 | 0.000e+00 | 2d_eval_detail.csv and summary JSON |
| instruction_only_prompt | 0.6548516843 | 0.2262443439 | 0.1798642534 | 0.000e+00 | 2d_eval_detail.csv and summary JSON |
| no_gaze | 0.6885980427 | 0.2026883622 | 0.1550524702 | 0.000e+00 | 2d_eval_detail.csv and summary JSON |
| no_gaze_text_prior | 0.6895615743 | 0.2198619225 | 0.1725379123 | 0.000e+00 | 2d_eval_detail.csv and summary JSON |

## Denominator and GT checks

- Anchor coverage is 3,972 mapped/evaluable interactions out of 4,000 for every variant. Unmapped interactions remain in Hit-All and validity denominators.
- Projected-2D contains 4,000 manifest/evaluator rows and 3,971 model prediction records for every variant. Missing records are retained by the evaluator as empty predictions.
- Anchor mapped GT sets are identical across variants for every interaction.
- Projected-2D uses one shared manifest, so GT panels/points and sample IDs are identical across variants.
- No duplicate interaction IDs were found in the unified sample-level exports. The key is `scene + row_index`; raw `event_id` alone is not globally unique across scene4 room partitions.
- Full JSON comparison found zero candidate-inventory mismatches and zero video-path mismatches between Anchor baseline and every variant.
- Percent and decimal representations are not mixed in generated CSVs; all metrics are decimal fractions.

## Parser/invalid checks

- Anchor parser/evaluator fields are shared in schema; validity is `response_status == ok` and invalid rows remain in denominators.
- Projected-2D uses the same JSON extraction/normalization implementation plus ablation-only input preparation. Baseline and instruction-only each have 7 parse failures; full-panels-no-crop, no-gaze, and no-gaze-text-prior have 0. Failures are distributed across scenes, with no scene-level collapse.
