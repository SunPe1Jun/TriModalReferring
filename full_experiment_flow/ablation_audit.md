# Modality Ablation Audit

## Purpose

Measure which input families affect Experiments 1 and 2, while auditing whether each run changes only one modality and whether results can support paired statistical claims.

## Experiment 1 Variants

| Variant | Category | Hit-All | Exact | Micro F1 | Delta Hit-All |
| --- | --- | ---: | ---: | ---: | ---: |
| full_baseline | baseline | 0.7660 | 0.3205 | 0.5326 | 0.0000 |
| language_anchors_only | multimodal input baseline | 0.6280 | 0.2553 | 0.4896 | -0.1380 |
| no_gaze | hybrid | 0.7170 | 0.2840 | 0.4890 | -0.0490 |
| no_hand | hybrid | 0.7665 | 0.3180 | 0.5326 | +0.0005 |
| no_visual | hybrid | 0.7303 | 0.3147 | 0.5425 | -0.0358 |
| no_gaze_hand | hybrid | 0.7140 | 0.2790 | 0.4877 | -0.0520 |

## Experiment 2 Variants

| Variant | Category | Temporal F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| full_baseline | baseline | 0.7333 | 0.2470 | 0.2038 | 0.0000 |
| full_panels_no_crop | preprocessing | 0.7010 | 0.2236 | 0.1759 | -0.0280 |
| instruction_only_prompt | prompt | 0.6549 | 0.2262 | 0.1799 | -0.0240 |
| no_gaze | hybrid | 0.6886 | 0.2027 | 0.1551 | -0.0488 |
| no_gaze_text_prior | hybrid | 0.6896 | 0.2199 | 0.1725 | -0.0313 |

## Audit Findings

- All variants contain the same 4,000 event ids as their full baseline, with no duplicate unified ids.
- Anchor GT sets and Projected-2D manifest GT are identical across variants.
- The correct unified interaction key is `scene + row_index`; raw `event_id` repeats across scene4 room partitions.
- No completed variant satisfies the strict single-modality standard used by the audit attachment.
- `no_gaze` and `no_hand` remove structured cues, but gaze markers or hands can remain visible in video.
- Experiment 2 `no_gaze` also changes crop preprocessing; `no_visual` changes media type/count/geometry and prompt text.
- The paired-bootstrap CSVs therefore contain headers only. These runs support descriptive hybrid-ablation comparisons, not strict single-modality significance claims.

## Authoritative Sources

- `analysis_outputs/ablation_audit/ABLATION_AUDIT.md`
- `analysis_outputs/ablation_audit/RESULT_VALIDATION.md`
- `analysis_outputs/ablation_audit/anchor_ablation_summary.csv`
- `analysis_outputs/ablation_audit/projected2d_ablation_summary.csv`
