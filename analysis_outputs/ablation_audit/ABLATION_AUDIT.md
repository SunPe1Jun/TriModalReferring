# Ablation Audit

## Scope and policy

This audit reads completed predictions and evaluations only. It did not modify paper files, rerun VLM inference, overwrite predictions, or overwrite existing evaluation outputs. Unified tables are recomputed in this independent directory.

## Completed experiments

All listed Anchor Selection variants contain 4,000 sample-level evaluation rows and all listed Projected-2D variants contain 4,000 evaluator rows. The Projected-2D prediction CSVs contain 3,971 model records; the 29 manifest events without records are retained as invalid/missing in the 4,000-event evaluation denominator.

Anchor Selection:

| Variant | Total | Hit-All | Hit-Mapped | Exact | Set-F1 macro | Micro F1 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| full_baseline | 4000 | 0.7660 | 0.7714 | 0.3205 | 0.5589 | 0.5326 | False |
| language_anchors_only | 4000 | 0.6280 | 0.6324 | 0.2553 | 0.4676 | 0.4896 | False |
| no_gaze | 4000 | 0.7170 | 0.7221 | 0.2840 | 0.5105 | 0.4890 | False |
| no_hand | 4000 | 0.7665 | 0.7719 | 0.3180 | 0.5591 | 0.5326 | False |
| no_visual | 4000 | 0.7302 | 0.7354 | 0.3147 | 0.5472 | 0.5425 | False |
| no_gaze_hand | 4000 | 0.7140 | 0.7190 | 0.2790 | 0.5076 | 0.4877 | False |

Projected-2D:

| Variant | Total | Valid rate | Temporal F1 | Point@100 F1 | Joint@100 F1 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| full_baseline | 4000 | 0.9910 | 0.7333 | 0.2470 | 0.2038 | False |
| full_panels_no_crop | 4000 | 0.9928 | 0.7010 | 0.2236 | 0.1759 | False |
| instruction_only_prompt | 4000 | 0.9910 | 0.6549 | 0.2262 | 0.1799 | False |
| no_gaze | 4000 | 0.9928 | 0.6886 | 0.2027 | 0.1551 | False |
| no_gaze_text_prior | 4000 | 0.9928 | 0.6896 | 0.2199 | 0.1725 | False |

The completed 3D Grounding VLM full run and three deterministic cue baselines were found. No completed 3D Grounding VLM modality-removal variant was found.

## Strict modality ablations

Under the conservative requirement that the target modality be fully removed while all other task/evaluation conditions remain fixed, the strict variants are: none.

No completed variant satisfies the attachment's strict standard. In particular, Anchor `no_visual` removes visual content but changes the input from video to one 64x64 image and adds ablation-specific prompt text; the image type/count/geometry and prompt are therefore not held fixed.

## Non-strict variants

- Anchor `no_gaze` removes structured gaze fields and gaze-derived proposals but can retain the visible gaze marker in video pixels.
- Anchor `no_hand` removes structured hand/ray fields but retains visible hands and gestures in video pixels.
- Anchor `no_gaze_hand` removes two structured cue families simultaneously.
- Anchor `language_anchors_only` removes several modalities and is a multimodal input baseline.
- Anchor `no_visual` removes visual content but also changes media type/count/geometry and prompt text, so it is classified as hybrid rather than strict.
- Projected-2D `full_panels_no_crop` is a preprocessing ablation.
- Projected-2D `instruction_only_prompt` is a prompt ablation.
- Projected-2D `no_gaze_text_prior` jointly changes gaze text and crop preprocessing.
- Projected-2D `no_gaze` removes gaze text and marker but also changes paired full+crop input to full-panel-only input; relative to the current full baseline it is a hybrid ablation.
- 3D `gaze_copy`, `hand_copy`, and `gaze_hand_fusion` are cue baselines, not VLM modality ablations.

## Controlled-comparison checks

- Anchor interaction-ID equality versus full: {"full_baseline": true, "language_anchors_only": true, "no_gaze": true, "no_gaze_hand": true, "no_hand": true, "no_visual": true}
- Anchor GT equality versus full: {"full_baseline": true, "language_anchors_only": true, "no_gaze": true, "no_gaze_hand": true, "no_hand": true, "no_visual": true}
- Projected-2D interaction-ID equality versus full: {"full_baseline": true, "full_panels_no_crop": true, "instruction_only_prompt": true, "no_gaze": true, "no_gaze_text_prior": true}
- Candidate anchor inventories are scene-level files shared by baseline and variants. Variant prediction JSON records the same `scene_anchor_csv`; the no-visual condition changes only visual input content.
- Projected-2D variants share the same manifest and evaluator. Their visual preprocessing/prompt differences are explicitly classified above.
- All runners use greedy decoding (`do_sample=False`). Anchor baseline and ablations use 1,536 max new tokens; Projected-2D uses 768.
- Sample-level predictions and raw model outputs are retained for all audited VLM runs.
- A full JSON-level scan found zero candidate-inventory mismatches and zero video-path mismatches between Anchor baseline and variants; each scene has exactly one stable anchor inventory.
- Anchor model path, dtype, input-mode metadata, frame limit, evidence-segment limit, prompt style, and prompt strategy are constant across all 4,000 rows of every variant.
- Projected-2D panel counts are identical by sample across variants (3,813 events with 3 panels, 152 with 2, and 6 with 1 among the 3,971 prediction records).

## Metric-scope conflicts

1. Historical Anchor Selection reports label the displayed `Set-F1` value ambiguously; the approximately 0.53 values are micro F1, not macro per-sample Set-F1.
2. The historical anchor evaluator's `macro_f1` averages only nonblank F1 values. It leaves no-overlap rows blank, which excludes zero-F1 samples and inflates the value (for example, scene-level values near 0.75). The unified export defines macro F1 over all evaluable samples with no-overlap rows equal to zero.
3. Anchor `Hit-All` uses all 4,000 interactions; `Hit-Mapped` and `Exact` use mapped/evaluable interactions. They must not be reported as sharing one denominator.
4. Projected-2D Temporal/Point/Joint F1 values are corpus-level F1 from aggregate TP/FP/FN, not mean per-sample F1.
5. All new exports use decimal fractions consistently.

## Re-evaluation and rerun requirements

No large inference rerun was performed. No completed run requires inference merely to reproduce the unified tables. The completed results can be reported under their audited categories (hybrid, preprocessing, prompt, multimodal-input, or cue baseline), but none supports a strict single-modality attribution under the requested definition.

For a claim of complete gaze or hand modality removal, new inference is required for Anchor `no_gaze` (mask/remove the video gaze marker) and Anchor `no_hand` (mask/remove visible hands/gesture evidence). For a strict Projected-2D gaze ablation relative to the paired-crop full baseline, new inference is required with the same paired image layout while removing gaze-derived crop selection/text/marker without changing image count or geometry. A full set of 3D Grounding VLM modality ablations also requires new inference because none was found.

The original completed evaluations do not need to be rerun for the reported legacy metrics. This audit recomputed unified metrics directly from sample-level files in a separate directory.

## Paired statistical analysis

No paired bootstrap was run because no completed variant meets the strict modality-ablation definition relative to its full baseline. Both bootstrap CSVs contain headers only.

## Subset readiness

- Stable: single-target versus multi-target, scenes, and room partitions.
- Not stable/available: discrete/location-like/region-like target type and instruction type. No name-based inference was used.
- Required for target-type analysis: an explicit interaction-level annotation or versioned mapping table with mutually exclusive target-type labels and documented handling of mixed-target interactions.

## Remaining unknowns

- Exact checkpoint revision/commit hashes are not recorded; only local model paths are available.
- A reliable semantic target-type field and instruction-type taxonomy are absent.
- The visible-pixel completeness of Anchor gaze/hand removal is known to be false from the implementation; no pixel-level audit of every video frame was attempted.
