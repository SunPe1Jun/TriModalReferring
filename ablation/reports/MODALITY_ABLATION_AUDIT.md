# Modality Ablation Implementation Audit

This audit documents the current implementation status of the VR-TriRef modality ablation workflows under `ablation/`. It covers experiment 1 and experiment 2 only. Experiment 3 is intentionally excluded from this ablation round.

## Scope

- Experiment 1: closed-set 3D anchor selection.
- Experiment 2: projected-2D point diagnostic.
- Baseline results are reused from the existing experiment outputs unless explicitly rerun.
- Generated model predictions, logs, masked images, CSV summaries, and large data files are ignored by git.

## Experiment 1

Runner: `ablation/exam1/run_exam1_ablation.sh`

Main model script: `ablation/exam1/scripts/grounding/run_qwen3vl_local_3d_batch_inprocess.py`

Prompt/parser support: `ablation/exam1/scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py`

Evaluator: `ablation/exam1/scripts/eval/evaluate_local_3d_object_match.py`

### no_visual

Status: implemented as a clean visual-evidence removal control.

- The original video is not passed to the model.
- The model receives a blank `64x64` RGB placeholder image.
- The prompt includes a visual-ablation note telling the model not to treat the blank image as scene evidence.
- Candidate anchors, language, gaze summaries, hand summaries, and structured fields remain available unless separately ablated.

Current full result:

- Baseline overall accuracy: `0.7660`
- `no_visual` overall accuracy: `0.7302`
- Delta overall accuracy: `-0.0358`
- Baseline micro F1: `0.5326`
- `no_visual` micro F1: `0.5425`

Interpretation note: overall closed-set accuracy drops after removing visual evidence, but set-level micro F1 does not drop in the current run. The paper should report both metrics rather than claiming a uniform degradation across all metrics.

### no_gaze

Status: structured gaze ablation is implemented; pixel-level gaze marker removal is not implemented for experiment 1 videos.

Removed or hidden from the prompt:

- `gaze_summary`
- `gaze_point`
- `gaze_vector`
- `gaze_origin`
- `camera_gaze_origin`
- `camera_gaze_direction`
- gaze-derived sparse timeline proposals

Limitation:

- The original video is still passed to the model.
- If the video contains a visible green gaze marker, the model may still see it in pixels.
- The prompt instructs the model to ignore visible gaze markers, but this is weaker than true pixel masking.

Recommended wording:

> Experiment-1 `no_gaze` removes structured gaze cues from the prompt/interface while keeping the visual stream unchanged.

### no_hand

Status: structured hand/gesture ablation is implemented; pixel-level hand removal is not implemented for experiment 1 videos.

Removed or hidden from the prompt:

- `hand_summary`
- hand/ray peak spatial fields, including right-index ray hit fields when present

Known minor prompt leakage:

- A generic instruction can still mention the field name `rightIndexFingerRayHitPoint` as a noisy auxiliary cue.
- No actual hand coordinate values are provided in `no_hand`.
- This generic field-name mention should be removed before a strict paper-facing rerun.

Limitation:

- The original video is still passed to the model.
- Any visible hand or gesture evidence in the video remains available to the model.

Recommended wording:

> Experiment-1 `no_hand` removes structured hand and ray cues from the prompt/interface while keeping the visual stream unchanged.

### Candidate Anchor Consistency

Candidate anchor lists are held fixed by scene across variants. Available output audits show the same candidate hash for the same scene across variants:

- `scene1`: `4663828e91852f61`
- `scene2`: `51008f9a91175b88`

This preserves the closed-set task definition. The `language_anchors_only` variant also keeps candidate anchors because removing them would change the task from closed-set selection to open-ended grounding.

### Decoding and Prompt Settings

Experiment-1 ablation variants use the same generation settings unless the ablation definition requires a media change:

- `do_sample=False`
- `max_new_tokens=1536`
- `prompt_style=full`
- `prompt_strategy=mention_first`
- default media mode is video
- `no_visual` intentionally replaces the video with a blank placeholder image

Baseline results are loaded from `data/match_eval_qwen3vl30b_mention_first_v3/`. For the strictest apples-to-apples audit, rerun baseline predictions through the ablation evaluator and compare summaries.

### Invalid Output Status

For the completed `no_visual` full run:

- Total predictions: `4000`
- `response_status_ok_count`: `3906`
- Parsed/status-missing outputs: `94 / 4000` (`2.35%`)

By scene:

- `scene1`: `22 / 800`
- `scene2`: `27 / 800`
- `scene3`: `23 / 800`
- `scene4_room1`: `3 / 200`
- `scene4_room2`: `0 / 200`
- `scene4_room3`: `1 / 200`
- `scene4_room4`: `0 / 200`
- `scene5`: `18 / 800`

This is higher than the baseline missing/non-ok count, but there is no single catastrophic scene-specific invalid-output spike.

## Experiment 2

Runner: `ablation/exam2/run_exam2_ablation.sh`

Main model script: `ablation/exam2/scripts/run_qwen3vl_2d_point_grounding.py`

Evaluator: `exam2/evaluate_2d_point_grounding.py`

### no_visual

The experiment-2 code supports a `blank_visual` sanity control that replaces panels with blank placeholders. This variant was not part of the completed full results summarized in the current report.

### no_gaze

Status: implemented as a clean gaze ablation for projected-2D panel evidence.

The `no_gaze` alias expands to:

- `gaze_text`
- `gaze_marker`

Effects:

- Gaze-specific prompt/system text is removed.
- The projected green gaze marker is masked in copied panel images.
- The current full run uses `PANEL_CONTEXT_MODE=full`, so no gaze-centered crop hints are provided.

Completed full result:

- Baseline Joint@100 F1: `0.2038`
- `no_gaze` Joint@100 F1: `0.1551`
- Delta Joint@100 F1: `-0.0488`
- Baseline Point@100 F1: `0.2470`
- `no_gaze` Point@100 F1: `0.2027`
- Delta Point@100 F1: `-0.0443`

This is the cleanest current evidence that gaze contributes to experiment-2 performance.

### no_gaze_text_prior

Status: implemented as a text-prior-only gaze ablation.

Effects:

- Gaze-specific prompt/system text is removed.
- Visible gaze markers are not masked.
- Full panels are used without gaze-centered crops.

Completed full result:

- Joint@100 F1: `0.1725`
- Delta Joint@100 F1: `-0.0313`
- Point@100 F1: `0.2199`
- Delta Point@100 F1: `-0.0271`

Interpretation:

- Removing only the gaze text prior hurts less than removing both gaze text and the marker.
- This suggests that visible gaze evidence contributes beyond the textual prior.

### full_panels_no_crop

Status: implemented as a crop-path ablation rather than a modality ablation.

Effects:

- Full panels are retained.
- Gaze-centered crop panels are removed.

Completed full result:

- Joint@100 F1: `0.1759`
- Delta Joint@100 F1: `-0.0280`
- Point@100 F1: `0.2236`
- Delta Point@100 F1: `-0.0234`

### instruction_only_prompt

Status: implemented as a prompt-format ablation, not a pure modality ablation.

Effects:

- Uses `PROMPT_MODE=instruction_only`.
- Keeps paired crop context.
- Should not be grouped as a primary modality-removal condition.

Completed full result:

- Joint@100 F1: `0.1799`
- Delta Joint@100 F1: `-0.0240`
- Point@100 F1: `0.2262`
- Delta Point@100 F1: `-0.0207`

### no_hand

Experiment 2 currently has no explicit hand summary or hand-joint interface in the manifest/prompt workflow. No primary hand ablation should be claimed from experiment 2.

### Decoding and Prompt Settings

For `full_panels_no_crop`, `no_gaze`, and `no_gaze_text_prior`:

- `do_sample=False`
- `max_new_tokens=768`
- `input_mode=multi_image`
- `PANELS=3`
- `PANEL_CONTEXT_MODE=full`
- same parser and evaluator

`instruction_only_prompt` intentionally changes prompt format and context mode, so it is not a pure modality ablation.

### Parser Consistency

All experiment-2 ablation variants use the same inference parser in `ablation/exam2/scripts/run_qwen3vl_2d_point_grounding.py` and the same evaluator interface as the baseline experiment-2 workflow.

### Invalid Output Status

Completed full-run prediction parse status:

- `full_panels_no_crop`: `0 / 3971` parse failures
- `no_gaze`: `0 / 3971` parse failures
- `no_gaze_text_prior`: `0 / 3971` parse failures
- `instruction_only_prompt`: `7 / 3971` parse failures

The `instruction_only_prompt` parse failures are sparse across scenes and do not indicate a single scene-level formatting collapse.

## Paper-Facing Recommendations

Use the current results as follows:

- Treat experiment-2 `no_gaze` as the strongest clean gaze-modality ablation.
- Treat experiment-1 `no_visual` as a clean visual-removal ablation.
- Treat experiment-1 `no_gaze` and `no_hand` as structured-cue ablations unless pixel masking is added.
- Do not claim an experiment-2 hand ablation from the current workflow.
- Report invalid outputs separately when using full-run metrics.

Recommended future cleanup before final paper tables:

- Remove the generic `rightIndexFingerRayHitPoint` mention in experiment-1 `no_hand`.
- If strict gaze/hand removal is required for experiment 1, implement video/frame-level masking or switch to masked evidence-frame inputs.
- Rerun baseline through the ablation evaluator to document exact parser/evaluator equivalence.
