#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MODEL_NAME="${MODEL_NAME:-/workspace/usr3/InternVL3-38B-Instruct}"
MODEL_TAG="${MODEL_TAG:-internvl3_38b_baseline}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/data/match_eval_qwen3vl30b_mention_first_v3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/internvl/outputs/exam2_${MODEL_TAG}}"
MANIFEST_DIR="${MANIFEST_DIR:-${OUTPUT_ROOT}/manifest}"
PRED_DIR="${PRED_DIR:-${OUTPUT_ROOT}/predictions}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/eval}"
DEBUG_OUTPUT_DIR="${DEBUG_OUTPUT_DIR:-${OUTPUT_ROOT}/debug_render}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${REPO_ROOT}/.offload_${MODEL_TAG}_exam2}"

START_INDEX="${START_INDEX:-0}"
# Default is smoke. Use LIMIT= for full run.
LIMIT="${LIMIT-5}"
PANELS="${PANELS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
DTYPE="${DTYPE:-bfloat16}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
PROMPT_MODE="${PROMPT_MODE:-expected_count}"
PANEL_WIDTH="${PANEL_WIDTH:-512}"
PANEL_HEIGHT="${PANEL_HEIGHT:-384}"
COLUMNS="${COLUMNS:-3}"
INPUT_MODE="${INPUT_MODE:-multi_image}"
PANEL_CONTEXT_MODE="${PANEL_CONTEXT_MODE:-paired_crop}"
PANEL_CAPTION_MODE="${PANEL_CAPTION_MODE:-none}"
PAIRED_CROP_COORDINATE_POLICY="${PAIRED_CROP_COORDINATE_POLICY:-paired_canvas_map}"
GAZE_CROP_RATIO="${GAZE_CROP_RATIO:-0.35}"
CROP_OUTPUT_SIZE="${CROP_OUTPUT_SIZE:-768}"
ACCEPTABLE_TOP_K="${ACCEPTABLE_TOP_K:-2}"
SAMPLE_INNER_MARGIN_RATIO="${SAMPLE_INNER_MARGIN_RATIO:-0.15}"
PANEL_SELECTION_STRATEGY="${PANEL_SELECTION_STRATEGY:-evidence}"
CANDIDATE_STEP_SECONDS="${CANDIDATE_STEP_SECONDS:-0.5}"
EVIDENCE_SEGMENT_DURATION_SECONDS="${EVIDENCE_SEGMENT_DURATION_SECONDS:-0.5}"
GAZE_STABILITY_WINDOW_SECONDS="${GAZE_STABILITY_WINDOW_SECONDS:-0.3}"
AUTO_OFFSET_SOURCE="${AUTO_OFFSET_SOURCE:-hybrid}"
HYBRID_THRESHOLD_SECONDS="${HYBRID_THRESHOLD_SECONDS:-1.0}"
HYBRID_VIDEO_BIAS_SECONDS="${HYBRID_VIDEO_BIAS_SECONDS:-0.5}"

BUILD_MANIFEST="${BUILD_MANIFEST:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_DEBUG_RENDER="${RUN_DEBUG_RENDER:-0}"
OVERWRITE_FRAMES="${OVERWRITE_FRAMES:-0}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-0}"
DEBUG_RENDER_MAX_EVENTS="${DEBUG_RENDER_MAX_EVENTS:-200}"

SCENES_ARGS=()
if [[ -n "${SCENES:-}" ]]; then
  # shellcheck disable=SC2206
  SCENES_ARGS=(--scenes ${SCENES})
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

OVERWRITE_FRAME_ARGS=()
if [[ "${OVERWRITE_FRAMES}" == "1" ]]; then
  OVERWRITE_FRAME_ARGS=(--overwrite_frames)
fi

OVERWRITE_PRED_ARGS=()
if [[ "${OVERWRITE_PREDICTIONS}" == "1" ]]; then
  OVERWRITE_PRED_ARGS=(--overwrite)
fi

KEEP_DUPLICATE_FRAME_ARGS=()
if [[ "${KEEP_DUPLICATE_FRAMES:-0}" == "1" ]]; then
  KEEP_DUPLICATE_FRAME_ARGS=(--keep_duplicate_frames)
fi

LOCAL_FILES_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARGS=(--local_files_only)
fi

LOAD_ARGS=()
if [[ "${LOAD_IN_8BIT}" == "1" ]]; then
  LOAD_ARGS=(--load_in_8bit)
else
  LOAD_ARGS=(--no_load_in_8bit)
fi

echo "Starting InternVL3-38B exam2 2D point workflow"
echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Start index: ${START_INDEX}, limit: ${LIMIT:-ALL}, panels: ${PANELS}, load_in_8bit=${LOAD_IN_8BIT}"
echo "Prompt mode: ${PROMPT_MODE}, input_mode=${INPUT_MODE}, panel_context_mode=${PANEL_CONTEXT_MODE}, coordinate_policy=${PAIRED_CROP_COORDINATE_POLICY}"

mkdir -p "${OUTPUT_ROOT}" "${PRED_DIR}" "${EVAL_OUTPUT_DIR}" "${DEBUG_OUTPUT_DIR}" "${OFFLOAD_FOLDER}"

if [[ "${BUILD_MANIFEST}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/build_2d_eval_manifest.py" \
    --repo_root "${REPO_ROOT}" \
    --eval_dir "${EVAL_DIR}" \
    --output_dir "${MANIFEST_DIR}" \
    --start_index "${START_INDEX}" \
    "${LIMIT_ARGS[@]}" \
    "${SCENES_ARGS[@]}" \
    --panels "${PANELS}" \
    --auto_video_time_offset \
    --auto_video_time_offset_source "${AUTO_OFFSET_SOURCE}" \
    --hybrid_offset_threshold_seconds "${HYBRID_THRESHOLD_SECONDS}" \
    --hybrid_video_time_bias_seconds "${HYBRID_VIDEO_BIAS_SECONDS}" \
    --acceptable_top_k "${ACCEPTABLE_TOP_K}" \
    --sample_inner_margin_ratio "${SAMPLE_INNER_MARGIN_RATIO}" \
    --panel_selection_strategy "${PANEL_SELECTION_STRATEGY}" \
    --candidate_step_seconds "${CANDIDATE_STEP_SECONDS}" \
    --evidence_segment_duration_seconds "${EVIDENCE_SEGMENT_DURATION_SECONDS}" \
    --gaze_stability_window_seconds "${GAZE_STABILITY_WINDOW_SECONDS}" \
    "${KEEP_DUPLICATE_FRAME_ARGS[@]}" \
    "${OVERWRITE_FRAME_ARGS[@]}"
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  python "${REPO_ROOT}/internvl/run_internvl_2d_point_grounding.py" \
    --repo_root "${REPO_ROOT}" \
    --manifest "${MANIFEST_DIR}/manifest_all.csv" \
    --output_csv "${PRED_DIR}/internvl_2d_predictions.csv" \
    --output_json_dir "${PRED_DIR}/json" \
    --model_input_dir "${PRED_DIR}/model_inputs" \
    --model_name "${MODEL_NAME}" \
    --dtype "${DTYPE}" \
    "${LOCAL_FILES_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    --device_map "${DEVICE_MAP}" \
    --image_size "${IMAGE_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --offload_folder "${OFFLOAD_FOLDER}" \
    --start_index "${START_INDEX}" \
    "${LIMIT_ARGS[@]}" \
    "${SCENES_ARGS[@]}" \
    --prompt_mode "${PROMPT_MODE}" \
    --input_mode "${INPUT_MODE}" \
    --panel_context_mode "${PANEL_CONTEXT_MODE}" \
    --panel_caption_mode "${PANEL_CAPTION_MODE}" \
    --paired_crop_coordinate_policy "${PAIRED_CROP_COORDINATE_POLICY}" \
    --gaze_crop_ratio "${GAZE_CROP_RATIO}" \
    --crop_output_size "${CROP_OUTPUT_SIZE}" \
    --panel_width "${PANEL_WIDTH}" \
    --panel_height "${PANEL_HEIGHT}" \
    --columns "${COLUMNS}" \
    --continue_on_error \
    "${OVERWRITE_PRED_ARGS[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/evaluate_2d_point_grounding.py" \
    --manifest "${MANIFEST_DIR}/manifest_all.csv" \
    --pred_csv "${PRED_DIR}/internvl_2d_predictions.csv" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --panel_width "${PANEL_WIDTH}" \
    --panel_height "${PANEL_HEIGHT}" \
    --columns "${COLUMNS}" \
    --coordinate_mode panel
fi

if [[ "${RUN_DEBUG_RENDER}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/render_prediction_debug.py" \
    --manifest "${MANIFEST_DIR}/manifest_all.csv" \
    --pred_csv "${PRED_DIR}/internvl_2d_predictions.csv" \
    --output_dir "${DEBUG_OUTPUT_DIR}" \
    --max_events "${DEBUG_RENDER_MAX_EVENTS}"
fi

echo "Finished InternVL3-38B exam2 workflow."
