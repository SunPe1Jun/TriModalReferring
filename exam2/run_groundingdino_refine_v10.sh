#!/usr/bin/env bash
set -euo pipefail
trap 'echo "ERROR: command failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# Refine Qwen3-VL v10 2D point predictions with GroundingDINO boxes.
#
# Smoke:
#   LIMIT=20 DINO_MODEL_NAME=/workspace/usr3/grounding-dino-base bash exam2/run_groundingdino_refine_v10.sh
#
# Full:
#   LIMIT= DINO_MODEL_NAME=/workspace/usr3/grounding-dino-base bash exam2/run_groundingdino_refine_v10.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

V10_ROOT="${V10_ROOT:-${REPO_ROOT}/exam2/outputs_qwen3vl30b_2d_point_hybrid_v10}"
MANIFEST="${MANIFEST:-${V10_ROOT}/manifest/manifest_all.csv}"
PRED_CSV="${PRED_CSV:-${V10_ROOT}/predictions/qwen3vl_2d_predictions.csv}"

DINO_MODEL_NAME="${DINO_MODEL_NAME:-/workspace/usr3/grounding-dino-base}"
DINO_TAG="${DINO_TAG:-groundingdino_base}"
POINT_MODE="${POINT_MODE:-box_center}"
BOX_SELECT_MODE="${BOX_SELECT_MODE:-vl_point_nearest}"
BOX_THRESHOLD="${BOX_THRESHOLD:-0.30}"
TEXT_THRESHOLD="${TEXT_THRESHOLD:-0.25}"
PROXIMITY_WEIGHT="${PROXIMITY_WEIGHT:-0.25}"
DEVICE="${DEVICE:-cuda}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
FALLBACK_TO_VL="${FALLBACK_TO_VL:-1}"
STRIP_DEICTIC="${STRIP_DEICTIC:-1}"

START_INDEX="${START_INDEX:-0}"
# Unset LIMIT keeps smoke default 20; explicitly passing LIMIT= means full run.
LIMIT="${LIMIT-20}"
SCENES="${SCENES:-}"
MAX_EVENTS="${MAX_EVENTS:-}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/exam2/outputs_${DINO_TAG}_refine_v10_${POINT_MODE}}"
PRED_DIR="${PRED_DIR:-${OUTPUT_ROOT}/predictions}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/eval}"
DEBUG_DIR="${DEBUG_DIR:-${OUTPUT_ROOT}/debug_render}"
JSON_DIR="${JSON_DIR:-${PRED_DIR}/json}"
DETECTION_JSON_DIR="${DETECTION_JSON_DIR:-${PRED_DIR}/dino_cache}"
OUTPUT_CSV="${OUTPUT_CSV:-${PRED_DIR}/dino_refined_predictions.csv}"

RUN_REFINE="${RUN_REFINE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_DEBUG_RENDER="${RUN_DEBUG_RENDER:-1}"
OVERWRITE="${OVERWRITE:-0}"
OVERWRITE_DETECTIONS="${OVERWRITE_DETECTIONS:-0}"
DEBUG_RENDER_MAX_EVENTS="${DEBUG_RENDER_MAX_EVENTS:-200}"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

SCENES_ARGS=()
if [[ -n "${SCENES}" ]]; then
  # shellcheck disable=SC2206
  SCENES_ARGS=(--scenes ${SCENES})
fi

MAX_EVENTS_ARGS=()
if [[ -n "${MAX_EVENTS}" ]]; then
  MAX_EVENTS_ARGS=(--max_events "${MAX_EVENTS}")
fi

LOCAL_FILES_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARGS=(--local_files_only)
fi

FALLBACK_ARGS=()
if [[ "${FALLBACK_TO_VL}" == "1" ]]; then
  FALLBACK_ARGS=(--fallback_to_vl)
fi

STRIP_DEICTIC_ARGS=(--no_strip_deictic)
if [[ "${STRIP_DEICTIC}" == "1" ]]; then
  STRIP_DEICTIC_ARGS=(--strip_deictic)
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

OVERWRITE_DET_ARGS=()
if [[ "${OVERWRITE_DETECTIONS}" == "1" ]]; then
  OVERWRITE_DET_ARGS=(--overwrite_detections)
fi

echo "Starting GroundingDINO refine for v10"
echo "Repo root: ${REPO_ROOT}"
echo "v10 root: ${V10_ROOT}"
echo "Manifest: ${MANIFEST}"
echo "Pred CSV: ${PRED_CSV}"
echo "DINO model: ${DINO_MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Start index: ${START_INDEX}, limit: ${LIMIT:-ALL}, scenes: ${SCENES:-ALL}, max_events: ${MAX_EVENTS:-NONE}"
echo "Point mode: ${POINT_MODE}, box_select_mode: ${BOX_SELECT_MODE}, box_threshold=${BOX_THRESHOLD}, text_threshold=${TEXT_THRESHOLD}"

mkdir -p "${PRED_DIR}" "${EVAL_DIR}" "${DEBUG_DIR}" "${JSON_DIR}" "${DETECTION_JSON_DIR}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Error: missing manifest: ${MANIFEST}" >&2
  exit 1
fi
if [[ ! -f "${PRED_CSV}" ]]; then
  echo "Error: missing v10 prediction CSV: ${PRED_CSV}" >&2
  exit 1
fi

if [[ "${RUN_REFINE}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/refine_v10_with_groundingdino.py" \
    --manifest "${MANIFEST}" \
    --pred_csv "${PRED_CSV}" \
    --output_csv "${OUTPUT_CSV}" \
    --output_json_dir "${JSON_DIR}" \
    --detection_json_dir "${DETECTION_JSON_DIR}" \
    --model_name "${DINO_MODEL_NAME}" \
    "${LOCAL_FILES_ARGS[@]}" \
    --device "${DEVICE}" \
    --box_threshold "${BOX_THRESHOLD}" \
    --text_threshold "${TEXT_THRESHOLD}" \
    --box_select_mode "${BOX_SELECT_MODE}" \
    --point_mode "${POINT_MODE}" \
    --proximity_weight "${PROXIMITY_WEIGHT}" \
    --start_index "${START_INDEX}" \
    "${LIMIT_ARGS[@]}" \
    "${SCENES_ARGS[@]}" \
    "${MAX_EVENTS_ARGS[@]}" \
    "${FALLBACK_ARGS[@]}" \
    "${STRIP_DEICTIC_ARGS[@]}" \
    "${OVERWRITE_ARGS[@]}" \
    "${OVERWRITE_DET_ARGS[@]}" \
    --continue_on_error
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/evaluate_2d_point_grounding.py" \
    --manifest "${MANIFEST}" \
    --pred_csv "${OUTPUT_CSV}" \
    --output_dir "${EVAL_DIR}" \
    --coordinate_mode panel \
    --start_index "${START_INDEX}" \
    "${LIMIT_ARGS[@]}" \
    "${SCENES_ARGS[@]}"
fi

if [[ "${RUN_DEBUG_RENDER}" == "1" ]]; then
  python "${REPO_ROOT}/exam2/render_prediction_debug.py" \
    --manifest "${MANIFEST}" \
    --pred_csv "${OUTPUT_CSV}" \
    --output_dir "${DEBUG_DIR}" \
    --max_events "${DEBUG_RENDER_MAX_EVENTS}" \
    --start_index "${START_INDEX}" \
    "${LIMIT_ARGS[@]}" \
    "${SCENES_ARGS[@]}"
fi

echo "Finished GroundingDINO refine."
