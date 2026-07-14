#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/exam3_point_grounding/outputs_full_v9_20260709}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL_NAME="${MODEL_NAME:-/workspace/usr3/InternVL3-38B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/internvl/outputs/exam3_internvl38b_point_grounding}"
PRED_CSV="${PRED_CSV:-${OUTPUT_ROOT}/predictions.csv}"
RAW_DIR="${RAW_DIR:-${OUTPUT_ROOT}/raw}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/eval}"
LIMIT="${LIMIT:-}"
SCENES="${SCENES:-scene1 scene2 scene3 scene4_room1 scene4_room2 scene4_room3 scene4_room4 scene5}"
START_INDEX="${START_INDEX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MAX_IMAGES="${MAX_IMAGES:-2}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
OVERWRITE_INFERENCE="${OVERWRITE_INFERENCE:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${REPO_ROOT}/.offload_exam3_internvl38b_point_grounding}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${RAW_DIR}" "${EVAL_DIR}" "${OFFLOAD_FOLDER}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

scene_args=()
if [[ -n "${SCENES}" ]]; then
  # shellcheck disable=SC2206
  scene_args=(--scenes ${SCENES})
fi

local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_args+=(--local_files_only)
fi

load_args=()
if [[ "${LOAD_IN_8BIT}" == "1" ]]; then
  load_args+=(--load_in_8bit)
else
  load_args+=(--no_load_in_8bit)
fi

overwrite_args=()
if [[ "${OVERWRITE_INFERENCE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  python "${REPO_ROOT}/internvl/run_internvl_point_grounding.py"             --repo_root "${REPO_ROOT}"             --manifest "${MANIFEST_ROOT}/manifest.csv"             --output_csv "${PRED_CSV}"             --output_json_dir "${RAW_DIR}"             --model_name "${MODEL_NAME}"             --dtype "${DTYPE}"             --device_map "${DEVICE_MAP}"             --image_size "${IMAGE_SIZE}"             --max_images "${MAX_IMAGES}"             --max_new_tokens "${MAX_NEW_TOKENS}"             --offload_folder "${OFFLOAD_FOLDER}"             "${local_args[@]}"             "${load_args[@]}"             --continue_on_error             "${overwrite_args[@]}"             --start_index "${START_INDEX}"             "${limit_args[@]}"             "${scene_args[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python "${REPO_ROOT}/exam3_point_grounding/evaluate_point_grounding.py"             --repo_root "${REPO_ROOT}"             --pred_csv "${PRED_CSV}"             --gt_manifest "${MANIFEST_ROOT}/gt_manifest_eval.csv"             --output_dir "${EVAL_DIR}"             --report_path "${OUTPUT_ROOT}/EXPERIMENT3_FULL_RESULTS_V9.md"
fi
