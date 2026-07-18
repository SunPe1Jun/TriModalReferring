#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${REPO_ROOT}/exam3_point_grounding/outputs_full_v9_20260709}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-8B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/qwen8/outputs/exam3_qwen3vl8b_point_grounding}"
PRED_CSV="${PRED_CSV:-${OUTPUT_ROOT}/predictions.csv}"
RAW_DIR="${RAW_DIR:-${OUTPUT_ROOT}/raw}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_ROOT}/eval}"
LIMIT="${LIMIT:-}"
SCENES="${SCENES:-scene1 scene2 scene3 scene4_room1 scene4_room2 scene4_room3 scene4_room4 scene5}"
START_INDEX="${START_INDEX:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DTYPE="${DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
OVERWRITE_INFERENCE="${OVERWRITE_INFERENCE:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${RAW_DIR}" "${EVAL_DIR}"

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

overwrite_args=()
if [[ "${OVERWRITE_INFERENCE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  python "${REPO_ROOT}/exam3_point_grounding/run_qwen3vl_point_grounding.py"             --repo_root "${REPO_ROOT}"             --manifest "${MANIFEST_ROOT}/manifest.csv"             --output_csv "${PRED_CSV}"             --output_json_dir "${RAW_DIR}"             --model_name "${MODEL_NAME}"             --dtype "${DTYPE}"             --max_new_tokens "${MAX_NEW_TOKENS}"             "${local_args[@]}"             --continue_on_error             "${overwrite_args[@]}"             --start_index "${START_INDEX}"             "${limit_args[@]}"             "${scene_args[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python "${REPO_ROOT}/exam3_point_grounding/evaluate_point_grounding.py"             --repo_root "${REPO_ROOT}"             --pred_csv "${PRED_CSV}"             --gt_manifest "${MANIFEST_ROOT}/gt_manifest_eval.csv"             --output_dir "${EVAL_DIR}"             --report_path "${OUTPUT_ROOT}/EXPERIMENT3_FULL_RESULTS_V9.md"
fi
