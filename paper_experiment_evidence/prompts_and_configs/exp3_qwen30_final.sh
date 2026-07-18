#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/workspace/usr3/miniconda3/envs/trimodal/bin}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-exam3_point_grounding/outputs}"
QWEN_OUTPUT_DIR="${QWEN_OUTPUT_DIR:-${OUTPUT_ROOT}/qwen3vl30b}"
SCENES="${SCENES:-scene1 scene2 scene3 scene4_room1 scene4_room2 scene4_room3 scene4_room4 scene5}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DTYPE="${DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
NO_EXTRACT_FRAMES="${NO_EXTRACT_FRAMES:-0}"
OVERWRITE_FRAMES="${OVERWRITE_FRAMES:-0}"
OVERWRITE_INFERENCE="${OVERWRITE_INFERENCE:-0}"
RUN_MANIFEST="${RUN_MANIFEST:-1}"
RUN_BASELINES="${RUN_BASELINES:-1}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

export CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_BIN}:${PATH}"

cd "${REPO_ROOT}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args+=(--limit "${LIMIT}")
fi

extract_args=()
if [[ "${NO_EXTRACT_FRAMES}" == "1" ]]; then
  extract_args+=(--no_extract_frames)
fi
if [[ "${OVERWRITE_FRAMES}" == "1" ]]; then
  extract_args+=(--overwrite_frames)
fi

local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_args+=(--local_files_only)
fi

overwrite_args=()
if [[ "${OVERWRITE_INFERENCE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

if [[ "${RUN_MANIFEST}" == "1" ]]; then
  python exam3_point_grounding/build_point_grounding_manifest.py \
    --repo_root "${REPO_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --scenes ${SCENES} \
    --start_index "${START_INDEX}" \
    "${limit_args[@]}" \
    "${extract_args[@]}"
fi

if [[ "${RUN_BASELINES}" == "1" ]]; then
  python exam3_point_grounding/run_cue_baselines.py \
    --repo_root "${REPO_ROOT}" \
    --manifest "${OUTPUT_ROOT}/manifest.csv" \
    --output_dir "${OUTPUT_ROOT}/cue_baselines"
fi

if [[ "${RUN_QWEN}" == "1" ]]; then
  mkdir -p "${QWEN_OUTPUT_DIR}/raw"
  python exam3_point_grounding/run_qwen3vl_point_grounding.py \
    --repo_root "${REPO_ROOT}" \
    --manifest "${OUTPUT_ROOT}/manifest.csv" \
    --output_csv "${QWEN_OUTPUT_DIR}/predictions.csv" \
    --output_json_dir "${QWEN_OUTPUT_DIR}/raw" \
    --model_name "${MODEL_NAME}" \
    --dtype "${DTYPE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    "${local_args[@]}" \
    --continue_on_error \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  if [[ -f "${QWEN_OUTPUT_DIR}/predictions.csv" ]]; then
    python exam3_point_grounding/evaluate_point_grounding.py \
      --repo_root "${REPO_ROOT}" \
      --pred_csv "${QWEN_OUTPUT_DIR}/predictions.csv" \
      --gt_manifest "${OUTPUT_ROOT}/gt_manifest_eval.csv" \
      --output_dir "${QWEN_OUTPUT_DIR}" \
      --report_path "exam3_point_grounding/RESULTS_POINT_3D_GROUNDING.md"
  fi
  for method in gaze_copy hand_copy gaze_hand_fusion; do
    if [[ -f "${OUTPUT_ROOT}/cue_baselines/${method}/predictions.csv" ]]; then
      python exam3_point_grounding/evaluate_point_grounding.py \
        --repo_root "${REPO_ROOT}" \
        --pred_csv "${OUTPUT_ROOT}/cue_baselines/${method}/predictions.csv" \
        --gt_manifest "${OUTPUT_ROOT}/gt_manifest_eval.csv" \
        --output_dir "${OUTPUT_ROOT}/cue_baselines/${method}/eval"
    fi
  done
fi
