#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/workspace/usr3/miniconda3/envs/trimodal/bin}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
MANIFEST="${MANIFEST:-exam3_point_grounding/outputs_full_v9_20260709/manifest.csv}"
GT_MANIFEST="${GT_MANIFEST:-exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation/exam3/outputs_qwen3vl30b_v9_input_mask_v3_full}"
VARIANT="${VARIANT:?Set VARIANT to no_visual, no_gaze, no_hand, no_gaze_hand, or no_instruction}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DTYPE="${DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SAMPLE_KEYS_FILE="${SAMPLE_KEYS_FILE:-}"
OVERWRITE_INFERENCE="${OVERWRITE_INFERENCE:-0}"
RUN_QWEN="${RUN_QWEN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

case "${VARIANT}" in
  no_visual|no_gaze|no_hand|no_gaze_hand|no_instruction) ;;
  *) echo "Unsupported VARIANT=${VARIANT}" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES
export PATH="${CONDA_ENV_BIN}:${PATH}"
cd "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_ROOT}/${VARIANT}"
PRED_CSV="${OUTPUT_DIR}/predictions.csv"
RAW_DIR="${OUTPUT_DIR}/raw"
EVAL_DIR="${OUTPUT_DIR}/eval"
mkdir -p "${RAW_DIR}" "${EVAL_DIR}"

key_args=()
eval_gt="${GT_MANIFEST}"
if [[ -n "${SAMPLE_KEYS_FILE}" ]]; then
  key_args+=(--sample_keys_file "${SAMPLE_KEYS_FILE}")
  eval_gt="${OUTPUT_DIR}/gt_manifest_eval_subset.csv"
  python ablation/exam3/subset_gt_manifest.py \
    --input "${GT_MANIFEST}" \
    --keys "${SAMPLE_KEYS_FILE}" \
    --output "${eval_gt}"
fi

local_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_args+=(--local_files_only)
fi

overwrite_args=()
if [[ "${OVERWRITE_INFERENCE}" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

printf '%s\n' \
  "variant=${VARIANT}" \
  "model_name=${MODEL_NAME}" \
  "manifest=${MANIFEST}" \
  "gt_manifest=${eval_gt}" \
  "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}" \
  "max_new_tokens=${MAX_NEW_TOKENS}" \
  "dtype=${DTYPE}" \
  "do_sample=false" \
  "prompt_template=exam3_point_grounding/prompts/qwen3vl_point_grounding.md" \
  > "${OUTPUT_DIR}/run_config.txt"

if [[ "${RUN_QWEN}" == "1" ]]; then
  python exam3_point_grounding/run_qwen3vl_point_grounding.py \
    --repo_root "${REPO_ROOT}" \
    --manifest "${MANIFEST}" \
    --output_csv "${PRED_CSV}" \
    --output_json_dir "${RAW_DIR}" \
    --model_name "${MODEL_NAME}" \
    --dtype "${DTYPE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --ablation_variant "${VARIANT}" \
    --flush_every 10 \
    --continue_on_error \
    "${local_args[@]}" \
    "${key_args[@]}" \
    "${overwrite_args[@]}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python exam3_point_grounding/evaluate_point_grounding.py \
    --repo_root "${REPO_ROOT}" \
    --pred_csv "${PRED_CSV}" \
    --gt_manifest "${eval_gt}" \
    --output_dir "${EVAL_DIR}" \
    --report_path "${OUTPUT_DIR}/RESULTS.md"
fi
