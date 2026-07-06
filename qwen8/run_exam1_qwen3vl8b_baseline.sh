#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-8B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen3vl8b_baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/qwen8/outputs/exam1_${MODEL_TAG}}"
PRED_ROOT="${PRED_ROOT:-${OUTPUT_ROOT}/predictions}"
EVAL_ROOT="${EVAL_ROOT:-${OUTPUT_ROOT}/eval}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${REPO_ROOT}/.offload_${MODEL_TAG}_exam1}"

START_INDEX="${START_INDEX:-0}"
# Default is smoke. Use LIMIT= for full run.
LIMIT="${LIMIT-5}"
SCENES="${SCENES:-scene1 scene2 scene3 scene4_room1 scene4_room2 scene4_room3 scene4_room4 scene5}"

RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-0}"

DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-mention_first}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"

SUMMARY_SCOPE_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  SUMMARY_SCOPE_ARGS=(--summary_scope predicted_rows)
fi

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE_PREDICTIONS}" == "1" ]]; then
  OVERWRITE_ARGS=(--overwrite)
fi

LOCAL_FILES_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARGS=(--local_files_only)
fi

FLASH_ATTN_ARGS=()
if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
  FLASH_ATTN_ARGS=(--use_flash_attn)
fi

gt_file() {
  case "$1" in
    scene1) printf '%s\n' "${DATA_DIR}/scene1_cleaned_v3.xlsx" ;;
    scene2) printf '%s\n' "${DATA_DIR}/scene2_cleaned_v2.xlsx" ;;
    scene3) printf '%s\n' "${DATA_DIR}/scene3.xlsx" ;;
    scene4_room1) printf '%s\n' "${DATA_DIR}/scene4_room1.xlsx" ;;
    scene4_room2) printf '%s\n' "${DATA_DIR}/scene4_room2.xlsx" ;;
    scene4_room3) printf '%s\n' "${DATA_DIR}/scene4_room3.xlsx" ;;
    scene4_room4) printf '%s\n' "${DATA_DIR}/scene4_room4.xlsx" ;;
    scene5) printf '%s\n' "${DATA_DIR}/scene5.xlsx" ;;
    *) echo "Unknown scene: $1" >&2; exit 1 ;;
  esac
}

mkdir -p "${PRED_ROOT}" "${EVAL_ROOT}" "${OFFLOAD_FOLDER}"

echo "Starting Qwen3-VL-8B exam1 closed-set 3D baseline"
echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Scenes: ${SCENES}"
echo "Start index: ${START_INDEX}, limit: ${LIMIT:-ALL}"

cd "${REPO_ROOT}"

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Error: model path does not exist: ${MODEL_NAME}" >&2
  exit 1
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  for scene in ${SCENES}; do
    out_dir="${PRED_ROOT}/${scene}"
    mkdir -p "${out_dir}"
    echo "[infer] ${scene}"
    python "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_qwen3vl_local_3d_batch_inprocess.py" \
      --input_csv "${DATA_DIR}/${scene}_api_input.csv" \
      --scene_anchor_csv "${DATA_DIR}/${scene}_anchor_table.tsv" \
      --output_dir "${out_dir}" \
      --model_name "${MODEL_NAME}" \
      --start_index "${START_INDEX}" \
      "${LIMIT_ARGS[@]}" \
      --dtype "${DTYPE}" \
      --input_mode "${INPUT_MODE}" \
      --max_video_frames "${MAX_VIDEO_FRAMES}" \
      --max_evidence_segments "${MAX_EVIDENCE_SEGMENTS}" \
      --evidence_segment_duration "${EVIDENCE_SEGMENT_DURATION}" \
      --prompt_style "${PROMPT_STYLE}" \
      --prompt_strategy "${PROMPT_STRATEGY}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --offload_folder "${OFFLOAD_FOLDER}" \
      --continue_on_error \
      "${LOCAL_FILES_ARGS[@]}" \
      "${FLASH_ATTN_ARGS[@]}" \
      "${OVERWRITE_ARGS[@]}"
  done
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  for scene in ${SCENES}; do
    pred_dir="${PRED_ROOT}/${scene}"
    if [[ ! -d "${pred_dir}" ]]; then
      echo "[skip eval] missing pred dir: ${pred_dir}"
      continue
    fi
    echo "[eval] ${scene}"
    python "${REPO_ROOT}/ablation/exam1/scripts/eval/evaluate_local_3d_object_match.py" \
      --pred_dir "${pred_dir}" \
      --gt_file "$(gt_file "${scene}")" \
      --anchor_csv "${DATA_DIR}/${scene}_anchor_table.tsv" \
      --output_csv "${EVAL_ROOT}/${scene}_match_eval.csv" \
      --output_json "${EVAL_ROOT}/${scene}_match_eval_summary.json" \
      "${SUMMARY_SCOPE_ARGS[@]}"
  done
  python "${REPO_ROOT}/ablation/exam1/scripts/eval/summarize_match_eval_summaries.py" \
    --input_dir "${EVAL_ROOT}" \
    --glob "*_match_eval_summary.json" \
    --output_csv "${EVAL_ROOT}/all_scene_match_eval_summary.csv" \
    --output_md "${EVAL_ROOT}/all_scene_match_eval_summary.md"
fi

echo "Finished Qwen3-VL-8B exam1 baseline."
