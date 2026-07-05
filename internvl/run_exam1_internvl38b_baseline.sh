#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"

MODEL_NAME="${MODEL_NAME:-/workspace/usr3/InternVL3-38B-Instruct}"
MODEL_TAG="${MODEL_TAG:-internvl3_38b_baseline}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/internvl/outputs/exam1_${MODEL_TAG}}"
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
LOAD_IN_8BIT="${LOAD_IN_8BIT:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-mention_first}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"

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

LOAD_ARGS=()
if [[ "${LOAD_IN_8BIT}" == "1" ]]; then
  LOAD_ARGS=(--load_in_8bit)
else
  LOAD_ARGS=(--no_load_in_8bit)
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

echo "Starting InternVL3-38B exam1 closed-set 3D baseline"
echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Scenes: ${SCENES}"
echo "Start index: ${START_INDEX}, limit: ${LIMIT:-ALL}, load_in_8bit=${LOAD_IN_8BIT}"

cd "${REPO_ROOT}"

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  for scene in ${SCENES}; do
    out_dir="${PRED_ROOT}/${scene}"
    mkdir -p "${out_dir}"
    if [[ "${OVERWRITE_PREDICTIONS}" != "1" && -n "${LIMIT}" && -f "${out_dir}/row_${START_INDEX}.json" ]]; then
      echo "[skip inference] ${scene}: existing row_${START_INDEX}.json"
      continue
    fi
    echo "[infer] ${scene}"
    python "${REPO_ROOT}/internvl/run_internvl_3d_batch.py" \
      --input_csv "${DATA_DIR}/${scene}_api_input.csv" \
      --scene_anchor_csv "${DATA_DIR}/${scene}_anchor_table.tsv" \
      --output_dir "${out_dir}" \
      --model_name "${MODEL_NAME}" \
      --start_index "${START_INDEX}" \
      "${LIMIT_ARGS[@]}" \
      --dtype "${DTYPE}" \
      "${LOCAL_FILES_ARGS[@]}" \
      "${LOAD_ARGS[@]}" \
      --device_map "${DEVICE_MAP}" \
      --image_size "${IMAGE_SIZE}" \
      --input_mode "${INPUT_MODE}" \
      --max_video_frames "${MAX_VIDEO_FRAMES}" \
      --max_evidence_segments "${MAX_EVIDENCE_SEGMENTS}" \
      --evidence_segment_duration "${EVIDENCE_SEGMENT_DURATION}" \
      --prompt_style "${PROMPT_STYLE}" \
      --prompt_strategy "${PROMPT_STRATEGY}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --offload_folder "${OFFLOAD_FOLDER}" \
      --continue_on_error \
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

echo "Finished InternVL3-38B exam1 baseline."
