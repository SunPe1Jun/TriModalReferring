#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/ablation/exam1/outputs}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/ablation/exam1/reports}"
OFFLOAD_BASE="${OFFLOAD_BASE:-${REPO_ROOT}/.offload_ablation_exam1}"

START_INDEX="${START_INDEX:-0}"
# Default is a smoke run. Use LIMIT= for full evaluation.
LIMIT="${LIMIT-5}"
VARIANTS="${VARIANTS:-no_visual no_gaze no_hand no_gaze_hand language_anchors_only}"
SCENES="${SCENES:-scene1 scene2 scene3 scene4_room1 scene4_room2 scene4_room3 scene4_room4 scene5}"

RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-0}"

SUMMARY_SCOPE_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  SUMMARY_SCOPE_ARGS=(--summary_scope predicted_rows)
fi

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

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}" "${OFFLOAD_BASE}"

variant_modalities() {
  case "$1" in
    no_visual) printf '%s\n' 'visual' ;;
    no_gaze) printf '%s\n' 'gaze' ;;
    no_hand) printf '%s\n' 'hand' ;;
    no_gaze_hand) printf '%s\n' 'gaze,hand' ;;
    language_anchors_only) printf '%s\n' 'visual,gaze,hand,structured_geometry,timeline' ;;
    no_structured_geometry) printf '%s\n' 'structured_geometry,timeline' ;;
    *) echo "Unknown exam1 ablation variant: $1" >&2; exit 1 ;;
  esac
}

scene_batch_script() {
  case "$1" in
    scene1) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene1_local_3d_batch.sh" ;;
    scene2) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene2_local_3d_batch.sh" ;;
    scene3) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene3_local_3d_batch.sh" ;;
    scene4_room1) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene4_room1_local_3d_batch.sh" ;;
    scene4_room2) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene4_room2_local_3d_batch.sh" ;;
    scene4_room3) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene4_room3_local_3d_batch.sh" ;;
    scene4_room4) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene4_room4_local_3d_batch.sh" ;;
    scene5) printf '%s\n' "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_scene5_local_3d_batch.sh" ;;
    *) echo "Unknown scene: $1" >&2; exit 1 ;;
  esac
}

scene_root() {
  case "$1" in
    scene1) printf '%s\n' "${SCENE1_ROOT:-/workspace/usr3/V3dMD/scene1}" ;;
    scene2) printf '%s\n' "${SCENE2_ROOT:-/workspace/usr3/V3dMD/scene2}" ;;
    scene3) printf '%s\n' "${SCENE3_ROOT:-/workspace/usr3/V3dMD/scene3}" ;;
    scene4_room1) printf '%s\n' "${SCENE4_ROOM1_ROOT:-/workspace/usr3/V3dMD/scene4/room1}" ;;
    scene4_room2) printf '%s\n' "${SCENE4_ROOM2_ROOT:-/workspace/usr3/V3dMD/scene4/room2}" ;;
    scene4_room3) printf '%s\n' "${SCENE4_ROOM3_ROOT:-/workspace/usr3/V3dMD/scene4/room3}" ;;
    scene4_room4) printf '%s\n' "${SCENE4_ROOM4_ROOT:-/workspace/usr3/V3dMD/scene4/room4}" ;;
    scene5) printf '%s\n' "${SCENE5_ROOT:-/workspace/usr3/V3dMD/scene5}" ;;
  esac
}

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
  esac
}

run_variant() {
  local variant="$1"
  local modalities
  modalities="$(variant_modalities "${variant}")"
  local variant_root="${OUTPUT_ROOT}/${variant}"
  local pred_root="${variant_root}/predictions"
  local eval_root="${variant_root}/eval"
  local offload_dir="${OFFLOAD_BASE}/${variant}"
  mkdir -p "${pred_root}" "${eval_root}" "${offload_dir}"

  echo
  echo "========== exam1 variant=${variant} modalities=${modalities} =========="
  echo "Output: ${variant_root}"

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    for scene in ${SCENES}; do
      local out_dir="${pred_root}/${scene}"
      mkdir -p "${out_dir}"
      if [[ "${OVERWRITE_PREDICTIONS}" != "1" && -n "${LIMIT}" && -f "${out_dir}/row_${START_INDEX}.json" ]]; then
        echo "[skip inference] ${variant}/${scene}: existing row_${START_INDEX}.json"
        continue
      fi
      echo "[infer] ${variant}/${scene}"
      CMD=(
        python "${REPO_ROOT}/ablation/exam1/scripts/grounding/run_qwen3vl_local_3d_batch_inprocess.py"
        --input_csv "${DATA_DIR}/${scene}_api_input.csv"
        --scene_anchor_csv "${DATA_DIR}/${scene}_anchor_table.tsv"
        --output_dir "${out_dir}"
        --model_name "${MODEL_NAME}"
        --start_index "${START_INDEX}"
        --dtype "${DTYPE}"
        --input_mode "${INPUT_MODE}"
        --max_video_frames "${MAX_VIDEO_FRAMES}"
        --max_evidence_segments "${MAX_EVIDENCE_SEGMENTS}"
        --evidence_segment_duration "${EVIDENCE_SEGMENT_DURATION}"
        --prompt_style "${PROMPT_STYLE}"
        --prompt_strategy "${PROMPT_STRATEGY}"
        --ablate_modalities "${modalities}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --offload_folder "${offload_dir}"
        --continue_on_error
      )
      if [[ -n "${LIMIT}" ]]; then
        CMD+=(--limit "${LIMIT}")
      fi
      if [[ "${OVERWRITE_PREDICTIONS}" == "1" ]]; then
        CMD+=(--overwrite)
      fi
      if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
        CMD+=(--local_files_only)
      fi
      if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
        CMD+=(--use_flash_attn)
      fi
      "${CMD[@]}"
    done
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    for scene in ${SCENES}; do
      local pred_dir="${pred_root}/${scene}"
      if [[ ! -d "${pred_dir}" ]]; then
        echo "[skip eval] missing pred dir: ${pred_dir}"
        continue
      fi
      echo "[eval] ${variant}/${scene}"
      python "${REPO_ROOT}/ablation/exam1/scripts/eval/evaluate_local_3d_object_match.py" \
        --pred_dir "${pred_dir}" \
        --gt_file "$(gt_file "${scene}")" \
        --anchor_csv "${DATA_DIR}/${scene}_anchor_table.tsv" \
        --output_csv "${eval_root}/${scene}_match_eval.csv" \
        --output_json "${eval_root}/${scene}_match_eval_summary.json" \
        "${SUMMARY_SCOPE_ARGS[@]}"
    done
    python "${REPO_ROOT}/ablation/exam1/scripts/eval/summarize_match_eval_summaries.py" \
      --input_dir "${eval_root}" \
      --glob "*_match_eval_summary.json" \
      --output_csv "${eval_root}/all_scene_match_eval_summary.csv" \
      --output_md "${eval_root}/all_scene_match_eval_summary.md"
  fi
}

cd "${REPO_ROOT}"
for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  python "${REPO_ROOT}/ablation/exam1/summarize_exam1_ablation.py" \
    --repo_root "${REPO_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --report_dir "${REPORT_DIR}"
fi

echo "Finished exam1 ablation workflow."
