#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/ablation/exam2/outputs}"
REPORT_DIR="${REPORT_DIR:-${REPO_ROOT}/ablation/exam2/reports}"
OFFLOAD_BASE="${OFFLOAD_BASE:-${REPO_ROOT}/.offload_ablation_exam2}"
BASE_MANIFEST="${BASE_MANIFEST:-${REPO_ROOT}/exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv}"
EVAL_DIR="${EVAL_DIR:-${REPO_ROOT}/data/match_eval_qwen3vl30b_mention_first_v3}"

START_INDEX="${START_INDEX:-0}"
# Default is a smoke run. Use LIMIT= for full evaluation.
LIMIT="${LIMIT-5}"
VARIANTS="${VARIANTS:-full_panels_no_crop no_gaze_text_prior no_gaze instruction_only_prompt}"
SCENES="${SCENES:-}"

BUILD_MANIFEST="${BUILD_MANIFEST:-0}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
RUN_DEBUG_RENDER="${RUN_DEBUG_RENDER:-0}"
RUN_SUMMARY="${RUN_SUMMARY:-1}"
OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS:-0}"
OVERWRITE_FRAMES="${OVERWRITE_FRAMES:-0}"

EVAL_LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  EVAL_LIMIT_ARGS=(--start_index "${START_INDEX}" --limit "${LIMIT}")
fi
EVAL_SCENES_ARGS=()
if [[ -n "${SCENES}" ]]; then
  # shellcheck disable=SC2206
  EVAL_SCENES_ARGS=(--scenes ${SCENES})
fi

DTYPE="${DTYPE:-auto}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
PANELS="${PANELS:-3}"
INPUT_MODE="${INPUT_MODE:-multi_image}"
PANEL_CAPTION_MODE="${PANEL_CAPTION_MODE:-none}"
PANEL_WIDTH="${PANEL_WIDTH:-512}"
PANEL_HEIGHT="${PANEL_HEIGHT:-384}"
COLUMNS="${COLUMNS:-3}"
PAIRED_CROP_COORDINATE_POLICY="${PAIRED_CROP_COORDINATE_POLICY:-paired_canvas_map}"
GAZE_MASK_RADIUS_RATIO="${GAZE_MASK_RADIUS_RATIO:-0.035}"

mkdir -p "${OUTPUT_ROOT}" "${REPORT_DIR}" "${OFFLOAD_BASE}"

variant_env() {
  case "$1" in
    full_panels_no_crop)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=full ABLATE_MODALITIES='
      ;;
    no_gaze_text_prior)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=full ABLATE_MODALITIES=gaze_text'
      ;;
    no_gaze)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=full ABLATE_MODALITIES=gaze'
      ;;
    no_hand_strict)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=full ABLATE_MODALITIES=hand_visual'
      ;;
    instruction_only_prompt)
      printf '%s\n' 'PROMPT_MODE=instruction_only PANEL_CONTEXT_MODE=paired_crop ABLATE_MODALITIES='
      ;;
    single_panel)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=paired_crop ABLATE_MODALITIES= PANELS=1'
      ;;
    blank_visual)
      printf '%s\n' 'PROMPT_MODE=expected_count PANEL_CONTEXT_MODE=full ABLATE_MODALITIES=visual'
      ;;
    *) echo "Unknown exam2 ablation variant: $1" >&2; exit 1 ;;
  esac
}

run_variant() {
  local variant="$1"
  local variant_root="${OUTPUT_ROOT}/${variant}"
  local manifest_dir="${variant_root}/manifest"
  local pred_dir="${variant_root}/predictions"
  local eval_dir="${variant_root}/eval"
  local debug_dir="${variant_root}/debug_render"
  local offload_dir="${OFFLOAD_BASE}/${variant}"
  mkdir -p "${variant_root}" "${manifest_dir}" "${pred_dir}" "${eval_dir}" "${debug_dir}" "${offload_dir}"

  local prompt_mode="expected_count"
  local panel_context_mode="paired_crop"
  local ablate_modalities=""
  local panels="${PANELS}"
  local env_line
  env_line="$(variant_env "${variant}")"
  for assignment in ${env_line}; do
    case "${assignment}" in
      PROMPT_MODE=*) prompt_mode="${assignment#PROMPT_MODE=}" ;;
      PANEL_CONTEXT_MODE=*) panel_context_mode="${assignment#PANEL_CONTEXT_MODE=}" ;;
      ABLATE_MODALITIES=*) ablate_modalities="${assignment#ABLATE_MODALITIES=}" ;;
      PANELS=*) panels="${assignment#PANELS=}" ;;
    esac
  done

  echo
  echo "========== exam2 variant=${variant} =========="
  echo "prompt_mode=${prompt_mode} panel_context_mode=${panel_context_mode} ablate_modalities=${ablate_modalities:-none} panels=${panels}"
  echo "Output: ${variant_root}"

  local variant_build_manifest="${BUILD_MANIFEST}"
  if [[ "${variant_build_manifest}" == "0" && -f "${BASE_MANIFEST}" && "${panels}" == "3" ]]; then
    mkdir -p "${manifest_dir}"
    ln -sfn "$(dirname "${BASE_MANIFEST}")" "${variant_root}/baseline_manifest_link"
    MANIFEST_DIR="$(dirname "${BASE_MANIFEST}")"
  else
    MANIFEST_DIR="${manifest_dir}"
    BUILD_MANIFEST=1 \
      RUN_INFERENCE=0 \
      RUN_EVAL=0 \
      RUN_DEBUG_RENDER=0 \
      MODEL_NAME="${MODEL_NAME}" \
      MODEL_TAG="ablation_${variant}_manifest" \
      EVAL_DIR="${EVAL_DIR}" \
      OUTPUT_ROOT="${variant_root}" \
      MANIFEST_DIR="${manifest_dir}" \
      START_INDEX="${START_INDEX}" \
      LIMIT="${LIMIT}" \
      SCENES="${SCENES}" \
      PANELS="${panels}" \
      LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY}" \
      OVERWRITE_FRAMES="${OVERWRITE_FRAMES}" \
      bash "${REPO_ROOT}/ablation/exam2/scripts/run_qwen3vl_30b_2d_full.sh"
  fi

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    RUN_INFERENCE=1 \
      BUILD_MANIFEST=0 \
      RUN_EVAL=0 \
      RUN_DEBUG_RENDER=0 \
      MODEL_NAME="${MODEL_NAME}" \
      MODEL_TAG="ablation_${variant}" \
      EVAL_DIR="${EVAL_DIR}" \
      OUTPUT_ROOT="${variant_root}" \
      MANIFEST_DIR="${MANIFEST_DIR}" \
      PRED_DIR="${pred_dir}" \
      EVAL_OUTPUT_DIR="${eval_dir}" \
      DEBUG_OUTPUT_DIR="${debug_dir}" \
      OFFLOAD_FOLDER="${offload_dir}" \
      START_INDEX="${START_INDEX}" \
      LIMIT="${LIMIT}" \
      SCENES="${SCENES}" \
      PANELS="${panels}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      DTYPE="${DTYPE}" \
      LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY}" \
      PROMPT_MODE="${prompt_mode}" \
      ABLATE_MODALITIES="${ablate_modalities}" \
      GAZE_MASK_RADIUS_RATIO="${GAZE_MASK_RADIUS_RATIO}" \
      INPUT_MODE="${INPUT_MODE}" \
      PANEL_CONTEXT_MODE="${panel_context_mode}" \
      PANEL_CAPTION_MODE="${PANEL_CAPTION_MODE}" \
      PAIRED_CROP_COORDINATE_POLICY="${PAIRED_CROP_COORDINATE_POLICY}" \
      PANEL_WIDTH="${PANEL_WIDTH}" \
      PANEL_HEIGHT="${PANEL_HEIGHT}" \
      COLUMNS="${COLUMNS}" \
      OVERWRITE_PREDICTIONS="${OVERWRITE_PREDICTIONS}" \
      bash "${REPO_ROOT}/ablation/exam2/scripts/run_qwen3vl_30b_2d_full.sh"
  fi

  if [[ "${RUN_EVAL}" == "1" ]]; then
    python "${REPO_ROOT}/exam2/evaluate_2d_point_grounding.py" \
      --manifest "${MANIFEST_DIR}/manifest_all.csv" \
      --pred_csv "${pred_dir}/qwen3vl_2d_predictions.csv" \
      --output_dir "${eval_dir}" \
      --panel_width "${PANEL_WIDTH}" \
      --panel_height "${PANEL_HEIGHT}" \
      --columns "${COLUMNS}" \
      "${EVAL_LIMIT_ARGS[@]}" \
      "${EVAL_SCENES_ARGS[@]}" \
      --coordinate_mode panel
  fi

  if [[ "${RUN_DEBUG_RENDER}" == "1" ]]; then
    python "${REPO_ROOT}/exam2/render_prediction_debug.py" \
      --manifest "${MANIFEST_DIR}/manifest_all.csv" \
      --pred_csv "${pred_dir}/qwen3vl_2d_predictions.csv" \
      --output_dir "${debug_dir}" \
      --max_events "${DEBUG_RENDER_MAX_EVENTS:-200}"
  fi
}

cd "${REPO_ROOT}"
for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

if [[ "${RUN_SUMMARY}" == "1" ]]; then
  python "${REPO_ROOT}/ablation/exam2/summarize_exam2_ablation.py" \
    --repo_root "${REPO_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --report_dir "${REPORT_DIR}"
fi

echo "Finished exam2 ablation workflow."
