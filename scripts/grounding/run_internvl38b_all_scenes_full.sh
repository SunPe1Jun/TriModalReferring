#!/usr/bin/env bash
set -euo pipefail

# Full workflow for InternVL3-38B across all prepared scenes/rooms:
# 1. optionally build scene input CSVs
# 2. run local 3D inference with the InternVL3-38B checkpoint
# 3. optionally summarize per-scene outputs
# 4. optionally run per-scene evaluation + final aggregate table
#
# Typical usage:
#   bash scripts/grounding/run_internvl38b_all_scenes_full.sh
#
# Smoke test first 10 rows for every scene:
#   LIMIT=10 START_INDEX=0 END_INDEX=9 bash scripts/grounding/run_internvl38b_all_scenes_full.sh
#
# Reuse existing input CSVs without rebuilding:
#   BUILD_INPUT=0 bash scripts/grounding/run_internvl38b_all_scenes_full.sh
#
# Only evaluate existing outputs:
#   RUN_INFERENCE=0 RUN_EVAL=1 bash scripts/grounding/run_internvl38b_all_scenes_full.sh

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/InternVL3-38B}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$REPO_ROOT/.offload_internvl38b}"
MODEL_TAG="${MODEL_TAG:-internvl38b}"

BUILD_INPUT="${BUILD_INPUT:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_SCENE_SUMMARY="${RUN_SCENE_SUMMARY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"

DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"

MATCH_EVAL_OUTPUT_DIR="${MATCH_EVAL_OUTPUT_DIR:-$DATA_DIR/match_eval_${MODEL_TAG}}"

scene1_gt_default="$DATA_DIR/scene1_cleaned_v3.xlsx"
scene2_gt_default="$DATA_DIR/scene2_cleaned_v2.xlsx"
scene3_gt_default="$DATA_DIR/scene3.xlsx"
scene4_room1_gt_default="$DATA_DIR/scene4_room1.xlsx"
scene4_room2_gt_default="$DATA_DIR/scene4_room2.xlsx"
scene4_room3_gt_default="$DATA_DIR/scene4_room3.xlsx"
scene4_room4_gt_default="$DATA_DIR/scene4_room4.xlsx"
scene5_gt_default="$DATA_DIR/scene5.xlsx"

SCENE1_GT="${SCENE1_GT:-$scene1_gt_default}"
SCENE2_GT="${SCENE2_GT:-$scene2_gt_default}"
SCENE3_GT="${SCENE3_GT:-$scene3_gt_default}"
SCENE4_ROOM1_GT="${SCENE4_ROOM1_GT:-$scene4_room1_gt_default}"
SCENE4_ROOM2_GT="${SCENE4_ROOM2_GT:-$scene4_room2_gt_default}"
SCENE4_ROOM3_GT="${SCENE4_ROOM3_GT:-$scene4_room3_gt_default}"
SCENE4_ROOM4_GT="${SCENE4_ROOM4_GT:-$scene4_room4_gt_default}"
SCENE5_GT="${SCENE5_GT:-$scene5_gt_default}"

mkdir -p "${OFFLOAD_FOLDER}"
mkdir -p "${MATCH_EVAL_OUTPUT_DIR}"

if [[ -n "${LIMIT}" && -z "${END_INDEX}" ]]; then
  if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "Error: LIMIT must be an integer." >&2
    exit 1
  fi
  END_INDEX="$((START_INDEX + LIMIT - 1))"
fi

echo "Repo root: ${REPO_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Model tag: ${MODEL_TAG}"
echo "Offload folder: ${OFFLOAD_FOLDER}"
echo "Build input: ${BUILD_INPUT}"
echo "Run inference: ${RUN_INFERENCE}"
echo "Run per-scene summary: ${RUN_SCENE_SUMMARY}"
echo "Run eval: ${RUN_EVAL}"
echo "Skip missing: ${SKIP_MISSING}"

build_scene_input() {
  local scene_key="$1"
  local scene_root="$2"
  local instruction_csv="$3"
  local input_csv="$4"
  local event_json_dir="$5"
  local scene_id="$6"
  local instruction_scene_id="$7"
  local instruction_start_order="$8"
  local sample_start_index="$9"

  local missing_items=()
  [[ -d "${scene_root}" ]] || missing_items+=("scene_root=${scene_root}")
  [[ -f "${instruction_csv}" ]] || missing_items+=("instruction_csv=${instruction_csv}")

  if (( ${#missing_items[@]} > 0 )); then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[Skip build] ${scene_key}"
      printf '  missing: %s\n' "${missing_items[@]}"
      return 0
    fi
    echo "Error: missing build inputs for ${scene_key}" >&2
    printf '  %s\n' "${missing_items[@]}" >&2
    exit 1
  fi

  local cmd=(
    python "${REPO_ROOT}/scripts/data_prep/build_scene_api_input.py"
    --scene-root "${scene_root}"
    --instruction-csv "${instruction_csv}"
    --output-csv "${input_csv}"
    --event-json-dir "${event_json_dir}"
    --scene-id "${scene_id}"
    --instruction-scene-id "${instruction_scene_id}"
    --instruction-start-order "${instruction_start_order}"
    --sample-start-index "${sample_start_index}"
    --output-profile "gaze_only_api"
    --overwrite
  )
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi

  echo
  echo "[Build] ${scene_key}"
  "${cmd[@]}"
}

run_scene_inference() {
  local scene_key="$1"
  local input_csv="$2"
  local anchor_csv="$3"
  local output_dir="$4"
  local start_index="$5"
  local end_index="$6"

  local missing_items=()
  [[ -f "${input_csv}" ]] || missing_items+=("input_csv=${input_csv}")
  [[ -f "${anchor_csv}" ]] || missing_items+=("anchor_csv=${anchor_csv}")
  [[ -d "${MODEL_NAME}" ]] || missing_items+=("model=${MODEL_NAME}")

  if (( ${#missing_items[@]} > 0 )); then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[Skip inference] ${scene_key}"
      printf '  missing: %s\n' "${missing_items[@]}"
      return 0
    fi
    echo "Error: missing inference inputs for ${scene_key}" >&2
    printf '  %s\n' "${missing_items[@]}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"
  mkdir -p "${OFFLOAD_FOLDER}"

  echo
  echo "[Inference] ${scene_key}"
  echo "  input_csv: ${input_csv}"
  echo "  anchor_csv: ${anchor_csv}"
  echo "  output_dir: ${output_dir}"
  echo "  rows: ${start_index}..${end_index}"

  for (( i=start_index; i<=end_index; i++ )); do
    local out_json="${output_dir}/row_${i}.json"
    echo
    echo "[${scene_key}] Row ${i} -> ${out_json}"

    local cmd=(
      python "${REPO_ROOT}/scripts/grounding/run_internvl_local_single_event_3d.py"
      --input_csv "${input_csv}"
      --row_index "${i}"
      --scene_anchor_csv "${anchor_csv}"
      --output_json "${out_json}"
      --model_name "${MODEL_NAME}"
      --dtype "${DTYPE}"
      --input_mode "${INPUT_MODE}"
      --max_video_frames "${MAX_VIDEO_FRAMES}"
      --prompt_style "${PROMPT_STYLE}"
      --max_new_tokens "${MAX_NEW_TOKENS}"
      --offload_folder "${OFFLOAD_FOLDER}"
    )
    if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
      cmd+=(--local_files_only)
    fi
    if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
      cmd+=(--use_flash_attn)
    fi
    "${cmd[@]}"
  done
}

summarize_scene_outputs() {
  local scene_key="$1"
  local output_dir="$2"
  local summary_csv="$3"
  local summary_md="$4"

  if [[ ! -d "${output_dir}" ]]; then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "[Skip summary] ${scene_key}: output_dir missing: ${output_dir}"
      return 0
    fi
    echo "Error: cannot summarize ${scene_key}, output_dir missing: ${output_dir}" >&2
    exit 1
  fi

  echo
  echo "[Summarize] ${scene_key}"
  python "${REPO_ROOT}/scripts/grounding/summarize_local_3d_batch.py" \
    --input_dir "${output_dir}" \
    --output_csv "${summary_csv}" \
    --output_md "${summary_md}"
}

scene_pipeline() {
  local scene_key="$1"
  local scene_root="$2"
  local instruction_csv="$3"
  local input_csv="$4"
  local event_json_dir="$5"
  local anchor_csv="$6"
  local output_dir="$7"
  local scene_id="$8"
  local instruction_scene_id="$9"
  local instruction_start_order="${10}"
  local sample_start_index="${11}"
  local default_end_index="${12}"

  local start_index="${START_INDEX}"
  local end_index="${END_INDEX:-$default_end_index}"

  if [[ "${BUILD_INPUT}" == "1" ]]; then
    build_scene_input \
      "${scene_key}" \
      "${scene_root}" \
      "${instruction_csv}" \
      "${input_csv}" \
      "${event_json_dir}" \
      "${scene_id}" \
      "${instruction_scene_id}" \
      "${instruction_start_order}" \
      "${sample_start_index}"
  fi

  if [[ "${RUN_INFERENCE}" == "1" ]]; then
    run_scene_inference "${scene_key}" "${input_csv}" "${anchor_csv}" "${output_dir}" "${start_index}" "${end_index}"
  fi

  if [[ "${RUN_SCENE_SUMMARY}" == "1" ]]; then
    summarize_scene_outputs \
      "${scene_key}" \
      "${output_dir}" \
      "${DATA_DIR}/${scene_key}_local_3d_summary_${MODEL_TAG}.csv" \
      "${DATA_DIR}/${scene_key}_local_3d_summary_${MODEL_TAG}.md"
  fi
}

scene_pipeline \
  "scene1" \
  "/workspace/usr3/V3dMD/scene1" \
  "${DATA_DIR}/scene1_instruction_set_merged.csv" \
  "${DATA_DIR}/scene1_api_input.csv" \
  "${DATA_DIR}/scene1_api_event_json" \
  "${DATA_DIR}/scene1_anchor_table.tsv" \
  "${DATA_DIR}/scene1_local_3d_outputs_${MODEL_TAG}" \
  "1" "1" "1" "0" "799"

scene_pipeline \
  "scene2" \
  "/workspace/usr3/V3dMD/scene2" \
  "${DATA_DIR}/scene2_instruction_set_merged.csv" \
  "${DATA_DIR}/scene2_api_input.csv" \
  "${DATA_DIR}/scene2_api_event_json" \
  "${DATA_DIR}/scene2_anchor_table.tsv" \
  "${DATA_DIR}/scene2_local_3d_outputs_${MODEL_TAG}" \
  "2" "2" "1" "0" "799"

scene_pipeline \
  "scene3" \
  "/workspace/usr3/V3dMD/scene3" \
  "${DATA_DIR}/scene3_instruction_set_merged.csv" \
  "${DATA_DIR}/scene3_api_input.csv" \
  "${DATA_DIR}/scene3_api_event_json" \
  "${DATA_DIR}/scene3_anchor_table.tsv" \
  "${DATA_DIR}/scene3_local_3d_outputs_${MODEL_TAG}" \
  "3" "3" "1" "0" "799"

scene_pipeline \
  "scene4_room1" \
  "/workspace/usr3/V3dMD/scene4/room1" \
  "${DATA_DIR}/scene4_room1_instruction_set_merged.csv" \
  "${DATA_DIR}/scene4_room1_api_input.csv" \
  "${DATA_DIR}/scene4_room1_api_event_json" \
  "${DATA_DIR}/scene4_room1_anchor_table.tsv" \
  "${DATA_DIR}/scene4_room1_local_3d_outputs_${MODEL_TAG}" \
  "4" "4" "1" "0" "199"

scene_pipeline \
  "scene4_room2" \
  "/workspace/usr3/V3dMD/scene4/room2" \
  "${DATA_DIR}/scene4_room2_instruction_set_merged.csv" \
  "${DATA_DIR}/scene4_room2_api_input.csv" \
  "${DATA_DIR}/scene4_room2_api_event_json" \
  "${DATA_DIR}/scene4_room2_anchor_table.tsv" \
  "${DATA_DIR}/scene4_room2_local_3d_outputs_${MODEL_TAG}" \
  "4" "4" "1" "0" "199"

scene_pipeline \
  "scene4_room3" \
  "/workspace/usr3/V3dMD/scene4/room3" \
  "${DATA_DIR}/scene4_room3_instruction_set_merged.csv" \
  "${DATA_DIR}/scene4_room3_api_input.csv" \
  "${DATA_DIR}/scene4_room3_api_event_json" \
  "${DATA_DIR}/scene4_room3_anchor_table.tsv" \
  "${DATA_DIR}/scene4_room3_local_3d_outputs_${MODEL_TAG}" \
  "4" "4" "1" "0" "199"

scene_pipeline \
  "scene4_room4" \
  "/workspace/usr3/V3dMD/scene4/room4" \
  "${DATA_DIR}/scene4_room4_instruction_set_merged.csv" \
  "${DATA_DIR}/scene4_room4_api_input.csv" \
  "${DATA_DIR}/scene4_room4_api_event_json" \
  "${DATA_DIR}/scene4_room4_anchor_table.tsv" \
  "${DATA_DIR}/scene4_room4_local_3d_outputs_${MODEL_TAG}" \
  "4" "4" "1" "0" "199"

scene_pipeline \
  "scene5" \
  "/workspace/usr3/V3dMD/scene5" \
  "${DATA_DIR}/scene5_instruction_set_merged.csv" \
  "${DATA_DIR}/scene5_api_input.csv" \
  "${DATA_DIR}/scene5_api_event_json" \
  "${DATA_DIR}/scene5_anchor_table.tsv" \
  "${DATA_DIR}/scene5_local_3d_outputs_${MODEL_TAG}" \
  "5" "5" "1" "0" "799"

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo
  echo "[Evaluation] Running per-scene match evaluation for ${MODEL_TAG}..."

  env \
    REPO_ROOT="${REPO_ROOT}" \
    DATA_DIR="${DATA_DIR}" \
    OUTPUT_DIR="${MATCH_EVAL_OUTPUT_DIR}" \
    SKIP_MISSING="${SKIP_MISSING}" \
    SCENE1_GT="${SCENE1_GT}" \
    SCENE2_GT="${SCENE2_GT}" \
    SCENE3_GT="${SCENE3_GT}" \
    SCENE4_ROOM1_GT="${SCENE4_ROOM1_GT}" \
    SCENE4_ROOM2_GT="${SCENE4_ROOM2_GT}" \
    SCENE4_ROOM3_GT="${SCENE4_ROOM3_GT}" \
    SCENE4_ROOM4_GT="${SCENE4_ROOM4_GT}" \
    SCENE5_GT="${SCENE5_GT}" \
    SCENE1_PRED_DIR="${DATA_DIR}/scene1_local_3d_outputs_${MODEL_TAG}" \
    SCENE2_PRED_DIR="${DATA_DIR}/scene2_local_3d_outputs_${MODEL_TAG}" \
    SCENE3_PRED_DIR="${DATA_DIR}/scene3_local_3d_outputs_${MODEL_TAG}" \
    SCENE4_ROOM1_PRED_DIR="${DATA_DIR}/scene4_room1_local_3d_outputs_${MODEL_TAG}" \
    SCENE4_ROOM2_PRED_DIR="${DATA_DIR}/scene4_room2_local_3d_outputs_${MODEL_TAG}" \
    SCENE4_ROOM3_PRED_DIR="${DATA_DIR}/scene4_room3_local_3d_outputs_${MODEL_TAG}" \
    SCENE4_ROOM4_PRED_DIR="${DATA_DIR}/scene4_room4_local_3d_outputs_${MODEL_TAG}" \
    SCENE5_PRED_DIR="${DATA_DIR}/scene5_local_3d_outputs_${MODEL_TAG}" \
    bash "${REPO_ROOT}/scripts/eval/run_all_match_eval_and_summarize.sh"
fi

echo
echo "Finished full ${MODEL_TAG} workflow."
