#!/usr/bin/env bash
set -euo pipefail

# Full workflow for Qwen3-VL-8B across all prepared scenes/rooms:
# 1. optionally build scene input CSVs
# 2. run local 3D inference with the 8B checkpoint
# 3. optionally summarize per-scene outputs
# 4. optionally run per-scene evaluation + final aggregate table
#
# Typical usage:
#   bash scripts/grounding/run_qwen3vl_8b_all_scenes_full.sh
#
# Smoke test first 10 rows for every scene:
#   LIMIT=10 START_INDEX=0 END_INDEX=9 bash scripts/grounding/run_qwen3vl_8b_all_scenes_full.sh
#
# Reuse existing input CSVs without rebuilding:
#   BUILD_INPUT=0 bash scripts/grounding/run_qwen3vl_8b_all_scenes_full.sh
#
# Only evaluate existing 8B outputs:
#   RUN_INFERENCE=0 RUN_EVAL=1 bash scripts/grounding/run_qwen3vl_8b_all_scenes_full.sh

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-8B-Instruct}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$REPO_ROOT/.offload_qwen3vl_8b}"
MODEL_TAG="${MODEL_TAG:-qwen3vl8b}"

BUILD_INPUT="${BUILD_INPUT:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"
RUN_SCENE_SUMMARY="${RUN_SCENE_SUMMARY:-1}"
RUN_EVAL="${RUN_EVAL:-1}"
SKIP_MISSING="${SKIP_MISSING:-0}"

DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-standard}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"
OUTPUT_PROFILE="${OUTPUT_PROFILE:-gaze_only_api}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

LIMIT="${LIMIT:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"

MATCH_EVAL_OUTPUT_DIR="${MATCH_EVAL_OUTPUT_DIR:-$DATA_DIR/match_eval_${MODEL_TAG}}"

SCENE1_ROOT="${SCENE1_ROOT:-/workspace/usr3/V3dMD/scene1}"
SCENE2_ROOT="${SCENE2_ROOT:-/workspace/usr3/V3dMD/scene2}"
SCENE3_ROOT="${SCENE3_ROOT:-/workspace/usr3/V3dMD/scene3}"
SCENE4_ROOM1_ROOT="${SCENE4_ROOM1_ROOT:-/workspace/usr3/V3dMD/scene4/room1}"
SCENE4_ROOM2_ROOT="${SCENE4_ROOM2_ROOT:-/workspace/usr3/V3dMD/scene4/room2}"
SCENE4_ROOM3_ROOT="${SCENE4_ROOM3_ROOT:-/workspace/usr3/V3dMD/scene4/room3}"
SCENE4_ROOM4_ROOT="${SCENE4_ROOM4_ROOT:-/workspace/usr3/V3dMD/scene4/room4}"
SCENE5_ROOT="${SCENE5_ROOT:-/workspace/usr3/V3dMD/scene5}"

SCENE1_INSTRUCTION_START_ORDER="${SCENE1_INSTRUCTION_START_ORDER:-1}"
SCENE2_INSTRUCTION_START_ORDER="${SCENE2_INSTRUCTION_START_ORDER:-1}"
SCENE3_INSTRUCTION_START_ORDER="${SCENE3_INSTRUCTION_START_ORDER:-1}"
SCENE4_ROOM1_INSTRUCTION_START_ORDER="${SCENE4_ROOM1_INSTRUCTION_START_ORDER:-1}"
SCENE4_ROOM2_INSTRUCTION_START_ORDER="${SCENE4_ROOM2_INSTRUCTION_START_ORDER:-1}"
SCENE4_ROOM3_INSTRUCTION_START_ORDER="${SCENE4_ROOM3_INSTRUCTION_START_ORDER:-1}"
SCENE4_ROOM4_INSTRUCTION_START_ORDER="${SCENE4_ROOM4_INSTRUCTION_START_ORDER:-1}"
SCENE5_INSTRUCTION_START_ORDER="${SCENE5_INSTRUCTION_START_ORDER:-1}"

SCENE1_SAMPLE_START_INDEX="${SCENE1_SAMPLE_START_INDEX:-0}"
SCENE2_SAMPLE_START_INDEX="${SCENE2_SAMPLE_START_INDEX:-0}"
SCENE3_SAMPLE_START_INDEX="${SCENE3_SAMPLE_START_INDEX:-0}"
SCENE4_ROOM1_SAMPLE_START_INDEX="${SCENE4_ROOM1_SAMPLE_START_INDEX:-0}"
SCENE4_ROOM2_SAMPLE_START_INDEX="${SCENE4_ROOM2_SAMPLE_START_INDEX:-0}"
SCENE4_ROOM3_SAMPLE_START_INDEX="${SCENE4_ROOM3_SAMPLE_START_INDEX:-0}"
SCENE4_ROOM4_SAMPLE_START_INDEX="${SCENE4_ROOM4_SAMPLE_START_INDEX:-0}"
SCENE5_SAMPLE_START_INDEX="${SCENE5_SAMPLE_START_INDEX:-0}"

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
echo "Output profile: ${OUTPUT_PROFILE}"
echo "Prompt strategy: ${PROMPT_STRATEGY}"
echo "Sparse evidence: max_segments=${MAX_EVIDENCE_SEGMENTS}, segment_duration=${EVIDENCE_SEGMENT_DURATION}s"

run_scene_batch() {
  local scene_key="$1"
  local batch_script="$2"
  local scene_root="$3"
  local instruction_csv="$4"
  local anchor_csv="$5"
  local input_csv="$6"
  local event_json_dir="$7"
  local output_dir="$8"
  local scene_id="$9"
  local instruction_scene_id="${10}"
  local instruction_start_order="${11}"
  local sample_start_index="${12}"

  local missing_items=()
  [[ -x "${batch_script}" || -f "${batch_script}" ]] || missing_items+=("batch_script=${batch_script}")
  [[ -d "${scene_root}" ]] || missing_items+=("scene_root=${scene_root}")
  [[ -f "${instruction_csv}" ]] || missing_items+=("instruction_csv=${instruction_csv}")
  [[ -f "${anchor_csv}" ]] || missing_items+=("anchor_csv=${anchor_csv}")
  [[ -d "${MODEL_NAME}" ]] || missing_items+=("model=${MODEL_NAME}")

  if (( ${#missing_items[@]} > 0 )); then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo
      echo "[Skip inference] ${scene_key}"
      printf '  missing: %s\n' "${missing_items[@]}"
      return 0
    fi
    echo "Error: missing required inputs for ${scene_key}" >&2
    printf '  %s\n' "${missing_items[@]}" >&2
    exit 1
  fi

  mkdir -p "${output_dir}"

  echo
  echo "[Inference] ${scene_key}"
  echo "  script: ${batch_script}"
  echo "  scene_root: ${scene_root}"
  echo "  input_csv: ${input_csv}"
  echo "  output_dir: ${output_dir}"

  env \
    REPO_ROOT="${REPO_ROOT}" \
    SCENE_ROOT="${scene_root}" \
    MODEL_NAME="${MODEL_NAME}" \
    INSTRUCTION_CSV="${instruction_csv}" \
    ANCHOR_CSV="${anchor_csv}" \
    INPUT_CSV="${input_csv}" \
    EVENT_JSON_DIR="${event_json_dir}" \
    OUTPUT_DIR="${output_dir}" \
    OFFLOAD_FOLDER="${OFFLOAD_FOLDER}" \
    SCENE_ID="${scene_id}" \
    INSTRUCTION_SCENE_ID="${instruction_scene_id}" \
    INSTRUCTION_START_ORDER="${instruction_start_order}" \
    SAMPLE_START_INDEX="${sample_start_index}" \
    BUILD_INPUT="${BUILD_INPUT}" \
    OUTPUT_PROFILE="${OUTPUT_PROFILE}" \
    DTYPE="${DTYPE}" \
    INPUT_MODE="${INPUT_MODE}" \
    MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES}" \
    MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS}" \
    EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION}" \
    PROMPT_STYLE="${PROMPT_STYLE}" \
    PROMPT_STRATEGY="${PROMPT_STRATEGY}" \
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
    USE_FLASH_ATTN="${USE_FLASH_ATTN}" \
    LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY}" \
    LIMIT="${LIMIT}" \
    START_INDEX="${START_INDEX}" \
    END_INDEX="${END_INDEX}" \
    bash "${batch_script}"
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

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  run_scene_batch \
    "scene1" \
    "${REPO_ROOT}/scripts/grounding/run_scene1_local_3d_batch.sh" \
    "${SCENE1_ROOT}" \
    "${DATA_DIR}/scene1_instruction_set_merged.csv" \
    "${DATA_DIR}/scene1_anchor_table.tsv" \
    "${DATA_DIR}/scene1_api_input.csv" \
    "${DATA_DIR}/scene1_api_event_json" \
    "${DATA_DIR}/scene1_local_3d_outputs_${MODEL_TAG}" \
    "1" "1" "${SCENE1_INSTRUCTION_START_ORDER}" "${SCENE1_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene2" \
    "${REPO_ROOT}/scripts/grounding/run_scene2_local_3d_batch.sh" \
    "${SCENE2_ROOT}" \
    "${DATA_DIR}/scene2_instruction_set_merged.csv" \
    "${DATA_DIR}/scene2_anchor_table.tsv" \
    "${DATA_DIR}/scene2_api_input.csv" \
    "${DATA_DIR}/scene2_api_event_json" \
    "${DATA_DIR}/scene2_local_3d_outputs_${MODEL_TAG}" \
    "2" "2" "${SCENE2_INSTRUCTION_START_ORDER}" "${SCENE2_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene3" \
    "${REPO_ROOT}/scripts/grounding/run_scene3_local_3d_batch.sh" \
    "${SCENE3_ROOT}" \
    "${DATA_DIR}/scene3_instruction_set_merged.csv" \
    "${DATA_DIR}/scene3_anchor_table.tsv" \
    "${DATA_DIR}/scene3_api_input.csv" \
    "${DATA_DIR}/scene3_api_event_json" \
    "${DATA_DIR}/scene3_local_3d_outputs_${MODEL_TAG}" \
    "3" "3" "${SCENE3_INSTRUCTION_START_ORDER}" "${SCENE3_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene4_room1" \
    "${REPO_ROOT}/scripts/grounding/run_scene4_room1_local_3d_batch.sh" \
    "${SCENE4_ROOM1_ROOT}" \
    "${DATA_DIR}/scene4_room1_instruction_set_merged.csv" \
    "${DATA_DIR}/scene4_room1_anchor_table.tsv" \
    "${DATA_DIR}/scene4_room1_api_input.csv" \
    "${DATA_DIR}/scene4_room1_api_event_json" \
    "${DATA_DIR}/scene4_room1_local_3d_outputs_${MODEL_TAG}" \
    "4" "4" "${SCENE4_ROOM1_INSTRUCTION_START_ORDER}" "${SCENE4_ROOM1_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene4_room2" \
    "${REPO_ROOT}/scripts/grounding/run_scene4_room2_local_3d_batch.sh" \
    "${SCENE4_ROOM2_ROOT}" \
    "${DATA_DIR}/scene4_room2_instruction_set_merged.csv" \
    "${DATA_DIR}/scene4_room2_anchor_table.tsv" \
    "${DATA_DIR}/scene4_room2_api_input.csv" \
    "${DATA_DIR}/scene4_room2_api_event_json" \
    "${DATA_DIR}/scene4_room2_local_3d_outputs_${MODEL_TAG}" \
    "4" "4" "${SCENE4_ROOM2_INSTRUCTION_START_ORDER}" "${SCENE4_ROOM2_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene4_room3" \
    "${REPO_ROOT}/scripts/grounding/run_scene4_room3_local_3d_batch.sh" \
    "${SCENE4_ROOM3_ROOT}" \
    "${DATA_DIR}/scene4_room3_instruction_set_merged.csv" \
    "${DATA_DIR}/scene4_room3_anchor_table.tsv" \
    "${DATA_DIR}/scene4_room3_api_input.csv" \
    "${DATA_DIR}/scene4_room3_api_event_json" \
    "${DATA_DIR}/scene4_room3_local_3d_outputs_${MODEL_TAG}" \
    "4" "4" "${SCENE4_ROOM3_INSTRUCTION_START_ORDER}" "${SCENE4_ROOM3_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene4_room4" \
    "${REPO_ROOT}/scripts/grounding/run_scene4_room4_local_3d_batch.sh" \
    "${SCENE4_ROOM4_ROOT}" \
    "${DATA_DIR}/scene4_room4_instruction_set_merged.csv" \
    "${DATA_DIR}/scene4_room4_anchor_table.tsv" \
    "${DATA_DIR}/scene4_room4_api_input.csv" \
    "${DATA_DIR}/scene4_room4_api_event_json" \
    "${DATA_DIR}/scene4_room4_local_3d_outputs_${MODEL_TAG}" \
    "4" "4" "${SCENE4_ROOM4_INSTRUCTION_START_ORDER}" "${SCENE4_ROOM4_SAMPLE_START_INDEX}"

  run_scene_batch \
    "scene5" \
    "${REPO_ROOT}/scripts/grounding/run_scene5_local_3d_batch.sh" \
    "${SCENE5_ROOT}" \
    "${DATA_DIR}/scene5_instruction_set_merged.csv" \
    "${DATA_DIR}/scene5_anchor_table.tsv" \
    "${DATA_DIR}/scene5_api_input.csv" \
    "${DATA_DIR}/scene5_api_event_json" \
    "${DATA_DIR}/scene5_local_3d_outputs_${MODEL_TAG}" \
    "5" "5" "${SCENE5_INSTRUCTION_START_ORDER}" "${SCENE5_SAMPLE_START_INDEX}"
fi

if [[ "${RUN_SCENE_SUMMARY}" == "1" ]]; then
  summarize_scene_outputs "scene1" "${DATA_DIR}/scene1_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene1_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene1_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene2" "${DATA_DIR}/scene2_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene2_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene2_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene3" "${DATA_DIR}/scene3_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene3_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene3_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene4_room1" "${DATA_DIR}/scene4_room1_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene4_room1_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene4_room1_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene4_room2" "${DATA_DIR}/scene4_room2_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene4_room2_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene4_room2_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene4_room3" "${DATA_DIR}/scene4_room3_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene4_room3_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene4_room3_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene4_room4" "${DATA_DIR}/scene4_room4_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene4_room4_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene4_room4_local_3d_summary_${MODEL_TAG}.md"
  summarize_scene_outputs "scene5" "${DATA_DIR}/scene5_local_3d_outputs_${MODEL_TAG}" "${DATA_DIR}/scene5_local_3d_summary_${MODEL_TAG}.csv" "${DATA_DIR}/scene5_local_3d_summary_${MODEL_TAG}.md"
fi

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
