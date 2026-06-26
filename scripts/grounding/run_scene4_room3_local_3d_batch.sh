#!/usr/bin/env bash
set -euo pipefail

# Remote local-model workflow for scene4 room3 3D referent selection.
#
# This runner assumes room3 has its own scene root and its own instruction workbook.
#
# Typical usage:
#   bash scripts/grounding/run_scene4_room3_local_3d_batch.sh
#
# Smoke test only:
#   LIMIT=10 START_INDEX=0 END_INDEX=9 bash scripts/grounding/run_scene4_room3_local_3d_batch.sh
#
# Skip rebuilding input csv:
#   BUILD_INPUT=0 bash scripts/grounding/run_scene4_room3_local_3d_batch.sh

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
SCENE_ROOT="${SCENE_ROOT:-/workspace/usr3/V3dMD/scene4/room3}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
INSTRUCTION_CSV="${INSTRUCTION_CSV:-$REPO_ROOT/data/scene4_room3_instruction_set_merged.csv}"
ANCHOR_CSV="${ANCHOR_CSV:-$REPO_ROOT/data/scene4_room3_anchor_table.tsv}"

INPUT_CSV="${INPUT_CSV:-$REPO_ROOT/data/scene4_room3_api_input.csv}"
EVENT_JSON_DIR="${EVENT_JSON_DIR:-$REPO_ROOT/data/scene4_room3_api_event_json}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/scene4_room3_local_3d_outputs}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$REPO_ROOT/.offload_qwen3vl_30b}"

SCENE_ID="${SCENE_ID:-4}"
INSTRUCTION_SCENE_ID="${INSTRUCTION_SCENE_ID:-4}"
INSTRUCTION_START_ORDER="${INSTRUCTION_START_ORDER:-1}"
SAMPLE_START_INDEX="${SAMPLE_START_INDEX:-0}"
OUTPUT_PROFILE="${OUTPUT_PROFILE:-gaze_only_api}"

DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
PROMPT_STRATEGY="${PROMPT_STRATEGY:-standard}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"

BUILD_INPUT="${BUILD_INPUT:-1}"
USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

LIMIT="${LIMIT:-200}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-199}"

if [[ -n "${LIMIT}" ]]; then
  if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "Error: LIMIT must be an integer." >&2
    exit 1
  fi
  END_INDEX="$((START_INDEX + LIMIT - 1))"
fi

if ! [[ "${START_INDEX}" =~ ^[0-9]+$ && "${END_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "Error: START_INDEX and END_INDEX must be integers." >&2
  exit 1
fi

if (( END_INDEX < START_INDEX )); then
  echo "Error: END_INDEX must be >= START_INDEX." >&2
  exit 1
fi

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OFFLOAD_FOLDER}"

if [[ ! -d "${SCENE_ROOT}" ]]; then
  echo "Error: scene root does not exist: ${SCENE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Error: model path does not exist: ${MODEL_NAME}" >&2
  exit 1
fi

if [[ ! -f "${INSTRUCTION_CSV}" ]]; then
  echo "Error: instruction CSV does not exist: ${INSTRUCTION_CSV}" >&2
  exit 1
fi

if [[ ! -f "${ANCHOR_CSV}" ]]; then
  echo "Error: anchor CSV does not exist: ${ANCHOR_CSV}" >&2
  exit 1
fi

echo "Repo root: ${REPO_ROOT}"
echo "Scene root: ${SCENE_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Instruction CSV: ${INSTRUCTION_CSV}"
echo "Anchor CSV: ${ANCHOR_CSV}"
echo "Input CSV: ${INPUT_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Instruction start order: ${INSTRUCTION_START_ORDER}"
echo "Sample start index: ${SAMPLE_START_INDEX}"
echo "Rows: ${START_INDEX}..${END_INDEX}"

if [[ "${BUILD_INPUT}" == "1" ]]; then
  BUILD_CMD=(
    python scripts/data_prep/build_scene_api_input.py
    --scene-root "${SCENE_ROOT}"
    --instruction-csv "${INSTRUCTION_CSV}"
    --output-csv "${INPUT_CSV}"
    --event-json-dir "${EVENT_JSON_DIR}"
    --scene-id "${SCENE_ID}"
    --instruction-scene-id "${INSTRUCTION_SCENE_ID}"
    --instruction-start-order "${INSTRUCTION_START_ORDER}"
    --sample-start-index "${SAMPLE_START_INDEX}"
    --output-profile "${OUTPUT_PROFILE}"
    --limit "${LIMIT}"
    --overwrite
  )
  echo
  echo "Building scene4 room3 input CSV..."
  "${BUILD_CMD[@]}"
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Error: input CSV does not exist after build step: ${INPUT_CSV}" >&2
  exit 1
fi

echo
echo "Running scene4 room3 local 3D inference..."

for (( i=START_INDEX; i<=END_INDEX; i++ )); do
  OUT_JSON="${OUTPUT_DIR}/row_${i}.json"
  echo
  echo "[Row ${i}] -> ${OUT_JSON}"

  CMD=(
    python scripts/grounding/run_qwen3vl_local_single_event_3d.py
    --input_csv "${INPUT_CSV}"
    --row_index "${i}"
    --scene_anchor_csv "${ANCHOR_CSV}"
    --output_json "${OUT_JSON}"
    --model_name "${MODEL_NAME}"
    --dtype "${DTYPE}"
    --input_mode "${INPUT_MODE}"
    --max_video_frames "${MAX_VIDEO_FRAMES}"
    --max_evidence_segments "${MAX_EVIDENCE_SEGMENTS}"
    --evidence_segment_duration "${EVIDENCE_SEGMENT_DURATION}"
    --prompt_style "${PROMPT_STYLE}"
    --prompt_strategy "${PROMPT_STRATEGY}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --offload_folder "${OFFLOAD_FOLDER}"
  )

  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    CMD+=(--local_files_only)
  fi
  if [[ "${USE_FLASH_ATTN}" == "1" ]]; then
    CMD+=(--use_flash_attn)
  fi

  "${CMD[@]}"
done

echo
echo "Finished scene4 room3 local 3D batch run."
