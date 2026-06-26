#!/usr/bin/env bash
set -euo pipefail

# Remote local-model inference-only workflow for scene4 room2 3D referent selection.
# This script assumes INPUT_CSV already exists and is ready to use.

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
INPUT_CSV="${INPUT_CSV:-$REPO_ROOT/data/scene4_room2_api_input.csv}"
ANCHOR_CSV="${ANCHOR_CSV:-$REPO_ROOT/data/scene4_room2_anchor_table.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/data/scene4_room2_local_3d_outputs}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-$REPO_ROOT/.offload_qwen3vl_30b}"

DTYPE="${DTYPE:-bfloat16}"
INPUT_MODE="${INPUT_MODE:-video}"
MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
PROMPT_STYLE="${PROMPT_STYLE:-full}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

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

if [[ ! -d "${MODEL_NAME}" ]]; then
  echo "Error: model path does not exist: ${MODEL_NAME}" >&2
  exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Error: input CSV does not exist: ${INPUT_CSV}" >&2
  echo "Hint: build it first with scripts/data_prep/build_scene_api_input.py" >&2
  exit 1
fi

if [[ ! -f "${ANCHOR_CSV}" ]]; then
  echo "Error: anchor CSV does not exist: ${ANCHOR_CSV}" >&2
  exit 1
fi

echo "Repo root: ${REPO_ROOT}"
echo "Model: ${MODEL_NAME}"
echo "Input CSV: ${INPUT_CSV}"
echo "Anchor CSV: ${ANCHOR_CSV}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Rows: ${START_INDEX}..${END_INDEX}"

echo
echo "Running scene4 room2 local 3D inference only..."

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
    --prompt_style "${PROMPT_STYLE}"
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
echo "Finished scene4 room2 local 3D inference-only run."
