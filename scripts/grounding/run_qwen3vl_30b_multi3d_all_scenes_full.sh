#!/usr/bin/env bash
set -euo pipefail

# Full workflow for Qwen3-VL-30B multi-referent 3D grounding across all scenes/rooms.
#
# This is a thin, experiment-specific wrapper around the generic all-scene Qwen3-VL
# workflow. The underlying runner now prompts for multiple selected objects and the
# evaluator reports both any-hit accuracy and set-level precision/recall/F1.
#
# Recommended nohup usage on the remote server:
#   cd /workspace/usr3/TriModal-Referring
#   nohup bash scripts/grounding/run_qwen3vl_30b_multi3d_all_scenes_full.sh \
#     > logs/qwen3vl30b_multi3d_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Smoke test first 3 rows per scene:
#   LIMIT=3 START_INDEX=0 bash scripts/grounding/run_qwen3vl_30b_multi3d_all_scenes_full.sh
#
# Continue from a row index within every scene/room:
#   START_INDEX=34 BUILD_INPUT=0 bash scripts/grounding/run_qwen3vl_30b_multi3d_all_scenes_full.sh
#
# Evaluate existing outputs only:
#   RUN_INFERENCE=0 RUN_SCENE_SUMMARY=1 RUN_EVAL=1 BUILD_INPUT=0 \
#     bash scripts/grounding/run_qwen3vl_30b_multi3d_all_scenes_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data}"
export MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
export OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-${REPO_ROOT}/.offload_qwen3vl_30b_multi3d}"
export MODEL_TAG="${MODEL_TAG:-qwen3vl30b_multi3d}"

export BUILD_INPUT="${BUILD_INPUT:-1}"
export RUN_INFERENCE="${RUN_INFERENCE:-1}"
export RUN_SCENE_SUMMARY="${RUN_SCENE_SUMMARY:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export SKIP_MISSING="${SKIP_MISSING:-0}"

export DTYPE="${DTYPE:-bfloat16}"
export INPUT_MODE="${INPUT_MODE:-video}"
export MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-16}"
export PROMPT_STYLE="${PROMPT_STYLE:-full}"
export PROMPT_STRATEGY="${PROMPT_STRATEGY:-standard}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-768}"
export MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
export EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"
export OUTPUT_PROFILE="${OUTPUT_PROFILE:-gaze_only_api}"
export USE_FLASH_ATTN="${USE_FLASH_ATTN:-0}"
export LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

export MATCH_EVAL_OUTPUT_DIR="${MATCH_EVAL_OUTPUT_DIR:-${DATA_DIR}/match_eval_${MODEL_TAG}}"

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${OFFLOAD_FOLDER}"
mkdir -p "${MATCH_EVAL_OUTPUT_DIR}"

echo "Starting Qwen3-VL-30B multi-referent 3D full workflow"
echo "Repo root: ${REPO_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Model: ${MODEL_NAME}"
echo "Model tag: ${MODEL_TAG}"
echo "Eval output dir: ${MATCH_EVAL_OUTPUT_DIR}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Prompt strategy: ${PROMPT_STRATEGY}"
echo "Sparse evidence: max_segments=${MAX_EVIDENCE_SEGMENTS}, segment_duration=${EVIDENCE_SEGMENT_DURATION}s"
echo
echo "Tip: use tail -f on the nohup log to monitor progress."
echo

bash "${SCRIPT_DIR}/run_qwen3vl_8b_all_scenes_full.sh"

echo
echo "Finished Qwen3-VL-30B multi-referent 3D full workflow."
echo "Per-scene outputs: ${DATA_DIR}/*_local_3d_outputs_${MODEL_TAG}"
echo "Evaluation summary: ${MATCH_EVAL_OUTPUT_DIR}/all_scene_match_eval_summary.md"
