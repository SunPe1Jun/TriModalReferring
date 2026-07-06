#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/qwen8/logs}"
MODEL_TAG="${MODEL_TAG:-qwen3vl8b_baseline}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

mkdir -p "${LOG_DIR}"

timestamp() {
  date +%Y-%m-%dT%H:%M:%S%z
}

log() {
  echo "[$(timestamp)] $*"
}

cd "${REPO_ROOT}"
log "Starting Qwen3-VL-8B full baseline sequence on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."

log "Running exam1 full."
env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
  LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 \
  MAX_VIDEO_FRAMES="${MAX_VIDEO_FRAMES:-8}" \
  MODEL_TAG="${MODEL_TAG}" \
  OUTPUT_ROOT="${REPO_ROOT}/qwen8/outputs/exam1_${MODEL_TAG}" \
  bash "${REPO_ROOT}/qwen8/run_exam1_qwen3vl8b_baseline.sh" \
  > "${LOG_DIR}/exam1_${MODEL_TAG}_full.log" 2>&1

log "Running exam2 full."
env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
  LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 \
  MODEL_TAG="${MODEL_TAG}_2d_point_hybrid_v10" \
  OUTPUT_ROOT="${REPO_ROOT}/qwen8/outputs/exam2_${MODEL_TAG}_2d_point_hybrid_v10" \
  bash "${REPO_ROOT}/qwen8/run_exam2_qwen3vl8b_baseline.sh" \
  > "${LOG_DIR}/exam2_${MODEL_TAG}_2d_point_hybrid_v10_full.log" 2>&1

log "Qwen3-VL-8B full baseline sequence finished."
