#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/internvl/logs}"
MODEL_TAG="${MODEL_TAG:-internvl3_38b_baseline}"
mkdir -p "${LOG_DIR}"

timestamp() {
  date +%Y-%m-%dT%H:%M:%S%z
}

log() {
  echo "[$(timestamp)] $*"
}

cd "${REPO_ROOT}"
log "Starting InternVL3-38B full baseline sequence."
log "Running exam1 full."
env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
  LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 LOAD_IN_8BIT=0 \
  OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam1_${MODEL_TAG}" \
  bash "${REPO_ROOT}/internvl/run_exam1_internvl38b_baseline.sh" \
  > "${LOG_DIR}/exam1_${MODEL_TAG}_full.log" 2>&1

log "Running exam2 full."
env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
  LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 LOAD_IN_8BIT=0 \
  OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam2_${MODEL_TAG}" \
  bash "${REPO_ROOT}/internvl/run_exam2_internvl38b_baseline.sh" \
  > "${LOG_DIR}/exam2_${MODEL_TAG}_full.log" 2>&1

log "InternVL3-38B full baseline sequence finished."
