#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

MODEL_DIR="${MODEL_DIR:-/workspace/usr3/InternVL3-38B-Instruct}"
DOWNLOAD_PID="${DOWNLOAD_PID:-}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/internvl/logs}"
MODEL_TAG="${MODEL_TAG:-internvl3_38b_baseline}"

mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

log() {
  echo "[$(timestamp)] $*"
}

model_ready() {
  [[ -f "${MODEL_DIR}/config.json" ]] || return 1
  [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || return 1
  local shard_count
  shard_count="$(ls "${MODEL_DIR}"/model-*.safetensors 2>/dev/null | wc -l || true)"
  [[ "${shard_count}" -ge 16 ]] || return 1
  ! ls "${MODEL_DIR}"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1
}

wait_for_download() {
  log "Waiting for InternVL model at ${MODEL_DIR}"
  while ! model_ready; do
    if [[ -n "${DOWNLOAD_PID}" ]] && ! kill -0 "${DOWNLOAD_PID}" >/dev/null 2>&1; then
      if ! model_ready; then
        log "Download process ${DOWNLOAD_PID} is not alive and model is incomplete."
        return 1
      fi
    fi
    du -sh "${MODEL_DIR}" 2>/dev/null || true
    sleep 300
  done
  log "InternVL model files look complete."
}

run_smoke() {
  log "Running InternVL exam1 smoke"
  env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
    LIMIT=1 SCENES=scene1 OVERWRITE_PREDICTIONS=1 \
    OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam1_${MODEL_TAG}_smoke" \
    bash "${REPO_ROOT}/internvl/run_exam1_internvl38b_baseline.sh" \
    > "${LOG_DIR}/exam1_${MODEL_TAG}_smoke.log" 2>&1

  log "Running InternVL exam2 smoke"
  env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
    LIMIT=1 SCENES=scene1 OVERWRITE_PREDICTIONS=1 RUN_DEBUG_RENDER=0 \
    OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam2_${MODEL_TAG}_smoke" \
    bash "${REPO_ROOT}/internvl/run_exam2_internvl38b_baseline.sh" \
    > "${LOG_DIR}/exam2_${MODEL_TAG}_smoke.log" 2>&1
}

run_full() {
  log "Running InternVL exam1 full baseline"
  env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
    LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 \
    OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam1_${MODEL_TAG}" \
    bash "${REPO_ROOT}/internvl/run_exam1_internvl38b_baseline.sh" \
    > "${LOG_DIR}/exam1_${MODEL_TAG}_full.log" 2>&1

  log "Running InternVL exam2 full baseline"
  env PATH=/workspace/usr3/miniconda3/envs/trimodal/bin:$PATH \
    LIMIT= RUN_DEBUG_RENDER=0 OVERWRITE_PREDICTIONS=0 \
    OUTPUT_ROOT="${REPO_ROOT}/internvl/outputs/exam2_${MODEL_TAG}" \
    bash "${REPO_ROOT}/internvl/run_exam2_internvl38b_baseline.sh" \
    > "${LOG_DIR}/exam2_${MODEL_TAG}_full.log" 2>&1
}

main() {
  log "InternVL baseline supervisor started."
  wait_for_download
  run_smoke
  run_full
  log "InternVL baseline supervisor finished."
}

main "$@"
