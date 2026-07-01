#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/ablation/logs}"
TRIMODAL_BIN="${TRIMODAL_BIN:-/workspace/usr3/miniconda3/envs/trimodal/bin}"
mkdir -p "${LOG_DIR}"

EXAM1_VARIANTS="${EXAM1_VARIANTS:-no_gaze no_hand}"
EXAM2_VARIANTS="${EXAM2_VARIANTS:-full_panels_no_crop no_gaze_text_prior}"
SMOKE_LIMIT="${SMOKE_LIMIT:-5}"
START_INDEX="${START_INDEX:-0}"

cd "${REPO_ROOT}"

echo "Launching smoke ablations. Logs: ${LOG_DIR}"

setsid bash -lc "cd '${REPO_ROOT}' && export PATH='${TRIMODAL_BIN}':$PATH && CUDA_VISIBLE_DEVICES=${EXAM1_CUDA_VISIBLE_DEVICES:-0} VARIANTS='${EXAM1_VARIANTS}' LIMIT='${SMOKE_LIMIT}' START_INDEX='${START_INDEX}' OVERWRITE_PREDICTIONS=1 bash ablation/exam1/run_exam1_ablation.sh" \
  > "${LOG_DIR}/exam1_smoke.log" 2>&1 < /dev/null &
EXAM1_PID=$!

setsid bash -lc "cd '${REPO_ROOT}' && export PATH='${TRIMODAL_BIN}':$PATH && CUDA_VISIBLE_DEVICES=${EXAM2_CUDA_VISIBLE_DEVICES:-1} VARIANTS='${EXAM2_VARIANTS}' LIMIT='${SMOKE_LIMIT}' START_INDEX='${START_INDEX}' OVERWRITE_PREDICTIONS=1 OVERWRITE_FRAMES=0 bash ablation/exam2/run_exam2_ablation.sh" \
  > "${LOG_DIR}/exam2_smoke.log" 2>&1 < /dev/null &
EXAM2_PID=$!

echo "exam1 smoke PID: ${EXAM1_PID}"
echo "exam2 smoke PID: ${EXAM2_PID}"
echo "Monitor with: tail -f ${LOG_DIR}/exam1_smoke.log ${LOG_DIR}/exam2_smoke.log"
