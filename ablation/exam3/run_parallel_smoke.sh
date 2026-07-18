#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation/exam3/outputs_qwen3vl30b_v9_input_mask_v3_smoke10}"
KEYS_FILE="${KEYS_FILE:-${REPO_ROOT}/${OUTPUT_ROOT}/smoke_keys.txt}"
mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}" "${REPO_ROOT}/ablation/exam3/logs"
cd "${REPO_ROOT}"

python ablation/exam3/make_sample_keys.py \
  --manifest "exam3_point_grounding/outputs_full_v9_20260709/manifest.csv" \
  --output "${KEYS_FILE}" \
  --limit 10

run_lane() {
  local gpu="$1"
  shift
  for variant in "$@"; do
    CUDA_VISIBLE_DEVICES="${gpu}" VARIANT="${variant}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      SAMPLE_KEYS_FILE="${KEYS_FILE}" OVERWRITE_INFERENCE="1" \
      bash "${REPO_ROOT}/ablation/exam3/run_variant.sh" \
      > "${REPO_ROOT}/ablation/exam3/logs/smoke_v3_${variant}.log" 2>&1
  done
}

run_lane 0 no_visual no_hand no_gaze_hand &
pid0=$!
run_lane 1 no_gaze no_instruction &
pid1=$!
wait "${pid0}" "${pid1}"

env PATH="/workspace/usr3/miniconda3/envs/trimodal/bin:${PATH}" \
  python "${REPO_ROOT}/ablation/exam3/summarize_exam3_ablation.py" \
  --output_root "${OUTPUT_ROOT}" \
  --expected_gt "${OUTPUT_ROOT}/no_visual/gt_manifest_eval_subset.csv" \
  --report_dir "ablation/exam3/reports/smoke10_v3"
