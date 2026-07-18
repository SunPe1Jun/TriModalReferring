#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation/exam3/outputs_qwen3vl30b_v9_input_mask_v3_full}"
mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}" "${REPO_ROOT}/ablation/exam3/logs"
cd "${REPO_ROOT}"

run_lane() {
  local gpu="$1"
  shift
  for variant in "$@"; do
    CUDA_VISIBLE_DEVICES="${gpu}" VARIANT="${variant}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      OVERWRITE_INFERENCE="0" bash "${REPO_ROOT}/ablation/exam3/run_variant.sh" \
      >> "${REPO_ROOT}/ablation/exam3/logs/full_v3_${variant}.log" 2>&1
  done
}

# The grouping is based on measured smoke throughput, not only variant count.
# It keeps the two sequential GPU lanes within roughly one full-run hour band.
run_lane 0 no_hand no_instruction &
pid0=$!
run_lane 1 no_visual no_gaze no_gaze_hand &
pid1=$!
wait "${pid0}" "${pid1}"

env PATH="/workspace/usr3/miniconda3/envs/trimodal/bin:${PATH}" \
  python "${REPO_ROOT}/ablation/exam3/summarize_exam3_ablation.py" \
  --output_root "${OUTPUT_ROOT}" \
  --expected_gt "exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv" \
  --report_dir "ablation/exam3/reports/full_v3"
