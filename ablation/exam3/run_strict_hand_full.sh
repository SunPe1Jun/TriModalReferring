#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
CONDA_ENV_BIN="${CONDA_ENV_BIN:-/workspace/usr3/miniconda3/envs/trimodal/bin}"
MANIFEST="${MANIFEST:-exam3_point_grounding/outputs_full_v9_20260709/manifest.csv}"
GT_MANIFEST="${GT_MANIFEST:-exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-ablation/exam3/outputs_qwen3vl30b_v9_strict_hand_v1_full}"
HAND_MASK_ROOT="${HAND_MASK_ROOT:-ablation/exam3/hand_masked_frames_v1_full}"
MASK_AUDIT="${MASK_AUDIT:-ablation/exam3/hand_masked_frames_v1_full_audit.json}"
SHARD_ROOT="${SHARD_ROOT:-${OUTPUT_ROOT}/shards}"
MASK_OVERWRITE="${MASK_OVERWRITE:-0}"

cd "${REPO_ROOT}"
export PATH="${CONDA_ENV_BIN}:${PATH}"
mkdir -p "${REPO_ROOT}/${OUTPUT_ROOT}" "${REPO_ROOT}/ablation/exam3/logs"

mask_overwrite_args=()
if [[ "${MASK_OVERWRITE}" == "1" ]]; then
  mask_overwrite_args+=(--overwrite)
fi
if [[ ! -f "${REPO_ROOT}/${MASK_AUDIT}" || "${MASK_OVERWRITE}" == "1" ]]; then
  python exam3_point_grounding/prepare_hand_masked_frames.py \
    --manifest "${MANIFEST}" \
    --output_dir "${HAND_MASK_ROOT}" \
    --audit_path "${MASK_AUDIT}" \
    "${mask_overwrite_args[@]}"
fi

python exam3_point_grounding/make_point_grounding_key_shards.py \
  --repo_root "${REPO_ROOT}" \
  --gt_manifest "${GT_MANIFEST}" \
  --output_dir "${SHARD_ROOT}/keys" \
  --prefix strict_hand \
  --num_shards 2 \
  --round_robin

run_shard() {
  local gpu="$1"
  local shard="$2"
  local shard_output="${SHARD_ROOT}/shard${shard}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    VARIANT=no_hand_strict \
    OUTPUT_ROOT="${shard_output}" \
    HAND_MASK_ROOT="${HAND_MASK_ROOT}" \
    SAMPLE_KEYS_FILE="${SHARD_ROOT}/keys/strict_hand_${shard}.txt" \
    RUN_EVAL=0 \
    OVERWRITE_INFERENCE=0 \
    bash "${REPO_ROOT}/ablation/exam3/run_variant.sh" \
    > "${REPO_ROOT}/ablation/exam3/logs/strict_hand_full_gpu${gpu}.log" 2>&1
}

run_shard 0 0 &
pid0=$!
run_shard 1 1 &
pid1=$!
wait "${pid0}" "${pid1}"

python ablation/exam3/merge_strict_hand_shards.py \
  --prediction_csvs \
    "${SHARD_ROOT}/shard0/no_hand_strict/predictions.csv" \
    "${SHARD_ROOT}/shard1/no_hand_strict/predictions.csv" \
  --output_csv "${OUTPUT_ROOT}/no_hand_strict/predictions.csv"

python exam3_point_grounding/evaluate_point_grounding.py \
  --repo_root "${REPO_ROOT}" \
  --pred_csv "${OUTPUT_ROOT}/no_hand_strict/predictions.csv" \
  --gt_manifest "${GT_MANIFEST}" \
  --output_dir "${OUTPUT_ROOT}/no_hand_strict/eval" \
  --report_path "${OUTPUT_ROOT}/no_hand_strict/RESULTS.md"

python ablation/exam3/summarize_exam3_ablation.py \
  --repo_root "${REPO_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --expected_gt "${GT_MANIFEST}" \
  --report_dir "ablation/exam3/reports/strict_hand_v1" \
  --variants no_hand_strict
