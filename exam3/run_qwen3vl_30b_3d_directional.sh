#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
MODEL_TAG="${MODEL_TAG:-qwen3vl30b_3d_directional_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/exam3/outputs_$MODEL_TAG}"
MANIFEST="${MANIFEST:-$REPO_ROOT/exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv}"
PROMPT_TEMPLATE="${PROMPT_TEMPLATE:-$REPO_ROOT/exam3/prompts/camera_centered_3d_directional_prompt.md}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
OVERWRITE="${OVERWRITE:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_MISSING_GT_ANCHORS="${SKIP_MISSING_GT_ANCHORS:-0}"
EVIDENCE_PANEL_STRATEGY="${EVIDENCE_PANEL_STRATEGY:-highest_score}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$OUTPUT_ROOT/predictions/raw" "$OUTPUT_ROOT/eval"

pred_args=(
  "$REPO_ROOT/exam3/run_qwen3vl_3d_directional.py"
  --repo_root "$REPO_ROOT"
  --manifest "$MANIFEST"
  --output_csv "$OUTPUT_ROOT/predictions/qwen3vl_3d_directional_predictions.csv"
  --output_json_dir "$OUTPUT_ROOT/predictions/raw"
  --prompt_template "$PROMPT_TEMPLATE"
  --model_name "$MODEL_NAME"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --start_index "$START_INDEX"
  --evidence_panel_strategy "$EVIDENCE_PANEL_STRATEGY"
)

if [[ -n "$LIMIT" ]]; then
  pred_args+=(--limit "$LIMIT")
fi

if [[ -n "${SCENES:-}" ]]; then
  read -r -a scene_array <<< "$SCENES"
  pred_args+=(--scenes "${scene_array[@]}")
fi

if [[ -n "${SAMPLE_KEYS:-}" ]]; then
  read -r -a sample_key_array <<< "$SAMPLE_KEYS"
  pred_args+=(--sample_keys "${sample_key_array[@]}")
fi

if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
  pred_args+=(--local_files_only)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  pred_args+=(--overwrite)
fi

if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
  pred_args+=(--continue_on_error)
fi

if [[ "$SKIP_MISSING_GT_ANCHORS" == "1" ]]; then
  pred_args+=(--skip_missing_gt_anchors)
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "${pred_args[@]}"

"$PYTHON_BIN" "$REPO_ROOT/exam3/evaluate_3d_directional.py" \
  --pred_csv "$OUTPUT_ROOT/predictions/qwen3vl_3d_directional_predictions.csv" \
  --output_dir "$OUTPUT_ROOT/eval" \
  --report_path "$OUTPUT_ROOT/report.md"
