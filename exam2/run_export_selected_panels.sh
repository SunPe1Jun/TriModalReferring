#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/usr3/TriModal-Referring}"
MANIFEST="${MANIFEST:-$REPO_ROOT/exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv}"
PRED_CSV="${PRED_CSV:-$REPO_ROOT/exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/predictions/qwen3vl_2d_predictions.csv}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/usr3/V3dMD}"
OUTPUT_MANIFEST="${OUTPUT_MANIFEST:-$REPO_ROOT/exam2/selected_panel_export_manifest.csv}"
SELECTION_POLICY="${SELECTION_POLICY:-all_unique}"
FOLDER_NAME="${FOLDER_NAME:-qwen_selected_panels}"
START_INDEX="${START_INDEX:-0}"
LIMIT="${LIMIT-20}"
DRY_RUN="${DRY_RUN:-1}"
OVERWRITE="${OVERWRITE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  "$REPO_ROOT/exam2/export_selected_panels_to_dataset.py"
  --manifest "$MANIFEST"
  --pred_csv "$PRED_CSV"
  --dataset_root "$DATASET_ROOT"
  --output_manifest "$OUTPUT_MANIFEST"
  --selection_policy "$SELECTION_POLICY"
  --folder_name "$FOLDER_NAME"
  --start_index "$START_INDEX"
)

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ -n "${SCENES:-}" ]]; then
  read -r -a scene_array <<< "$SCENES"
  args+=(--scenes "${scene_array[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  args+=(--dry_run)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "${args[@]}"
