#!/usr/bin/env bash
set -euo pipefail

# Run per-scene/room local-3D match evaluation and then aggregate all summaries.
#
# Typical usage:
#   bash scripts/eval/run_all_match_eval_and_summarize.sh
#
# Skip scenes whose inputs are temporarily missing:
#   SKIP_MISSING=1 bash scripts/eval/run_all_match_eval_and_summarize.sh
#
# Override a single path if needed:
#   SCENE4_ROOM3_PRED_DIR=/ai/data/TriModal-Referring/data/scene4_room3_local_3d_outputs \
#   bash scripts/eval/run_all_match_eval_and_summarize.sh

REPO_ROOT="${REPO_ROOT:-/ai/data/TriModal-Referring}"
DATA_DIR="${DATA_DIR:-$REPO_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/match_eval}"
SKIP_MISSING="${SKIP_MISSING:-0}"

first_existing() {
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  printf '%s\n' ""
  return 0
}

SCENE1_GT="${SCENE1_GT:-$(first_existing "$DATA_DIR/scene1_cleaned_v3.xlsx" "$DATA_DIR/scene1_cleaned.xlsx" "$REPO_ROOT/scene1.xlsx")}"
SCENE2_GT="${SCENE2_GT:-$(first_existing "$DATA_DIR/scene2_cleaned_v2.xlsx" "$DATA_DIR/scene2_cleaned.xlsx" "$REPO_ROOT/scene2.xlsx")}"
SCENE3_GT="${SCENE3_GT:-$(first_existing "$DATA_DIR/scene3.xlsx" "$REPO_ROOT/scene3.xlsx")}"
SCENE4_ROOM1_GT="${SCENE4_ROOM1_GT:-$(first_existing "$DATA_DIR/scene4_room1.xlsx" "$REPO_ROOT/scene4_room1.xlsx")}"
SCENE4_ROOM2_GT="${SCENE4_ROOM2_GT:-$(first_existing "$DATA_DIR/scene4_room2.xlsx" "$REPO_ROOT/scene4_room2.xlsx")}"
SCENE4_ROOM3_GT="${SCENE4_ROOM3_GT:-$(first_existing "$DATA_DIR/scene4_room3.xlsx" "$REPO_ROOT/scene4_room3.xlsx")}"
SCENE4_ROOM4_GT="${SCENE4_ROOM4_GT:-$(first_existing "$DATA_DIR/scene4_room4.xlsx" "$REPO_ROOT/scene4_room4.xlsx")}"
SCENE5_GT="${SCENE5_GT:-$(first_existing "$DATA_DIR/scene5.xlsx" "$REPO_ROOT/scene5.xlsx")}"

SCENE1_PRED_DIR="${SCENE1_PRED_DIR:-$DATA_DIR/scene1_local_3d_outputs}"
SCENE2_PRED_DIR="${SCENE2_PRED_DIR:-$DATA_DIR/scene2_local_3d_outputs}"
SCENE3_PRED_DIR="${SCENE3_PRED_DIR:-$DATA_DIR/scene3_local_3d_outputs}"
SCENE4_ROOM1_PRED_DIR="${SCENE4_ROOM1_PRED_DIR:-$DATA_DIR/scene4_room1_local_3d_outputs}"
SCENE4_ROOM2_PRED_DIR="${SCENE4_ROOM2_PRED_DIR:-$DATA_DIR/scene4_room2_local_3d_outputs}"
SCENE4_ROOM3_PRED_DIR="${SCENE4_ROOM3_PRED_DIR:-$DATA_DIR/scene4_room3_local_3d_outputs}"
SCENE4_ROOM4_PRED_DIR="${SCENE4_ROOM4_PRED_DIR:-$DATA_DIR/scene4_room4_local_3d_outputs}"
SCENE5_PRED_DIR="${SCENE5_PRED_DIR:-$DATA_DIR/scene5_local_3d_outputs}"

SCENE1_ANCHOR="${SCENE1_ANCHOR:-$DATA_DIR/scene1_anchor_table.tsv}"
SCENE2_ANCHOR="${SCENE2_ANCHOR:-$DATA_DIR/scene2_anchor_table.tsv}"
SCENE3_ANCHOR="${SCENE3_ANCHOR:-$DATA_DIR/scene3_anchor_table.tsv}"
SCENE4_ROOM1_ANCHOR="${SCENE4_ROOM1_ANCHOR:-$DATA_DIR/scene4_room1_anchor_table.tsv}"
SCENE4_ROOM2_ANCHOR="${SCENE4_ROOM2_ANCHOR:-$DATA_DIR/scene4_room2_anchor_table.tsv}"
SCENE4_ROOM3_ANCHOR="${SCENE4_ROOM3_ANCHOR:-$DATA_DIR/scene4_room3_anchor_table.tsv}"
SCENE4_ROOM4_ANCHOR="${SCENE4_ROOM4_ANCHOR:-$DATA_DIR/scene4_room4_anchor_table.tsv}"
SCENE5_ANCHOR="${SCENE5_ANCHOR:-$DATA_DIR/scene5_anchor_table.tsv}"

mkdir -p "${OUTPUT_DIR}"

echo "Repo root: ${REPO_ROOT}"
echo "Data dir: ${DATA_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Skip missing: ${SKIP_MISSING}"

declare -a SCENES=(
  "scene1|${SCENE1_PRED_DIR}|${SCENE1_GT}|${SCENE1_ANCHOR}"
  "scene2|${SCENE2_PRED_DIR}|${SCENE2_GT}|${SCENE2_ANCHOR}"
  "scene3|${SCENE3_PRED_DIR}|${SCENE3_GT}|${SCENE3_ANCHOR}"
  "scene4_room1|${SCENE4_ROOM1_PRED_DIR}|${SCENE4_ROOM1_GT}|${SCENE4_ROOM1_ANCHOR}"
  "scene4_room2|${SCENE4_ROOM2_PRED_DIR}|${SCENE4_ROOM2_GT}|${SCENE4_ROOM2_ANCHOR}"
  "scene4_room3|${SCENE4_ROOM3_PRED_DIR}|${SCENE4_ROOM3_GT}|${SCENE4_ROOM3_ANCHOR}"
  "scene4_room4|${SCENE4_ROOM4_PRED_DIR}|${SCENE4_ROOM4_GT}|${SCENE4_ROOM4_ANCHOR}"
  "scene5|${SCENE5_PRED_DIR}|${SCENE5_GT}|${SCENE5_ANCHOR}"
)

RAN_COUNT=0

for entry in "${SCENES[@]}"; do
  IFS="|" read -r SCENE_KEY PRED_DIR GT_FILE ANCHOR_CSV <<< "${entry}"

  MISSING_ITEMS=()
  [[ -d "${PRED_DIR}" ]] || MISSING_ITEMS+=("pred_dir=${PRED_DIR}")
  [[ -f "${GT_FILE}" ]] || MISSING_ITEMS+=("gt_file=${GT_FILE}")
  [[ -f "${ANCHOR_CSV}" ]] || MISSING_ITEMS+=("anchor_csv=${ANCHOR_CSV}")

  if (( ${#MISSING_ITEMS[@]} > 0 )); then
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo
      echo "[Skip] ${SCENE_KEY}"
      printf '  missing: %s\n' "${MISSING_ITEMS[@]}"
      continue
    fi
    echo "Error: missing required inputs for ${SCENE_KEY}" >&2
    printf '  %s\n' "${MISSING_ITEMS[@]}" >&2
    exit 1
  fi

  OUT_CSV="${OUTPUT_DIR}/${SCENE_KEY}_match_eval.csv"
  OUT_JSON="${OUTPUT_DIR}/${SCENE_KEY}_match_eval_summary.json"

  echo
  echo "[Eval] ${SCENE_KEY}"
  echo "  pred_dir: ${PRED_DIR}"
  echo "  gt_file: ${GT_FILE}"
  echo "  anchor_csv: ${ANCHOR_CSV}"

  python "${REPO_ROOT}/scripts/eval/evaluate_local_3d_object_match.py" \
    --pred_dir "${PRED_DIR}" \
    --gt_file "${GT_FILE}" \
    --anchor_csv "${ANCHOR_CSV}" \
    --output_csv "${OUT_CSV}" \
    --output_json "${OUT_JSON}"

  RAN_COUNT=$((RAN_COUNT + 1))
done

if (( RAN_COUNT == 0 )); then
  echo "Error: no scene evaluations were run." >&2
  exit 1
fi

echo
echo "[Aggregate] Combining per-scene summaries..."

python "${REPO_ROOT}/scripts/eval/summarize_match_eval_summaries.py" \
  --input_dir "${OUTPUT_DIR}" \
  --glob "*_match_eval_summary.json" \
  --output_csv "${OUTPUT_DIR}/all_scene_match_eval_summary.csv" \
  --output_md "${OUTPUT_DIR}/all_scene_match_eval_summary.md"

echo
echo "Finished all scene evaluations and aggregation."
