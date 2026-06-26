#!/usr/bin/env bash
set -euo pipefail

# Full workflow for Qwen3-VL-30B mention-first multi-referent 3D grounding.
# This branch keeps one model call per instruction, but changes the prompt
# structure so the model first extracts all language referent mentions and then
# matches each mention to scene anchor labels.
#
# Recommended remote usage:
#   cd /workspace/usr3/TriModal-Referring
#   nohup bash scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh \
#     > logs/qwen3vl30b_mention_first_v3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Smoke test:
#   LIMIT=10 START_INDEX=0 bash scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

export REPO_ROOT
export MODEL_NAME="${MODEL_NAME:-/workspace/usr3/Qwen3-VL-30B-A3B-Instruct}"
export MODEL_TAG="${MODEL_TAG:-qwen3vl30b_mention_first_v3}"
export PROMPT_STRATEGY="${PROMPT_STRATEGY:-mention_first}"
export MAX_EVIDENCE_SEGMENTS="${MAX_EVIDENCE_SEGMENTS:-0}"
export EVIDENCE_SEGMENT_DURATION="${EVIDENCE_SEGMENT_DURATION:-0.5}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"

bash "${SCRIPT_DIR}/run_qwen3vl_30b_multi3d_all_scenes_full.sh"
