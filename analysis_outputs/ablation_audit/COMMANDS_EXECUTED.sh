#!/usr/bin/env bash
set -euo pipefail

# Read-only inventory/configuration checks used during the audit included rg,
# find, sed, head, ls, sha256sum, git diff --no-index, and git log.
# No model inference and no original evaluator output overwrite was performed.

python /workspace/usr3/TriModal-Referring/analysis_outputs/ablation_audit/run_ablation_audit.py --repo-root /workspace/usr3/TriModal-Referring --output-dir /workspace/usr3/TriModal-Referring/analysis_outputs/ablation_audit
