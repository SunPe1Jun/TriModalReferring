#!/usr/bin/env bash
set -euo pipefail

# Create a dedicated conda environment for InternVL3 local experiments.
#
# Typical usage:
#   bash scripts/utils/create_internvl_conda_env.sh
#
# Custom env name:
#   ENV_NAME=internvl38b bash scripts/utils/create_internvl_conda_env.sh
#
# This script intentionally avoids flash-attn because the current server
# does not expose nvcc. The goal is a stable, reproducible baseline env.

REPO_ROOT="${REPO_ROOT:-/ai/data/TriModal-Referring}"
ENV_NAME="${ENV_NAME:-internvl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.5.1}"
PYTORCH_CUDA_VERSION="${PYTORCH_CUDA_VERSION:-12.1}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-$REPO_ROOT/scripts/grounding/requirements_internvl.txt}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda was not found in PATH." >&2
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Error: requirements file does not exist: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

echo "Repo root: ${REPO_ROOT}"
echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PYTHON_VERSION}"
echo "PyTorch version: ${PYTORCH_VERSION}"
echo "PyTorch CUDA runtime: ${PYTORCH_CUDA_VERSION}"
echo "Requirements: ${REQUIREMENTS_FILE}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda environment already exists: ${ENV_NAME}"
else
  echo
  echo "[1/4] Creating conda environment ${ENV_NAME}..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo
echo "[2/4] Installing PyTorch + CUDA runtime..."
conda install -y \
  -c pytorch \
  -c nvidia \
  "pytorch=${PYTORCH_VERSION}" \
  "torchvision" \
  "pytorch-cuda=${PYTORCH_CUDA_VERSION}"

echo
echo "[3/4] Installing Python requirements..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQUIREMENTS_FILE}"

echo
echo "[4/4] Verifying core imports..."
python - <<'PY'
import torch
import transformers
import accelerate
import sentencepiece
import google.protobuf
import einops
import timm

print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_device_count =", torch.cuda.device_count())
print("transformers =", transformers.__version__)
print("accelerate =", accelerate.__version__)
print("protobuf =", google.protobuf.__version__)
print("einops ok")
print("timm ok")
PY

echo
echo "InternVL environment is ready: ${ENV_NAME}"
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
