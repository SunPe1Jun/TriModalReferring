#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="python"
CUDA_VARIANT="cu121"
INSTALL_FLASH_ATTN="0"
UPGRADE_PIP="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --cuda)
      CUDA_VARIANT="$2"
      shift 2
      ;;
    --use-flash-attn)
      INSTALL_FLASH_ATTN="1"
      shift
      ;;
    --no-upgrade-pip)
      UPGRADE_PIP="0"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements_qwen3vl.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 1
fi

if [[ "$UPGRADE_PIP" == "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
fi

TORCH_INDEX_URL=""
case "$CUDA_VARIANT" in
  cu118)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
    ;;
  cu121)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
    ;;
  cu124)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
    ;;
  cpu)
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    ;;
  *)
    echo "Unsupported --cuda value: $CUDA_VARIANT" >&2
    echo "Supported values: cu118, cu121, cu124, cpu" >&2
    exit 1
    ;;
esac

echo "[1/4] Installing PyTorch from: $TORCH_INDEX_URL"
"$PYTHON_BIN" -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

echo "[2/4] Installing Python requirements"
"$PYTHON_BIN" -m pip install -r "$REQ_FILE"

echo "[3/4] Verifying core imports"
"$PYTHON_BIN" - <<'PY'
import importlib
modules = [
    'torch',
    'transformers',
    'accelerate',
    'huggingface_hub',
    'PIL',
    'safetensors',
]
missing = []
for name in modules:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append((name, str(exc)))
if missing:
    for name, exc in missing:
        print(f'Missing or broken module: {name} | {exc}')
    raise SystemExit(1)
print('Core dependencies imported successfully.')
PY

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "[4/4] Installing flash-attn (optional)"
  set +e
  "$PYTHON_BIN" -m pip install flash-attn --no-build-isolation
  FLASH_EXIT=$?
  set -e
  if [[ $FLASH_EXIT -ne 0 ]]; then
    echo "flash-attn installation failed. You can still run without --use_flash_attn." >&2
  else
    echo "flash-attn installed successfully."
  fi
else
  echo "[4/4] Skipping flash-attn installation"
fi

echo "Environment setup completed."
