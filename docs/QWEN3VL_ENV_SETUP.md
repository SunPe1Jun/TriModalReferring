# Qwen3-VL Environment Setup

## Overview

这份文档说明如何在远程服务器上为本地 `Qwen/Qwen3-VL-8B-Instruct` 推理脚本补齐运行环境。

对应脚本：

- `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`

对应依赖文件：

- `scripts/grounding/requirements_qwen3vl.txt`
- `scripts/grounding/install_qwen3vl_requirements.sh`

---

## Required Python Packages

核心 Python 依赖已经写入：

- `transformers`
- `accelerate`
- `huggingface_hub`
- `pillow`
- `safetensors`
- `sentencepiece`

注意：

- `torch` 不放在 `requirements_qwen3vl.txt` 里，因为它通常需要按 CUDA 版本单独安装
- `flash-attn` 是可选项，不是必须

---

## Recommended Install Script

在远程服务器上推荐直接运行：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh --cuda cu124
```

可选参数：

- `--python /path/to/python`
- `--cuda cu118|cu121|cu124|cpu`
- `--use-flash-attn`
- `--no-upgrade-pip`

例如：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh \
  --python /ai/data/miniconda3/envs/qwen/bin/python \
  --cuda cu121
```

如果你想尝试安装 `flash-attn`：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh \
  --python /ai/data/miniconda3/envs/qwen/bin/python \
  --cuda cu121 \
  --use-flash-attn
```

如果 `flash-attn` 安装失败，不影响基础推理，只是后续不要加 `--use_flash_attn`。

---

## Manual Install

如果你不想用安装脚本，也可以手动安装。

### Example: CUDA 12.1

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r scripts/grounding/requirements_qwen3vl.txt
```

### Example: CPU only

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r scripts/grounding/requirements_qwen3vl.txt
```

---

## Verify Environment

先检查 `torch` 是否正常：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

再检查核心依赖：

```bash
python -c "import transformers, accelerate, PIL, huggingface_hub, safetensors; print('ok')"
```

---

## Run Local Grounding Script

如果你已经把模型仓库下载到远程本地目录，例如：

- `/ai/data/models/Qwen3-VL-8B-Instruct`

那么推荐这样运行：

```bash
python scripts/grounding/run_qwen3vl_local_keyframe_grounding.py \
  --input_csv data/keyframe_grounding_input.csv \
  --output_csv data/qwen3vl_grounding_output.csv \
  --model_name /ai/data/models/Qwen3-VL-8B-Instruct \
  --local_files_only \
  --dtype auto \
  --max_new_tokens 128 \
  --continue_on_error
```

如果 `flash-attn` 已安装成功，也可以加：

```bash
--use_flash_attn
```

---

## Common Issues

### 1. `Missing dependency: torch`

说明当前 Python 环境还没有装 PyTorch。

优先使用：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh --cuda cu121
```

### 2. `flash-attn` 安装失败

很常见，通常和以下因素有关：

- CUDA 版本
- 编译环境
- PyTorch 版本
- GCC / nvcc 环境

如果失败，可以先不用它，照样能推理。

### 3. 本地模型目录无法加载

请确认本地目录里至少包含：

- `config.json`
- tokenizer / processor 相关文件
- 权重文件，例如 `*.safetensors`

### 4. 显存不足

可以尝试：

- `--dtype float16`
- 不使用 `--use_flash_attn`
- 减小 batch 规模（当前脚本本来就是逐条处理）

---

## Files

- `scripts/grounding/requirements_qwen3vl.txt`
- `scripts/grounding/install_qwen3vl_requirements.sh`
- `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`
