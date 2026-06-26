# Full Workflow Guide

## Overview

这份文档说明当前仓库里从原始多模态样本到本地 Qwen3-VL 非实体空间 grounding 预测的完整流程。

任务目标是：

- 输入关键帧图像、语言信息、gaze / hand 信息、事件级 JSON 空间线索
- 本地加载 `Qwen/Qwen3-VL-8B-Instruct`
- 输出图像中的 2D grounding point 和可选 3D world point
- 预测用户当前指向的 non-entity spatial referent

---

## Folder Layout

### `scripts/visualization`
- `enhance_gaze_overlay.py`
- `batch_enhance_gaze_overlay.py`

### `scripts/data_prep`
- `generate_event_annotations_template.py`
- `extract_event_keyframes.py`
- `build_event_manifest.py`
- `build_keyframe_grounding_input.py`

### `scripts/grounding`
- `download_hf_model_via_mirror.py`
- `run_qwen3vl_local_keyframe_grounding.py`
- `requirements_qwen3vl.txt`
- `install_qwen3vl_requirements.sh`
- `render_grounding_result_overlays.py`
- `summarize_grounding_results.py`

### `docs`
- `SCRIPT_USAGE.md`
- `FULL_WORKFLOW.md`
- `QWEN3VL_ENV_SETUP.md`
- `QWEN3VL_RUN_AUDIT.md`

---

## Recommended Pipeline

推荐顺序如下：

1. 可选：先对视频中的 gaze 点进行增强显示
2. 生成 `event_annotations.csv` 模板
3. 补充或修改事件时间和 GT 信息
4. 根据 `t_peak` 抽取关键帧
5. 自动构建 `event_manifest.csv`
6. 将 `event_manifest.csv` 转成 grounding 输入 CSV
7. 下载或准备本地 Qwen3-VL 权重
8. 配置推理环境
9. 运行本地 grounding 推理脚本
10. 检查 overlay 图和结果汇总

---

## Step 0: Environment

### Python dependencies

基础数据脚本通常只需要 Python 标准库。

可视化和 grounding 还需要：

- `ffmpeg`
- `ffprobe`
- `torch`
- `transformers`
- `accelerate`
- `pillow`
- `huggingface_hub`
- `safetensors`
- `sentencepiece`

### Check ffmpeg

```bash
ffmpeg -version
ffprobe -version
```

### Qwen3-VL environment setup

推荐参考：

- `docs/QWEN3VL_ENV_SETUP.md`

或者直接执行：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh --cuda cu121
```

---

## Step 1: Enhance Gaze Overlay (Optional)

如果你希望后续关键帧来自已经增强过 gaze marker 的视频，可以先运行这一阶段。

### 1.1 Single video

```bash
python scripts/visualization/enhance_gaze_overlay.py \
  --input-video data/1/ScreenRecord_2025-10-11-20_53_14.mp4 \
  --input-json data/1/multimodal_data.json \
  --metadata-json data/1/metadata_2025-10-11_20-53-14-907.json \
  --output-video data/1/ScreenRecord_2025-10-11-20_53_14_gaze_enhanced.mp4 \
  --point-source gazePoint \
  --marker-size 32 \
  --pixel-offset-x -20 \
  --overwrite
```

### 1.2 Batch videos

```bash
python scripts/visualization/batch_enhance_gaze_overlay.py \
  --input-root data \
  --point-source gazePoint \
  --marker-size 32 \
  --pixel-offset-x -20 \
  --overwrite \
  --continue-on-error
```

说明：后续数据准备脚本默认优先使用 `*gaze_enhanced*.mp4`。

---

## Step 2: Generate Event Annotation Template

```bash
python scripts/data_prep/generate_event_annotations_template.py \
  --input-root data \
  --output-csv data/event_annotations.csv \
  --overwrite
```

生成后建议人工检查：

- `t_start`
- `t_peak`
- `t_end`
- `gt_type`
- `gt_anchor_x`
- `gt_anchor_y`
- `gt_anchor_z`

---

## Step 3: Extract Event Keyframes

```bash
python scripts/data_prep/extract_event_keyframes.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --output-root data \
  --output-template "{scene_id}/keyframes/{t_peak}.jpg" \
  --overwrite
```

---

## Step 4: Build Event Manifest

```bash
python scripts/data_prep/build_event_manifest.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --json-root data \
  --output-csv data/event_manifest.csv
```

输出会包含：

- `event_id`
- `keyframe_path`
- `json_path`
- 时间窗字段
- 视频路径和 GT anchor 字段

---

## Step 5: Build Grounding Input CSV

脚本：`scripts/data_prep/build_keyframe_grounding_input.py`

作用：

- 读取 `event_manifest.csv`
- 读取每行对应的 `json_path`
- 自动从时间窗中汇总 `gaze_summary`
- 自动从时间窗中汇总 `hand_summary`
- 自动提取 `t_peak` 附近的事件级 JSON 片段
- 自动生成 `event_json_path`
- 自动计算可投影到关键帧的 `spatial prior`
- 输出 grounding 推理输入 CSV

```bash
python scripts/data_prep/build_keyframe_grounding_input.py \
  --input-csv data/event_manifest.csv \
  --output-csv data/keyframe_grounding_input.csv \
  --overwrite
```

当前默认输出列：

- `event_id`
- `keyframe_path`
- `gaze_summary`
- `hand_summary`
- `instruction_text`
- `utterance_text`
- `target_description`
- `event_json_path`
- `spatial_context_text`
- `spatial_context_json`
- `spatial_prior_u_norm`
- `spatial_prior_v_norm`
- `spatial_prior_source`

其中：

- `event_json_path` 指向自动生成的事件级 JSON 片段
- `spatial_prior_*` 是从 JSON 投影得到的 2D 先验
- `target_description` 会在原地重建时自动保留

---

## Step 6: Download or Prepare Local Qwen3-VL Weights

### 6.1 Download with hf-mirror

```bash
python scripts/grounding/download_hf_model_via_mirror.py \
  --output_dir E:\models\Qwen3-VL-8B-Instruct \
  --allow_patterns "*" \
  --resume_download
```

如果你已经在远程把模型仓库下好了，这一步可以跳过。

---

## Step 7: Run Local Qwen3-VL Grounding

脚本：`scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`

### Input CSV

兼容旧版输入列：

- `event_id`
- `keyframe_path`
- `gaze_summary`
- `hand_summary`

同时支持新增可选列：

- `instruction_text`
- `utterance_text`
- `target_description`
- `event_json_path`

### What the script now uses

它现在会把以下信息同时送入模型：

- `event_id`
- `task instruction`
- `utterance text`
- `target description`
- `gaze summary`
- `hand summary`
- `event_json_path` 中抽取出来的空间信息
- `spatial prior` 的 `u_norm / v_norm / source`
- keyframe image

### Output CSV

输出列包括：

- `event_id`
- `prompt_text`
- `model_raw_output`
- `parsed_json`
- `u_norm`
- `v_norm`
- `x_world`
- `y_world`
- `z_world`
- `referent_text`
- `reasoning_summary`
- `confidence`
- `parse_ok`
- `error_message`
- `event_json_path`
- `spatial_context_text`
- `spatial_context_json`
- `spatial_prior_u_norm`
- `spatial_prior_v_norm`
- `spatial_prior_source`

### Example: image mode

```bash
python scripts/grounding/run_qwen3vl_local_keyframe_grounding.py \
  --input_csv data/keyframe_grounding_input.csv \
  --output_csv data/qwen3vl_grounding_output.csv \
  --model_name /path/to/Qwen3-VL-8B-Instruct \
  --local_files_only \
  --dtype auto \
  --prompt_variant debug \
  --max_new_tokens 256 \
  --continue_on_error \
  --vis_dir data/qwen3vl_vis
```

### Example: video mode

```bash
python scripts/grounding/run_qwen3vl_local_keyframe_grounding.py \
  --input_csv data/keyframe_grounding_input_with_video.csv \
  --output_csv data/qwen3vl_grounding_output_video.csv \
  --model_name /path/to/Qwen3-VL-8B-Instruct \
  --local_files_only \
  --input_mode video \
  --max_video_frames 16 \
  --prompt_variant debug \
  --max_new_tokens 256 \
  --continue_on_error \
  --vis_dir data/qwen3vl_vis_video
```

### Overlay visualization

如果提供：

```bash
--vis_dir data/qwen3vl_vis
```

脚本会：

- 在关键帧上绘制模型预测 2D 点
- 同时绘制 JSON `spatial prior` 点
- 标注 `referent_text`
- 标注 `confidence`
- 每个 `event_id` 保存一张 overlay 图

---

## Step 8: Summarize and Visualize Results

### 8.1 Render result overlays from prediction CSV

```bash
python scripts/grounding/render_grounding_result_overlays.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --source-csv data/keyframe_grounding_input.csv \
  --output-dir data/grounding_overlays \
  --overwrite
```

### 8.2 Summarize grounding results

```bash
python scripts/grounding/summarize_grounding_results.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --output-csv data/qwen3vl_grounding_summary.csv \
  --output-md data/qwen3vl_grounding_summary.md
```

---

## Typical End-to-End Command Sequence

### 1. 批量增强 gaze 视频

```bash
python scripts/visualization/batch_enhance_gaze_overlay.py \
  --input-root data \
  --point-source gazePoint \
  --pixel-offset-x -20 \
  --overwrite \
  --continue-on-error
```

### 2. 生成事件标注模板

```bash
python scripts/data_prep/generate_event_annotations_template.py \
  --input-root data \
  --output-csv data/event_annotations.csv \
  --overwrite
```

### 3. 修改 `event_annotations.csv`

### 4. 抽取关键帧

```bash
python scripts/data_prep/extract_event_keyframes.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --output-root data \
  --output-template "{scene_id}/keyframes/{t_peak}.jpg" \
  --overwrite
```

### 5. 构建 manifest

```bash
python scripts/data_prep/build_event_manifest.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --json-root data \
  --output-csv data/event_manifest.csv
```

### 6. 构建 grounding 输入 CSV

```bash
python scripts/data_prep/build_keyframe_grounding_input.py \
  --input-csv data/event_manifest.csv \
  --output-csv data/keyframe_grounding_input.csv \
  --overwrite
```

### 7. 可选补充 instruction / utterance / target_description

### 8. 本地运行 Qwen3-VL

```bash
python scripts/grounding/run_qwen3vl_local_keyframe_grounding.py \
  --input_csv data/keyframe_grounding_input.csv \
  --output_csv data/qwen3vl_grounding_output.csv \
  --model_name /path/to/Qwen3-VL-8B-Instruct \
  --local_files_only \
  --dtype auto \
  --max_new_tokens 256 \
  --continue_on_error \
  --vis_dir data/qwen3vl_vis
```

### 9. 生成结果可视化和汇总

```bash
python scripts/grounding/render_grounding_result_overlays.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --source-csv data/keyframe_grounding_input.csv \
  --output-dir data/grounding_overlays \
  --overwrite
```

```bash
python scripts/grounding/summarize_grounding_results.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --output-csv data/qwen3vl_grounding_summary.csv \
  --output-md data/qwen3vl_grounding_summary.md
```

---

## Common Issues

### 1. Multiple video files matched

当前脚本已经默认优先使用：

- `*gaze_enhanced*.mp4`

### 2. ffmpeg not found

确保下面命令可运行：

```bash
ffmpeg -version
ffprobe -version
```

### 3. Missing dependency: torch

请参考：

- `docs/QWEN3VL_ENV_SETUP.md`

或者直接执行：

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh --cuda cu121
```

### 4. 模型输出不是 JSON

当前 grounding 脚本已经会：

- 保留 `model_raw_output`
- 尝试从代码块或额外文本中解析 JSON
- 输出 `parsed_json`
- 写出 `parse_ok` 和 `error_message`

### 5. 本地模型目录无法加载

请确认目录中包含：

- `config.json`
- tokenizer / processor 相关文件
- `*.safetensors`

并优先使用：

```bash
--local_files_only
```

---

## Related Docs

- `docs/SCRIPT_USAGE.md`
- `docs/QWEN3VL_ENV_SETUP.md`
- `docs/QWEN3VL_RUN_AUDIT.md`
- `SCRIPT_INDEX.md`
