# TriModal Referring Scripts Guide

## Overview

本说明文件介绍当前仓库中的脚本，主要用于：

- event-level multimodal non-entity grounding 数据准备
- gaze 可视化增强
- 本地 Qwen3-VL grounding 推理
- 结果可视化与汇总

所有说明使用中文，但脚本参数名、文件名、列名保持英文。

---

## 1. `scripts/data_prep/build_event_manifest.py`

### Purpose

根据事件标注 CSV、视频目录和 JSON 目录，生成 `event_manifest.csv`。

### Output columns

- `event_id`
- `scene_id`
- `video_id`
- `video_path`
- `json_path`
- `t_start`
- `t_peak`
- `t_end`
- `keyframe_path`
- `camera_pose_json`
- `gt_anchor_x`
- `gt_anchor_y`
- `gt_anchor_z`
- `gt_type`
- `has_gt_anchor`

### Features

- 使用 Python 标准库
- 支持自动检索样本目录中的视频、JSON、关键帧
- 默认优先使用 `*gaze_enhanced*.mp4`
- 对缺失文件和错误输入有清晰报错

### Example

```bash
python scripts/data_prep/build_event_manifest.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --json-root data \
  --output-csv data/event_manifest.csv
```

---

## 2. `scripts/data_prep/extract_event_keyframes.py`

### Purpose

根据 `event_annotations.csv` 中的 `t_peak`，从每条事件对应的视频中抽取关键帧。

### Dependencies

- `ffmpeg`

### Example

```bash
python scripts/data_prep/extract_event_keyframes.py \
  --annotation-csv data/event_annotations.csv \
  --video-root data \
  --output-root data \
  --output-template "{scene_id}/keyframes/{t_peak}.jpg" \
  --overwrite
```

---

## 3. `scripts/data_prep/generate_event_annotations_template.py`

### Purpose

批量扫描样本目录并生成 `event_annotations.csv` 模板。

### Example

```bash
python scripts/data_prep/generate_event_annotations_template.py \
  --input-root data \
  --output-csv data/event_annotations.csv \
  --overwrite
```

---

## 4. `scripts/data_prep/build_keyframe_grounding_input.py`

### Purpose

将 `event_manifest.csv` 自动转换成 grounding 模型输入 CSV。

### Input

输入 CSV 至少需要包含：

- `event_id`
- `keyframe_path`
- `json_path`
- `t_start`
- `t_peak`
- `t_end`

### Output files and columns

默认会生成两份文件：

- `keyframe_grounding_input.csv`
- `keyframe_grounding_input_with_video.csv`

`keyframe_grounding_input.csv` 列：

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

`keyframe_grounding_input_with_video.csv` 额外列：

- `video_path`
- `t_start`
- `t_end`

### Features

- 自动读取 `json_path` 指向的 `multimodal_data.json`
- 按事件时间窗筛选样本
- 自动构建可直接送入模型的 `gaze_summary`
- 自动构建可直接送入模型的 `hand_summary`
- 自动提取 `t_peak` 附近的事件级 JSON 片段
- 自动生成 `event_json_path`
- 自动计算可投影到关键帧的 `spatial prior`

### Example

```bash
python scripts/data_prep/build_keyframe_grounding_input.py \
  --input-csv data/event_manifest.csv \
  --output-csv data/keyframe_grounding_input.csv \
  --overwrite
```

---

## 5. `scripts/visualization/enhance_gaze_overlay.py`

### Purpose

对单条视频中的 gaze 点做增强显示。

### Dependencies

- `ffmpeg`
- `ffprobe`

### Example

```bash
python scripts/visualization/enhance_gaze_overlay.py \
  --input-video data/1/ScreenRecord_2025-10-11-20_53_14.mp4 \
  --input-json data/1/multimodal_data.json \
  --metadata-json data/1/metadata_2025-10-11_20-53-14-907.json \
  --output-video data/1/ScreenRecord_2025-10-11-20_53_14_gaze_enhanced.mp4 \
  --point-source gazePoint \
  --pixel-offset-x -20 \
  --overwrite
```

---

## 6. `scripts/visualization/batch_enhance_gaze_overlay.py`

### Purpose

批量增强整个数据目录下的视频 gaze marker。

### Example

```bash
python scripts/visualization/batch_enhance_gaze_overlay.py \
  --input-root data \
  --point-source gazePoint \
  --pixel-offset-x -20 \
  --overwrite \
  --continue-on-error
```

---

## 7. `scripts/grounding/download_hf_model_via_mirror.py`

### Purpose

通过 `hf-mirror` 下载 `Qwen3-VL-30B-A3B-Instruct` 或其他 Hugging Face 仓库。

### Example

```bash
python scripts/grounding/download_hf_model_via_mirror.py \
  --output_dir /ai/data/Qwen3-VL-30B-A3B-Instruct \
  --allow_patterns "*" \
  --resume_download
```

---

## 8. `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`

### Purpose

本地加载 Qwen3-VL，对“语言 + gaze + hand + JSON 空间信息”的多模态非实体指代 grounding 任务做推理。

### Backward compatibility

兼容旧版输入 CSV：

- `event_id`
- `keyframe_path`
- `gaze_summary`
- `hand_summary`

也支持新增可选列：

- `instruction_text`
- `utterance_text`
- `target_description`
- `event_json_path`

### Output columns

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

### Features

- prompt 中显式加入 task instruction / utterance / target description / gaze / hand / JSON 空间线索
- prompt 中显式加入 `spatial prior` 的 `u_norm / v_norm / source`
- 明确指定这是非实体空间指代 grounding，不是 object grounding
- JSON 解析兼容代码块和额外文本
- 保留原始模型输出和解析结果
- 支持 `--vis_dir` 生成关键帧 overlay 图
- overlay 同时绘制模型预测点和 JSON prior 点

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

---

## 9. `scripts/grounding/render_grounding_result_overlays.py`

### Purpose

把预测结果中的候选点重新绘制到 keyframe 图片上，便于人工检查。

### Example

```bash
python scripts/grounding/render_grounding_result_overlays.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --source-csv data/keyframe_grounding_input.csv \
  --output-dir data/grounding_overlays \
  --overwrite
```

---

## 10. `scripts/grounding/summarize_grounding_results.py`

### Purpose

把 grounding 输出 CSV 汇总成更易读的 CSV 和 Markdown 报告。

### Example

```bash
python scripts/grounding/summarize_grounding_results.py \
  --pred-csv data/qwen3vl_grounding_output.csv \
  --output-csv data/qwen3vl_grounding_summary.csv \
  --output-md data/qwen3vl_grounding_summary.md
```

---

## 11. `scripts/grounding/install_qwen3vl_requirements.sh`

### Purpose

在远程服务器上为 Qwen3-VL grounding 补齐推理环境。

### Example

```bash
bash scripts/grounding/install_qwen3vl_requirements.sh \
  --python /path/to/python \
  --cuda cu121
```

---

## Related docs

- `docs/FULL_WORKFLOW.md`
- `docs/QWEN3VL_ENV_SETUP.md`
- `docs/QWEN3VL_RUN_AUDIT.md`
- `SCRIPT_INDEX.md`

