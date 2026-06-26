# Script Index

## Folder Layout

### `scripts/data_prep`
数据准备相关脚本，推荐先运行这一组。

- `generate_event_annotations_template.py`
  - 批量生成 `event_annotations.csv` 模板
- `extract_event_keyframes.py`
  - 根据 `t_peak` 从视频中抽取关键帧
- `build_event_manifest.py`
  - 自动检索样本目录并生成 `event_manifest.csv`
- `build_keyframe_grounding_input.py`
  - 将 `event_manifest.csv` 转成 grounding 模型输入 CSV，并自动构建 gaze/hand 文本摘要

### `scripts/visualization`
视频中的 gaze 点增强与可视化脚本。

- `enhance_gaze_overlay.py`
  - 单条视频增强 gaze marker
- `batch_enhance_gaze_overlay.py`
  - 批量增强整个数据目录下的视频

### `scripts/grounding`
下游 grounding 推理与模型准备脚本。

- `download_hf_model_via_mirror.py`
  - 通过 hf-mirror 下载 Hugging Face 模型仓库
- `run_qwen3vl_local_keyframe_grounding.py`
  - 本地加载 `Qwen/Qwen3-VL-8B-Instruct` 做语言 + gaze + hand + JSON 空间信息联合 grounding
- `requirements_qwen3vl.txt`
  - Qwen3-VL 推理依赖清单
- `install_qwen3vl_requirements.sh`
  - 远程环境依赖安装脚本

### `docs`
说明文档。

- `SCRIPT_USAGE.md`
- `FULL_WORKFLOW.md`
- `QWEN3VL_ENV_SETUP.md`

---

## Recommended Order

推荐顺序如下：

1. 如果需要先增强 gaze 可视化视频：
   - `scripts/visualization/enhance_gaze_overlay.py`
   - 或 `scripts/visualization/batch_enhance_gaze_overlay.py`
2. 生成事件标注模板：
   - `scripts/data_prep/generate_event_annotations_template.py`
3. 抽取关键帧：
   - `scripts/data_prep/extract_event_keyframes.py`
4. 构建 manifest：
   - `scripts/data_prep/build_event_manifest.py`
5. 构建 grounding 输入 CSV：
   - `scripts/data_prep/build_keyframe_grounding_input.py`
6. 配置 Qwen3-VL 运行环境：
   - `scripts/grounding/install_qwen3vl_requirements.sh`
7. 跑本地 Qwen3-VL grounding：
   - `scripts/grounding/run_qwen3vl_local_keyframe_grounding.py`

---

## Common Commands

生成标注模板：

```bash
python scripts/data_prep/generate_event_annotations_template.py ^
  --input-root data ^
  --output-csv data/event_annotations.csv ^
  --overwrite
```

抽取关键帧：

```bash
python scripts/data_prep/extract_event_keyframes.py ^
  --annotation-csv data/event_annotations.csv ^
  --video-root data ^
  --output-root data ^
  --output-template "{scene_id}/keyframes/{t_peak}.jpg"
```

构建 manifest：

```bash
python scripts/data_prep/build_event_manifest.py ^
  --annotation-csv data/event_annotations.csv ^
  --video-root data ^
  --json-root data ^
  --output-csv data/event_manifest.csv
```

构建 grounding 输入 CSV：

```bash
python scripts/data_prep/build_keyframe_grounding_input.py ^
  --input-csv data/event_manifest.csv ^
  --output-csv data/keyframe_grounding_input.csv ^
  --overwrite
```

批量增强 gaze 视频：

```bash
python scripts/visualization/batch_enhance_gaze_overlay.py ^
  --input-root data ^
  --point-source gazePoint ^
  --pixel-offset-x -20 ^
  --overwrite ^
  --continue-on-error
```

运行本地 Qwen3-VL grounding：

```bash
python scripts/grounding/run_qwen3vl_local_keyframe_grounding.py ^
  --input_csv data/keyframe_grounding_input.csv ^
  --output_csv data/qwen3vl_grounding_output.csv ^
  --vis_dir data/qwen3vl_vis
```
