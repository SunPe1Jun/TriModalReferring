# TriModalReferring

Experiment code for VR-TriRef multimodal referring-grounding diagnostics.

This repository contains scripts and prompt templates for:

- closed-set 3D anchor selection with Qwen3-VL;
- projected-2D point grounding diagnostics;
- post-hoc selected-panel export from the projected-2D experiment;
- camera-centered 3D directional point diagnostics.

Large assets are intentionally excluded from git, including V3dMD data, model weights, extracted frames, model outputs, logs, and evaluation artifacts.

Default remote-server paths used by the scripts:

- repo: `/workspace/usr3/TriModal-Referring`
- dataset: `/workspace/usr3/V3dMD`
- Qwen3-VL-30B: `/workspace/usr3/Qwen3-VL-30B-A3B-Instruct`
- GroundingDINO: `/workspace/usr3/grounding-dino-base`

Activate the configured environment before running experiments:

```bash
conda activate trimodal
```

Key entry points:

```bash
# Experiment 1: closed-set 3D anchor selection
bash scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh

# Experiment 2: projected-2D point diagnostic
bash exam2/run_qwen3vl_30b_2d_full.sh

# Export post-hoc selected panels into each V3dMD sample folder
bash exam2/run_export_selected_panels.sh

# Experiment 3: camera-centered 3D directional diagnostic
bash exam3/run_qwen3vl_30b_3d_directional.sh
```

Use `LIMIT=20` for smoke tests and `LIMIT=` for full runs in the current shell wrappers.
