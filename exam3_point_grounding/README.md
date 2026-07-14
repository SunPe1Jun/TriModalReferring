# Experiment 3: Point-Supervised 3D Referent Grounding

This directory replaces the archived camera-centered angular diagnostic under `exam3/`.

The new task is candidate-free, point-supervised 3D anchor-point grounding:

- the model does not receive candidate anchor names or coordinates;
- the model predicts a variable-size set of 3D world-coordinate points;
- GT supervision is the released scene-level anchor-point set;
- evaluation happens after inference by mapping predictions to nearest anchors and by scale-normalized point distances.

The task must not be described as 3D box grounding, 3D IoU grounding, object-extent localization, or ScanRefer-style box evaluation.

## Main Commands

Smoke manifest and cue baselines:

```bash
cd /workspace/usr3/TriModal-Referring
CONDA_ENV_BIN=/workspace/usr3/miniconda3/envs/trimodal/bin \
LIMIT=10 RUN_QWEN=0 RUN_EVAL=1 \
bash exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh
```

Qwen smoke:

```bash
cd /workspace/usr3/TriModal-Referring
CONDA_ENV_BIN=/workspace/usr3/miniconda3/envs/trimodal/bin \
LIMIT=10 RUN_MANIFEST=1 RUN_BASELINES=1 RUN_QWEN=1 RUN_EVAL=1 \
bash exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh
```

Full run:

```bash
cd /workspace/usr3/TriModal-Referring
CONDA_ENV_BIN=/workspace/usr3/miniconda3/envs/trimodal/bin \
RUN_MANIFEST=1 RUN_BASELINES=1 RUN_QWEN=1 RUN_EVAL=1 \
bash exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh
```

## Outputs

- `outputs/manifest.csv`: model-facing manifest, no GT/candidate anchor fields.
- `outputs/gt_manifest_eval.csv`: evaluator-only GT anchor ids and points.
- `outputs/manifest_summary.json`: audit and coverage summary.
- `outputs/cue_baselines/*/predictions.csv`: gaze/hand/fusion baselines.
- `outputs/qwen3vl30b/predictions.csv`: Qwen parsed predictions.
- `outputs/qwen3vl30b/raw/`: raw model responses and parsed payloads.
- `outputs/qwen3vl30b/evaluation_summary.json`: final metrics.
- `outputs/qwen3vl30b/evaluation_detail.csv`: per-sample metrics.

## Metric Summary

1. Candidate-free nearest-anchor set recovery: map each predicted point to the nearest scene anchor in the evaluator, then compare predicted anchor set with GT anchor set.
2. Nearest-distractor margin normalization: normalize point error by half the nearest negative-anchor distance for each GT point and compute Margin-F1@0.5, @1.0, and @2.0 with Hungarian matching.
3. Robust scene-normalized distance: divide matched Euclidean error by the scene robust 5th/95th percentile diagonal.

Invalid model outputs are treated as empty prediction sets for end-to-end metrics and are not removed from the sample denominator.
