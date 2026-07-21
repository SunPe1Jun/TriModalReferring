# Modality Ablation Workflows

This folder contains the completed VR-TriRef ablation workflows:

- `exam1`: closed-set 3D anchor selection.
- `exam2`: projected-2D point diagnostic.
- `exam3`: candidate-free measured point-hypothesis diagnostic.

The ablation workflows reuse the existing dataset files, prompt/parser logic, and evaluators. Baseline outputs are not rerun by default and are read from the existing experiment folders:

- Experiment 1 baseline: `data/match_eval_qwen3vl30b_mention_first_v3/`
- Experiment 2 baseline: `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`

Generated outputs live under `ablation/exam*/outputs/` and logs under `ablation/logs/`. Experiment 3 final reports and compact evidence are under `ablation/exam3/reports/full_v3/`.

## Quick Smoke Test

Run both experiments in parallel on two GPUs:

```bash
cd /workspace/usr3/TriModal-Referring
bash ablation/run_parallel_smoke.sh
```

## Full Runs

After smoke outputs look sane:

```bash
cd /workspace/usr3/TriModal-Referring
bash ablation/run_parallel_full.sh
```

Each experiment can also be run independently through its own script.
