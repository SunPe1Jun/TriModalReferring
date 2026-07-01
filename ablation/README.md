# Modality Ablation Workflows

This folder contains ablation workflows for the two established VR-TriRef experiments only:

- `exam1`: closed-set 3D anchor selection.
- `exam2`: projected-2D point diagnostic.

Experiment 3 is intentionally excluded from this ablation round because its feasibility and paper framing are still under discussion.

The ablation workflows reuse the existing dataset files, prompt/parser logic, and evaluators. Baseline outputs are not rerun by default and are read from the existing experiment folders:

- Experiment 1 baseline: `data/match_eval_qwen3vl30b_mention_first_v3/`
- Experiment 2 baseline: `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`

Generated outputs live under `ablation/exam*/outputs/` and logs under `ablation/logs/`.

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
