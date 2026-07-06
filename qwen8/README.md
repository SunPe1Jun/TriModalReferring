# Qwen3-VL-8B Baselines

This folder contains isolated Qwen3-VL-8B runners for the experiment 1 and
experiment 2 baselines. The scripts reuse the existing Qwen3-VL prompt builders,
parsers, manifests, and evaluators; only the model path, output roots, and default
GPU binding are changed.

Default model path:

```bash
/workspace/usr3/Qwen3-VL-8B-Instruct
```

Default GPU binding:

```bash
CUDA_VISIBLE_DEVICES=1
```

Smoke tests:

```bash
LIMIT=10 SCENES=scene1 bash qwen8/run_exam1_qwen3vl8b_baseline.sh
LIMIT=10 SCENES=scene1 bash qwen8/run_exam2_qwen3vl8b_baseline.sh
```

Full run:

```bash
LIMIT= bash qwen8/run_qwen3vl8b_full_sequence.sh
```

Outputs are written under `qwen8/outputs/`, and logs under `qwen8/logs/`.
