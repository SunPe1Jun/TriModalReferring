# InternVL Baseline Workflows

This folder adds local InternVL baselines for the established VR-TriRef experiments:

- Experiment 1: closed-set 3D anchor selection.
- Experiment 2: projected-2D point diagnostic.

The scripts reuse the existing Qwen prompt builders, parsers, manifests, and evaluators. Only the local model adapter is changed to `OpenGVLab/InternVL3-38B-Instruct`.

The intended local model path is:

```bash
/workspace/usr3/InternVL3-38B-Instruct
```

On the current 2xA100 80GB machine, the 38B model is loaded in BF16 by default (`LOAD_IN_8BIT=0`). The 8-bit path currently needs extra transformers 5.x compatibility work.

## Smoke Tests

```bash
cd /workspace/usr3/TriModal-Referring

LIMIT=2 SCENES=scene1 bash internvl/run_exam1_internvl38b_baseline.sh

LIMIT=2 SCENES=scene1 RUN_DEBUG_RENDER=0 bash internvl/run_exam2_internvl38b_baseline.sh
```

## Full Runs

```bash
cd /workspace/usr3/TriModal-Referring

LIMIT= bash internvl/run_exam1_internvl38b_baseline.sh

LIMIT= RUN_DEBUG_RENDER=0 bash internvl/run_exam2_internvl38b_baseline.sh
```

Outputs are written under:

- `internvl/outputs/exam1_internvl3_38b_baseline/`
- `internvl/outputs/exam2_internvl3_38b_baseline/`
