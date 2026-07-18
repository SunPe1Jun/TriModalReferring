# Experiment 3 Qwen3-VL-30B Ablation

This workflow evaluates controlled model-input ablations for the frozen v9 candidate-free measured point-hypothesis diagnostic. It does not modify the baseline manifest, selected frames, GT, or raw Qwen3-VL-30B outputs.

## Variants

- `no_visual`
- `no_gaze`
- `no_hand`
- `no_gaze_hand`
- `no_instruction`

The repaired workflow uses the same 4,000 sample IDs, Qwen3-VL-30B-A3B-Instruct checkpoint, greedy decoding, parser, and Experiment 3 evaluator throughout. Behavioral cues are masked after the target-free frame selector, so these are descriptive input ablations rather than strict single-modality causal interventions. Earlier 3,971-row partial outputs predate GT completion and are not paper results.

The final v3 masking implementation operates on the frozen expanded `prompt_text` exactly as it was parsed by the baseline runner. It removes only bounded field spans and preserves all remaining characters. Earlier local v1/v2 smoke and partial directories are implementation diagnostics and are not paper results.

## Smoke Test

```bash
bash ablation/exam3/run_parallel_smoke.sh
```

## Full Run

Use `setsid` so both GPU lanes survive an SSH disconnect:

```bash
setsid bash ablation/exam3/run_parallel_full.sh \
  > ablation/exam3/logs/full_parallel.log 2>&1 < /dev/null &
```

The full launcher assigns one sequential lane to each A100. Existing per-sample JSON outputs are reused on restart unless `OVERWRITE_INFERENCE=1` is explicitly set.

Progress can be checked without touching the running jobs:

```bash
python ablation/exam3/monitor_full.py
```
