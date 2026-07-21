# Experiment 3 Qwen3-VL-30B Ablation

This workflow evaluates controlled model-input ablations for the frozen v9 candidate-free measured point-hypothesis diagnostic. It does not modify the baseline manifest, selected frames, GT, or raw Qwen3-VL-30B outputs.

## Variants

- `no_visual`
- `no_gaze`
- `no_hand`
- `no_hand_strict`
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

## Final Results

The five 4,000-sample Qwen3-VL-30B runs are complete. The authoritative report is
`ablation/exam3/reports/full_v3/EXPERIMENT3_QWEN30B_ABLATION.md`.

| Variant | Valid | Anchor F1 | Exact | Margin-F1@1.0 | Margin-F1@2.0 | Mean scene-normalized error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 4,000 | 0.4326 | 0.0185 | 0.2503 | 0.4105 | 0.1569 |
| no_visual | 4,000 | 0.4341 | 0.0180 | 0.2518 | 0.4120 | 0.1560 |
| no_gaze | 4,000 | 0.0673 | 0.0013 | 0.0055 | 0.0263 | 0.3771 |
| no_hand | 4,000 | 0.4353 | 0.0187 | 0.2527 | 0.4134 | 0.1558 |
| no_gaze_hand | 3,999 | 0.0542 | 0.0000 | 0.0000 | 0.0037 | 0.5376 |
| no_instruction | 4,000 | 0.4332 | 0.0177 | 0.2510 | 0.4111 | 0.1565 |

These are descriptive post-selection input ablations. The frozen frame selector used gaze and hand availability, and the v9 prompt asks the model to copy measured gaze hypotheses by default. The sharp `no_gaze` degradation therefore measures dependence on exposed gaze hypotheses under this protocol; it is not a strict causal estimate of gaze contribution in general referential grounding.

Export compact sample-level evidence without changing raw outputs:

```bash
python ablation/exam3/export_compact_evidence.py
```

## Strict Hand-Input Ablation

The earlier `no_hand` condition removes structured hand fields only. The separate `no_hand_strict` condition also projects tracked left/right hand joints into each frozen evidence panel and replaces the expanded hand regions with a neutral fill before inference. It preserves the original panel selection, panel count, image size, prompt template, decoding, parser, and evaluator. It is a strict **model-input** hand ablation after frame selection, not a strict intervention on the hand-aware frame selector.

Run the two-GPU full workflow with SSH-safe detachment:

```bash
setsid bash ablation/exam3/run_strict_hand_full.sh \
  > ablation/exam3/logs/strict_hand_full_launcher.log 2>&1 < /dev/null &
```

The mask geometry and per-panel coverage audit are stored under `ablation/exam3/hand_masked_frames_v1_full_audit.json`; raw model outputs remain in the strict run output directory and are separate from the earlier descriptive ablation.
