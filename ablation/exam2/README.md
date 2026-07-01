# Experiment 2 Ablation: Projected-2D Point Diagnostic

This workflow evaluates how projected-2D point grounding changes when model-visible evidence is hidden. The manifest builder, parser, and evaluator are reused from `exam2/`.

Baseline is reused from `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`.

## Variants

| Variant | Change |
| --- | --- |
| `full_panels_no_crop` | use full panels only, removing the gaze-centered crop path |
| `no_gaze_text_prior` | remove gaze-specific prompt/system text and use full panels only |
| `no_gaze` | remove gaze-specific text and mask the visible green gaze marker in copied panel images; use full panels only |
| `instruction_only_prompt` | keep visual setup but use `PROMPT_MODE=instruction_only` |
| `single_panel` | use one evidence-selected panel instead of three |
| `blank_visual` | replace panels with blank placeholders as a sanity control |

Experiment 2 has no explicit hand summary in the current manifest/prompt interface. Therefore this workflow does not report a primary `no_hand` ablation for experiment 2.

## Smoke

```bash
cd /workspace/usr3/TriModal-Referring
CUDA_VISIBLE_DEVICES=1 VARIANTS="full_panels_no_crop no_gaze_text_prior" LIMIT=5 bash ablation/exam2/run_exam2_ablation.sh
```

## Full

```bash
cd /workspace/usr3/TriModal-Referring
CUDA_VISIBLE_DEVICES=1 LIMIT= bash ablation/exam2/run_exam2_ablation.sh
```
