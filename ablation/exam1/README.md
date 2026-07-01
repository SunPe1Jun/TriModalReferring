# Experiment 1 Ablation: Closed-set 3D Anchor Selection

This workflow evaluates how closed-set 3D anchor selection changes when model-visible modalities are hidden. The candidate anchor table and evaluator are never changed.

Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`.

## Variants

| Variant | Hidden model-visible evidence |
| --- | --- |
| `no_visual` | video/image evidence; model receives a blank placeholder image |
| `no_gaze` | gaze summary, gazePoint/gazeVector/gazeOrigin, gaze-derived sparse timeline proposals |
| `no_hand` | hand summary and hand/ray prompt fields |
| `no_gaze_hand` | gaze and hand cues |
| `language_anchors_only` | visual, gaze, hand, structured geometry, and sparse timeline; candidate anchors remain |
| `no_structured_geometry` | structured world/camera geometry except the candidate anchor table |

Important limitation: if the original video already contains a visible green gaze marker, `no_gaze` instructs the model to ignore it but does not edit the video frames. `no_visual` is the clean control that removes the video entirely.

## Smoke

```bash
cd /workspace/usr3/TriModal-Referring
CUDA_VISIBLE_DEVICES=0 VARIANTS="no_gaze no_hand" LIMIT=5 bash ablation/exam1/run_exam1_ablation.sh
```

## Full

```bash
cd /workspace/usr3/TriModal-Referring
CUDA_VISIBLE_DEVICES=0 LIMIT= bash ablation/exam1/run_exam1_ablation.sh
```
