# Experiment 1 Strict Model-Input Hand Ablation

This compact bundle records the completed Qwen3-VL-30B
`no_hand_strict` feasibility run for closed-set 3D anchor selection. The
runner removes structured hand fields from the prompt and masks projected
tracked-hand regions in every decoded video frame before constructing the
model's video tensor. Candidate anchors, parser, evaluator, decoding policy,
and the 4,000-sample set are unchanged.

| Condition | Hit-All | Exact | Macro set F1 | Micro P/R/F1 | Valid |
| --- | ---: | ---: | ---: | --- | ---: |
| Full | 0.77975 | 0.32875 | 0.56570 | 0.56493 / 0.51691 / 0.53986 | 3,970/4,000 |
| `no_hand_strict` | 0.77900 | 0.32675 | 0.56519 | 0.57367 / 0.51003 / 0.53999 | 3,969/4,000 |

`predictions/qwen3_vl_30b_no_hand_strict.csv` contains one row per
`scene::row_index`. `hand_mask_audit.csv` contains 60,988 per-frame mask
records; `validation.json` confirms complete sample coverage and zero detected
structured hand-prompt leaks.

The implementation verifies strict model-input ablation feasibility, not
pixel-perfect hand segmentation. Read `../STRICT_HAND_ABLATION_AUDIT.md`
before using the metrics.
