# Experiment 2 Strict Model-Input Hand Ablation

This compact bundle records the completed Qwen3-VL-30B
`no_hand_strict` feasibility run for the projected-2D diagnostic. It reuses
the frozen panel manifest and saves a hand-masked replacement for every panel
before the image paths are passed to the model. The prompt, panel order,
parser, evaluator, and 4,000-sample set are otherwise unchanged.

| Condition | Temporal F1 | Point@100 F1 | Joint@100 F1 | Valid |
| --- | ---: | ---: | ---: | ---: |
| Full | 0.73475 | 0.24968 | 0.20669 | 3,998/4,000 |
| `no_hand_strict` | 0.69352 | 0.13497 | 0.08529 | 4,000/4,000 |

`predictions/qwen3_vl_30b_no_hand_strict.csv` contains one row per
`scene::row_index`. The 11,834 entries in `hand_mask_audit.csv` all reference
existing files under the hand-masked model-input path.

The implementation verifies strict model-input ablation feasibility, not
pixel-perfect hand segmentation. Read `../STRICT_HAND_ABLATION_AUDIT.md`
before interpreting the larger Exp.2 drop, because rectangular masking can
also remove nearby target context.
