# V3 Full Evaluation Statistics

This file records compact statistics for the Qwen3-VL-30B experiment 3 V3 full run. Source files are under:

- `exam3/outputs_qwen3vl30b_3d_directional_v3_full_evaluable/eval/`

## Sample Accounting

| item | count |
|---|---:|
| manifest event groups | 4000 |
| skipped groups without valid GT anchors | 29 |
| evaluated samples | 3971 |
| valid predictions | 3861 |
| invalid predictions | 110 |

The 29 skipped groups are not evaluable because they do not provide valid GT anchor coordinates. The 110 invalid model predictions are inside the evaluated set and are included in all-sample denominators.

## Threshold Summary

| threshold | all samples | valid only |
|---:|---:|---:|
| 5 deg | 43.24% | 44.47% |
| 10 deg | 49.86% | 51.28% |
| 15 deg | 58.78% | 60.45% |
| 20 deg | 68.55% | 70.50% |
| 30 deg | 79.65% | 81.92% |

The primary reporting range is 5 to 20 degrees. The 30 degree threshold is a loose supplementary reference.

## Invalid Counts By Partition

| partition | invalid count |
|---|---:|
| scene1 | 2 |
| scene2 | 12 |
| scene3 | 36 |
| scene4_room1 | 17 |
| scene4_room2 | 23 |
| scene4_room3 | 19 |
| scene4_room4 | 1 |

All invalid predictions have reason `point_3d_wrong_dimension`.

## Worst Valid Angular Errors

These rows are the largest valid angular errors in `3d_directional_eval_detail.csv`; invalid rows are excluded because they have no defined angle.

| angular error deg | scene | row_index | panel | matched GT | GT ids |
|---:|---|---:|---|---|---|
| 122.47 | scene2 | 5 | P1 | door1 | door1 |
| 120.10 | scene3 | 383 | P2 | truck1 | truck1 |
| 119.34 | scene3 | 370 | P1 | truck1 | truck1 |
| 116.95 | scene3 | 415 | P1 | person3 | van1;person3 |
| 114.45 | scene4_room4 | 188 | P1 | bedside_lamp | bedside_lamp;wall_picture2 |
| 112.44 | scene4_room4 | 118 | P1 | bedside_lamp | bedside_lamp;wall_picture2 |
| 108.99 | scene3 | 324 | P3 | person2 | truck1;person2 |
| 108.15 | scene2 | 210 | P3 | laptop1 | laptop1;desk6 |
| 107.45 | scene1 | 84 | P3 | cargo1 | cargo1 |
| 106.56 | scene4_room1 | 103 | P2 | bookshelf | bookshelf |
| 106.41 | scene4_room4 | 175 | P1 | wall_picture1 | wall_picture1 |
| 106.37 | scene3 | 274 | P2 | truck1 | truck1 |

## Per-Scene Directional Profile

| partition | median deg | @20 all | note |
|---|---:|---:|---|
| scene1 | 10.44 | 71.25% | good directional performance and near-perfect valid rate |
| scene2 | 15.59 | 59.67% | near the 20 degree gate, with several large wrong-referent errors |
| scene3 | 19.42 | 48.75% | hardest scene; weaker direction accuracy and more invalid outputs |
| scene4_room1 | 0.00 | 75.27% | many exact-anchor hits, but schema valid rate is low |
| scene4_room2 | 0.00 | 85.50% | strong direction accuracy, schema failures dominate residual loss |
| scene4_room3 | 0.00 | 83.92% | strong direction accuracy, schema failures dominate residual loss |
| scene4_room4 | 0.00 | 96.98% | strongest partition |
| scene5 | 7.46 | 77.76% | no invalid outputs and good @20 accuracy |

## Acceptance Notes

V3 passes the paper-facing directional criterion:

- median angular error is under 10 degrees;
- all-sample `@20` is above 60%;
- `@5`, `@10`, `@15`, and `@20` are all reported directly;
- `@30` is retained only as a loose reference.

V3 does not meet the earlier 98% engineering valid-rate gate because of 110 wrong-dimension JSON outputs. The final decision is to stop iteration and report this as a limitation rather than running V4.
