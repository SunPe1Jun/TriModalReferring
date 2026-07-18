# Experiment 3 Full Results (4,000 interactions)

Experiment 3 is a candidate-free measured point-hypothesis diagnostic. The model receives language, target-free evidence frames, camera/gaze/hand telemetry, and scene-scale context, then emits measured Unity-world 3D points. Hidden anchors are used only by the evaluator.

| Model or baseline | Valid | Anchor F1 | Exact | Margin-F1@1.0 | Scene-normalized error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL-30B-A3B | 4000/4000 | 0.4326 | 0.0185 | 0.2503 | 0.1569 |
| Qwen3-VL-8B | 4000/4000 | 0.4213 | 0.0225 | 0.2440 | 0.1550 |
| InternVL3-38B | 4000/4000 | 0.3931 | 0.0480 | 0.2310 | 0.1615 |
| gaze_copy | 4000/4000 | 0.4325 | 0.0435 | 0.2547 | 0.1519 |
| hand_copy | 4000/4000 | 0.0914 | 0.0008 | 0.0069 | 0.8462 |
| gaze_hand_fusion | 4000/4000 | 0.4069 | 0.0090 | 0.2302 | 0.2258 |

The gaze-copy baseline is reported because gaze hypotheses are exposed by the task input and provide a strong copy-based control. The diagnostic does not measure object extents, boxes, 3D IoU, or unconstrained reconstruction.
