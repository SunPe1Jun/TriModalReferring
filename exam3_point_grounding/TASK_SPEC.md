# Task Spec: Point-Supervised 3D Referent Grounding

## Task

For each referential interaction, the model receives:

- instruction text;
- optional utterance text;
- up to three chronological evidence frames;
- per-frame relative time;
- camera position, forward/right/up basis, and FOV;
- gaze validity, origin, direction, and ray endpoint;
- hand validity, approximate ray origin, direction, and ray endpoint;
- scene robust coordinate bounds and distance-scale cues.

The model predicts:

```json
{
  "points_3d": [
    {
      "referent": "short object or spatial description",
      "point": [0.0, 0.0, 0.0],
      "confidence": 0.5
    }
  ]
}
```

An empty `points_3d` array is valid but scores as an empty prediction set.

## Leakage Boundary

The prompt and model-facing manifest must not contain:

- candidate anchor names or coordinates;
- GT anchor ids or coordinates;
- GT referent count;
- `target_description`;
- Exam2 projected GT coordinates;
- Exam2 model predictions;
- old Experiment 3 candidate-anchor lists.

The evaluator may load anchor tables and evaluator-only GT files after inference.

## Evidence Selection

Evidence selection is target-free:

1. sample candidate timestamps every 0.5 seconds across the event window;
2. score a local +/-0.3s window using gaze validity/stability, hand validity/stability, and camera stability;
3. apply temporal NMS with 0.85s minimum separation;
4. select up to three frames, preferring one gaze-supported frame, one hand-supported frame, and one distinct high-quality frame;
5. fall back to 25/50/75% event times if no cue-supported candidates exist.

## Evaluation

Let `A_s` be all released anchors in scene `s`, `G_i` the GT anchor set, and `P_i` the predicted point set.

### Candidate-Free Nearest-Anchor Set Recovery

For each predicted point:

```text
nearest(p) = argmin_a in A_s ||p - a||_2
```

Compare the nearest-anchor set with `G_i`. Report precision, recall, F1, exact set match, cardinality error, and duplicate nearest-anchor rate.

### Nearest-Distractor Margin Normalization

For each GT anchor `g`:

```text
m(g) = min_{a in A_s - G_i} ||g - a||_2
r(g) = 0.5 * m(g)
normalized_error(p, g) = ||p - g||_2 / (r(g) + eps)
```

Use Hungarian matching over the full prediction-GT matrix. Report Margin-F1@0.5, Margin-F1@1.0, Margin-F1@2.0, and matched normalized-error mean/median.

### Robust Scene-Normalized Distance

For scene robust diagonal:

```text
D_s = sqrt((q95_x-q05_x)^2 + (q95_y-q05_y)^2 + (q95_z-q05_z)^2)
```

Report matched `||p-g||_2 / D_s` mean and median.

## Invalid Outputs

Malformed JSON, non-array `points_3d`, wrong dimensions, NaN/Inf, invalid confidence, and non-numeric coordinates are marked invalid. Invalid outputs are kept in the all-sample denominator and evaluated as empty prediction sets.
