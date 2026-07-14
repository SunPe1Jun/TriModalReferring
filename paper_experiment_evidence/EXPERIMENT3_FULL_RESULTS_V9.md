# Experiment 3: Candidate-Free Measured Point-Hypothesis Diagnostic

The model receives language, up to three evidence frames, camera/gaze/hand telemetry, and broad scene bounds. It outputs one or more measured Unity-world 3D points. Candidate anchor IDs and hidden GT points are not supplied to the model. The evaluator subsequently associates each predicted point with its nearest hidden anchor, computes set metrics, local distractor-margin metrics, and scene-normalized error.

The final manifest contains 3,971 rows. Qwen3-VL-30B, Qwen3-VL-8B, and InternVL3-38B each have 3,971 valid parsed outputs. Single-target and multi-target partitions are preserved in the sample files. The gaze-copy baseline is a required diagnostic because the v9 prompt exposes copyable gaze hypotheses; it ties Qwen on anchor F1 and is slightly stronger on Margin-F1.

These results support analysis of measured behavioral point hypotheses. They do not support claims of unconstrained 3D reconstruction, 3D boxes, object extents, or 3D IoU.
