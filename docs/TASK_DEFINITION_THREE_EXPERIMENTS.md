# VR-TriRef Task Definition: Three Diagnostic Experiments

This document summarizes the three grounding diagnostics used in VR-TriRef. The experiments are designed as complementary views of the same referential interaction problem: given a user's language and egocentric multimodal evidence, identify the intended referent in a 3D VR scene.

## Overall Problem

For each interaction event, the input consists of a natural-language referring expression, first-person visual evidence, and synchronized behavioral telemetry such as gaze, hand, and camera pose. The target may be a single object or a set of objects. The central question is whether a vision-language model can use these multimodal cues to recover the user's intended referent under different output constraints.

The three diagnostics progressively vary the model's output space. Experiment 1 evaluates closed-set 3D selection, Experiment 2 evaluates image-plane point grounding after 3D-to-2D projection, and Experiment 3 evaluates candidate-free 3D point grounding.

## Experiment 1: Closed-Set 3D Anchor Selection

The first diagnostic formulates referring grounding as closed-set selection over scene-level 3D anchors. For a given event, the model receives the referring expression, egocentric evidence, behavioral cues, and a list of candidate anchors from the scene. It must return the anchor or anchors that correspond to the user's intended referent.

Formally, let $A_s = \{a_1, \ldots, a_n\}$ denote the candidate anchor set for scene $s$, and let $G_i \subseteq A_s$ denote the ground-truth referent set for event $i$. The model predicts a subset $P_i \subseteq A_s$. Evaluation compares $P_i$ with $G_i$, allowing both single-target and multi-target interactions.

This experiment measures whether the model can associate the user's multimodal referring behavior with the correct semantic object when the possible 3D referents are explicitly enumerated. Candidate anchors are part of the task input, so this setting primarily tests referent selection rather than open-ended localization.

Primary outcomes include any-hit accuracy, exact-set accuracy, mapped-only accuracy, micro precision, micro recall, micro F1, and per-scene performance.

## Experiment 2: Projected-2D Point Diagnostic

The second diagnostic evaluates whether the model can identify the relevant temporal evidence and localize the referent in the egocentric image plane. Ground-truth 3D anchors are projected into selected first-person image panels using camera pose, producing 2D point supervision for visible evidence frames.

For each event, the model observes the referring expression and a small set of image panels sampled from the interaction. It predicts which panel contains the most relevant visual evidence and outputs one or more 2D points for the intended referent. Let $q_i$ be the projected 2D location of a ground-truth anchor in a valid panel, and let $p_i$ be a predicted 2D point. A point prediction is correct when its image-plane distance to a valid projected target is within a fixed pixel threshold.

This experiment isolates two capabilities that are not visible in closed-set anchor selection: temporal evidence selection and spatial localization in the user's field of view. Because the target is evaluated in image coordinates, the diagnostic is sensitive to whether the model attends to the correct moment and visible referent region rather than merely exploiting candidate-anchor names.

Primary outcomes include temporal selection F1, point F1 at pixel thresholds such as 50 and 100 pixels, joint temporal-and-point F1, matched point distance, and per-scene performance.

## Experiment 3: Candidate-Free Point-Supervised 3D Grounding

The third diagnostic removes the candidate anchor list from the model input. The model receives the referring expression, target-free egocentric evidence frames, camera pose, gaze telemetry, hand telemetry, and broad scene-scale context in a shared Unity world coordinate system. It must output a variable-size set of 3D world-coordinate points corresponding to the intended referent or referents.

Let $G_i = \{g_1, \ldots, g_m\}$ be the hidden set of ground-truth anchor points for event $i$, and let $P_i = \{p_1, \ldots, p_k\}$ be the model's predicted 3D point set. Candidate anchors and ground-truth coordinates are not exposed to the model. They are used only after inference to evaluate whether the predicted points recover the intended referents.

The current protocol frames this experiment as candidate-free point-supervised grounding rather than unconstrained 3D object reconstruction. Gaze and hand endpoints are treated as measured behavioral point hypotheses: informative cues that may support grounding, but not ground truth object centers. This distinction is important because the model is evaluated on referent recovery, not on reconstructing object extent or metric 3D shape.

Evaluation maps each predicted 3D point to its nearest scene anchor and compares the resulting anchor set with the hidden ground-truth set. In addition, point errors are normalized by local anchor ambiguity margins and by scene scale. Invalid or malformed model outputs are retained in the denominator and treated as empty predictions.

Primary outcomes include nearest-anchor set precision, recall, F1, exact-set accuracy, margin-normalized F1, scene-normalized point error, valid-output rate, and per-scene performance.

## Relationship Among the Diagnostics

The three experiments are not redundant. They evaluate the same underlying referential grounding problem under increasingly less constrained output spaces.

Experiment 1 asks whether the model can choose the correct referent when all candidate 3D anchors are provided. Experiment 2 asks whether the model can localize the referent in egocentric visual evidence and select the relevant temporal panel. Experiment 3 asks whether the model can produce candidate-free 3D point predictions using language, visual evidence, and behavioral telemetry, with anchors hidden until evaluation.

Together, these diagnostics separate semantic selection, image-plane grounding, temporal evidence use, and target-free 3D point grounding. This structure makes it possible to analyze not only whether a model identifies the correct referent, but also which form of multimodal evidence and spatial supervision supports that decision.
