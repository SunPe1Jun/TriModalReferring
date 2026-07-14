import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_point_grounding import margin_matching_metrics
from point_grounding_common import Anchor


def scale_anchor(anchor, factor):
    return Anchor(anchor.anchor_id, tuple(value * factor for value in anchor.point))


def test_margin_errors_are_scale_invariant():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (2.0, 0.0, 0.0)),
        Anchor("c", (5.0, 0.0, 0.0)),
    ]
    gt = [anchors[0]]
    pred = [(0.25, 0.0, 0.0)]
    base = margin_matching_metrics(pred, gt, anchors)["matched_margin_errors"][0]

    factor = 7.0
    scaled_anchors = [scale_anchor(anchor, factor) for anchor in anchors]
    scaled_gt = [scaled_anchors[0]]
    scaled_pred = [(pred[0][0] * factor, 0.0, 0.0)]
    scaled = margin_matching_metrics(scaled_pred, scaled_gt, scaled_anchors)["matched_margin_errors"][0]
    assert abs(base - scaled) < 1e-9
