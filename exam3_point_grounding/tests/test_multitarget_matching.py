import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_point_grounding import margin_matching_metrics, nearest_anchor_set_metrics
from point_grounding_common import Anchor


def test_multitarget_matching_recovers_both_targets():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (5.0, 0.0, 0.0)),
        Anchor("c", (10.0, 0.0, 0.0)),
        Anchor("d", (0.0, 5.0, 0.0)),
    ]
    gt = [anchors[0], anchors[1]]
    pred = [(0.1, 0.0, 0.0), (5.1, 0.0, 0.0)]
    set_metrics = nearest_anchor_set_metrics(pred, gt, anchors)
    margin_metrics = margin_matching_metrics(pred, gt, anchors)
    assert set_metrics["exact"]
    assert margin_metrics["tp@1.0"] == 2
    assert margin_metrics["fp@1.0"] == 0
    assert margin_metrics["fn@1.0"] == 0


def test_missing_prediction_is_false_negative():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (5.0, 0.0, 0.0)),
        Anchor("c", (10.0, 0.0, 0.0)),
    ]
    gt = [anchors[0], anchors[1]]
    pred = [(0.0, 0.0, 0.0)]
    metrics = margin_matching_metrics(pred, gt, anchors)
    assert metrics["tp@1.0"] == 1
    assert metrics["fp@1.0"] == 0
    assert metrics["fn@1.0"] == 1
