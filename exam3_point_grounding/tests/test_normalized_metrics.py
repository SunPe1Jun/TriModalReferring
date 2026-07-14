import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_point_grounding import margin_matching_metrics, nearest_anchor_set_metrics
from point_grounding_common import Anchor


def test_margin_f1_perfect_single_target():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (2.0, 0.0, 0.0)),
        Anchor("c", (5.0, 0.0, 0.0)),
    ]
    gt = [anchors[0]]
    pred = [(0.0, 0.0, 0.0)]
    metrics = margin_matching_metrics(pred, gt, anchors)
    assert metrics["tp@1.0"] == 1
    assert metrics["fp@1.0"] == 0
    assert metrics["fn@1.0"] == 0
    assert metrics["f1@1.0"] == 1.0


def test_extra_prediction_is_false_positive():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (2.0, 0.0, 0.0)),
        Anchor("c", (5.0, 0.0, 0.0)),
    ]
    gt = [anchors[0]]
    pred = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0)]
    metrics = margin_matching_metrics(pred, gt, anchors)
    assert metrics["tp@1.0"] == 1
    assert metrics["fp@1.0"] == 1
    assert metrics["fn@1.0"] == 0
    assert metrics["f1@1.0"] < 1.0


def test_nearest_anchor_set_recovery():
    anchors = [
        Anchor("a", (0.0, 0.0, 0.0)),
        Anchor("b", (10.0, 0.0, 0.0)),
    ]
    gt = [anchors[1]]
    pred = [(9.9, 0.0, 0.0)]
    metrics = nearest_anchor_set_metrics(pred, gt, anchors)
    assert metrics["nearest_ids"] == ["b"]
    assert metrics["exact"]
    assert metrics["f1"] == 1.0
