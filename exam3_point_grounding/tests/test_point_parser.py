import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from point_parser import parse_points_3d_output


def test_valid_points_parse():
    ok, parsed, reason = parse_points_3d_output('{"points_3d":[{"referent":"box","point":[1,2.5,-3],"confidence":0.7}]}')
    assert ok
    assert reason == ""
    assert parsed["points_3d"][0]["point"] == [1.0, 2.5, -3.0]


def test_empty_array_is_valid():
    ok, parsed, reason = parse_points_3d_output('{"points_3d":[]}')
    assert ok
    assert reason == ""
    assert parsed == {"points_3d": []}


def test_extract_json_from_markdown_wrapped_text():
    ok, parsed, reason = parse_points_3d_output('```json\n{"points_3d":[]}\n```')
    assert ok
    assert reason == ""
    assert parsed == {"points_3d": []}


def test_reject_wrong_dimension():
    ok, parsed, reason = parse_points_3d_output('{"points_3d":[{"point":[1,2]}]}')
    assert not ok
    assert "wrong_dimension" in reason
    assert parsed == {"points_3d": []}


def test_reject_nan():
    ok, parsed, reason = parse_points_3d_output('{"points_3d":[{"point":[1, NaN, 3]}]}')
    assert not ok
    assert reason in {"invalid_json", "point_entry_0_nonfinite_or_not_number"}
    assert parsed == {"points_3d": []}
