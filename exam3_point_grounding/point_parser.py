#!/usr/bin/env python3
"""Strict output parser for point-supervised 3D grounding."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple


def finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def extract_first_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == "\"":
                in_string = False
            continue
        if char == "\"":
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_points_3d_output(raw_output: str) -> Tuple[bool, Dict[str, Any], str]:
    """Parse the required JSON object.

    Returns (parse_ok, parsed_payload, invalid_reason). Invalid outputs are
    represented as {"points_3d": []} for end-to-end scoring.
    """
    text = "" if raw_output is None else str(raw_output).strip()
    if not text:
        return False, {"points_3d": []}, "empty_output"
    json_text = extract_first_json_object(text)
    if json_text is None:
        return False, {"points_3d": []}, "invalid_json"
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError:
        return False, {"points_3d": []}, "invalid_json"
    if not isinstance(payload, dict):
        return False, {"points_3d": []}, "json_not_object"
    raw_points = payload.get("points_3d")
    if not isinstance(raw_points, list):
        return False, {"points_3d": []}, "missing_or_nonarray_points_3d"

    parsed_points: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_points):
        if not isinstance(item, Mapping):
            return False, {"points_3d": []}, f"point_entry_{idx}_not_object"
        raw_point = item.get("point")
        if not isinstance(raw_point, list):
            return False, {"points_3d": []}, f"point_entry_{idx}_missing_point"
        if len(raw_point) != 3:
            return False, {"points_3d": []}, f"point_entry_{idx}_wrong_dimension"
        point: List[float] = []
        for value in raw_point:
            parsed_value = finite_float(value)
            if parsed_value is None:
                return False, {"points_3d": []}, f"point_entry_{idx}_nonfinite_or_not_number"
            point.append(parsed_value)
        confidence = item.get("confidence")
        parsed_confidence = None
        if confidence is not None:
            parsed_confidence = finite_float(confidence)
            if parsed_confidence is None:
                return False, {"points_3d": []}, f"point_entry_{idx}_invalid_confidence"
            parsed_confidence = max(0.0, min(1.0, parsed_confidence))
        parsed_entry: Dict[str, Any] = {
            "referent": str(item.get("referent", ""))[:120],
            "point": point,
        }
        if parsed_confidence is not None:
            parsed_entry["confidence"] = parsed_confidence
        parsed_points.append(parsed_entry)
    return True, {"points_3d": parsed_points}, ""
