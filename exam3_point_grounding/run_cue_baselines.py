#!/usr/bin/env python3
"""Cue-copy baselines for candidate-free point-supervised 3D grounding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import normalize_text, read_csv_rows, vector_distance, write_csv, write_json  # noqa: E402


OUTPUT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "raw_json_path",
    "model_raw_output",
    "parsed_json",
    "parse_ok",
    "invalid_reason",
    "pred_point_count",
    "pred_points_json",
    "error_message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gaze/hand/fusion cue baselines.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", default="exam3_point_grounding/outputs/manifest.csv")
    parser.add_argument("--output_dir", default="exam3_point_grounding/outputs/cue_baselines")
    parser.add_argument("--dedup_distance", type=float, default=0.5)
    parser.add_argument("--max_points", type=int, default=3)
    return parser.parse_args()


def parse_evidence(row: Mapping[str, str]) -> List[Mapping[str, Any]]:
    try:
        payload = json.loads(normalize_text(row.get("evidence_json")) or "[]")
    except json.JSONDecodeError:
        return []
    return payload if isinstance(payload, list) else []


def telemetry_points(row: Mapping[str, str], key: str) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    for item in parse_evidence(row):
        telemetry = item.get("telemetry") if isinstance(item, Mapping) else None
        if not isinstance(telemetry, Mapping):
            continue
        valid_key = "gaze_valid" if key == "gaze_hit" else "hand_valid"
        if telemetry.get(valid_key) is False:
            continue
        raw_point = telemetry.get(key)
        if not isinstance(raw_point, list) or len(raw_point) != 3:
            continue
        try:
            point = (float(raw_point[0]), float(raw_point[1]), float(raw_point[2]))
        except (TypeError, ValueError):
            continue
        points.append(point)
    return points


def dedup(points: Sequence[Tuple[float, float, float]], threshold: float, max_points: int) -> List[Tuple[float, float, float]]:
    result: List[Tuple[float, float, float]] = []
    for point in points:
        if any(vector_distance(point, used) < threshold for used in result):
            continue
        result.append(point)
        if len(result) >= max_points:
            break
    return result


def fuse_points(gaze_points: Sequence[Tuple[float, float, float]], hand_points: Sequence[Tuple[float, float, float]], threshold: float, max_points: int) -> List[Tuple[float, float, float]]:
    fused: List[Tuple[float, float, float]] = []
    used_hand = set()
    for gaze in gaze_points:
        best_idx = None
        best_dist = None
        for idx, hand in enumerate(hand_points):
            if idx in used_hand:
                continue
            dist = vector_distance(gaze, hand)
            if best_dist is None or dist < best_dist:
                best_idx = idx
                best_dist = dist
        if best_idx is not None and best_dist is not None and best_dist <= max(1.0, threshold * 2.0):
            hand = hand_points[best_idx]
            fused.append(((gaze[0] + hand[0]) / 2.0, (gaze[1] + hand[1]) / 2.0, (gaze[2] + hand[2]) / 2.0))
            used_hand.add(best_idx)
        else:
            fused.append(gaze)
    for idx, hand in enumerate(hand_points):
        if idx not in used_hand:
            fused.append(hand)
    return dedup(fused, threshold, max_points)


def point_entries(points: Sequence[Tuple[float, float, float]], source: str) -> List[Dict[str, Any]]:
    return [
        {
            "referent": source,
            "point": [float(point[0]), float(point[1]), float(point[2])],
            "confidence": 0.5,
        }
        for point in points
    ]


def make_row(row: Mapping[str, str], raw_path: Path, points: Sequence[Tuple[float, float, float]], method: str) -> Dict[str, Any]:
    entries = point_entries(points, method)
    parsed = {"points_3d": entries}
    return {
        "scene": normalize_text(row.get("scene")),
        "row_index": normalize_text(row.get("row_index")),
        "event_id": normalize_text(row.get("event_id")),
        "instruction": normalize_text(row.get("instruction")),
        "raw_json_path": str(raw_path),
        "model_raw_output": json.dumps(parsed, ensure_ascii=False),
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
        "parse_ok": "True",
        "invalid_reason": "",
        "pred_point_count": len(entries),
        "pred_points_json": json.dumps(entries, ensure_ascii=False),
        "error_message": "",
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    rows = read_csv_rows((repo_root / args.manifest).resolve())
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, List[Dict[str, Any]]] = {"gaze_copy": [], "hand_copy": [], "gaze_hand_fusion": []}
    for row in rows:
        gaze_points = dedup(telemetry_points(row, "gaze_hit"), args.dedup_distance, args.max_points)
        hand_points = dedup(telemetry_points(row, "hand_hit"), args.dedup_distance, args.max_points)
        fusion_points = fuse_points(gaze_points, hand_points, args.dedup_distance, args.max_points)
        for method, points in (("gaze_copy", gaze_points), ("hand_copy", hand_points), ("gaze_hand_fusion", fusion_points)):
            raw_dir = output_dir / method / "raw"
            raw_path = raw_dir / f"{normalize_text(row.get('scene'))}_row_{normalize_text(row.get('row_index'))}.json"
            payload = {
                "scene": normalize_text(row.get("scene")),
                "row_index": normalize_text(row.get("row_index")),
                "event_id": normalize_text(row.get("event_id")),
                "method": method,
                "parsed_json": {"points_3d": point_entries(points, method)},
                "parse_ok": True,
            }
            write_json(raw_path, payload)
            outputs[method].append(make_row(row, raw_path, points, method))

    for method, out_rows in outputs.items():
        write_csv(output_dir / method / "predictions.csv", OUTPUT_COLUMNS, out_rows)
    print(json.dumps({method: len(rows) for method, rows in outputs.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
