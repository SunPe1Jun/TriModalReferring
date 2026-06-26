#!/usr/bin/env python3
"""Evaluate camera-centered 3D directional point predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DETAIL_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "evidence_panel_id",
    "parse_ok",
    "eval_valid",
    "invalid_reason",
    "point_x",
    "point_y",
    "point_z",
    "camera_x",
    "camera_y",
    "camera_z",
    "matched_gt_anchor_id",
    "matched_gt_anchor_x",
    "matched_gt_anchor_y",
    "matched_gt_anchor_z",
    "angular_error_deg",
    "gt_anchor_ids",
    "raw_json_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate camera-centered 3D directional predictions.")
    parser.add_argument("--pred_csv", required=True, help="Prediction CSV from run_qwen3vl_3d_directional.py.")
    parser.add_argument("--output_dir", required=True, help="Evaluation output directory.")
    parser.add_argument("--report_path", help="Optional markdown report path.")
    parser.add_argument("--thresholds", default="5,10,15,30", help="Angular thresholds in degrees.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_float(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def parse_point(row: Mapping[str, str], prefix: str) -> Optional[Tuple[float, float, float]]:
    values = [parse_float(row.get(f"{prefix}_{axis}")) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    return float(values[0]), float(values[1]), float(values[2])  # type: ignore[arg-type]


def vector_norm(vector: Tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])


def angular_error_deg(camera: Tuple[float, float, float], point: Tuple[float, float, float], gt: Tuple[float, float, float]) -> Optional[float]:
    pred_vec = (point[0] - camera[0], point[1] - camera[1], point[2] - camera[2])
    gt_vec = (gt[0] - camera[0], gt[1] - camera[1], gt[2] - camera[2])
    pred_norm = vector_norm(pred_vec)
    gt_norm = vector_norm(gt_vec)
    if pred_norm <= 1e-9 or gt_norm <= 1e-9:
        return None
    dot = (
        pred_vec[0] * gt_vec[0]
        + pred_vec[1] * gt_vec[1]
        + pred_vec[2] * gt_vec[2]
    ) / (pred_norm * gt_norm)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def parse_gt_anchors(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(normalize_text(row.get("gt_anchor_points_json")) or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    anchors: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        x = parse_float(item.get("x"))
        y = parse_float(item.get("y"))
        z = parse_float(item.get("z"))
        name = normalize_text(item.get("id"))
        if name and x is not None and y is not None and z is not None:
            anchors.append({"id": name, "x": x, "y": y, "z": z})
    return anchors


def best_gt_match(
    camera: Tuple[float, float, float],
    point: Tuple[float, float, float],
    anchors: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[Mapping[str, Any]], Optional[float], str]:
    scored: List[Tuple[float, Mapping[str, Any]]] = []
    for anchor in anchors:
        gt = (float(anchor["x"]), float(anchor["y"]), float(anchor["z"]))
        angle = angular_error_deg(camera, point, gt)
        if angle is not None:
            scored.append((angle, anchor))
    if not scored:
        return None, None, "degenerate_camera_or_gt_direction"
    angle, anchor = min(scored, key=lambda item: item[0])
    return anchor, angle, ""


def accuracy(errors: Sequence[float], threshold: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return sum(1 for error in errors if error <= threshold) / denominator


def summarize_group(rows: Sequence[Mapping[str, Any]], thresholds: Sequence[float]) -> Dict[str, Any]:
    total = len(rows)
    valid_rows = [row for row in rows if row.get("eval_valid") == "True"]
    errors = [float(row["angular_error_deg"]) for row in valid_rows if row.get("angular_error_deg") != ""]
    summary: Dict[str, Any] = {
        "total_samples": total,
        "valid_prediction_count": len(errors),
        "invalid_count": total - len(errors),
        "valid_rate": len(errors) / total if total else 0.0,
        "mean_angular_error_deg_valid_only": mean(errors) if errors else None,
        "median_angular_error_deg_valid_only": median(errors) if errors else None,
    }
    for threshold in thresholds:
        key = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        hits = sum(1 for error in errors if error <= threshold)
        summary[f"angular_accuracy_at_{key}_deg_all_samples"] = hits / total if total else 0.0
        summary[f"angular_accuracy_at_{key}_deg_valid_only"] = hits / len(errors) if errors else 0.0
    invalid_reasons = Counter(normalize_text(row.get("invalid_reason")) or "unknown" for row in rows if row.get("eval_valid") != "True")
    summary["invalid_reason_counts"] = dict(invalid_reasons)
    return summary


def summary_row(name: str, summary: Mapping[str, Any], thresholds: Sequence[float]) -> Dict[str, Any]:
    row = {
        "partition": name,
        "total_samples": summary.get("total_samples", 0),
        "valid_prediction_count": summary.get("valid_prediction_count", 0),
        "invalid_count": summary.get("invalid_count", 0),
        "valid_rate": summary.get("valid_rate", 0.0),
        "mean_angular_error_deg_valid_only": summary.get("mean_angular_error_deg_valid_only"),
        "median_angular_error_deg_valid_only": summary.get("median_angular_error_deg_valid_only"),
    }
    for threshold in thresholds:
        key = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        row[f"acc_at_{key}_deg_all"] = summary.get(f"angular_accuracy_at_{key}_deg_all_samples", 0.0)
        row[f"acc_at_{key}_deg_valid"] = summary.get(f"angular_accuracy_at_{key}_deg_valid_only", 0.0)
    return row


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}"


def write_report(path: Path, pred_csv: Path, summary: Mapping[str, Any], per_scene: Mapping[str, Any], thresholds: Sequence[float]) -> None:
    overall = summary["overall"]
    threshold_lines = []
    for threshold in thresholds:
        key = str(int(threshold)) if float(threshold).is_integer() else str(threshold)
        threshold_lines.append(
            f"- @{key} deg: all={format_percent(overall.get(f'angular_accuracy_at_{key}_deg_all_samples'))}, "
            f"valid-only={format_percent(overall.get(f'angular_accuracy_at_{key}_deg_valid_only'))}"
        )
    scene_lines = []
    for scene, scene_summary in sorted(per_scene.items()):
        scene_lines.append(
            f"| {scene} | {scene_summary.get('total_samples', 0)} | {scene_summary.get('valid_prediction_count', 0)} | "
            f"{format_number(scene_summary.get('mean_angular_error_deg_valid_only'))} | "
            f"{format_number(scene_summary.get('median_angular_error_deg_valid_only'))} |"
        )
    invalid_counts = overall.get("invalid_reason_counts", {})
    invalid_text = ", ".join(f"{key}: {value}" for key, value in invalid_counts.items()) if invalid_counts else "none"
    content = f"""# Camera-Centered 3D Directional Point Diagnostic Report

## Data Input

- Prediction CSV: `{pred_csv}`
- Evidence frames: one frame per event from the exam2 v10 manifest.
- Candidate anchors: scene-level anchor tables in `data/*_anchor_table.tsv`.
- GT anchors: valid mapped anchor coordinates serialized from the exam2 v10 manifest.

## Evidence Panel Rule

The default rule is `highest_score`: choose the event panel with the largest exam2 `panel_selection_score`, breaking ties by panel index and time. This reuses the projected-2D diagnostic evidence selector and does not use GT anchor identity or model output.

## Evaluation Formula

For camera position `c`, GT anchor `g`, and model point `p_hat`, compute `u = normalize(p_hat - c)` and `v = normalize(g - c)`. The angular error is `acos(clip(u dot v, -1, 1))` in degrees. For multi-anchor GT, the minimum angle over valid GT anchors is used and the matched anchor id is retained.

## Invalid Outputs

Invalid model outputs are kept in the all-sample denominator. Mean and median angular error are valid-only because invalid outputs have no defined angle. Invalid reason counts: {invalid_text}.

## Overall Results

- total samples: {overall.get('total_samples', 0)}
- valid predictions: {overall.get('valid_prediction_count', 0)}
- invalid predictions: {overall.get('invalid_count', 0)}
- mean angular error (valid-only): {format_number(overall.get('mean_angular_error_deg_valid_only'))} deg
- median angular error (valid-only): {format_number(overall.get('median_angular_error_deg_valid_only'))} deg

Angular accuracy:
{chr(10).join(threshold_lines)}

## Per-Scene Results

| partition | samples | valid | mean deg | median deg |
|---|---:|---:|---:|---:|
{chr(10).join(scene_lines)}

## Main Failure Modes

- Invalid or non-JSON outputs reduce all-sample accuracy directly.
- Large angular errors indicate wrong referent direction even when the predicted depth may look plausible.
- Multi-referent instructions are evaluated as a single directional point matched to the nearest valid GT direction, so they do not measure full set recovery.

## Current Limitations

- The evidence frame is a selected diagnostic panel rather than a continuous video input.
- GT is still anchor-based and may be coarse for parts, surfaces, grouped objects, and large objects.
- The metric evaluates direction from the selected camera frame; it does not verify precise depth.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    pred_csv = Path(args.pred_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]

    detail_rows: List[Dict[str, Any]] = []
    for row in read_csv_rows(pred_csv):
        scene = normalize_text(row.get("scene"))
        parse_ok = normalize_text(row.get("parse_ok")) == "True"
        camera = parse_point(row, "camera")
        point = parse_point(row, "point")
        anchors = parse_gt_anchors(row)
        invalid_reason = normalize_text(row.get("invalid_reason"))
        eval_valid = False
        matched: Optional[Mapping[str, Any]] = None
        angle: Optional[float] = None
        if not parse_ok:
            invalid_reason = invalid_reason or "parse_invalid"
        elif point is None:
            invalid_reason = invalid_reason or "missing_point"
        elif camera is None:
            invalid_reason = "missing_camera"
        elif not anchors:
            invalid_reason = "missing_gt_anchors"
        else:
            matched, angle, match_error = best_gt_match(camera, point, anchors)
            if angle is None or matched is None:
                invalid_reason = match_error
            else:
                eval_valid = True
                invalid_reason = ""
        detail_rows.append(
            {
                "scene": scene,
                "row_index": normalize_text(row.get("row_index")),
                "event_id": normalize_text(row.get("event_id")),
                "evidence_panel_id": normalize_text(row.get("evidence_panel_id")),
                "parse_ok": str(parse_ok),
                "eval_valid": str(eval_valid),
                "invalid_reason": invalid_reason,
                "point_x": normalize_text(row.get("point_x")),
                "point_y": normalize_text(row.get("point_y")),
                "point_z": normalize_text(row.get("point_z")),
                "camera_x": normalize_text(row.get("camera_x")),
                "camera_y": normalize_text(row.get("camera_y")),
                "camera_z": normalize_text(row.get("camera_z")),
                "matched_gt_anchor_id": normalize_text(matched.get("id")) if matched else "",
                "matched_gt_anchor_x": f"{float(matched['x']):.9f}" if matched else "",
                "matched_gt_anchor_y": f"{float(matched['y']):.9f}" if matched else "",
                "matched_gt_anchor_z": f"{float(matched['z']):.9f}" if matched else "",
                "angular_error_deg": f"{angle:.9f}" if angle is not None else "",
                "gt_anchor_ids": normalize_text(row.get("gt_anchor_ids")),
                "raw_json_path": normalize_text(row.get("raw_json_path")),
            }
        )

    by_scene_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        by_scene_rows[normalize_text(row.get("scene"))].append(row)
    per_scene = {scene: summarize_group(rows, thresholds) for scene, rows in by_scene_rows.items()}
    overall = summarize_group(detail_rows, thresholds)
    summary = {
        "pred_csv": str(pred_csv),
        "thresholds_deg": thresholds,
        "overall": overall,
        "per_scene": per_scene,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "3d_directional_eval_detail.csv"
    summary_json_path = output_dir / "3d_directional_eval_summary.json"
    per_scene_csv_path = output_dir / "3d_directional_eval_by_scene.csv"
    write_csv(detail_path, DETAIL_COLUMNS, detail_rows)
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_columns = list(summary_row("overall", overall, thresholds).keys())
    summary_rows = [summary_row("overall", overall, thresholds)]
    summary_rows.extend(summary_row(scene, per_scene[scene], thresholds) for scene in sorted(per_scene))
    write_csv(per_scene_csv_path, summary_columns, summary_rows)
    if args.report_path:
        write_report(Path(args.report_path).resolve(), pred_csv, summary, per_scene, thresholds)
    print(f"Wrote detail: {detail_path}")
    print(f"Wrote summary: {summary_json_path}")
    print(f"Wrote per-scene CSV: {per_scene_csv_path}")
    if args.report_path:
        print(f"Wrote report: {Path(args.report_path).resolve()}")


if __name__ == "__main__":
    main()
