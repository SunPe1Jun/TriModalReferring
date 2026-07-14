#!/usr/bin/env python3
"""Evaluate candidate-free point-supervised 3D referent grounding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import (  # noqa: E402
    Anchor,
    load_anchor_table,
    nearest_anchor,
    nearest_negative_distance,
    normalize_text,
    parse_float,
    read_csv_rows,
    robust_bounds,
    split_names,
    vector_distance,
    write_csv,
    write_json,
)


DETAIL_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "parse_ok",
    "invalid_reason",
    "gt_count",
    "pred_count",
    "nearest_pred_anchor_ids",
    "gt_anchor_ids",
    "set_tp",
    "set_fp",
    "set_fn",
    "anchor_set_precision",
    "anchor_set_recall",
    "anchor_set_f1",
    "anchor_set_exact",
    "cardinality_error",
    "duplicate_nearest_anchor_count",
    "margin_tp_at_0_5",
    "margin_fp_at_0_5",
    "margin_fn_at_0_5",
    "margin_f1_at_0_5",
    "margin_tp_at_1_0",
    "margin_fp_at_1_0",
    "margin_fn_at_1_0",
    "margin_f1_at_1_0",
    "margin_tp_at_2_0",
    "margin_fp_at_2_0",
    "margin_fn_at_2_0",
    "margin_f1_at_2_0",
    "matched_margin_errors_json",
    "matched_euclidean_errors_json",
    "mean_matched_euclidean_error",
    "median_matched_euclidean_error",
    "mean_scene_normalized_error",
    "median_scene_normalized_error",
    "margin_undefined_gt_count",
    "raw_json_path",
)

SUMMARY_COLUMNS = (
    "partition",
    "total_samples",
    "valid_output_count",
    "invalid_output_count",
    "valid_output_rate",
    "anchor_set_precision_micro",
    "anchor_set_recall_micro",
    "anchor_set_f1_micro",
    "anchor_set_exact_rate",
    "mean_cardinality_error",
    "duplicate_nearest_anchor_rate",
    "margin_f1_at_0_5",
    "margin_f1_at_1_0",
    "margin_f1_at_2_0",
    "mean_matched_margin_error",
    "median_matched_margin_error",
    "mean_matched_euclidean_error",
    "median_matched_euclidean_error",
    "mean_scene_normalized_error",
    "median_scene_normalized_error",
    "margin_undefined_gt_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate point-supervised 3D grounding predictions.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--gt_manifest", default="exam3_point_grounding/outputs/gt_manifest_eval.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_path")
    return parser.parse_args()


def parse_points_json(value: Any) -> List[Tuple[float, float, float]]:
    text = normalize_text(value)
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = payload.get("points_3d")
    if not isinstance(payload, list):
        return []
    points: List[Tuple[float, float, float]] = []
    for item in payload:
        raw_point = item.get("point") if isinstance(item, Mapping) else None
        if not isinstance(raw_point, list) or len(raw_point) != 3:
            continue
        values = [parse_float(part) for part in raw_point]
        if any(value is None for value in values):
            continue
        points.append((float(values[0]), float(values[1]), float(values[2])))  # type: ignore[arg-type]
    return points


def parse_gt_points(row: Mapping[str, str]) -> List[Anchor]:
    text = normalize_text(row.get("gt_points_json"))
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    anchors: List[Anchor] = []
    if not isinstance(payload, list):
        return anchors
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        anchor_id = normalize_text(item.get("id"))
        raw_point = item.get("point")
        if not isinstance(raw_point, list) or len(raw_point) != 3:
            continue
        values = [parse_float(part) for part in raw_point]
        if anchor_id and all(value is not None for value in values):
            anchors.append(Anchor(anchor_id, (float(values[0]), float(values[1]), float(values[2]))))  # type: ignore[arg-type]
    return anchors


def f1_from_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def min_cost_matching(costs: Sequence[Sequence[float]]) -> List[Tuple[int, int]]:
    if not costs or not costs[0]:
        return []
    rows = len(costs)
    cols = len(costs[0])
    if rows <= cols and cols <= 14:
        dp: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]] = {(0, 0): (0.0, [])}
        for row_idx in range(rows):
            next_dp: Dict[Tuple[int, int], Tuple[float, List[Tuple[int, int]]]] = {}
            for (_old_row, mask), (cost, pairs) in dp.items():
                for col_idx in range(cols):
                    if mask & (1 << col_idx):
                        continue
                    next_mask = mask | (1 << col_idx)
                    next_cost = cost + costs[row_idx][col_idx]
                    key = (row_idx + 1, next_mask)
                    if key not in next_dp or next_cost < next_dp[key][0]:
                        next_dp[key] = (next_cost, pairs + [(row_idx, col_idx)])
            dp = next_dp
        return min(dp.values(), key=lambda item: item[0])[1]
    if cols < rows and rows <= 14:
        transposed = [[costs[row][col] for row in range(rows)] for col in range(cols)]
        return [(row, col) for col, row in min_cost_matching(transposed)]

    # Greedy fallback for pathological over-prediction; normal outputs are small.
    candidates = sorted((costs[row][col], row, col) for row in range(rows) for col in range(cols))
    used_rows = set()
    used_cols = set()
    pairs = []
    for _cost, row, col in candidates:
        if row in used_rows or col in used_cols:
            continue
        pairs.append((row, col))
        used_rows.add(row)
        used_cols.add(col)
        if len(used_rows) == rows or len(used_cols) == cols:
            break
    return pairs


def nearest_anchor_set_metrics(pred_points: Sequence[Tuple[float, float, float]], gt_anchors: Sequence[Anchor], all_anchors: Sequence[Anchor]) -> Dict[str, Any]:
    nearest_ids: List[str] = []
    for point in pred_points:
        anchor, _distance = nearest_anchor(point, all_anchors)
        if anchor is not None:
            nearest_ids.append(anchor.anchor_id)
    pred_set = set(nearest_ids)
    gt_set = {anchor.anchor_id for anchor in gt_anchors}
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    duplicate_count = len(nearest_ids) - len(pred_set)
    return {
        "nearest_ids": nearest_ids,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": pred_set == gt_set and duplicate_count == 0,
        "cardinality_error": abs(len(nearest_ids) - len(gt_set)),
        "duplicate_count": duplicate_count,
    }


def margin_matching_metrics(
    pred_points: Sequence[Tuple[float, float, float]],
    gt_anchors: Sequence[Anchor],
    all_anchors: Sequence[Anchor],
    thresholds: Sequence[float] = (0.5, 1.0, 2.0),
) -> Dict[str, Any]:
    gt_ids = {anchor.anchor_id for anchor in gt_anchors}
    margins: List[Optional[float]] = []
    for gt in gt_anchors:
        nearest_negative = nearest_negative_distance(gt, all_anchors, gt_ids)
        margins.append(0.5 * nearest_negative if nearest_negative is not None else None)

    defined_gt_indices = [idx for idx, margin in enumerate(margins) if margin is not None and margin > 1e-12]
    undefined_count = len(gt_anchors) - len(defined_gt_indices)
    if not pred_points or not defined_gt_indices:
        result = {
            "matched_margin_errors": [],
            "matched_euclidean_errors": [],
            "undefined_gt_count": undefined_count,
        }
        for threshold in thresholds:
            result[f"tp@{threshold}"] = 0
            result[f"fp@{threshold}"] = len(pred_points)
            result[f"fn@{threshold}"] = len(defined_gt_indices)
            result[f"f1@{threshold}"] = 0.0
        return result

    gt_subset = [gt_anchors[idx] for idx in defined_gt_indices]
    margin_subset = [margins[idx] for idx in defined_gt_indices]
    cost_matrix: List[List[float]] = []
    for pred in pred_points:
        row = []
        for gt, margin in zip(gt_subset, margin_subset):
            assert margin is not None
            row.append(vector_distance(pred, gt.point) / (margin + 1e-12))
        cost_matrix.append(row)
    pairs = min_cost_matching(cost_matrix)
    matched_errors = [cost_matrix[pred_idx][gt_idx] for pred_idx, gt_idx in pairs]
    matched_euclidean = [vector_distance(pred_points[pred_idx], gt_subset[gt_idx].point) for pred_idx, gt_idx in pairs]
    result = {
        "matched_margin_errors": matched_errors,
        "matched_euclidean_errors": matched_euclidean,
        "undefined_gt_count": undefined_count,
    }
    for threshold in thresholds:
        tp = sum(1 for error in matched_errors if error <= threshold)
        fp = len(pred_points) - tp
        fn = len(defined_gt_indices) - tp
        _precision, _recall, f1 = f1_from_counts(tp, fp, fn)
        result[f"tp@{threshold}"] = tp
        result[f"fp@{threshold}"] = fp
        result[f"fn@{threshold}"] = fn
        result[f"f1@{threshold}"] = f1
    return result


def scene_normalized_errors(
    pred_points: Sequence[Tuple[float, float, float]],
    gt_anchors: Sequence[Anchor],
    robust_diagonal: float,
) -> List[float]:
    if not pred_points or not gt_anchors or robust_diagonal <= 1e-12:
        return []
    cost_matrix = [[vector_distance(pred, gt.point) for gt in gt_anchors] for pred in pred_points]
    pairs = min_cost_matching(cost_matrix)
    return [cost_matrix[pred_idx][gt_idx] / robust_diagonal for pred_idx, gt_idx in pairs]


def summarize(rows: Sequence[Mapping[str, Any]], name: str) -> Dict[str, Any]:
    total = len(rows)
    valid = sum(1 for row in rows if str(row.get("parse_ok")).lower() == "true")
    set_tp = sum(int(row.get("set_tp", 0) or 0) for row in rows)
    set_fp = sum(int(row.get("set_fp", 0) or 0) for row in rows)
    set_fn = sum(int(row.get("set_fn", 0) or 0) for row in rows)
    precision, recall, f1 = f1_from_counts(set_tp, set_fp, set_fn)
    margin_errors: List[float] = []
    euclidean_errors: List[float] = []
    scene_norm_errors: List[float] = []
    for row in rows:
        for key, target in (
            ("matched_margin_errors_json", margin_errors),
            ("matched_euclidean_errors_json", euclidean_errors),
        ):
            try:
                values = json.loads(str(row.get(key, "[]")))
            except json.JSONDecodeError:
                values = []
            if isinstance(values, list):
                target.extend(float(value) for value in values if isinstance(value, (int, float)))
        try:
            scene_values = json.loads(str(row.get("scene_normalized_errors_json", "[]")))
        except json.JSONDecodeError:
            scene_values = []
        if isinstance(scene_values, list):
            scene_norm_errors.extend(float(value) for value in scene_values if isinstance(value, (int, float)))

    summary: Dict[str, Any] = {
        "partition": name,
        "total_samples": total,
        "valid_output_count": valid,
        "invalid_output_count": total - valid,
        "valid_output_rate": valid / total if total else 0.0,
        "anchor_set_precision_micro": precision,
        "anchor_set_recall_micro": recall,
        "anchor_set_f1_micro": f1,
        "anchor_set_exact_rate": sum(1 for row in rows if str(row.get("anchor_set_exact")).lower() == "true") / total if total else 0.0,
        "mean_cardinality_error": mean([float(row.get("cardinality_error", 0) or 0) for row in rows]) if rows else 0.0,
        "duplicate_nearest_anchor_rate": sum(float(row.get("duplicate_nearest_anchor_count", 0) or 0) for row in rows) / max(1, sum(float(row.get("pred_count", 0) or 0) for row in rows)),
        "mean_matched_margin_error": mean(margin_errors) if margin_errors else None,
        "median_matched_margin_error": median(margin_errors) if margin_errors else None,
        "mean_matched_euclidean_error": mean(euclidean_errors) if euclidean_errors else None,
        "median_matched_euclidean_error": median(euclidean_errors) if euclidean_errors else None,
        "mean_scene_normalized_error": mean(scene_norm_errors) if scene_norm_errors else None,
        "median_scene_normalized_error": median(scene_norm_errors) if scene_norm_errors else None,
        "margin_undefined_gt_count": sum(int(row.get("margin_undefined_gt_count", 0) or 0) for row in rows),
    }
    for threshold_key, output_key in (("0_5", "0.5"), ("1_0", "1.0"), ("2_0", "2.0")):
        tp = sum(int(row.get(f"margin_tp_at_{threshold_key}", 0) or 0) for row in rows)
        fp = sum(int(row.get(f"margin_fp_at_{threshold_key}", 0) or 0) for row in rows)
        fn = sum(int(row.get(f"margin_fn_at_{threshold_key}", 0) or 0) for row in rows)
        _p, _r, mf1 = f1_from_counts(tp, fp, fn)
        summary[f"margin_f1_at_{threshold_key}"] = mf1
    return summary


def read_predictions(path: Path) -> Dict[Tuple[str, str], Mapping[str, str]]:
    rows = read_csv_rows(path)
    result = {}
    for row in rows:
        key = (normalize_text(row.get("scene")), normalize_text(row.get("row_index")))
        result[key] = row
    return result


def maybe_float_list(values: Sequence[float]) -> str:
    return json.dumps([float(value) for value in values], ensure_ascii=False)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    pred_rows = read_predictions(Path(args.pred_csv))
    gt_rows = read_csv_rows((repo_root / args.gt_manifest).resolve())
    output_dir = Path(args.output_dir).resolve()
    scene_anchor_cache: Dict[str, List[Anchor]] = {}
    scene_bounds_cache: Dict[str, Dict[str, Any]] = {}

    detail_rows: List[Dict[str, Any]] = []
    for gt_row in gt_rows:
        scene = normalize_text(gt_row.get("scene"))
        row_index = normalize_text(gt_row.get("row_index"))
        pred_row = pred_rows.get((scene, row_index), {})
        gt_anchors = parse_gt_points(gt_row)
        if scene not in scene_anchor_cache:
            scene_anchor_cache[scene] = load_anchor_table(repo_root, scene)
            scene_bounds_cache[scene] = robust_bounds(scene_anchor_cache[scene])
        all_anchors = scene_anchor_cache[scene]
        pred_points = parse_points_json(pred_row.get("parsed_json") or pred_row.get("pred_points_json"))
        parse_ok = normalize_text(pred_row.get("parse_ok")).lower() == "true"
        invalid_reason = normalize_text(pred_row.get("invalid_reason")) or ("missing_prediction" if not pred_row else "")

        set_metrics = nearest_anchor_set_metrics(pred_points, gt_anchors, all_anchors)
        margin_metrics = margin_matching_metrics(pred_points, gt_anchors, all_anchors)
        scene_norms = scene_normalized_errors(pred_points, gt_anchors, float(scene_bounds_cache[scene]["robust_diagonal"]))
        matched_euclidean = margin_metrics["matched_euclidean_errors"]
        margin_errors = margin_metrics["matched_margin_errors"]

        def threshold_value(base: str, threshold: float) -> Any:
            return margin_metrics[f"{base}@{threshold}"]

        row = {
            "scene": scene,
            "row_index": row_index,
            "event_id": normalize_text(gt_row.get("event_id")),
            "parse_ok": str(parse_ok),
            "invalid_reason": invalid_reason,
            "gt_count": len(gt_anchors),
            "pred_count": len(pred_points),
            "nearest_pred_anchor_ids": ",".join(set_metrics["nearest_ids"]),
            "gt_anchor_ids": ",".join(anchor.anchor_id for anchor in gt_anchors),
            "set_tp": set_metrics["tp"],
            "set_fp": set_metrics["fp"],
            "set_fn": set_metrics["fn"],
            "anchor_set_precision": set_metrics["precision"],
            "anchor_set_recall": set_metrics["recall"],
            "anchor_set_f1": set_metrics["f1"],
            "anchor_set_exact": str(set_metrics["exact"]),
            "cardinality_error": set_metrics["cardinality_error"],
            "duplicate_nearest_anchor_count": set_metrics["duplicate_count"],
            "margin_tp_at_0_5": threshold_value("tp", 0.5),
            "margin_fp_at_0_5": threshold_value("fp", 0.5),
            "margin_fn_at_0_5": threshold_value("fn", 0.5),
            "margin_f1_at_0_5": threshold_value("f1", 0.5),
            "margin_tp_at_1_0": threshold_value("tp", 1.0),
            "margin_fp_at_1_0": threshold_value("fp", 1.0),
            "margin_fn_at_1_0": threshold_value("fn", 1.0),
            "margin_f1_at_1_0": threshold_value("f1", 1.0),
            "margin_tp_at_2_0": threshold_value("tp", 2.0),
            "margin_fp_at_2_0": threshold_value("fp", 2.0),
            "margin_fn_at_2_0": threshold_value("fn", 2.0),
            "margin_f1_at_2_0": threshold_value("f1", 2.0),
            "matched_margin_errors_json": maybe_float_list(margin_errors),
            "matched_euclidean_errors_json": maybe_float_list(matched_euclidean),
            "mean_matched_euclidean_error": mean(matched_euclidean) if matched_euclidean else "",
            "median_matched_euclidean_error": median(matched_euclidean) if matched_euclidean else "",
            "mean_scene_normalized_error": mean(scene_norms) if scene_norms else "",
            "median_scene_normalized_error": median(scene_norms) if scene_norms else "",
            "scene_normalized_errors_json": maybe_float_list(scene_norms),
            "margin_undefined_gt_count": margin_metrics["undefined_gt_count"],
            "raw_json_path": normalize_text(pred_row.get("raw_json_path")),
        }
        detail_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_detail.csv", DETAIL_COLUMNS + ("scene_normalized_errors_json",), detail_rows)
    partitions: Dict[str, List[Mapping[str, Any]]] = {"overall": detail_rows, "single_target": [], "multi_target": []}
    by_scene: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        by_scene[str(row["scene"])].append(row)
        gt_count = int(row.get("gt_count", 0) or 0)
        partitions["single_target" if gt_count == 1 else "multi_target"].append(row)
    summaries = {"overall": summarize(detail_rows, "overall")}
    summaries["per_scene"] = {scene: summarize(rows, scene) for scene, rows in sorted(by_scene.items())}
    summaries["single_target"] = summarize(partitions.get("single_target", []), "single_target")
    summaries["multi_target"] = summarize(partitions.get("multi_target", []), "multi_target")
    summaries["invalid_reason_counts"] = dict(Counter(row.get("invalid_reason") or "none" for row in detail_rows if str(row.get("parse_ok")).lower() != "true"))
    write_json(output_dir / "evaluation_summary.json", summaries)

    summary_rows = [summaries["overall"], summaries["single_target"], summaries["multi_target"]] + list(summaries["per_scene"].values())
    write_csv(output_dir / "evaluation_summary.csv", SUMMARY_COLUMNS, summary_rows)

    if args.report_path:
        write_report(Path(args.report_path), summaries, Path(args.pred_csv))
    print(json.dumps({"samples": len(detail_rows), "summary": str(output_dir / "evaluation_summary.json")}, ensure_ascii=False))


def format_float(value: Any) -> str:
    if value is None or value == "":
        return "n/a"
    return f"{float(value):.4f}"


def write_report(path: Path, summaries: Mapping[str, Any], pred_csv: Path) -> None:
    overall = summaries["overall"]
    per_scene = summaries["per_scene"]
    scene_lines = []
    for scene, item in per_scene.items():
        scene_lines.append(
            f"| {scene} | {item['total_samples']} | {format_float(item['anchor_set_f1_micro'])} | "
            f"{format_float(item['margin_f1_at_1_0'])} | {format_float(item['mean_scene_normalized_error'])} |"
        )
    content = f"""# Point-Supervised 3D Referent Grounding Results

Prediction CSV: `{pred_csv}`

## Overall

- samples: {overall['total_samples']}
- valid output rate: {format_float(overall['valid_output_rate'])}
- nearest-anchor set F1: {format_float(overall['anchor_set_f1_micro'])}
- nearest-anchor exact rate: {format_float(overall['anchor_set_exact_rate'])}
- primary Margin-F1@1.0: {format_float(overall['margin_f1_at_1_0'])}
- Margin-F1@0.5: {format_float(overall['margin_f1_at_0_5'])}
- Margin-F1@2.0: {format_float(overall['margin_f1_at_2_0'])}
- mean scene-normalized error: {format_float(overall['mean_scene_normalized_error'])}
- mean Euclidean error in world units: {format_float(overall['mean_matched_euclidean_error'])}
- invalid reason counts: {summaries.get('invalid_reason_counts', {})}

## Per Scene

| scene | samples | anchor-set F1 | Margin-F1@1.0 | mean scene-normalized error |
|---|---:|---:|---:|---:|
{chr(10).join(scene_lines)}

## Notes

Malformed model outputs are treated as empty prediction sets in all end-to-end metrics. Candidate anchors are used only by the evaluator after inference to map predicted points to nearest anchors.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
