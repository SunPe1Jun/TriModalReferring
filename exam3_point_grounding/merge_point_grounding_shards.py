#!/usr/bin/env python3
"""Merge Experiment 3 raw JSON shard outputs into one prediction CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import normalize_text, read_csv_rows, write_csv, write_json  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Merge point-grounding raw JSON shards.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", default="exam3_point_grounding/outputs_full_v9_20260709/manifest.csv")
    parser.add_argument("--gt_manifest", default="exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv")
    parser.add_argument("--raw_dirs", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--report_json", required=True)
    parser.add_argument("--include_missing", action="store_true")
    return parser.parse_args()


def key(scene: Any, row_index: Any) -> Tuple[str, str]:
    return normalize_text(scene), normalize_text(row_index)


def raw_filename(scene: str, row_index: str) -> str:
    return f"{scene}_row_{row_index}.json"


def find_raw(scene: str, row_index: str, raw_dirs: Sequence[Path]) -> Path | None:
    name = raw_filename(scene, row_index)
    for raw_dir in raw_dirs:
        path = raw_dir / name
        if path.exists():
            return path
    return None


def output_row(manifest_row: Mapping[str, str], raw_path: Path | None, payload: Mapping[str, Any] | None) -> Dict[str, Any]:
    scene = normalize_text(manifest_row.get("scene"))
    row_index = normalize_text(manifest_row.get("row_index"))
    if payload is None:
        return {
            "scene": scene,
            "row_index": row_index,
            "event_id": normalize_text(manifest_row.get("event_id")),
            "instruction": normalize_text(manifest_row.get("instruction")),
            "raw_json_path": "",
            "model_raw_output": "",
            "parsed_json": json.dumps({"points_3d": []}, ensure_ascii=False),
            "parse_ok": "False",
            "invalid_reason": "missing_prediction",
            "pred_point_count": 0,
            "pred_points_json": json.dumps([], ensure_ascii=False),
            "error_message": "",
        }
    parsed = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else {"points_3d": []}
    points = parsed.get("points_3d") if isinstance(parsed, Mapping) else []
    if not isinstance(points, list):
        points = []
    return {
        "scene": scene,
        "row_index": row_index,
        "event_id": normalize_text(payload.get("event_id") or manifest_row.get("event_id")),
        "instruction": normalize_text(manifest_row.get("instruction")),
        "raw_json_path": str(raw_path) if raw_path is not None else "",
        "model_raw_output": normalize_text(payload.get("model_raw_output")),
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
        "parse_ok": str(bool(payload.get("parse_ok"))),
        "invalid_reason": normalize_text(payload.get("invalid_reason")),
        "pred_point_count": len(points),
        "pred_points_json": json.dumps(points, ensure_ascii=False),
        "error_message": normalize_text(payload.get("error_message")),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    raw_dirs = [Path(item).resolve() for item in args.raw_dirs]
    manifest_rows = read_csv_rows((repo_root / args.manifest).resolve())
    manifest_by_key = {key(row.get("scene"), row.get("row_index")): row for row in manifest_rows}
    gt_rows = read_csv_rows((repo_root / args.gt_manifest).resolve())
    out_rows = []
    missing = []
    duplicate_sources = 0
    for gt_row in gt_rows:
        scene, row_index = key(gt_row.get("scene"), gt_row.get("row_index"))
        manifest_row = manifest_by_key.get((scene, row_index), gt_row)
        matching = [raw_dir / raw_filename(scene, row_index) for raw_dir in raw_dirs if (raw_dir / raw_filename(scene, row_index)).exists()]
        if len(matching) > 1:
            duplicate_sources += 1
        raw_path = matching[0] if matching else None
        if raw_path is None:
            missing.append(f"{scene}:{row_index}")
            if args.include_missing:
                out_rows.append(output_row(manifest_row, None, None))
            continue
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        out_rows.append(output_row(manifest_row, raw_path, payload))
    output_csv = Path(args.output_csv).resolve()
    write_csv(output_csv, OUTPUT_COLUMNS, out_rows)
    report = {
        "gt_total": len(gt_rows),
        "prediction_rows": len(out_rows),
        "raw_dirs": [str(path) for path in raw_dirs],
        "missing_count": len(missing),
        "missing_keys_preview": missing[:50],
        "duplicate_source_key_count": duplicate_sources,
        "output_csv": str(output_csv),
    }
    write_json(Path(args.report_json).resolve(), report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
