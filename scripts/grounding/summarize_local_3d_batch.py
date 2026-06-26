#!/usr/bin/env python3
"""Summarize batch local 3D grounding JSON outputs into CSV and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


class SummaryError(Exception):
    """Raised when batch summary generation fails."""


CSV_FIELDS: Sequence[str] = (
    "row_index",
    "event_id",
    "response_status",
    "referent_type",
    "primary_source",
    "selected_object_name",
    "selected_object_rank",
    "best_timestamp_seconds",
    "x_world",
    "y_world",
    "z_world",
    "projected_u_norm",
    "projected_v_norm",
    "projection_valid",
    "confidence",
    "validation_warning_count",
    "validation_warnings",
    "referent_text",
    "validation_note",
    "reasoning_summary",
    "video_path",
    "input_csv",
    "scene_anchor_csv",
    "prompt_style",
    "model_name",
    "output_json_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local 3D batch JSON outputs.")
    parser.add_argument("--input_dir", required=True, help="Directory containing per-row JSON outputs.")
    parser.add_argument("--output_csv", required=True, help="Path to save the merged CSV summary.")
    parser.add_argument("--output_md", required=True, help="Path to save the Markdown report.")
    parser.add_argument(
        "--glob",
        default="row_*.json",
        help="Glob pattern used to collect JSON files inside input_dir. Default: row_*.json",
    )
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"Failed to parse JSON: {path} ({exc})") from exc


def collect_files(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SummaryError(f"Input directory does not exist: {input_dir}")
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise SummaryError(f"No JSON files matched {pattern!r} in {input_dir}")
    return files


def flatten_record(payload: Dict[str, Any], source_path: Path) -> Dict[str, Any]:
    adjusted = payload.get("adjusted_response") or {}
    parsed = payload.get("parsed_response") or {}
    chosen = adjusted if adjusted else parsed
    resolved = payload.get("resolved_object_row") or {}
    warnings = payload.get("validation_warnings") or []

    x_world = chosen.get("x_world")
    y_world = chosen.get("y_world")
    z_world = chosen.get("z_world")
    if x_world in (None, "") and resolved:
        x_world = resolved.get("x_world")
        y_world = resolved.get("y_world")
        z_world = resolved.get("z_world")

    return {
        "row_index": payload.get("row_index"),
        "event_id": normalize_text(payload.get("event_id")),
        "response_status": normalize_text(payload.get("response_status")),
        "referent_type": normalize_text(chosen.get("referent_type")),
        "primary_source": normalize_text(chosen.get("primary_source")),
        "selected_object_name": normalize_text(chosen.get("selected_object_name")),
        "selected_object_rank": chosen.get("selected_object_rank"),
        "best_timestamp_seconds": chosen.get("best_timestamp_seconds"),
        "x_world": x_world,
        "y_world": y_world,
        "z_world": z_world,
        "projected_u_norm": payload.get("projected_u_norm"),
        "projected_v_norm": payload.get("projected_v_norm"),
        "projection_valid": payload.get("projection_valid"),
        "confidence": chosen.get("confidence"),
        "validation_warning_count": len(warnings),
        "validation_warnings": " | ".join(str(item) for item in warnings),
        "referent_text": normalize_text(chosen.get("referent_text")),
        "validation_note": normalize_text(chosen.get("validation_note")),
        "reasoning_summary": normalize_text(chosen.get("reasoning_summary")),
        "video_path": normalize_text(payload.get("video_path")),
        "input_csv": normalize_text(payload.get("input_csv")),
        "scene_anchor_csv": normalize_text(payload.get("scene_anchor_csv")),
        "prompt_style": normalize_text(payload.get("prompt_style")),
        "model_name": normalize_text(payload.get("model_name") or payload.get("model")),
        "output_json_path": str(source_path.resolve()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def build_markdown(rows: Sequence[Dict[str, Any]], input_dir: Path, pattern: str) -> str:
    total = len(rows)
    ok_rows = [row for row in rows if normalize_text(row.get("response_status")) == "ok"]
    none_rows = [row for row in rows if normalize_text(row.get("referent_type")) == "none"]
    warning_rows = [row for row in rows if int(row.get("validation_warning_count") or 0) > 0]
    projection_ok_rows = [row for row in rows if parse_bool(row.get("projection_valid"))]

    confidences = [parse_float(row.get("confidence")) for row in rows]
    confidences = [value for value in confidences if value is not None]

    referent_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    object_counts: Dict[str, int] = {}
    for row in rows:
        referent = normalize_text(row.get("referent_type"), "(empty)")
        referent_counts[referent] = referent_counts.get(referent, 0) + 1

        source = normalize_text(row.get("primary_source"), "(empty)")
        source_counts[source] = source_counts.get(source, 0) + 1

        object_name = normalize_text(row.get("selected_object_name"))
        if object_name:
            object_counts[object_name] = object_counts.get(object_name, 0) + 1

    top_objects = sorted(object_counts.items(), key=lambda item: (-item[1], item[0]))[:15]

    lines: List[str] = [
        "# Local 3D Batch Summary",
        "",
        f"- Input directory: `{input_dir}`",
        f"- File pattern: `{pattern}`",
        f"- Total JSON files: {total}",
        f"- response_status = ok: {len(ok_rows)}",
        f"- referent_type = none: {len(none_rows)}",
        f"- Rows with validation warnings: {len(warning_rows)}",
        f"- Rows with valid projection: {len(projection_ok_rows)}",
        f"- Average confidence: {statistics.mean(confidences):.4f}" if confidences else "- Average confidence: n/a",
        "",
        "## Referent Type Counts",
        "",
        "| referent_type | count |",
        "| --- | ---: |",
    ]
    for key, count in sorted(referent_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {count} |")

    lines.extend([
        "",
        "## Primary Source Counts",
        "",
        "| primary_source | count |",
        "| --- | ---: |",
    ])
    for key, count in sorted(source_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {count} |")

    lines.extend([
        "",
        "## Top Selected Objects",
        "",
        "| selected_object_name | count |",
        "| --- | ---: |",
    ])
    if top_objects:
        for object_name, count in top_objects:
            lines.append(f"| {object_name} | {count} |")
    else:
        lines.append("| (none) | 0 |")

    lines.extend([
        "",
        "## Rows With Warnings",
        "",
        "| row_index | event_id | response_status | selected_object_name | warnings |",
        "| ---: | --- | --- | --- | --- |",
    ])
    if warning_rows:
        for row in warning_rows[:50]:
            lines.append(
                f"| {row.get('row_index', '')} | {row.get('event_id', '')} | {row.get('response_status', '')} | "
                f"{row.get('selected_object_name', '')} | {row.get('validation_warnings', '')} |"
            )
    else:
        lines.append("|  |  |  |  | none |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        input_dir = Path(args.input_dir).expanduser().resolve()
        files = collect_files(input_dir, args.glob)
        rows = [flatten_record(load_json(path), path) for path in files]

        output_csv = Path(args.output_csv).expanduser().resolve()
        output_md = Path(args.output_md).expanduser().resolve()
        write_csv(output_csv, rows)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(rows, input_dir, args.glob), encoding="utf-8")

        print(f"Saved merged CSV to: {output_csv}")
        print(f"Saved Markdown report to: {output_md}")
        return 0
    except SummaryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
