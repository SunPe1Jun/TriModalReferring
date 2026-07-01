#!/usr/bin/env python3
"""Aggregate multiple match-eval summary JSON files into one CSV/Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


class SummaryAggregationError(Exception):
    """Raised when summary aggregation cannot proceed."""


DEFAULT_FIELDS = (
    "scene_key",
    "source_json",
    "total_gt_rows",
    "total_prediction_files",
    "response_status_ok_count",
    "storyboard_fallback_count",
    "match_count_overall",
    "overall_accuracy",
    "evaluable_row_count",
    "evaluable_coverage",
    "unevaluable_row_count",
    "match_count_evaluable_only",
    "mapped_only_accuracy",
    "unsupported_gt_row_count",
    "unsupported_gt_referent_mention_count",
    "unsupported_gt_distinct_label_count",
    "predicted_label_count",
    "matched_label_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate scene/room match-eval summary JSON files into a single CSV and optional Markdown table."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *_match_eval_summary*.json files.",
    )
    parser.add_argument(
        "--output_csv",
        required=True,
        help="Path to save the aggregated CSV table.",
    )
    parser.add_argument(
        "--output_md",
        help="Optional path to save a Markdown report.",
    )
    parser.add_argument(
        "--glob",
        default="*match_eval*summary*.json",
        help="Glob pattern for summary JSON files. Default: *match_eval*summary*.json",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def safe_int(value: Any) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def infer_scene_key(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"_match_eval.*$", "", stem)
    stem = re.sub(r"_summary.*$", "", stem)
    return stem


def collect_summary_files(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise SummaryAggregationError(f"Input directory does not exist: {input_dir}")
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not files:
        raise SummaryAggregationError(f"No summary JSON files matched {pattern!r} in {input_dir}")
    return files


def load_summary_row(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SummaryAggregationError(f"Failed to parse summary JSON: {path} ({exc})") from exc

    unsupported = payload.get("unsupported_gt_referent_counts") or {}
    predicted = payload.get("predicted_referent_counts") or {}
    matched = payload.get("matched_referent_counts") or {}

    return {
        "scene_key": infer_scene_key(path),
        "source_json": str(path.resolve()),
        "total_gt_rows": safe_int(payload.get("total_gt_rows")),
        "total_prediction_files": safe_int(payload.get("total_prediction_files")),
        "response_status_ok_count": safe_int(payload.get("response_status_ok_count")),
        "storyboard_fallback_count": safe_int(payload.get("storyboard_fallback_count")),
        "match_count_overall": safe_int(payload.get("match_count_overall")),
        "overall_accuracy": safe_float(payload.get("overall_accuracy")),
        "evaluable_row_count": safe_int(payload.get("evaluable_row_count")),
        "evaluable_coverage": safe_float(payload.get("evaluable_coverage")),
        "unevaluable_row_count": safe_int(payload.get("unevaluable_row_count")),
        "match_count_evaluable_only": safe_int(payload.get("match_count_evaluable_only")),
        "mapped_only_accuracy": safe_float(payload.get("mapped_only_accuracy")),
        "unsupported_gt_row_count": safe_int(payload.get("unsupported_gt_row_count")),
        "unsupported_gt_referent_mention_count": safe_int(payload.get("unsupported_gt_referent_mention_count")),
        "unsupported_gt_distinct_label_count": safe_int(payload.get("unsupported_gt_distinct_label_count", len(unsupported))),
        "predicted_label_count": len(predicted),
        "matched_label_count": len(matched),
        "_unsupported_gt_referent_counts": unsupported,
        "_predicted_referent_counts": predicted,
        "_matched_referent_counts": matched,
    }


def sort_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda item: (
            item["scene_key"],
        ),
    )


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(DEFAULT_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in DEFAULT_FIELDS})


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def build_totals_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_gt_rows = sum(safe_int(row["total_gt_rows"]) for row in rows)
    total_prediction_files = sum(safe_int(row["total_prediction_files"]) for row in rows)
    response_status_ok_count = sum(safe_int(row["response_status_ok_count"]) for row in rows)
    storyboard_fallback_count = sum(safe_int(row["storyboard_fallback_count"]) for row in rows)
    match_count_overall = sum(safe_int(row["match_count_overall"]) for row in rows)
    evaluable_row_count = sum(safe_int(row["evaluable_row_count"]) for row in rows)
    match_count_evaluable_only = sum(safe_int(row["match_count_evaluable_only"]) for row in rows)

    overall_accuracy = None if total_gt_rows <= 0 else match_count_overall / total_gt_rows
    mapped_only_accuracy = None if evaluable_row_count <= 0 else match_count_evaluable_only / evaluable_row_count
    evaluable_coverage = None if total_gt_rows <= 0 else evaluable_row_count / total_gt_rows

    return {
        "scene_key": "ALL",
        "source_json": "",
        "total_gt_rows": total_gt_rows,
        "total_prediction_files": total_prediction_files,
        "response_status_ok_count": response_status_ok_count,
        "storyboard_fallback_count": storyboard_fallback_count,
        "match_count_overall": match_count_overall,
        "overall_accuracy": overall_accuracy,
        "evaluable_row_count": evaluable_row_count,
        "evaluable_coverage": evaluable_coverage,
        "unevaluable_row_count": total_gt_rows - evaluable_row_count,
        "match_count_evaluable_only": match_count_evaluable_only,
        "mapped_only_accuracy": mapped_only_accuracy,
        "unsupported_gt_row_count": sum(safe_int(row["unsupported_gt_row_count"]) for row in rows),
        "unsupported_gt_referent_mention_count": sum(safe_int(row["unsupported_gt_referent_mention_count"]) for row in rows),
        "unsupported_gt_distinct_label_count": sum(safe_int(row["unsupported_gt_distinct_label_count"]) for row in rows),
        "predicted_label_count": 0,
        "matched_label_count": 0,
    }


def write_markdown(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Match Evaluation Summary",
        "",
        "| Scene | GT Rows | Pred Files | OK | Fallback | Hits | Overall Acc | Evaluable | Evaluable Coverage | Evaluable Hits | Mapped Acc | Unsupported Rows | Unsupported Mentions | Unsupported Distinct Labels |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {scene} | {gt} | {pred} | {ok} | {fallback} | {hits} | {overall} | {evaluable} | {coverage} | {mapped_hits} | {mapped_acc} | {unsupported_rows} | {unsupported_mentions} | {unsupported_distinct} |".format(
                scene=row["scene_key"],
                gt=row["total_gt_rows"],
                pred=row["total_prediction_files"],
                ok=row["response_status_ok_count"],
                fallback=row["storyboard_fallback_count"],
                hits=row["match_count_overall"],
                overall=format_float(row["overall_accuracy"]),
                evaluable=row["evaluable_row_count"],
                coverage=format_float(row["evaluable_coverage"]),
                mapped_hits=row["match_count_evaluable_only"],
                mapped_acc=format_float(row["mapped_only_accuracy"]),
                unsupported_rows=row["unsupported_gt_row_count"],
                unsupported_mentions=row["unsupported_gt_referent_mention_count"],
                unsupported_distinct=row["unsupported_gt_distinct_label_count"],
            )
        )

    totals = build_totals_row(rows)
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Total GT rows: {totals['total_gt_rows']}",
            f"- Total prediction files: {totals['total_prediction_files']}",
            f"- Total overall hits: {totals['match_count_overall']}",
            f"- Aggregate overall accuracy: {format_float(totals['overall_accuracy'])}",
            f"- Aggregate evaluable rows: {totals['evaluable_row_count']}",
            f"- Aggregate evaluable coverage: {format_float(totals['evaluable_coverage'])}",
            f"- Aggregate evaluable hits: {totals['match_count_evaluable_only']}",
            f"- Aggregate mapped-only accuracy: {format_float(totals['mapped_only_accuracy'])}",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_md = Path(args.output_md).resolve() if args.output_md else None

    rows = sort_rows([load_summary_row(path) for path in collect_summary_files(input_dir, args.glob)])
    rows_with_total = rows + [build_totals_row(rows)]

    write_csv(output_csv, rows_with_total)
    if output_md is not None:
        write_markdown(output_md, rows)

    print(f"Saved aggregate CSV to: {output_csv}")
    if output_md is not None:
        print(f"Saved Markdown report to: {output_md}")

    totals = rows_with_total[-1]
    print(
        f"Aggregate overall accuracy: {totals['match_count_overall']}/{totals['total_gt_rows']} = "
        f"{totals['overall_accuracy']}"
    )
    print(
        f"Aggregate mapped-only accuracy: {totals['match_count_evaluable_only']}/{totals['evaluable_row_count']} = "
        f"{totals['mapped_only_accuracy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
