#!/usr/bin/env python3
"""Generate an event annotation CSV template from sample directories."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


OUTPUT_COLUMNS = (
    "scene_id",
    "video_id",
    "t_start",
    "t_peak",
    "t_end",
    "gt_anchor_x",
    "gt_anchor_y",
    "gt_anchor_z",
    "gt_type",
)


class AnnotationTemplateError(Exception):
    """Raised when the annotation template cannot be generated."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an event annotation CSV template from sample directories."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory that contains sample subdirectories such as data/1, data/2, ...",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the generated event annotation CSV.",
    )
    parser.add_argument(
        "--sample-pattern",
        default="*",
        help="Glob pattern used to select sample directories under input root.",
    )
    parser.add_argument(
        "--video-pattern",
        default="*gaze_enhanced*.mp4",
        help="Glob pattern used to locate the source video inside each sample directory.",
    )
    parser.add_argument(
        "--metadata-pattern",
        default="metadata_*.json",
        help="Optional glob pattern used to locate the metadata JSON inside each sample directory.",
    )
    parser.add_argument(
        "--default-peak-ratio",
        type=float,
        default=0.5,
        help="If metadata duration exists, t_peak is initialized as duration * default_peak_ratio.",
    )
    parser.add_argument(
        "--default-gt-type",
        default="unknown",
        help="Default gt_type written into the template.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()


def find_single_match(directory: Path, pattern: str, label: str, required: bool) -> Optional[Path]:
    matches = sorted(Path(item).resolve() for item in glob.glob(str(directory / pattern)))
    if not matches:
        if required:
            raise AnnotationTemplateError(f"Missing {label} in sample directory: {directory}")
        return None
    if len(matches) > 1:
        joined = ", ".join(str(match) for match in matches[:5])
        raise AnnotationTemplateError(
            f"Multiple {label} files matched in {directory}: {joined}. Please use a more specific pattern."
        )
    return matches[0]


def iter_sample_dirs(input_root: Path, sample_pattern: str) -> Iterable[Path]:
    matches = sorted(Path(item).resolve() for item in glob.glob(str(input_root / sample_pattern)))
    for match in matches:
        if match.is_dir():
            yield match


def read_duration_from_metadata(metadata_path: Optional[Path]) -> Optional[float]:
    if metadata_path is None:
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise AnnotationTemplateError(f"Failed to parse metadata JSON: {metadata_path}. {exc}") from exc

    duration = payload.get("duration")
    if duration is None:
        return None
    try:
        duration_value = float(duration)
    except (TypeError, ValueError) as exc:
        raise AnnotationTemplateError(
            f"Invalid duration field in metadata JSON: {metadata_path}"
        ) from exc
    if duration_value < 0:
        raise AnnotationTemplateError(f"Negative duration found in metadata JSON: {metadata_path}")
    return duration_value


def format_seconds(value: float) -> str:
    return f"{value:.3f}"


def build_row(
    sample_dir: Path,
    video_path: Path,
    metadata_path: Optional[Path],
    default_peak_ratio: float,
    default_gt_type: str,
) -> Dict[str, str]:
    duration = read_duration_from_metadata(metadata_path)
    if duration is None:
        t_start = 0.0
        t_peak = 0.0
        t_end = 0.0
    else:
        t_start = 0.0
        t_end = duration
        t_peak = duration * default_peak_ratio
        if t_peak < t_start:
            t_peak = t_start
        if t_peak > t_end:
            t_peak = t_end

    return {
        "scene_id": sample_dir.name,
        "video_id": video_path.stem,
        "t_start": format_seconds(t_start),
        "t_peak": format_seconds(t_peak),
        "t_end": format_seconds(t_end),
        "gt_anchor_x": "",
        "gt_anchor_y": "",
        "gt_anchor_z": "",
        "gt_type": default_gt_type,
    }


def write_rows(output_csv: Path, rows: Sequence[Dict[str, str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()

    try:
        input_root = Path(args.input_root).resolve()
        output_csv = Path(args.output_csv).resolve()
        if not input_root.exists() or not input_root.is_dir():
            raise AnnotationTemplateError(
                f"Input root does not exist or is not a directory: {input_root}"
            )
        if output_csv.exists() and not args.overwrite:
            raise AnnotationTemplateError(
                f"Output CSV already exists. Use --overwrite to replace it: {output_csv}"
            )
        if not (0.0 <= args.default_peak_ratio <= 1.0):
            raise AnnotationTemplateError("default-peak-ratio must be in [0, 1].")

        sample_dirs = list(iter_sample_dirs(input_root, args.sample_pattern))
        if not sample_dirs:
            raise AnnotationTemplateError(
                f"No sample directories matched under {input_root} with pattern: {args.sample_pattern}"
            )

        rows: List[Dict[str, str]] = []
        for sample_dir in sample_dirs:
            video_path = find_single_match(sample_dir, args.video_pattern, "video", True)
            metadata_path = find_single_match(sample_dir, args.metadata_pattern, "metadata JSON", False)
            rows.append(
                build_row(
                    sample_dir=sample_dir,
                    video_path=video_path,
                    metadata_path=metadata_path,
                    default_peak_ratio=args.default_peak_ratio,
                    default_gt_type=args.default_gt_type,
                )
            )

        write_rows(output_csv, rows)
        print(f"Annotation template written to: {output_csv}")
        print(f"Total rows: {len(rows)}")
        return 0
    except AnnotationTemplateError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
