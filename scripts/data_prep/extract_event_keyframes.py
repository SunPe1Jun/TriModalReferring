#!/usr/bin/env python3
"""Extract event keyframes from videos using ffmpeg."""

from __future__ import annotations

import argparse
import csv
import glob
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


DEFAULT_INPUT_COLUMNS = (
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


class KeyframeExtractionError(Exception):
    """Raised when keyframe extraction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract event keyframes from videos based on t_peak."
    )
    parser.add_argument(
        "--annotation-csv",
        required=True,
        help="Path to the event annotation CSV file.",
    )
    parser.add_argument(
        "--video-root",
        required=True,
        help="Root directory that stores source videos.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory where keyframe images will be written.",
    )
    parser.add_argument(
        "--video-template",
        default="{scene_id}/{video_id}.mp4",
        help="Relative path template from video root to the source video file. Wildcards are supported.",
    )
    parser.add_argument(
        "--output-template",
        default="{scene_id}/{video_id}/{t_peak}.jpg",
        help="Relative path template from output root to the extracted keyframe image.",
    )
    parser.add_argument(
        "--event-id-template",
        default="event_{row_index:06d}_{scene_id}_{video_id}_{t_start}_{t_end}",
        help="Template used to build a readable event identifier for logs.",
    )
    parser.add_argument(
        "--input-columns",
        default=",".join(DEFAULT_INPUT_COLUMNS),
        help="Comma-separated input CSV columns in the expected semantic order.",
    )
    parser.add_argument(
        "--image-quality",
        type=int,
        default=2,
        help="JPEG quality passed to ffmpeg -q:v. Lower is higher quality.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without extracting images.",
    )
    return parser.parse_args()


def split_columns(raw_columns: str, expected_min_count: int, label: str) -> List[str]:
    columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
    if len(columns) < expected_min_count:
        raise KeyframeExtractionError(
            f"{label} must contain at least {expected_min_count} column names, but received {len(columns)}."
        )
    return columns


def validate_input_columns(column_names: Sequence[str]) -> None:
    missing = [column for column in DEFAULT_INPUT_COLUMNS if column not in column_names]
    if missing:
        raise KeyframeExtractionError(
            "input-columns is missing required names: " + ", ".join(missing)
        )


def sanitize_token(value: object) -> str:
    text = str(value).strip()
    sanitized_chars: List[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            sanitized_chars.append(char)
        else:
            sanitized_chars.append("_")
    sanitized = "".join(sanitized_chars).strip("_")
    return sanitized or "empty"


def is_missing_value(value: object) -> bool:
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "null", "none", "na"}


def parse_peak_time(raw_value: str, row_index: int) -> float:
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise KeyframeExtractionError(
            f"Row {row_index} has invalid t_peak value: {raw_value}"
        ) from exc
    if value < 0:
        raise KeyframeExtractionError(
            f"Row {row_index} has negative t_peak value: {raw_value}"
        )
    return value


def build_template_context(row: Mapping[str, str], row_index: int) -> Dict[str, str]:
    context = {key: str(value).strip() for key, value in row.items()}
    context["row_index"] = str(row_index)
    context["scene_id"] = sanitize_token(context["scene_id"])
    context["video_id"] = sanitize_token(context["video_id"])
    context["t_start"] = sanitize_token(context["t_start"])
    context["t_peak"] = sanitize_token(context["t_peak"])
    context["t_end"] = sanitize_token(context["t_end"])
    context["gt_type"] = sanitize_token(context.get("gt_type", ""))
    return context


def format_template(template: str, context: Mapping[str, str], label: str) -> str:
    try:
        return template.format(
            row_index=int(context["row_index"]),
            scene_id=context["scene_id"],
            video_id=context["video_id"],
            t_start=context["t_start"],
            t_peak=context["t_peak"],
            t_end=context["t_end"],
            gt_type=context["gt_type"],
        )
    except KeyError as exc:
        raise KeyframeExtractionError(
            f"Unknown placeholder {exc!s} in {label}: {template}"
        ) from exc
    except ValueError as exc:
        raise KeyframeExtractionError(
            f"Invalid formatting in {label}: {template}. Details: {exc}"
        ) from exc


def resolve_single_match(
    root_dir: Path,
    relative_template: str,
    context: Mapping[str, str],
    label: str,
    row_index: int,
) -> Path:
    relative_path = format_template(relative_template, context, label)
    pattern_path = root_dir / Path(relative_path)
    matches = sorted(Path(item).resolve() for item in glob.glob(str(pattern_path)))

    if not matches:
        raise KeyframeExtractionError(
            f"Missing {label} for row {row_index}: {pattern_path.resolve()}"
        )
    if len(matches) > 1:
        formatted_matches = ", ".join(str(match) for match in matches[:5])
        raise KeyframeExtractionError(
            f"{label} matched multiple files for row {row_index}: {formatted_matches}. "
            f"Please make the template more specific: {relative_template}"
        )
    return matches[0]


def load_rows(annotation_csv: Path, input_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not annotation_csv.exists():
        raise KeyframeExtractionError(f"Annotation CSV does not exist: {annotation_csv}")
    if not annotation_csv.is_file():
        raise KeyframeExtractionError(f"Annotation CSV is not a file: {annotation_csv}")

    with annotation_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise KeyframeExtractionError(f"Annotation CSV has no header row: {annotation_csv}")

        missing_columns = [column for column in input_columns if column not in reader.fieldnames]
        if missing_columns:
            raise KeyframeExtractionError(
                "Annotation CSV is missing required columns: " + ", ".join(missing_columns)
            )

        rows: List[Dict[str, str]] = []
        for row_index, raw_row in enumerate(reader, start=1):
            normalized_row = {
                column: (raw_row.get(column, "") or "").strip()
                for column in input_columns
            }
            if not any(normalized_row.values()):
                continue

            required_missing = [
                column
                for column in ("scene_id", "video_id", "t_peak")
                if is_missing_value(normalized_row[column])
            ]
            if required_missing:
                raise KeyframeExtractionError(
                    f"Row {row_index} is missing required values: {', '.join(required_missing)}"
                )

            parse_peak_time(normalized_row["t_peak"], row_index)
            rows.append(normalized_row)

    if not rows:
        raise KeyframeExtractionError(f"Annotation CSV contains no valid data rows: {annotation_csv}")
    return rows


def ensure_ffmpeg_available() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise KeyframeExtractionError(
            "ffmpeg was not found in PATH. Please install ffmpeg before running this script."
        )
    return ffmpeg_path


def extract_keyframe(
    ffmpeg_path: str,
    video_path: Path,
    output_path: Path,
    t_peak_seconds: float,
    image_quality: int,
    overwrite: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{t_peak_seconds:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        str(image_quality),
        "-y" if overwrite else "-n",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown ffmpeg error."
        raise KeyframeExtractionError(
            f"ffmpeg failed for video {video_path}: {stderr}"
        )
    if not output_path.exists():
        raise KeyframeExtractionError(
            f"ffmpeg did not produce an output image: {output_path}"
        )


def main() -> int:
    args = parse_args()

    try:
        input_columns = split_columns(
            args.input_columns, len(DEFAULT_INPUT_COLUMNS), "input-columns"
        )
        validate_input_columns(input_columns)

        annotation_csv = Path(args.annotation_csv).resolve()
        video_root = Path(args.video_root).resolve()
        output_root = Path(args.output_root).resolve()

        if not video_root.exists() or not video_root.is_dir():
            raise KeyframeExtractionError(
                f"Video root directory does not exist or is not a directory: {video_root}"
            )

        ffmpeg_path = ensure_ffmpeg_available()
        source_rows = load_rows(annotation_csv, input_columns)

        extracted_count = 0
        skipped_count = 0

        for row_index, row in enumerate(source_rows, start=1):
            context = build_template_context(row, row_index)
            event_id = format_template(args.event_id_template, context, "event-id-template")
            video_path = resolve_single_match(
                video_root, args.video_template, context, "video-template", row_index
            )
            output_relative = format_template(args.output_template, context, "output-template")
            output_path = (output_root / Path(output_relative)).resolve()
            t_peak_seconds = parse_peak_time(row["t_peak"], row_index)

            if output_path.exists() and not args.overwrite:
                skipped_count += 1
                print(f"Skip existing keyframe: {event_id} -> {output_path}")
                continue

            if args.dry_run:
                print(
                    f"Plan keyframe: {event_id} | video={video_path} | t_peak={t_peak_seconds:.3f} | output={output_path}"
                )
                extracted_count += 1
                continue

            extract_keyframe(
                ffmpeg_path=ffmpeg_path,
                video_path=video_path,
                output_path=output_path,
                t_peak_seconds=t_peak_seconds,
                image_quality=args.image_quality,
                overwrite=args.overwrite,
            )
            extracted_count += 1
            print(f"Extracted keyframe: {event_id} -> {output_path}")

        print(f"Processed events: {len(source_rows)}")
        print(f"Extracted keyframes: {extracted_count}")
        print(f"Skipped existing: {skipped_count}")
        return 0
    except KeyframeExtractionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
