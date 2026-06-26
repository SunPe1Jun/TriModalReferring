#!/usr/bin/env python3
"""Build an event-level multimodal non-entity grounding manifest."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


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

DEFAULT_OUTPUT_COLUMNS = (
    "event_id",
    "scene_id",
    "video_id",
    "video_path",
    "json_path",
    "t_start",
    "t_peak",
    "t_end",
    "keyframe_path",
    "camera_pose_json",
    "gt_anchor_x",
    "gt_anchor_y",
    "gt_anchor_z",
    "gt_type",
    "has_gt_anchor",
)

DEFAULT_AUTO_VIDEO_PATTERNS = (
    "*gaze_enhanced*.mp4",
    "ScreenRecord_*.mp4",
    "*.mp4",
)

DEFAULT_AUTO_JSON_PATTERNS = (
    "multimodal_data.json",
)

DEFAULT_AUTO_CAMERA_POSE_PATTERNS = (
    "multimodal_data.json",
    "*camera_pose*.json",
)

DEFAULT_AUTO_KEYFRAME_PATTERNS = (
    "keyframes/{t_peak}.jpg",
    "keyframes/{t_peak}.png",
    "keyframes/*{t_peak}*.jpg",
    "keyframes/*{t_peak}*.png",
    "*{t_peak}*.jpg",
    "*{t_peak}*.png",
)


class ManifestBuildError(Exception):
    """Raised when the manifest cannot be built from the given inputs."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an event manifest for multimodal non-entity grounding."
    )
    parser.add_argument(
        "--annotation-csv",
        required=True,
        help="Path to the event annotation CSV file.",
    )
    parser.add_argument(
        "--video-root",
        required=True,
        help="Root directory that stores sample directories, videos, and optionally keyframes.",
    )
    parser.add_argument(
        "--json-root",
        required=True,
        help="Root directory that stores sample directories and JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Path to the output event manifest CSV file.",
    )
    parser.add_argument(
        "--auto-discover",
        dest="auto_discover",
        action="store_true",
        help="Automatically locate files inside sample directories such as data/<scene_id>/.",
    )
    parser.add_argument(
        "--no-auto-discover",
        dest="auto_discover",
        action="store_false",
        help="Disable automatic file discovery and use explicit templates only.",
    )
    parser.set_defaults(auto_discover=True)
    parser.add_argument(
        "--sample-dir-template",
        default="{scene_id}",
        help="Relative sample directory template from the roots when auto-discover is enabled.",
    )
    parser.add_argument(
        "--auto-video-patterns",
        default=",".join(DEFAULT_AUTO_VIDEO_PATTERNS),
        help="Comma-separated filename patterns searched inside each sample directory for the source video.",
    )
    parser.add_argument(
        "--auto-json-patterns",
        default=",".join(DEFAULT_AUTO_JSON_PATTERNS),
        help="Comma-separated filename patterns searched inside each sample directory for gaze or hand JSON.",
    )
    parser.add_argument(
        "--auto-camera-pose-patterns",
        default=",".join(DEFAULT_AUTO_CAMERA_POSE_PATTERNS),
        help="Comma-separated filename patterns searched inside each sample directory for camera pose JSON.",
    )
    parser.add_argument(
        "--auto-keyframe-patterns",
        default=",".join(DEFAULT_AUTO_KEYFRAME_PATTERNS),
        help="Comma-separated filename patterns searched inside each sample directory for keyframes.",
    )
    parser.add_argument(
        "--video-template",
        default="{scene_id}/{video_id}.mp4",
        help="Relative path template from video root to the source video file. Wildcards are supported.",
    )
    parser.add_argument(
        "--json-template",
        default="{scene_id}/{video_id}.json",
        help="Relative path template from json root to the gaze or hand JSON file. Wildcards are supported.",
    )
    parser.add_argument(
        "--keyframe-template",
        default="{scene_id}/{video_id}/{t_peak}.jpg",
        help="Relative path template from video root to the keyframe image file. Wildcards are supported.",
    )
    parser.add_argument(
        "--camera-pose-template",
        default="{scene_id}/{video_id}_camera_pose.json",
        help="Relative path template from json root to the camera pose JSON file. Wildcards are supported.",
    )
    parser.add_argument(
        "--event-id-template",
        default="event_{row_index:06d}_{scene_id}_{video_id}_{t_start}_{t_end}",
        help="Template used to build event_id.",
    )
    parser.add_argument(
        "--input-columns",
        default=",".join(DEFAULT_INPUT_COLUMNS),
        help="Comma-separated input CSV columns in the expected semantic order.",
    )
    parser.add_argument(
        "--output-columns",
        default=",".join(DEFAULT_OUTPUT_COLUMNS),
        help="Comma-separated output manifest columns.",
    )
    return parser.parse_args()


def split_columns(raw_columns: str, expected_min_count: int, label: str) -> List[str]:
    columns = [item.strip() for item in raw_columns.split(",") if item.strip()]
    if len(columns) < expected_min_count:
        raise ManifestBuildError(
            f"{label} must contain at least {expected_min_count} comma-separated column names, "
            f"but received {len(columns)}."
        )
    return columns


def parse_patterns(raw_patterns: str, label: str) -> List[str]:
    patterns = [item.strip() for item in raw_patterns.split(",") if item.strip()]
    if not patterns:
        raise ManifestBuildError(f"{label} must contain at least one pattern.")
    return patterns


def validate_input_columns(column_names: Sequence[str]) -> None:
    missing = [column for column in DEFAULT_INPUT_COLUMNS if column not in column_names]
    if missing:
        raise ManifestBuildError(
            "input-columns is missing required names: " + ", ".join(missing)
        )


def ensure_file(path: Path, description: str, row_index: int) -> None:
    if not path.exists():
        raise ManifestBuildError(f"Missing {description} for row {row_index}: {path}")
    if not path.is_file():
        raise ManifestBuildError(
            f"Expected {description} to be a file for row {row_index}, but found: {path}"
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
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "null", "none", "na"}


def has_gt_anchor(row: Mapping[str, str]) -> bool:
    return not any(
        is_missing_value(row[column])
        for column in ("gt_anchor_x", "gt_anchor_y", "gt_anchor_z")
    )


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
        raise ManifestBuildError(f"Unknown placeholder {exc!s} in {label}: {template}") from exc
    except ValueError as exc:
        raise ManifestBuildError(
            f"Invalid formatting in {label}: {template}. Details: {exc}"
        ) from exc


def resolve_file_path(
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
        raise ManifestBuildError(f"Missing {label} for row {row_index}: {pattern_path.resolve()}")
    if len(matches) > 1:
        formatted_matches = ", ".join(str(match) for match in matches[:5])
        raise ManifestBuildError(
            f"{label} matched multiple files for row {row_index}: {formatted_matches}. "
            f"Please make the template more specific: {relative_template}"
        )
    return matches[0]


def resolve_auto_sample_dir(
    root_dir: Path,
    sample_dir_template: str,
    context: Mapping[str, str],
    label: str,
    row_index: int,
) -> Path:
    relative_dir = format_template(sample_dir_template, context, label)
    sample_dir = (root_dir / relative_dir).resolve()
    if not sample_dir.exists() or not sample_dir.is_dir():
        raise ManifestBuildError(
            f"Missing sample directory for row {row_index}: {sample_dir}"
        )
    return sample_dir


def resolve_from_patterns(
    sample_dir: Path,
    patterns: Sequence[str],
    context: Mapping[str, str],
    label: str,
    row_index: int,
) -> Path:
    all_matches: List[Path] = []
    for pattern in patterns:
        formatted_pattern = format_template(pattern, context, label)
        search_pattern = sample_dir / formatted_pattern
        matches = sorted(Path(item).resolve() for item in glob.glob(str(search_pattern)))
        if matches:
            all_matches = matches
            break

    if not all_matches:
        raise ManifestBuildError(
            f"Missing {label} for row {row_index} under sample directory: {sample_dir}"
        )
    if len(all_matches) > 1:
        formatted_matches = ", ".join(str(match) for match in all_matches[:5])
        raise ManifestBuildError(
            f"{label} matched multiple files for row {row_index}: {formatted_matches}. "
            f"Please narrow the auto pattern list."
        )
    return all_matches[0]


def load_rows(annotation_csv: Path, input_columns: Sequence[str]) -> List[Dict[str, str]]:
    if not annotation_csv.exists():
        raise ManifestBuildError(f"Annotation CSV does not exist: {annotation_csv}")
    if not annotation_csv.is_file():
        raise ManifestBuildError(f"Annotation CSV is not a file: {annotation_csv}")

    with annotation_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestBuildError(f"Annotation CSV has no header row: {annotation_csv}")

        missing_columns = [column for column in input_columns if column not in reader.fieldnames]
        if missing_columns:
            raise ManifestBuildError(
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
                for column in ("scene_id", "video_id", "t_start", "t_peak", "t_end", "gt_type")
                if is_missing_value(normalized_row[column])
            ]
            if required_missing:
                raise ManifestBuildError(
                    f"Row {row_index} is missing required values: {', '.join(required_missing)}"
                )
            rows.append(normalized_row)

    if not rows:
        raise ManifestBuildError(f"Annotation CSV contains no valid data rows: {annotation_csv}")
    return rows


def build_3d_anchor_interface(row: Mapping[str, str]) -> Dict[str, str]:
    return {
        "gt_anchor_x": row["gt_anchor_x"],
        "gt_anchor_y": row["gt_anchor_y"],
        "gt_anchor_z": row["gt_anchor_z"],
        "gt_type": row["gt_type"],
        "has_gt_anchor": str(has_gt_anchor(row)).lower(),
    }


def build_manifest_rows(
    rows: Iterable[Mapping[str, str]],
    video_root: Path,
    json_root: Path,
    auto_discover: bool,
    sample_dir_template: str,
    auto_video_patterns: Sequence[str],
    auto_json_patterns: Sequence[str],
    auto_camera_pose_patterns: Sequence[str],
    auto_keyframe_patterns: Sequence[str],
    video_template: str,
    json_template: str,
    keyframe_template: str,
    camera_pose_template: str,
    event_id_template: str,
) -> List[Dict[str, str]]:
    manifest_rows: List[Dict[str, str]] = []
    seen_event_ids = set()

    for row_index, row in enumerate(rows, start=1):
        context = build_template_context(row, row_index)
        event_id = format_template(event_id_template, context, "event-id-template")
        if event_id in seen_event_ids:
            raise ManifestBuildError(
                f"Duplicate event_id generated for row {row_index}: {event_id}. "
                "Please change --event-id-template."
            )
        seen_event_ids.add(event_id)

        if auto_discover:
            video_sample_dir = resolve_auto_sample_dir(
                video_root, sample_dir_template, context, "sample-dir-template", row_index
            )
            json_sample_dir = resolve_auto_sample_dir(
                json_root, sample_dir_template, context, "sample-dir-template", row_index
            )
            video_path = resolve_from_patterns(
                video_sample_dir, auto_video_patterns, context, "auto-video-patterns", row_index
            )
            json_path = resolve_from_patterns(
                json_sample_dir, auto_json_patterns, context, "auto-json-patterns", row_index
            )
            camera_pose_json = resolve_from_patterns(
                json_sample_dir,
                auto_camera_pose_patterns,
                context,
                "auto-camera-pose-patterns",
                row_index,
            )
            keyframe_path = resolve_from_patterns(
                video_sample_dir, auto_keyframe_patterns, context, "auto-keyframe-patterns", row_index
            )
        else:
            video_path = resolve_file_path(
                video_root, video_template, context, "video-template", row_index
            )
            json_path = resolve_file_path(
                json_root, json_template, context, "json-template", row_index
            )
            keyframe_path = resolve_file_path(
                video_root, keyframe_template, context, "keyframe-template", row_index
            )
            camera_pose_json = resolve_file_path(
                json_root, camera_pose_template, context, "camera-pose-template", row_index
            )

        ensure_file(video_path, "video file", row_index)
        ensure_file(json_path, "gaze or hand JSON file", row_index)
        ensure_file(keyframe_path, "keyframe file", row_index)
        ensure_file(camera_pose_json, "camera pose JSON file", row_index)

        anchor_fields = build_3d_anchor_interface(row)
        manifest_rows.append(
            {
                "event_id": event_id,
                "scene_id": row["scene_id"],
                "video_id": row["video_id"],
                "video_path": str(video_path),
                "json_path": str(json_path),
                "t_start": row["t_start"],
                "t_peak": row["t_peak"],
                "t_end": row["t_end"],
                "keyframe_path": str(keyframe_path),
                "camera_pose_json": str(camera_pose_json),
                "gt_anchor_x": anchor_fields["gt_anchor_x"],
                "gt_anchor_y": anchor_fields["gt_anchor_y"],
                "gt_anchor_z": anchor_fields["gt_anchor_z"],
                "gt_type": anchor_fields["gt_type"],
                "has_gt_anchor": anchor_fields["has_gt_anchor"],
            }
        )

    return manifest_rows


def write_manifest(
    output_csv: Path, rows: Sequence[Mapping[str, str]], output_columns: Sequence[str]
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in output_columns})


def main() -> int:
    args = parse_args()

    try:
        input_columns = split_columns(args.input_columns, len(DEFAULT_INPUT_COLUMNS), "input-columns")
        validate_input_columns(input_columns)
        output_columns = split_columns(args.output_columns, len(DEFAULT_OUTPUT_COLUMNS), "output-columns")

        annotation_csv = Path(args.annotation_csv).resolve()
        video_root = Path(args.video_root).resolve()
        json_root = Path(args.json_root).resolve()
        output_csv = Path(args.output_csv).resolve()

        if not video_root.exists() or not video_root.is_dir():
            raise ManifestBuildError(
                f"Video root directory does not exist or is not a directory: {video_root}"
            )
        if not json_root.exists() or not json_root.is_dir():
            raise ManifestBuildError(
                f"JSON root directory does not exist or is not a directory: {json_root}"
            )

        source_rows = load_rows(annotation_csv, input_columns)
        manifest_rows = build_manifest_rows(
            rows=source_rows,
            video_root=video_root,
            json_root=json_root,
            auto_discover=args.auto_discover,
            sample_dir_template=args.sample_dir_template,
            auto_video_patterns=parse_patterns(args.auto_video_patterns, "auto-video-patterns"),
            auto_json_patterns=parse_patterns(args.auto_json_patterns, "auto-json-patterns"),
            auto_camera_pose_patterns=parse_patterns(
                args.auto_camera_pose_patterns, "auto-camera-pose-patterns"
            ),
            auto_keyframe_patterns=parse_patterns(args.auto_keyframe_patterns, "auto-keyframe-patterns"),
            video_template=args.video_template,
            json_template=args.json_template,
            keyframe_template=args.keyframe_template,
            camera_pose_template=args.camera_pose_template,
            event_id_template=args.event_id_template,
        )
        write_manifest(output_csv, manifest_rows, output_columns)
        print(f"Manifest written to: {output_csv}")
        print(f"Total events: {len(manifest_rows)}")
        return 0
    except ManifestBuildError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
