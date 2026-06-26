#!/usr/bin/env python3
"""Batch render enhanced gaze overlays for dataset samples."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import enhance_gaze_overlay as overlay


class BatchOverlayError(Exception):
    """Raised when batch overlay execution cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch render enhanced gaze overlay videos for sample directories."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root directory that contains sample subdirectories such as data/1, data/2, ...",
    )
    parser.add_argument(
        "--sample-pattern",
        default="*",
        help="Glob pattern used to select sample directories under input root.",
    )
    parser.add_argument(
        "--video-pattern",
        default="ScreenRecord_*.mp4",
        help="Glob pattern used to locate the source video inside each sample directory.",
    )
    parser.add_argument(
        "--json-name",
        default="multimodal_data.json",
        help="File name of the multimodal gaze JSON inside each sample directory.",
    )
    parser.add_argument(
        "--metadata-pattern",
        default="metadata_*.json",
        help="Optional glob pattern used to locate the metadata JSON inside each sample directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_gaze_enhanced",
        help="Suffix inserted before the output video extension.",
    )
    parser.add_argument(
        "--output-root",
        help="Optional separate output root. If omitted, outputs are written into each sample directory.",
    )
    parser.add_argument(
        "--point-source",
        choices=("gazePoint", "cameraHitPoint", "auto"),
        default="gazePoint",
        help="3D point source used to compute the 2D overlay position.",
    )
    parser.add_argument(
        "--time-offset-seconds",
        type=float,
        default=0.0,
        help="Manual offset added to every overlay timestamp after alignment.",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=32,
        help="Marker font size used in the ASS overlay.",
    )
    parser.add_argument(
        "--pixel-offset-x",
        type=float,
        default=0.0,
        help="Global horizontal pixel offset applied to every rendered marker. Negative moves left.",
    )
    parser.add_argument(
        "--pixel-offset-y",
        type=float,
        default=0.0,
        help="Global vertical pixel offset applied to every rendered marker. Negative moves up.",
    )
    parser.add_argument(
        "--marker-color",
        default="00FF00",
        help="Marker color in RRGGBB format.",
    )
    parser.add_argument(
        "--outline-color",
        default="FFFFFF",
        help="Outline color in RRGGBB format.",
    )
    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=0.03,
        help="Minimum overlay duration per gaze sample.",
    )
    parser.add_argument(
        "--max-segment-seconds",
        type=float,
        default=0.20,
        help="Maximum overlay duration per gaze sample.",
    )
    parser.add_argument(
        "--keep-ass",
        action="store_true",
        help="Keep the generated ASS files next to each output video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output videos.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining samples if one sample fails.",
    )
    return parser.parse_args()


def find_single_match(directory: Path, pattern: str, label: str, required: bool) -> Optional[Path]:
    matches = sorted(Path(path).resolve() for path in glob.glob(str(directory / pattern)))
    if not matches:
        if required:
            raise BatchOverlayError(f"Missing {label} in sample directory: {directory}")
        return None
    if len(matches) > 1:
        joined = ", ".join(str(match) for match in matches[:5])
        raise BatchOverlayError(
            f"Multiple {label} files matched in {directory}: {joined}. Please use a more specific pattern."
        )
    return matches[0]


def iter_sample_dirs(input_root: Path, sample_pattern: str) -> Iterable[Path]:
    matches = sorted(Path(path).resolve() for path in glob.glob(str(input_root / sample_pattern)))
    for match in matches:
        if match.is_dir():
            yield match


def build_output_path(sample_dir: Path, output_root: Optional[Path], output_suffix: str, video_path: Path) -> Path:
    stem = video_path.stem + output_suffix
    filename = stem + video_path.suffix
    if output_root is None:
        return (sample_dir / filename).resolve()
    return (output_root / sample_dir.name / filename).resolve()


def render_one_sample(
    sample_dir: Path,
    output_root: Optional[Path],
    args: argparse.Namespace,
    ffmpeg_path: str,
    ffprobe_path: str,
) -> Path:
    input_video = find_single_match(sample_dir, args.video_pattern, "video", True)
    input_json = sample_dir / args.json_name
    metadata_json = find_single_match(sample_dir, args.metadata_pattern, "metadata JSON", False)
    output_video = build_output_path(sample_dir, output_root, args.output_suffix, input_video)

    overlay.ensure_file_exists(input_video, "input video")
    overlay.ensure_file_exists(input_json, "input JSON")
    if output_video.exists() and not args.overwrite:
        raise BatchOverlayError(
            f"Output video already exists. Use --overwrite to replace it: {output_video}"
        )

    width, height = overlay.run_ffprobe(ffprobe_path, input_video)
    samples = overlay.load_multimodal_samples(input_json)
    start_time = overlay.load_metadata_start_time(metadata_json)
    ass_text, valid_count = overlay.build_ass_overlay(
        samples=samples,
        width=width,
        height=height,
        point_source=args.point_source,
        start_time=start_time,
        time_offset_seconds=args.time_offset_seconds,
        marker_size=args.marker_size,
        pixel_offset_x=args.pixel_offset_x,
        pixel_offset_y=args.pixel_offset_y,
        marker_color=args.marker_color,
        outline_color=args.outline_color,
        min_segment_seconds=args.min_segment_seconds,
        max_segment_seconds=args.max_segment_seconds,
    )

    if args.keep_ass:
        ass_path = output_video.with_suffix(".ass")
        ass_path.parent.mkdir(parents=True, exist_ok=True)
        ass_path.write_text(ass_text, encoding="utf-8")
        overlay.render_overlay_video(ffmpeg_path, input_video, ass_path, output_video, args.overwrite)
    else:
        import tempfile

        with tempfile.TemporaryDirectory(prefix="gaze_overlay_batch_") as temp_dir:
            ass_path = Path(temp_dir) / "gaze_overlay.ass"
            ass_path.write_text(ass_text, encoding="utf-8")
            overlay.render_overlay_video(ffmpeg_path, input_video, ass_path, output_video, args.overwrite)

    print(f"Rendered sample: {sample_dir.name} -> {output_video} | markers={valid_count}")
    return output_video


def main() -> int:
    args = parse_args()

    try:
        input_root = Path(args.input_root).resolve()
        output_root = Path(args.output_root).resolve() if args.output_root else None
        if not input_root.exists() or not input_root.is_dir():
            raise BatchOverlayError(f"Input root does not exist or is not a directory: {input_root}")
        if output_root is not None:
            output_root.mkdir(parents=True, exist_ok=True)

        ffmpeg_path = overlay.ensure_tool_available("ffmpeg")
        ffprobe_path = overlay.ensure_tool_available("ffprobe")

        sample_dirs = list(iter_sample_dirs(input_root, args.sample_pattern))
        if not sample_dirs:
            raise BatchOverlayError(
                f"No sample directories matched under {input_root} with pattern: {args.sample_pattern}"
            )

        success_count = 0
        failure_count = 0
        failed_samples: List[Tuple[str, str]] = []

        for sample_dir in sample_dirs:
            try:
                render_one_sample(sample_dir, output_root, args, ffmpeg_path, ffprobe_path)
                success_count += 1
            except (BatchOverlayError, overlay.GazeOverlayError) as exc:
                failure_count += 1
                failed_samples.append((sample_dir.name, str(exc)))
                print(f"Failed sample: {sample_dir.name} | {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    break

        print(f"Processed samples: {success_count + failure_count}")
        print(f"Succeeded: {success_count}")
        print(f"Failed: {failure_count}")
        if failed_samples:
            print("Failure details:")
            for sample_name, message in failed_samples:
                print(f"  {sample_name}: {message}")

        return 0 if failure_count == 0 else 1
    except BatchOverlayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
