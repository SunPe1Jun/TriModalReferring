#!/usr/bin/env python3
"""Render a timeline diagnostic video with projected GT anchors and gaze.

Use this when video time and multimodal JSON time may be offset. The script
samples an event timeline, extracts video frames at
video_time = json_sample_time + offset, projects GT anchors and gazePoint using
the JSON camera pose, overlays them, and optionally encodes an MP4.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

import build_2d_eval_manifest as prep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render timeline projection diagnostics for one event row.")
    parser.add_argument("--repo_root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument("--eval_dir", default=None, help="Directory containing *_match_eval.csv.")
    parser.add_argument("--scene", required=True, choices=prep.SCENES, help="Scene key, e.g. scene2.")
    parser.add_argument("--row_index", type=int, required=True, help="Zero-based row index.")
    parser.add_argument("--output_dir", default="exam2/outputs/timeline_diagnostics", help="Output directory.")
    parser.add_argument("--video_time_offset_seconds", type=float, default=0.0, help="video_time = json_time + offset. With --auto_video_time_offset, this is an extra adjustment.")
    parser.add_argument("--auto_video_time_offset", action="store_true", help="Estimate this row's offset from metadata_*.json and the first multimodal timestamp.")
    parser.add_argument("--auto_video_time_offset_source", choices=("metadata", "video_filename", "hybrid"), default="metadata", help="Auto offset source. Default: metadata.")
    parser.add_argument("--hybrid_offset_threshold_seconds", type=float, default=1.0, help="Hybrid threshold for metadata/video start disagreement. Default: 1.0.")
    parser.add_argument("--hybrid_video_time_bias_seconds", type=float, default=0.5, help="Hybrid video filename fallback bias. Default: 0.5.")
    parser.add_argument("--fps", type=float, default=2.0, help="Diagnostic sampling/encoding FPS. Default: 2.")
    parser.add_argument("--max_duration", type=float, help="Optional max seconds to render from event window.")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", help="ffmpeg executable.")
    parser.add_argument("--ffprobe_path", default="ffprobe", help="ffprobe executable.")
    parser.add_argument("--path-rewrite", action="append", default=[], metavar="OLD=NEW", help="Rewrite path prefixes; can be repeated.")
    parser.add_argument("--no_video", action="store_true", help="Only render annotated JPEG frames, do not encode MP4.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite extracted/annotated frames.")
    return parser.parse_args()


def read_row(path: Path, row_index: int) -> Dict[str, str]:
    rows = prep.read_csv_rows(path)
    if row_index < 0 or row_index >= len(rows):
        raise prep.Exam2Error(f"row_index out of range for {path}: {row_index}")
    return rows[row_index]


def load_font(size: int = 18) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def color_for_index(index: int) -> Tuple[int, int, int]:
    palette = [
        (255, 59, 48),
        (0, 122, 255),
        (255, 149, 0),
        (175, 82, 222),
        (90, 200, 250),
        (255, 45, 85),
    ]
    return palette[index % len(palette)]


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], size: int = 12, width: int = 3) -> None:
    draw.line((x - size, y, x + size, y), fill=color, width=width)
    draw.line((x, y - size, x, y + size), fill=color, width=width)


def extract_raw_frame(ffmpeg_path: str, video_path: Path, video_time: float, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{max(0.0, video_time):.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def encode_video(ffmpeg_path: str, frames_dir: Path, output_mp4: Path, fps: float) -> bool:
    command = [
        ffmpeg_path,
        "-y",
        "-framerate",
        f"{fps:.3f}",
        "-i",
        str(frames_dir / "annotated_%05d.jpg"),
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr)
    return completed.returncode == 0 and output_mp4.exists()


def sample_timeline(t_start: float, t_end: float, fps: float, max_duration: Optional[float]) -> List[float]:
    if fps <= 0:
        raise prep.Exam2Error("--fps must be positive")
    if t_end < t_start:
        t_end = t_start
    if max_duration is not None:
        t_end = min(t_end, t_start + max_duration)
    step = 1.0 / fps
    times: List[float] = []
    t = t_start
    while t <= t_end + 1e-6:
        times.append(t)
        t += step
    if not times:
        times.append(t_start)
    return times


def gaze_world_point(sample: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    eye = sample.get("eyeGaze")
    if not isinstance(eye, Mapping):
        return None
    if eye.get("isEyeOpen") is False:
        return None
    return prep.point3(eye.get("gazePoint"))


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    eval_dir = prep.resolve_eval_dir(repo_root, args.eval_dir).resolve()
    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    rewrites = prep.parse_rewrites(args.path_rewrite)

    paths = prep.scene_paths(repo_root, eval_dir, args.scene)
    api_row = read_row(paths.api_input_csv, args.row_index)
    match_by_row = prep.build_match_eval_by_row(paths.match_eval_csv)
    eval_row = match_by_row.get(args.row_index, {})
    referents = prep.split_names(eval_row.get("gt_referents_mapped"))
    if not referents:
        raise prep.Exam2Error(f"No mapped GT referents for {args.scene} row {args.row_index}")

    anchors = prep.load_anchor_table(paths.anchor_csv)
    video_path = prep.path_from_text(api_row.get("video_path", ""), rewrites)
    json_path = prep.path_from_text(api_row.get("json_path", ""), rewrites)
    if not video_path.exists() or not json_path.exists():
        raise prep.Exam2Error(f"Missing media: video_exists={video_path.exists()} json_exists={json_path.exists()}")
    width, height = prep.ffprobe_size(args.ffprobe_path, video_path)
    multimodal_samples = prep.load_multimodal_samples(json_path)
    timed_samples = prep.collect_timed_samples(multimodal_samples)
    if not timed_samples:
        raise prep.Exam2Error(f"No timed samples in {json_path}")
    video_time_offset, video_time_offset_source = prep.effective_video_time_offset(
        json_path=json_path,
        video_path=video_path,
        samples=multimodal_samples,
        manual_adjustment=float(args.video_time_offset_seconds),
        auto_enabled=bool(args.auto_video_time_offset),
        source_strategy=args.auto_video_time_offset_source,
        threshold_seconds=float(args.hybrid_offset_threshold_seconds),
        video_bias_seconds=float(args.hybrid_video_time_bias_seconds),
    )

    t_start = prep.parse_float(api_row.get("t_start"))
    t_end = prep.parse_float(api_row.get("t_end"))
    if t_start is None:
        t_start = timed_samples[0][0]
    if t_end is None:
        t_end = timed_samples[-1][0]
    json_times = sample_timeline(t_start, t_end, args.fps, args.max_duration)

    diag_dir = output_root / f"{args.scene}_row_{args.row_index}_offset_{video_time_offset:+.3f}".replace(".", "p")
    raw_dir = diag_dir / "raw"
    annotated_dir = diag_dir / "annotated"
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    font = load_font(18)
    small_font = load_font(14)

    referent_points: List[Tuple[str, Tuple[float, float, float]]] = []
    for name in referents:
        anchor = anchors.get(name)
        if anchor is not None:
            referent_points.append((name, anchor))

    for frame_idx, json_time in enumerate(json_times, start=1):
        sample_time, sample = prep.nearest_sample(timed_samples, json_time)
        video_time = max(0.0, sample_time + video_time_offset)
        raw_path = raw_dir / f"raw_{frame_idx:05d}.jpg"
        annotated_path = annotated_dir / f"annotated_{frame_idx:05d}.jpg"
        ok = extract_raw_frame(args.ffmpeg_path, video_path, video_time, raw_path, args.overwrite)
        if not ok:
            continue
        if annotated_path.exists() and not args.overwrite:
            continue
        image = Image.open(raw_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        overlay_lines = [
            f"{args.scene} row={args.row_index} event={api_row.get('event_id','')}",
            f"json_t={sample_time:.3f}s video_t={video_time:.3f}s offset={video_time_offset:+.3f}s",
            f"offset_source: {video_time_offset_source[:120]}",
            f"instruction: {prep.normalize_text(api_row.get('instruction_text'))[:120]}",
        ]
        y = 8
        for line in overlay_lines:
            draw.rectangle((6, y - 2, min(width - 6, 14 + len(line) * 9), y + 22), fill=(0, 0, 0))
            draw.text((10, y), line, fill=(255, 255, 255), font=small_font)
            y += 24

        for ref_idx, (name, anchor) in enumerate(referent_points):
            projected = prep.project_world_point(sample, anchor, width, height)
            if projected is None:
                continue
            u, v = projected
            x = u * width
            yy = v * height
            color = color_for_index(ref_idx)
            radius = 9
            draw.ellipse((x - radius, yy - radius, x + radius, yy + radius), outline=color, width=4)
            draw.text((x + 12, yy - 10), name, fill=color, font=font)

        gaze = gaze_world_point(sample)
        if gaze is not None:
            projected_gaze = prep.project_world_point(sample, gaze, width, height)
            if projected_gaze is not None:
                gx = projected_gaze[0] * width
                gy = projected_gaze[1] * height
                draw_cross(draw, gx, gy, (0, 255, 0), size=14, width=4)
                draw.text((gx + 14, gy + 8), "gazePoint", fill=(0, 255, 0), font=font)

        image.save(annotated_path, quality=92)

    output_mp4 = diag_dir / f"{args.scene}_row_{args.row_index}_offset_{video_time_offset:+.3f}.mp4".replace(".", "p")
    if not args.no_video:
        encoded = encode_video(args.ffmpeg_path, annotated_dir, output_mp4, args.fps)
        if encoded:
            print(f"Wrote diagnostic video: {output_mp4}")
        else:
            print(f"Failed to encode video. Annotated frames are in: {annotated_dir}")
    print(f"Annotated frames: {annotated_dir}")
    print(f"Referents: {', '.join(referents)}")
    print(f"Video time offset: {video_time_offset:.6f} ({video_time_offset_source})")


if __name__ == "__main__":
    try:
        main()
    except prep.Exam2Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
