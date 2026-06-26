#!/usr/bin/env python3
"""Enhance the gaze marker overlay in a video using multimodal gaze JSON."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


class GazeOverlayError(Exception):
    """Raised when gaze overlay generation fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance gaze marker rendering in a video using multimodal gaze JSON."
    )
    parser.add_argument("--input-video", required=True, help="Path to the source video.")
    parser.add_argument("--input-json", required=True, help="Path to the multimodal gaze JSON file.")
    parser.add_argument(
        "--output-video",
        required=True,
        help="Path to the rendered output video with enhanced gaze markers.",
    )
    parser.add_argument(
        "--metadata-json",
        help="Optional metadata JSON used to align gaze timestamps with the recording start time.",
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
        default=28,
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
        help="Maximum overlay duration per gaze sample to avoid long stale markers.",
    )
    parser.add_argument(
        "--keep-ass",
        action="store_true",
        help="Keep the generated ASS subtitle file next to the output video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output video if it already exists.",
    )
    return parser.parse_args()


def ensure_tool_available(tool_name: str) -> str:
    tool_path = shutil.which(tool_name)
    if tool_path is None:
        raise GazeOverlayError(
            f"{tool_name} was not found in PATH. Please install {tool_name} before running this script."
        )
    return tool_path


def ensure_file_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise GazeOverlayError(f"Missing {description}: {path}")
    if not path.is_file():
        raise GazeOverlayError(f"Expected {description} to be a file, but found: {path}")


def normalize_iso_timestamp(raw_value: str) -> str:
    text = raw_value.strip()
    if not text:
        return text

    time_start = text.find("T")
    if time_start < 0:
        return text

    fraction_start = text.find(".", time_start)
    if fraction_start < 0:
        return text

    timezone_start = len(text)
    for marker in ("+", "-", "Z"):
        marker_index = text.find(marker, time_start + 1)
        if marker_index != -1:
            timezone_start = min(timezone_start, marker_index)

    fraction = text[fraction_start + 1 : timezone_start]
    if not fraction.isdigit() or len(fraction) <= 6:
        return text

    return text[: fraction_start + 1] + fraction[:6] + text[timezone_start:]


def parse_timestamp(raw_value: str, label: str) -> datetime:
    normalized = normalize_iso_timestamp(raw_value)
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise GazeOverlayError(f"Invalid ISO timestamp in {label}: {raw_value}") from exc


def load_metadata_start_time(metadata_path: Optional[Path]) -> Optional[datetime]:
    if metadata_path is None:
        return None
    ensure_file_exists(metadata_path, "metadata JSON file")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise GazeOverlayError(f"Failed to parse metadata JSON: {metadata_path}. {exc}") from exc
    timestamp = payload.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise GazeOverlayError(f"metadata JSON does not contain a valid timestamp field: {metadata_path}")
    return parse_timestamp(timestamp, f"metadata JSON {metadata_path}")


def repair_json_array_if_needed(raw_text: str) -> str:
    stripped = raw_text.strip()
    if not stripped:
        raise GazeOverlayError("Input JSON file is empty.")

    open_brackets = stripped.count("[")
    close_brackets = stripped.count("]")
    if stripped.startswith("[") and open_brackets == close_brackets + 1 and stripped.endswith("}"):
        return stripped + "\n]"
    return stripped


def load_multimodal_samples(json_path: Path) -> List[Dict[str, object]]:
    ensure_file_exists(json_path, "multimodal JSON file")
    raw_text = json_path.read_text(encoding="utf-8-sig")

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        repaired_text = repair_json_array_if_needed(raw_text)
        try:
            payload = json.loads(repaired_text)
        except json.JSONDecodeError as exc:
            raise GazeOverlayError(f"Failed to parse multimodal JSON: {json_path}. {exc}") from exc

    if not isinstance(payload, list):
        raise GazeOverlayError(f"Expected multimodal JSON to be a list, but found: {type(payload).__name__}")
    if not payload:
        raise GazeOverlayError(f"Multimodal JSON contains no gaze samples: {json_path}")
    return payload


def run_ffprobe(ffprobe_path: str, video_path: Path) -> Tuple[int, int]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GazeOverlayError(result.stderr.strip() or f"ffprobe failed for {video_path}")

    try:
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GazeOverlayError(f"Failed to read video resolution from ffprobe for {video_path}") from exc

    if width <= 0 or height <= 0:
        raise GazeOverlayError(f"Invalid video resolution detected for {video_path}: {width}x{height}")
    return width, height


def vector_subtract(a: Dict[str, object], b: Dict[str, object]) -> Tuple[float, float, float]:
    return (float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]), float(a["z"]) - float(b["z"]))


def quaternion_conjugate(q: Dict[str, object]) -> Tuple[float, float, float, float]:
    return (-float(q["x"]), -float(q["y"]), -float(q["z"]), float(q["w"]))


def quaternion_multiply(
    left: Tuple[float, float, float, float], right: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector_by_quaternion(
    vector: Tuple[float, float, float], quaternion: Tuple[float, float, float, float]
) -> Tuple[float, float, float]:
    vx, vy, vz = vector
    vector_quaternion = (vx, vy, vz, 0.0)
    quaternion_inverse = (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3])
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, vector_quaternion),
        quaternion_inverse,
    )
    return rotated[0], rotated[1], rotated[2]


def world_to_camera(sample: Dict[str, object], point: Dict[str, object]) -> Tuple[float, float, float]:
    camera_position = sample.get("cameraPosition")
    camera_rotation = sample.get("cameraRotation")
    if not isinstance(camera_position, dict) or not isinstance(camera_rotation, dict):
        raise GazeOverlayError("Missing cameraPosition or cameraRotation in gaze sample.")

    translated = vector_subtract(point, camera_position)
    inverse_rotation = quaternion_conjugate(camera_rotation)
    return rotate_vector_by_quaternion(translated, inverse_rotation)


def project_camera_point(
    camera_point: Tuple[float, float, float], width: int, height: int, vertical_fov_degrees: float
) -> Optional[Tuple[float, float]]:
    x_value, y_value, z_value = camera_point
    if z_value <= 1e-6:
        return None

    vertical_fov_radians = math.radians(vertical_fov_degrees)
    if vertical_fov_radians <= 0 or vertical_fov_radians >= math.pi:
        return None

    aspect_ratio = width / float(height)
    tan_vertical = math.tan(vertical_fov_radians / 2.0)
    tan_horizontal = tan_vertical * aspect_ratio
    if tan_vertical <= 0 or tan_horizontal <= 0:
        return None

    x_ndc = x_value / (z_value * tan_horizontal)
    y_ndc = y_value / (z_value * tan_vertical)
    if not math.isfinite(x_ndc) or not math.isfinite(y_ndc):
        return None
    if abs(x_ndc) > 1.2 or abs(y_ndc) > 1.2:
        return None

    pixel_x = (x_ndc + 1.0) * 0.5 * width
    pixel_y = (1.0 - y_ndc) * 0.5 * height
    if pixel_x < 0 or pixel_x > width or pixel_y < 0 or pixel_y > height:
        return None
    return pixel_x, pixel_y


def get_point_from_sample(sample: Dict[str, object], point_source: str) -> Tuple[Tuple[float, float], str]:
    eye_gaze = sample.get("eyeGaze")
    if not isinstance(eye_gaze, dict):
        raise GazeOverlayError("Missing eyeGaze object in gaze sample.")

    camera_fov = sample.get("cameraFOV")
    if camera_fov is None:
        raise GazeOverlayError("Missing cameraFOV in gaze sample.")

    if point_source == "cameraHitPoint":
        point = eye_gaze.get("cameraHitPoint")
        if not isinstance(point, dict):
            raise GazeOverlayError("cameraHitPoint is missing in gaze sample.")
        return (float(point["x"]), float(point["y"]), float(point["z"])), "camera"

    if point_source == "gazePoint":
        point = eye_gaze.get("gazePoint")
        if not isinstance(point, dict):
            raise GazeOverlayError("gazePoint is missing in gaze sample.")
        return world_to_camera(sample, point), "camera"

    gaze_point = eye_gaze.get("gazePoint")
    if isinstance(gaze_point, dict):
        return world_to_camera(sample, gaze_point), "camera"

    camera_hit_point = eye_gaze.get("cameraHitPoint")
    if isinstance(camera_hit_point, dict):
        return (float(camera_hit_point["x"]), float(camera_hit_point["y"]), float(camera_hit_point["z"])), "camera"

    raise GazeOverlayError("Neither gazePoint nor cameraHitPoint is available in gaze sample.")


def format_ass_timestamp(seconds_value: float) -> str:
    if seconds_value < 0:
        seconds_value = 0.0
    hours = int(seconds_value // 3600)
    minutes = int((seconds_value % 3600) // 60)
    seconds = int(seconds_value % 60)
    centiseconds = int(round((seconds_value - math.floor(seconds_value)) * 100))
    if centiseconds == 100:
        seconds += 1
        centiseconds = 0
    if seconds == 60:
        minutes += 1
        seconds = 0
    if minutes == 60:
        hours += 1
        minutes = 0
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def rgb_to_ass_bgr(color_value: str) -> str:
    cleaned = color_value.strip().lstrip("#")
    if len(cleaned) != 6 or any(char not in "0123456789abcdefABCDEF" for char in cleaned):
        raise GazeOverlayError(f"Invalid color value, expected RRGGBB: {color_value}")
    red = cleaned[0:2]
    green = cleaned[2:4]
    blue = cleaned[4:6]
    return f"&H{blue}{green}{red}&"


def build_ass_overlay(
    samples: Sequence[Dict[str, object]],
    width: int,
    height: int,
    point_source: str,
    start_time: Optional[datetime],
    time_offset_seconds: float,
    marker_size: int,
    pixel_offset_x: float,
    pixel_offset_y: float,
    marker_color: str,
    outline_color: str,
    min_segment_seconds: float,
    max_segment_seconds: float,
) -> Tuple[str, int]:
    marker_primary = rgb_to_ass_bgr(marker_color)
    marker_outline = rgb_to_ass_bgr(outline_color)

    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        f"Style: GazeMarker,Arial,{marker_size},{marker_primary},{marker_primary},{marker_outline},&H00000000&,1,0,0,0,100,100,0,0,1,3,0,5,0,0,0,1",
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]

    event_lines: List[str] = []
    valid_count = 0
    sample_times: List[datetime] = []
    for sample in samples:
        timestamp_value = sample.get("timestamp")
        if isinstance(timestamp_value, str) and timestamp_value.strip():
            sample_times.append(parse_timestamp(timestamp_value, "multimodal JSON sample"))
        else:
            sample_times.append(sample_times[-1] if sample_times else datetime.fromtimestamp(0))

    inferred_start = start_time or sample_times[0]

    for index, sample in enumerate(samples):
        eye_gaze = sample.get("eyeGaze")
        if not isinstance(eye_gaze, dict):
            continue
        if eye_gaze.get("isEyeOpen") is False:
            continue

        try:
            camera_point, _space = get_point_from_sample(sample, point_source)
            camera_fov = float(sample["cameraFOV"])
        except (GazeOverlayError, KeyError, TypeError, ValueError):
            continue

        projected = project_camera_point(camera_point, width, height, camera_fov)
        if projected is None:
            continue

        current_time = (sample_times[index] - inferred_start).total_seconds() + time_offset_seconds
        if current_time < 0:
            continue

        if index + 1 < len(sample_times):
            next_time = (sample_times[index + 1] - inferred_start).total_seconds() + time_offset_seconds
            duration = max(min_segment_seconds, min(max_segment_seconds, next_time - current_time))
        else:
            duration = min_segment_seconds
        if duration <= 0:
            duration = min_segment_seconds

        start_text = format_ass_timestamp(current_time)
        end_text = format_ass_timestamp(current_time + duration)
        x_value = int(round(projected[0] + pixel_offset_x))
        y_value = int(round(projected[1] + pixel_offset_y))
        text = f"{{\\pos({x_value},{y_value})\\an5}}●"
        event_lines.append(
            f"Dialogue: 0,{start_text},{end_text},GazeMarker,,0,0,0,,{text}"
        )
        valid_count += 1

    if not event_lines:
        raise GazeOverlayError("No valid gaze overlay points could be projected onto the video.")

    return "\n".join(header + event_lines) + "\n", valid_count


def render_overlay_video(
    ffmpeg_path: str,
    input_video: Path,
    ass_path: Path,
    output_video: Path,
    overwrite: bool,
) -> None:
    output_video.parent.mkdir(parents=True, exist_ok=True)
    ass_filter_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_video),
        "-vf",
        f"ass='{ass_filter_path}'",
        "-c:a",
        "copy",
        "-y" if overwrite else "-n",
        str(output_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "Unknown ffmpeg error."
        raise GazeOverlayError(f"ffmpeg failed to render the overlay video: {stderr}")
    if not output_video.exists():
        raise GazeOverlayError(f"ffmpeg did not produce the output video: {output_video}")


def main() -> int:
    args = parse_args()

    try:
        ffmpeg_path = ensure_tool_available("ffmpeg")
        ffprobe_path = ensure_tool_available("ffprobe")

        input_video = Path(args.input_video).resolve()
        input_json = Path(args.input_json).resolve()
        output_video = Path(args.output_video).resolve()
        metadata_json = Path(args.metadata_json).resolve() if args.metadata_json else None

        ensure_file_exists(input_video, "input video")
        ensure_file_exists(input_json, "input JSON")
        if output_video.exists() and not args.overwrite:
            raise GazeOverlayError(
                f"Output video already exists. Use --overwrite to replace it: {output_video}"
            )

        width, height = run_ffprobe(ffprobe_path, input_video)
        samples = load_multimodal_samples(input_json)
        start_time = load_metadata_start_time(metadata_json)

        ass_text, valid_count = build_ass_overlay(
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
            ass_path.write_text(ass_text, encoding="utf-8")
            render_overlay_video(ffmpeg_path, input_video, ass_path, output_video, args.overwrite)
        else:
            with tempfile.TemporaryDirectory(prefix="gaze_overlay_") as temp_dir:
                ass_path = Path(temp_dir) / "gaze_overlay.ass"
                ass_path.write_text(ass_text, encoding="utf-8")
                render_overlay_video(ffmpeg_path, input_video, ass_path, output_video, args.overwrite)

        print(f"Output video written to: {output_video}")
        print(f"Projected gaze markers: {valid_count}")
        return 0
    except GazeOverlayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
