#!/usr/bin/env python3
"""Build a 2D point-grounding manifest from 3D referent anchors.

This script prepares the second experiment. It does not call a model. It reads
the current v3 match-eval files, scene API input CSVs, anchor tables, videos,
and multimodal JSON files; samples K candidate frames per event; projects GT
referent anchor points into those frames; and writes a manifest CSV.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENES = (
    "scene1",
    "scene2",
    "scene3",
    "scene4_room1",
    "scene4_room2",
    "scene4_room3",
    "scene4_room4",
    "scene5",
)

MANIFEST_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "referent_name",
    "referent_index",
    "panel_id",
    "panel_index",
    "frame_time",
    "json_sample_time",
    "video_frame_time",
    "requested_video_frame_time",
    "video_duration",
    "video_time_offset_seconds",
    "video_time_offset_source",
    "frame_duplicate_of",
    "panel_selection_score",
    "panel_selection_reason",
    "frame_path",
    "video_path",
    "json_path",
    "image_width",
    "image_height",
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "gt_u_norm",
    "gt_v_norm",
    "gt_x",
    "gt_y",
    "projection_valid",
    "gaze_u_norm",
    "gaze_v_norm",
    "gaze_x",
    "gaze_y",
    "gaze_projection_valid",
    "gaze_anchor_distance_px",
    "gaze_source",
    "evidence_rank",
    "evidence_acceptable",
    "frame_extracted",
    "status",
    "status_detail",
)


class Exam2Error(Exception):
    """Raised when exam2 manifest construction cannot continue."""


@dataclass(frozen=True)
class ScenePaths:
    scene: str
    api_input_csv: Path
    match_eval_csv: Path
    anchor_csv: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D point-grounding manifest and candidate frames.")
    parser.add_argument("--repo_root", default=".", help="Repository root. Default: current directory.")
    parser.add_argument(
        "--eval_dir",
        default=None,
        help="Directory containing *_match_eval.csv. Default: <repo_root>/data/match_eval_qwen3vl30b_mention_first_v3, falling back to <repo_root>/new/...",
    )
    parser.add_argument("--output_dir", default="exam2/outputs", help="Output directory. Default: exam2/outputs")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES), help="Scenes to process. Default: all scenes.")
    parser.add_argument("--start_index", type=int, default=0, help="First row_index per scene. Default: 0.")
    parser.add_argument("--limit", type=int, help="Maximum rows per scene for smoke tests.")
    parser.add_argument("--panels", type=int, default=5, help="Number of candidate frames per event. Default: 5.")
    parser.add_argument("--ffmpeg_path", default="ffmpeg", help="ffmpeg executable. Default: ffmpeg")
    parser.add_argument("--ffprobe_path", default="ffprobe", help="ffprobe executable. Default: ffprobe")
    parser.add_argument(
        "--video_time_offset_seconds",
        type=float,
        default=0.0,
        help="Offset added when extracting video frames. With --auto_video_time_offset, this is an extra adjustment. Default: 0.",
    )
    parser.add_argument(
        "--auto_video_time_offset",
        action="store_true",
        help="Estimate per-sample offset from metadata/video start time to the first multimodal timestamp.",
    )
    parser.add_argument(
        "--auto_video_time_offset_source",
        choices=("metadata", "video_filename", "hybrid"),
        default="metadata",
        help="Auto offset source. metadata keeps the old behavior; hybrid falls back toward video filename when metadata and video time differ too much. Default: metadata.",
    )
    parser.add_argument(
        "--hybrid_offset_threshold_seconds",
        type=float,
        default=1.0,
        help="Hybrid source threshold for metadata/video start disagreement. Default: 1.0.",
    )
    parser.add_argument(
        "--hybrid_video_time_bias_seconds",
        type=float,
        default=0.5,
        help="When hybrid falls back to video filename, shift the video start by this amount toward metadata. Default: 0.5.",
    )
    parser.add_argument("--no_extract_frames", action="store_true", help="Do not call ffmpeg; still build projections if metadata is available.")
    parser.add_argument("--overwrite_frames", action="store_true", help="Re-extract frames even when image files already exist.")
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite path prefixes in video/json paths. Can be repeated.",
    )
    parser.add_argument("--image_width", type=int, help="Override image width when ffprobe is unavailable.")
    parser.add_argument("--image_height", type=int, help="Override image height when ffprobe is unavailable.")
    parser.add_argument(
        "--min_panel_time_gap_seconds",
        type=float,
        default=0.25,
        help="Drop candidate panels closer than this in requested video time. Default: 0.25.",
    )
    parser.add_argument(
        "--keep_duplicate_frames",
        action="store_true",
        help="Keep panels whose extracted image bytes duplicate an earlier panel. Default: drop duplicates.",
    )
    parser.add_argument(
        "--acceptable_top_k",
        type=int,
        default=2,
        help="Mark the top-K gaze-nearest panels per referent as weak acceptable evidence. Default: 2.",
    )
    parser.add_argument(
        "--sample_inner_margin_ratio",
        type=float,
        default=0.0,
        help="Trim this ratio from both ends of the event window before sampling panels. Default: 0.0.",
    )
    parser.add_argument(
        "--panel_selection_strategy",
        choices=("uniform", "evidence"),
        default="uniform",
        help="uniform samples evenly; evidence ranks dense candidates by multimodal cue quality. Default: uniform.",
    )
    parser.add_argument(
        "--candidate_step_seconds",
        type=float,
        default=0.5,
        help="Dense candidate step for evidence panel selection. Default: 0.5.",
    )
    parser.add_argument(
        "--evidence_segment_duration_seconds",
        type=float,
        default=0.5,
        help="Minimum temporal separation between selected evidence panels. Default: 0.5.",
    )
    parser.add_argument(
        "--gaze_stability_window_seconds",
        type=float,
        default=0.3,
        help="Neighbor window for gaze stability scoring. Default: 0.3.",
    )
    return parser.parse_args()


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise Exam2Error(f"Missing CSV: {path}")
    text = path.read_text(encoding="utf-8-sig")
    delimiter = detect_delimiter(text)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if reader.fieldnames is None:
        raise Exam2Error(f"CSV has no header: {path}")
    return [dict(row) for row in reader]


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_float(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def split_names(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [part.strip() for part in text.replace(";", ",").replace("|", ",").split(",") if part.strip()]


def parse_rewrites(items: Sequence[str]) -> List[Tuple[str, str]]:
    rewrites: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise Exam2Error(f"Invalid --path-rewrite value, expected OLD=NEW: {item}")
        old, new = item.split("=", 1)
        if not old:
            raise Exam2Error(f"Invalid --path-rewrite with empty OLD: {item}")
        rewrites.append((old, new))
    return rewrites


def rewrite_path_text(path_text: str, rewrites: Sequence[Tuple[str, str]]) -> str:
    result = path_text
    for old, new in rewrites:
        if result.startswith(old):
            return new + result[len(old) :]
    return result


def path_from_text(path_text: str, rewrites: Sequence[Tuple[str, str]]) -> Path:
    rewritten = rewrite_path_text(normalize_text(path_text), rewrites)
    return Path(rewritten)


def scene_paths(repo_root: Path, eval_dir: Path, scene: str) -> ScenePaths:
    return ScenePaths(
        scene=scene,
        api_input_csv=repo_root / "data" / f"{scene}_api_input.csv",
        match_eval_csv=eval_dir / f"{scene}_match_eval.csv",
        anchor_csv=repo_root / "data" / f"{scene}_anchor_table.tsv",
    )


def resolve_eval_dir(repo_root: Path, requested: Optional[str]) -> Path:
    if requested:
        return Path(requested)
    primary = repo_root / "data" / "match_eval_qwen3vl30b_mention_first_v3"
    if primary.exists():
        return primary
    fallback = repo_root / "new" / "match_eval_qwen3vl30b_mention_first_v3"
    return fallback


def load_anchor_table(path: Path) -> Dict[str, Tuple[float, float, float]]:
    rows = read_csv_rows(path)
    anchors: Dict[str, Tuple[float, float, float]] = {}
    for row in rows:
        name = normalize_text(row.get("object_name") or row.get("物体名称"))
        if not name:
            continue
        x = parse_float(row.get("x_world") or row.get("location_x") or row.get("x"))
        y = parse_float(row.get("y_world") or row.get("location_y") or row.get("y"))
        z = parse_float(row.get("z_world") or row.get("location_z") or row.get("z"))
        if x is None or y is None or z is None:
            continue
        anchors[name] = (x, y, z)
    if not anchors:
        raise Exam2Error(f"No anchors loaded from: {path}")
    return anchors


def normalize_iso_timestamp(value: Any) -> str:
    text = normalize_text(value)
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


def parse_sample_datetime(value: Any) -> Optional[dt.datetime]:
    text = normalize_iso_timestamp(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:
        return None


def repair_json_array_text(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("["):
        # Some captured multimodal files are array-like logs where adjacent
        # objects are written as "}\n{" instead of "},\n{".
        stripped = stripped.replace("}\n{", "},\n{")
        stripped = stripped.replace("}\r\n{", "},\r\n{")
        if stripped.count("[") == stripped.count("]") + 1 and stripped.endswith("}"):
            stripped += "\n]"
    return stripped


def load_multimodal_samples(path: Path) -> List[Mapping[str, Any]]:
    raw_text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = json.loads(repair_json_array_text(raw_text))
    if not isinstance(payload, list):
        raise Exam2Error(f"Multimodal JSON is not a list: {path}")
    return [item for item in payload if isinstance(item, Mapping)]


def first_sample_datetime(samples: Sequence[Mapping[str, Any]]) -> Optional[dt.datetime]:
    sample_datetimes = [parse_sample_datetime(sample.get("timestamp")) for sample in samples]
    valid_datetimes = [sample_dt for sample_dt in sample_datetimes if sample_dt is not None]
    if not valid_datetimes:
        return None
    return min(valid_datetimes)


def read_metadata_datetime(media_dir: Path) -> Tuple[Optional[dt.datetime], str]:
    metadata_paths = sorted(media_dir.glob("metadata_*.json"))
    if not metadata_paths:
        return None, "auto_unavailable:no_metadata_json"
    errors: List[str] = []
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"{metadata_path.name}:{type(exc).__name__}")
            continue
        if not isinstance(payload, Mapping):
            errors.append(f"{metadata_path.name}:not_object")
            continue
        metadata_dt = parse_sample_datetime(payload.get("timestamp"))
        if metadata_dt is None:
            errors.append(f"{metadata_path.name}:missing_timestamp")
            continue
        return metadata_dt, f"metadata:{metadata_path.name}"
    return None, "auto_unavailable:" + ";".join(errors[:3])


def read_video_filename_datetime(video_path: Path, reference_dt: Optional[dt.datetime]) -> Tuple[Optional[dt.datetime], str]:
    import re

    match = re.search(
        r"ScreenRecord_(\d{4})-(\d{2})-(\d{2})-(\d{2})_(\d{2})_(\d{2})",
        video_path.name,
    )
    if not match:
        return None, "auto_unavailable:no_screenrecord_timestamp"
    year, month, day, hour, minute, second = [int(part) for part in match.groups()]
    parsed = dt.datetime(year, month, day, hour, minute, second)
    if reference_dt is not None and reference_dt.tzinfo is not None:
        parsed = parsed.replace(tzinfo=reference_dt.tzinfo)
    return parsed, f"video_filename:{video_path.name}"


def choose_auto_start_datetime(
    metadata_dt: Optional[dt.datetime],
    metadata_source: str,
    video_dt: Optional[dt.datetime],
    video_source: str,
    source_strategy: str,
    threshold_seconds: float,
    video_bias_seconds: float,
) -> Tuple[Optional[dt.datetime], str]:
    if source_strategy == "metadata":
        return metadata_dt, metadata_source
    if source_strategy == "video_filename":
        return video_dt, video_source
    if source_strategy != "hybrid":
        return None, f"auto_unavailable:unknown_source:{source_strategy}"
    if metadata_dt is None and video_dt is None:
        return None, f"auto_unavailable:no_metadata_or_video_time:{metadata_source};{video_source}"
    if metadata_dt is None:
        return video_dt, f"hybrid:fallback_video_no_metadata:{video_source}"
    if video_dt is None:
        return metadata_dt, f"hybrid:fallback_metadata_no_video_time:{metadata_source}"
    delta = (metadata_dt - video_dt).total_seconds()
    if abs(delta) <= threshold_seconds:
        return metadata_dt, f"hybrid:metadata_within_{threshold_seconds:.3f}s:{metadata_source};{video_source};delta={delta:.3f}"
    if delta < 0:
        adjusted = video_dt - dt.timedelta(seconds=video_bias_seconds)
        return adjusted, f"hybrid:metadata_early_use_video_minus_{video_bias_seconds:.3f}s:{metadata_source};{video_source};delta={delta:.3f}"
    adjusted = video_dt + dt.timedelta(seconds=video_bias_seconds)
    return adjusted, f"hybrid:metadata_late_use_video_plus_{video_bias_seconds:.3f}s:{metadata_source};{video_source};delta={delta:.3f}"


def effective_video_time_offset(
    json_path: Path,
    video_path: Optional[Path],
    samples: Sequence[Mapping[str, Any]],
    manual_adjustment: float,
    auto_enabled: bool,
    source_strategy: str = "metadata",
    threshold_seconds: float = 1.0,
    video_bias_seconds: float = 0.5,
) -> Tuple[float, str]:
    if not auto_enabled:
        return manual_adjustment, "manual"
    metadata_dt, metadata_source = read_metadata_datetime(json_path.parent)
    first_dt = first_sample_datetime(samples)
    if first_dt is None:
        return manual_adjustment, "auto_unavailable:no_multimodal_timestamp+manual_adjustment"
    video_dt, video_source = read_video_filename_datetime(video_path, first_dt) if video_path is not None else (None, "auto_unavailable:no_video_path")
    start_dt, start_source = choose_auto_start_datetime(
        metadata_dt=metadata_dt,
        metadata_source=metadata_source,
        video_dt=video_dt,
        video_source=video_source,
        source_strategy=source_strategy,
        threshold_seconds=threshold_seconds,
        video_bias_seconds=video_bias_seconds,
    )
    if start_dt is None:
        return manual_adjustment, start_source + "+manual_adjustment"
    auto_offset = (first_dt - start_dt).total_seconds()
    return auto_offset + manual_adjustment, f"auto:{start_source}+manual_adjustment:{manual_adjustment:.3f}"


def collect_timed_samples(samples: Sequence[Mapping[str, Any]]) -> List[Tuple[float, Mapping[str, Any]]]:
    dated: List[Tuple[dt.datetime, Mapping[str, Any]]] = []
    for sample in samples:
        sample_dt = parse_sample_datetime(sample.get("timestamp"))
        if sample_dt is not None:
            dated.append((sample_dt, sample))
    if not dated:
        return []
    dated.sort(key=lambda item: item[0])
    base = dated[0][0]
    return [((sample_dt - base).total_seconds(), sample) for sample_dt, sample in dated]


def sample_times(t_start: float, t_end: float, panels: int) -> List[float]:
    if panels <= 1:
        return [(t_start + t_end) / 2.0]
    if t_end <= t_start:
        return [t_start for _ in range(panels)]
    return [t_start + (t_end - t_start) * idx / (panels - 1) for idx in range(panels)]


def apply_inner_margin(t_start: float, t_end: float, ratio: float) -> Tuple[float, float]:
    if ratio <= 0.0 or t_end <= t_start:
        return t_start, t_end
    clamped_ratio = min(0.45, max(0.0, ratio))
    duration = t_end - t_start
    return t_start + duration * clamped_ratio, t_end - duration * clamped_ratio


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def nearest_sample(timed_samples: Sequence[Tuple[float, Mapping[str, Any]]], target_time: float) -> Tuple[float, Mapping[str, Any]]:
    return min(timed_samples, key=lambda item: abs(item[0] - target_time))


def point3(mapping: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(mapping, Mapping):
        return None
    x = parse_float(mapping.get("x"))
    y = parse_float(mapping.get("y"))
    z = parse_float(mapping.get("z"))
    if x is None or y is None or z is None:
        return None
    return x, y, z


def quat4(mapping: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(mapping, Mapping):
        return None
    x = parse_float(mapping.get("x"))
    y = parse_float(mapping.get("y"))
    z = parse_float(mapping.get("z"))
    w = parse_float(mapping.get("w"))
    if x is None or y is None or z is None or w is None:
        return None
    return x, y, z, w


def quaternion_multiply(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quaternion_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return -q[0], -q[1], -q[2], q[3]


def rotate_vector_by_quaternion(
    vector: Tuple[float, float, float],
    quaternion: Tuple[float, float, float, float],
) -> Tuple[float, float, float]:
    vx, vy, vz = vector
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (vx, vy, vz, 0.0)),
        quaternion_conjugate(quaternion),
    )
    return rotated[0], rotated[1], rotated[2]


def world_to_camera(sample: Mapping[str, Any], point: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
    camera_position = point3(sample.get("cameraPosition"))
    camera_rotation = quat4(sample.get("cameraRotation"))
    if camera_position is None or camera_rotation is None:
        return None
    translated = (
        point[0] - camera_position[0],
        point[1] - camera_position[1],
        point[2] - camera_position[2],
    )
    return rotate_vector_by_quaternion(translated, quaternion_conjugate(camera_rotation))


def project_camera_point(
    camera_point: Tuple[float, float, float],
    width: int,
    height: int,
    vertical_fov_degrees: float,
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
    u_norm = (x_ndc + 1.0) * 0.5
    v_norm = (1.0 - y_ndc) * 0.5
    if u_norm < 0.0 or u_norm > 1.0 or v_norm < 0.0 or v_norm > 1.0:
        return None
    return u_norm, v_norm


def project_world_point(
    sample: Mapping[str, Any],
    point: Tuple[float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    camera_point = world_to_camera(sample, point)
    if camera_point is None:
        return None
    fov = parse_float(sample.get("cameraFOV"))
    if fov is None:
        return None
    return project_camera_point(camera_point, width, height, fov)


def is_nonzero_point(point: Tuple[float, float, float]) -> bool:
    return any(abs(value) > 1e-6 for value in point)


def gaze_world_point(sample: Mapping[str, Any]) -> Tuple[Optional[Tuple[float, float, float]], str]:
    eye_gaze = sample.get("eyeGaze")
    if not isinstance(eye_gaze, Mapping):
        return None, ""
    if eye_gaze.get("isEyeOpen") is False:
        return None, ""
    for key in ("gazePoint", "cameraHitPoint"):
        point = point3(eye_gaze.get(key))
        if point is not None and is_nonzero_point(point):
            return point, key
    return None, ""


def gaze_projection(sample: Mapping[str, Any], width: int, height: int) -> Tuple[Optional[Tuple[float, float]], str]:
    gaze_point, gaze_source = gaze_world_point(sample)
    if gaze_point is None:
        return None, ""
    projected = project_world_point(sample, gaze_point, width, height)
    return projected, gaze_source if projected is not None else ""


def camera_position_distance(left_sample: Mapping[str, Any], right_sample: Mapping[str, Any]) -> Optional[float]:
    left = point3(left_sample.get("cameraPosition"))
    right = point3(right_sample.get("cameraPosition"))
    if left is None or right is None:
        return None
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2 + (left[2] - right[2]) ** 2)


def dense_candidate_times(sample_start: float, sample_end: float, step_seconds: float) -> List[float]:
    if sample_end <= sample_start:
        return [sample_start]
    step = max(0.05, step_seconds)
    times: List[float] = []
    current = sample_start
    while current <= sample_end + 1e-6:
        times.append(current)
        current += step
    midpoint = (sample_start + sample_end) / 2.0
    if all(abs(midpoint - item) > step * 0.25 for item in times):
        times.append(midpoint)
    return sorted(times)


def evidence_candidate_score(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    sample_time: float,
    sample: Mapping[str, Any],
    width: int,
    height: int,
    stability_window_seconds: float,
) -> Tuple[float, str]:
    projected, gaze_source = gaze_projection(sample, width, height)
    if projected is None:
        return 0.0, "no_projected_gaze"

    score = 1.0
    reasons = [f"gaze:{gaze_source}"]
    neighbor_distances: List[float] = []
    camera_distances: List[float] = []
    window = max(0.05, stability_window_seconds)
    for neighbor_time, neighbor in timed_samples:
        if neighbor is sample:
            continue
        delta = abs(neighbor_time - sample_time)
        if delta <= 1e-6 or delta > window:
            continue
        neighbor_projected, _source = gaze_projection(neighbor, width, height)
        if neighbor_projected is not None:
            neighbor_distances.append(
                math.hypot((neighbor_projected[0] - projected[0]) * width, (neighbor_projected[1] - projected[1]) * height)
            )
        camera_distance = camera_position_distance(sample, neighbor)
        if camera_distance is not None:
            camera_distances.append(camera_distance)

    if neighbor_distances:
        avg_gaze_motion = sum(neighbor_distances) / len(neighbor_distances)
        stability = max(0.0, 1.0 - avg_gaze_motion / 280.0)
        score += 0.9 * stability
        reasons.append(f"gaze_stability:{stability:.2f}")
    else:
        score += 0.15
        reasons.append("gaze_stability:unknown")

    if camera_distances:
        avg_camera_motion = sum(camera_distances) / len(camera_distances)
        camera_stability = max(0.0, 1.0 - avg_camera_motion / 0.35)
        score += 0.25 * camera_stability
        reasons.append(f"camera_stability:{camera_stability:.2f}")
    else:
        reasons.append("camera_stability:unknown")

    return score, ";".join(reasons)


def select_panel_targets(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    video_start: float,
    video_end: float,
    video_time_offset: float,
    panels: int,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> List[Tuple[float, float, float, str]]:
    if args.panel_selection_strategy == "uniform":
        return [(video_time, 0.0, video_time - video_time_offset, "uniform") for video_time in sample_times(video_start, video_end, panels)]

    sample_start = max(timed_samples[0][0], video_start - video_time_offset)
    sample_end = min(timed_samples[-1][0], video_end - video_time_offset)
    candidates: List[Tuple[float, float, float, str]] = []
    for target_sample_time in dense_candidate_times(sample_start, sample_end, float(args.candidate_step_seconds)):
        sample_time, sample = nearest_sample(timed_samples, target_sample_time)
        video_time = sample_time + video_time_offset
        if video_time < video_start - 1e-6 or video_time > video_end + 1e-6:
            continue
        score, reason = evidence_candidate_score(
            timed_samples,
            sample_time,
            sample,
            width,
            height,
            float(args.gaze_stability_window_seconds),
        )
        candidates.append((video_time, score, sample_time, reason))

    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected: List[Tuple[float, float, float, str]] = []
    min_gap = max(float(args.min_panel_time_gap_seconds), float(args.evidence_segment_duration_seconds))
    for video_time, score, sample_time, reason in candidates:
        if score <= 0.0:
            continue
        if any(abs(video_time - used_video_time) < min_gap for used_video_time, _score, _sample_time, _reason in selected):
            continue
        selected.append((video_time, score, sample_time, reason))
        if len(selected) >= panels:
            break
    if not selected:
        return [(video_time, 0.0, video_time - video_time_offset, "uniform_fallback:no_positive_evidence") for video_time in sample_times(video_start, video_end, panels)]
    selected.sort(key=lambda item: item[0])
    return selected


def annotate_evidence_rows(rows: Sequence[Dict[str, Any]], acceptable_top_k: int) -> None:
    for row in rows:
        row["evidence_rank"] = ""
        row["evidence_acceptable"] = "False"
    if acceptable_top_k <= 0:
        return
    candidates: List[Tuple[float, int, Dict[str, Any]]] = []
    for row in rows:
        if normalize_text(row.get("projection_valid")) != "True":
            continue
        if normalize_text(row.get("gaze_projection_valid")) != "True":
            continue
        distance_px = parse_float(row.get("gaze_anchor_distance_px"))
        panel_index = parse_int(row.get("panel_index")) or 999
        if distance_px is None:
            continue
        candidates.append((distance_px, panel_index, row))
    candidates.sort(key=lambda item: (item[0], item[1]))
    for rank, (_distance_px, _panel_index, row) in enumerate(candidates, start=1):
        row["evidence_rank"] = rank
        row["evidence_acceptable"] = str(rank <= acceptable_top_k)


def ffprobe_size(ffprobe_path: str, video_path: Path) -> Tuple[int, int]:
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
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise Exam2Error(f"ffprobe failed for {video_path}: {completed.stderr.strip()}")
    payload = json.loads(completed.stdout)
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not streams:
        raise Exam2Error(f"ffprobe returned no video stream for {video_path}")
    width = int(streams[0]["width"])
    height = int(streams[0]["height"])
    return width, height


def ffprobe_duration(ffprobe_path: str, video_path: Path) -> Optional[float]:
    command = [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    if not text:
        return None
    try:
        duration = float(text)
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0.0 else None


def file_digest(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_frame(ffmpeg_path: str, video_path: Path, time_seconds: float, output_path: Path, overwrite: bool) -> bool:
    if output_path.exists() and not overwrite:
        return True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-ss",
        f"{time_seconds:.3f}",
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


def build_match_eval_by_row(path: Path) -> Dict[int, Dict[str, str]]:
    rows = read_csv_rows(path)
    result: Dict[int, Dict[str, str]] = {}
    for row in rows:
        row_index = parse_int(row.get("row_index"))
        if row_index is None:
            continue
        result[row_index] = row
    return result


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def choose_rows(api_rows: List[Dict[str, str]], start_index: int, limit: Optional[int]) -> Iterable[Tuple[int, Dict[str, str]]]:
    end_index = len(api_rows) if limit is None else min(len(api_rows), start_index + limit)
    for row_index in range(start_index, end_index):
        yield row_index, api_rows[row_index]


def make_empty_manifest_row(
    scene: str,
    row_index: int,
    api_row: Mapping[str, str],
    referent_name: str,
    referent_index: int,
    status: str,
    detail: str,
) -> Dict[str, Any]:
    return {
        "scene": scene,
        "row_index": row_index,
        "event_id": normalize_text(api_row.get("event_id")),
        "instruction": normalize_text(api_row.get("instruction_text") or api_row.get("target_description")),
        "referent_name": referent_name,
        "referent_index": referent_index,
        "panel_id": "",
        "panel_index": "",
        "frame_time": "",
        "json_sample_time": "",
        "video_frame_time": "",
        "requested_video_frame_time": "",
        "video_duration": "",
        "video_time_offset_seconds": "",
        "video_time_offset_source": "",
        "frame_duplicate_of": "",
        "panel_selection_score": "",
        "panel_selection_reason": "",
        "frame_path": "",
        "video_path": normalize_text(api_row.get("video_path")),
        "json_path": normalize_text(api_row.get("json_path")),
        "image_width": "",
        "image_height": "",
        "anchor_x": "",
        "anchor_y": "",
        "anchor_z": "",
        "gt_u_norm": "",
        "gt_v_norm": "",
        "gt_x": "",
        "gt_y": "",
        "projection_valid": "False",
        "gaze_u_norm": "",
        "gaze_v_norm": "",
        "gaze_x": "",
        "gaze_y": "",
        "gaze_projection_valid": "False",
        "gaze_anchor_distance_px": "",
        "gaze_source": "",
        "evidence_rank": "",
        "evidence_acceptable": "False",
        "frame_extracted": "False",
        "status": status,
        "status_detail": detail,
    }


def process_scene(
    paths: ScenePaths,
    output_dir: Path,
    args: argparse.Namespace,
    rewrites: Sequence[Tuple[str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    api_rows = read_csv_rows(paths.api_input_csv)
    match_by_row = build_match_eval_by_row(paths.match_eval_csv)
    anchors = load_anchor_table(paths.anchor_csv)
    frame_root = output_dir / "frames" / paths.scene

    manifest_rows: List[Dict[str, Any]] = []
    summary = {
        "scene": paths.scene,
        "api_rows_seen": 0,
        "referent_count": 0,
        "manifest_rows": 0,
        "valid_projection_rows": 0,
        "rows_with_valid_projection": 0,
        "missing_gt_rows": 0,
        "missing_anchor_referents": 0,
        "missing_media_rows": 0,
        "auto_offset_rows": 0,
        "auto_offset_unavailable_rows": 0,
        "duplicate_frame_panels_dropped": 0,
        "close_time_panels_dropped": 0,
        "rows_without_panels": 0,
        "evidence_acceptable_rows": 0,
        "gaze_projection_rows": 0,
    }

    for row_index, api_row in choose_rows(api_rows, args.start_index, args.limit):
        summary["api_rows_seen"] += 1
        eval_row = match_by_row.get(row_index)
        gt_names = split_names(eval_row.get("gt_referents_mapped") if eval_row else "")
        if not gt_names:
            summary["missing_gt_rows"] += 1
            manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, "", 0, "missing_gt_referents", "No mapped GT referents found."))
            continue

        video_path = path_from_text(api_row.get("video_path", ""), rewrites)
        json_path = path_from_text(api_row.get("json_path", ""), rewrites)
        if not video_path.exists() or not json_path.exists():
            summary["missing_media_rows"] += 1
            detail = f"video_exists={video_path.exists()} json_exists={json_path.exists()}"
            for referent_index, referent_name in enumerate(gt_names, start=1):
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "missing_media", detail))
            continue

        try:
            width, height = (
                (args.image_width, args.image_height)
                if args.image_width and args.image_height
                else ffprobe_size(args.ffprobe_path, video_path)
            )
            video_duration = ffprobe_duration(args.ffprobe_path, video_path)
        except Exception as exc:
            detail = f"image_size_error:{type(exc).__name__}:{exc}"
            for referent_index, referent_name in enumerate(gt_names, start=1):
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "image_size_error", detail))
            continue

        try:
            multimodal_samples = load_multimodal_samples(json_path)
            timed_samples = collect_timed_samples(multimodal_samples)
        except Exception as exc:
            detail = f"json_error:{type(exc).__name__}:{exc}"
            for referent_index, referent_name in enumerate(gt_names, start=1):
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "json_error", detail))
            continue
        if not timed_samples:
            for referent_index, referent_name in enumerate(gt_names, start=1):
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "no_timed_samples", "No timestamped multimodal samples."))
            continue

        video_time_offset, video_time_offset_source = effective_video_time_offset(
            json_path=json_path,
            video_path=video_path,
            samples=multimodal_samples,
            manual_adjustment=float(args.video_time_offset_seconds),
            auto_enabled=bool(args.auto_video_time_offset),
            source_strategy=args.auto_video_time_offset_source,
            threshold_seconds=float(args.hybrid_offset_threshold_seconds),
            video_bias_seconds=float(args.hybrid_video_time_bias_seconds),
        )
        if args.auto_video_time_offset:
            if video_time_offset_source.startswith("auto:"):
                summary["auto_offset_rows"] += 1
            else:
                summary["auto_offset_unavailable_rows"] += 1

        t_start = parse_float(api_row.get("t_start"))
        t_end = parse_float(api_row.get("t_end"))
        if t_start is None:
            t_start = timed_samples[0][0]
        if t_end is None:
            t_end = timed_samples[-1][0]
        raw_video_start = max(0.0, t_start + video_time_offset)
        raw_video_end = max(0.0, t_end + video_time_offset)
        if video_duration is not None:
            max_frame_time = max(0.0, video_duration - 0.05)
            video_start = clamp(raw_video_start, 0.0, max_frame_time)
            video_end = clamp(raw_video_end, 0.0, max_frame_time)
            if video_end < video_start:
                video_start, video_end = video_end, video_start
        else:
            video_start, video_end = raw_video_start, raw_video_end
        video_start, video_end = apply_inner_margin(video_start, video_end, float(args.sample_inner_margin_ratio))
        panel_targets = select_panel_targets(timed_samples, video_start, video_end, video_time_offset, args.panels, width, height, args)
        panels: List[Tuple[str, int, float, float, float, Mapping[str, Any], Path, bool, str, float, str]] = []
        used_video_times: List[float] = []
        seen_frame_digests: Dict[str, str] = {}
        for raw_panel_index, (requested_video_time, selection_score, target_sample_time, selection_reason) in enumerate(panel_targets, start=1):
            if any(abs(requested_video_time - used) < float(args.min_panel_time_gap_seconds) for used in used_video_times):
                summary["close_time_panels_dropped"] += 1
                continue
            sample_time, sample = nearest_sample(timed_samples, target_sample_time)
            panel_id = f"P{len(panels) + 1}"
            video_time = requested_video_time
            frame_path = frame_root / f"row_{row_index}" / f"{panel_id}.jpg"
            extracted = False if args.no_extract_frames else extract_frame(args.ffmpeg_path, video_path, video_time, frame_path, args.overwrite_frames)
            duplicate_of = ""
            if extracted and not args.keep_duplicate_frames:
                digest = file_digest(frame_path)
                if digest and digest in seen_frame_digests:
                    duplicate_of = seen_frame_digests[digest]
                    summary["duplicate_frame_panels_dropped"] += 1
                    continue
                if digest:
                    seen_frame_digests[digest] = panel_id
            used_video_times.append(requested_video_time)
            panels.append((panel_id, len(panels) + 1, sample_time, video_time, requested_video_time, sample, frame_path, extracted, duplicate_of, selection_score, selection_reason))

        if not panels:
            summary["rows_without_panels"] += 1
            for referent_index, referent_name in enumerate(gt_names, start=1):
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "no_valid_panels", "No usable non-duplicate video panels were extracted."))
            continue

        row_has_valid_projection = False
        for referent_index, referent_name in enumerate(gt_names, start=1):
            summary["referent_count"] += 1
            anchor = anchors.get(referent_name)
            if anchor is None:
                summary["missing_anchor_referents"] += 1
                manifest_rows.append(make_empty_manifest_row(paths.scene, row_index, api_row, referent_name, referent_index, "missing_anchor", f"Referent not in anchor table: {referent_name}"))
                continue
            referent_rows: List[Dict[str, Any]] = []
            for panel_id, panel_index, sample_time, video_time, requested_video_time, sample, frame_path, extracted, duplicate_of, selection_score, selection_reason in panels:
                projected = project_world_point(sample, anchor, width, height)
                valid = projected is not None
                if valid:
                    row_has_valid_projection = True
                    summary["valid_projection_rows"] += 1
                    u_norm, v_norm = projected
                    gt_x = u_norm * width
                    gt_y = v_norm * height
                    status = "point_evaluable"
                    detail = ""
                else:
                    u_norm = v_norm = gt_x = gt_y = ""
                    status = "projection_invalid"
                    detail = "Anchor projects outside image or camera metadata is invalid."
                gaze_point, gaze_source = gaze_world_point(sample)
                gaze_projected = project_world_point(sample, gaze_point, width, height) if gaze_point is not None else None
                gaze_valid = gaze_projected is not None
                if gaze_valid:
                    summary["gaze_projection_rows"] += 1
                    gaze_u_norm, gaze_v_norm = gaze_projected
                    gaze_x = gaze_u_norm * width
                    gaze_y = gaze_v_norm * height
                else:
                    gaze_u_norm = gaze_v_norm = gaze_x = gaze_y = ""
                    gaze_source = ""
                if valid and gaze_valid:
                    gaze_anchor_distance_px = math.hypot(float(gaze_x) - float(gt_x), float(gaze_y) - float(gt_y))
                else:
                    gaze_anchor_distance_px = ""
                referent_rows.append(
                    {
                        "scene": paths.scene,
                        "row_index": row_index,
                        "event_id": normalize_text(api_row.get("event_id")),
                        "instruction": normalize_text(api_row.get("instruction_text") or api_row.get("target_description")),
                        "referent_name": referent_name,
                        "referent_index": referent_index,
                        "panel_id": panel_id,
                        "panel_index": panel_index,
                        "frame_time": f"{video_time:.3f}",
                        "json_sample_time": f"{sample_time:.3f}",
                        "video_frame_time": f"{video_time:.3f}",
                        "requested_video_frame_time": f"{requested_video_time:.3f}",
                        "video_duration": f"{video_duration:.3f}" if video_duration is not None else "",
                        "video_time_offset_seconds": f"{video_time_offset:.3f}",
                        "video_time_offset_source": video_time_offset_source,
                        "frame_duplicate_of": duplicate_of,
                        "panel_selection_score": f"{selection_score:.3f}",
                        "panel_selection_reason": selection_reason,
                        "frame_path": str(frame_path),
                        "video_path": str(video_path),
                        "json_path": str(json_path),
                        "image_width": width,
                        "image_height": height,
                        "anchor_x": anchor[0],
                        "anchor_y": anchor[1],
                        "anchor_z": anchor[2],
                        "gt_u_norm": f"{u_norm:.6f}" if valid else "",
                        "gt_v_norm": f"{v_norm:.6f}" if valid else "",
                        "gt_x": f"{gt_x:.2f}" if valid else "",
                        "gt_y": f"{gt_y:.2f}" if valid else "",
                        "projection_valid": str(valid),
                        "gaze_u_norm": f"{gaze_u_norm:.6f}" if gaze_valid else "",
                        "gaze_v_norm": f"{gaze_v_norm:.6f}" if gaze_valid else "",
                        "gaze_x": f"{gaze_x:.2f}" if gaze_valid else "",
                        "gaze_y": f"{gaze_y:.2f}" if gaze_valid else "",
                        "gaze_projection_valid": str(gaze_valid),
                        "gaze_anchor_distance_px": f"{gaze_anchor_distance_px:.2f}" if valid and gaze_valid else "",
                        "gaze_source": gaze_source,
                        "evidence_rank": "",
                        "evidence_acceptable": "False",
                        "frame_extracted": str(extracted),
                        "status": status,
                        "status_detail": detail,
                    }
                )
            annotate_evidence_rows(referent_rows, int(args.acceptable_top_k))
            summary["evidence_acceptable_rows"] += sum(
                1 for row in referent_rows if normalize_text(row.get("evidence_acceptable")) == "True"
            )
            manifest_rows.extend(referent_rows)
        if row_has_valid_projection:
            summary["rows_with_valid_projection"] += 1

    summary["manifest_rows"] = len(manifest_rows)
    return manifest_rows, summary


def write_manifest(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def main() -> None:
    args = parse_args()
    if args.panels <= 0:
        raise SystemExit("--panels must be positive")
    if args.acceptable_top_k < 0:
        raise SystemExit("--acceptable_top_k must be non-negative")
    if args.candidate_step_seconds <= 0:
        raise SystemExit("--candidate_step_seconds must be positive")
    if args.evidence_segment_duration_seconds < 0:
        raise SystemExit("--evidence_segment_duration_seconds must be non-negative")
    if args.gaze_stability_window_seconds <= 0:
        raise SystemExit("--gaze_stability_window_seconds must be positive")
    repo_root = Path(args.repo_root).resolve()
    eval_dir = resolve_eval_dir(repo_root, args.eval_dir).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    rewrites = parse_rewrites(args.path_rewrite)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for scene in args.scenes:
        if scene not in SCENES:
            raise SystemExit(f"Unknown scene: {scene}")
        paths = scene_paths(repo_root, eval_dir, scene)
        print(f"[{scene}] building manifest from {paths.api_input_csv}")
        rows, summary = process_scene(paths, output_dir, args, rewrites)
        all_rows.extend(rows)
        summaries.append(summary)
        scene_manifest = output_dir / f"manifest_{scene}.csv"
        write_manifest(scene_manifest, rows)
        print(f"[{scene}] wrote {len(rows)} rows -> {scene_manifest}")

    all_manifest = output_dir / "manifest_all.csv"
    write_manifest(all_manifest, all_rows)
    summary_payload = {
        "repo_root": str(repo_root),
        "eval_dir": str(eval_dir),
        "output_dir": str(output_dir),
        "scenes": summaries,
        "total_manifest_rows": len(all_rows),
        "total_valid_projection_rows": sum(item["valid_projection_rows"] for item in summaries),
        "total_rows_with_valid_projection": sum(item["rows_with_valid_projection"] for item in summaries),
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote all manifest: {all_manifest}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exam2Error as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
