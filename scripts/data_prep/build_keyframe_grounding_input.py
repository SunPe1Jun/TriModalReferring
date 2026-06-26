#!/usr/bin/env python3
"""Convert event_manifest.csv into keyframe grounding input CSV for grounding workflows."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import struct
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MANIFEST_REQUIRED_COLUMNS = (
    "event_id",
    "keyframe_path",
    "json_path",
    "t_start",
    "t_peak",
    "t_end",
)

OUTPUT_COLUMNS = (
    "event_id",
    "keyframe_path",
    "gaze_summary",
    "hand_summary",
    "instruction_text",
    "utterance_text",
    "target_description",
    "event_json_path",
    "spatial_context_text",
    "spatial_context_json",
    "spatial_prior_u_norm",
    "spatial_prior_v_norm",
    "spatial_prior_source",
)

OUTPUT_WITH_VIDEO_COLUMNS = (
    "event_id",
    "keyframe_path",
    "video_path",
    "json_path",
    "t_start",
    "t_end",
    "gaze_summary",
    "hand_summary",
    "instruction_text",
    "utterance_text",
    "target_description",
    "event_json_path",
    "spatial_context_text",
    "spatial_context_json",
    "spatial_prior_u_norm",
    "spatial_prior_v_norm",
    "spatial_prior_source",
)

PRESERVED_OPTIONAL_COLUMNS = (
    "instruction_text",
    "utterance_text",
    "target_description",
)


class ManifestConversionError(Exception):
    """Raised when manifest conversion cannot proceed."""


class SummaryBuilder:
    def __init__(
        self,
        max_points: int,
        peak_window_seconds: float,
        prior_source_order: Sequence[str],
        output_profile: str,
    ) -> None:
        self.max_points = max_points
        self.peak_window_seconds = peak_window_seconds
        self.prior_source_order = tuple(prior_source_order)
        self.output_profile = output_profile

    def build_record(
        self,
        event_id: str,
        keyframe_path: Path,
        video_path: Path,
        json_path: Path,
        t_start: float,
        t_peak: float,
        t_end: float,
        preserved_fields: Mapping[str, str],
        event_json_dir: Path,
    ) -> Dict[str, str]:
        samples = load_multimodal_samples(json_path)
        timed_samples = collect_timed_samples(samples)
        window_samples = select_window_samples(timed_samples, t_start, t_peak, t_end)
        if not window_samples:
            raise ManifestConversionError(
                f"No timed multimodal samples found in event window for event_id={event_id}."
            )

        peak_sample = select_peak_sample(timed_samples, t_peak)
        if peak_sample is None:
            peak_sample = window_samples[len(window_samples) // 2]
        peak_window_samples = select_peak_window_samples(
            timed_samples,
            t_peak=t_peak,
            peak_window_seconds=self.peak_window_seconds,
            fallback_sample=peak_sample,
        )

        gaze_summary = build_gaze_summary(window_samples, self.max_points)
        hand_summary = build_hand_summary(window_samples)
        image_width, image_height = get_image_size(keyframe_path)
        include_prior_metadata = self.output_profile == "legacy"
        spatial_prior = (
            compute_spatial_prior(
                peak_window_samples,
                image_width,
                image_height,
                self.prior_source_order,
            )
            if include_prior_metadata
            else {"source": "none", "u_norm": None, "v_norm": None, "world_point": None, "sample_count": 0}
        )
        spatial_payload = build_event_json_payload(
            event_id=event_id,
            json_path=json_path,
            t_start=t_start,
            t_peak=t_peak,
            t_end=t_end,
            peak_window_seconds=self.peak_window_seconds,
            window_samples=window_samples,
            peak_window_samples=peak_window_samples,
            peak_sample=peak_sample,
            image_width=image_width,
            image_height=image_height,
            spatial_prior=spatial_prior,
            include_prior_metadata=include_prior_metadata,
        )
        event_json_path = write_event_json(event_json_dir, event_id, spatial_payload)
        spatial_context_text = build_spatial_context_text(
            spatial_payload,
            output_profile=self.output_profile,
        )

        target_description = preserved_fields.get("target_description", "")
        instruction_text = build_instruction_text(
            preserved_fields.get("instruction_text", ""),
            target_description,
        )
        return {
            "event_id": event_id,
            "keyframe_path": str(keyframe_path),
            "video_path": str(video_path),
            "json_path": str(json_path),
            "t_start": format_time_seconds(t_start),
            "t_end": format_time_seconds(t_end),
            "gaze_summary": gaze_summary,
            "hand_summary": hand_summary,
            "instruction_text": instruction_text,
            "utterance_text": preserved_fields.get("utterance_text", ""),
            "target_description": target_description,
            "event_json_path": str(event_json_path),
            "spatial_context_text": spatial_context_text,
            "spatial_context_json": json.dumps(spatial_payload, ensure_ascii=False),
            "spatial_prior_u_norm": format_optional_float(spatial_prior.get("u_norm")) if include_prior_metadata else "",
            "spatial_prior_v_norm": format_optional_float(spatial_prior.get("v_norm")) if include_prior_metadata else "",
            "spatial_prior_source": str(spatial_prior.get("source", "none")) if include_prior_metadata else "none",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert event_manifest.csv into keyframe grounding model input CSV."
    )
    parser.add_argument("--input-csv", required=True, help="Path to event_manifest.csv.")
    parser.add_argument("--output-csv", required=True, help="Path to the output keyframe grounding input CSV.")
    parser.add_argument(
        "--event-json-dir",
        help="Directory for saving compact event-level JSON files. Defaults to <output_csv_stem>_event_json next to the output CSV.",
    )
    parser.add_argument(
        "--max-gaze-points",
        type=int,
        default=3,
        help="Maximum number of representative gaze points to include in the gaze summary.",
    )
    parser.add_argument(
        "--peak-window-seconds",
        type=float,
        default=0.25,
        help="Half window size around t_peak used to build the compact event JSON snippet.",
    )
    parser.add_argument(
        "--prior-source-order",
        default="gazePoint,cameraHitPoint",
        help="Comma-separated source priority for spatial prior selection. Default prefers gaze point over camera hit and excludes hand ray.",
    )
    parser.add_argument(
        "--output-profile",
        choices=("gaze_only_api", "legacy"),
        default="gaze_only_api",
        help="Controls whether the generated CSV keeps legacy prior-driven context or emits gaze-only API-friendly context. Default: gaze_only_api.",
    )
    parser.add_argument(
        "--with-video-output-csv",
        help="Optional path for a second CSV that also includes video_path, t_start, and t_end. Defaults to <output_stem>_with_video.csv.",
    )
    parser.add_argument("--instruction-xlsx", help="Optional XLSX file used to populate instruction_text in bulk.")
    parser.add_argument("--instruction-sheet", help="Worksheet name inside the XLSX file. Defaults to the first worksheet.")
    parser.add_argument("--instruction-column", default="C", help="Excel column letter that contains instruction text. Default: C.")
    parser.add_argument("--instruction-start-row", type=int, default=2, help="1-based start row for instruction extraction. Default: 2.")
    parser.add_argument("--instruction-key-mode", choices=("row_order", "scene_id"), default="row_order", help="How to align XLSX rows to manifest rows. Default: row_order.")
    parser.add_argument("--instruction-key-column", help="Excel column letter for the alignment key when instruction-key-mode=scene_id.")
    parser.add_argument("--instruction-assignment-csv", help="Optional explicit assignment CSV generated by build_instruction_assignment_csv.py.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output CSV if it already exists.")
    return parser.parse_args()


def split_columns(raw_columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = [column for column in required if column not in raw_columns]
    if missing:
        raise ManifestConversionError(f"{label} is missing required columns: {', '.join(missing)}")


def parse_float(raw_value: str, label: str, row_index: int) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ManifestConversionError(f"Row {row_index} has invalid {label}: {raw_value}") from exc


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


def repair_json_array_if_needed(raw_text: str) -> str:
    stripped = raw_text.strip()
    if not stripped:
        raise ManifestConversionError("Input multimodal JSON file is empty.")
    if stripped.startswith("[") and stripped.count("[") == stripped.count("]") + 1 and stripped.endswith("}"):
        return stripped + "\n]"
    return stripped


def load_multimodal_samples(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise ManifestConversionError(f"Missing multimodal JSON file: {json_path}")
    if not json_path.is_file():
        raise ManifestConversionError(f"Expected multimodal JSON to be a file: {json_path}")
    raw_text = json_path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        repaired = repair_json_array_if_needed(raw_text)
        try:
            payload = json.loads(repaired)
        except json.JSONDecodeError as exc:
            raise ManifestConversionError(f"Failed to parse multimodal JSON: {json_path}. {exc}") from exc
    if not isinstance(payload, list):
        raise ManifestConversionError(
            f"Expected multimodal JSON to be a list, but found: {type(payload).__name__}"
        )
    return payload


def parse_sample_datetime(timestamp_value: Any) -> Optional[dt.datetime]:
    if not isinstance(timestamp_value, str) or not timestamp_value.strip():
        return None
    normalized = normalize_iso_timestamp(timestamp_value)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return dt.datetime.fromisoformat(normalized)
    except ValueError:
        return None


def collect_timed_samples(samples: Sequence[Mapping[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    dated_samples: List[Tuple[dt.datetime, Dict[str, Any]]] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        sample_datetime = parse_sample_datetime(sample.get("timestamp"))
        if sample_datetime is None:
            continue
        dated_samples.append((sample_datetime, dict(sample)))
    if not dated_samples:
        return []

    distinct_dates = sorted({sample_datetime.date().isoformat() for sample_datetime, _ in dated_samples})
    if len(distinct_dates) > 1:
        print(
            "Warning: Cross-day timestamps detected in multimodal JSON. "
            f"Found multiple dates: {', '.join(distinct_dates)}. "
            "Continuing by sorting full datetimes and computing elapsed seconds across day boundaries.",
            file=sys.stderr,
        )

    dated_samples.sort(key=lambda item: item[0])
    base_time = dated_samples[0][0]
    return [((sample_datetime - base_time).total_seconds(), sample) for sample_datetime, sample in dated_samples]


def select_window_samples(
    timed_samples: Sequence[Tuple[float, Dict[str, Any]]],
    t_start: float,
    t_peak: float,
    t_end: float,
) -> List[Dict[str, Any]]:
    windowed = [sample for sample_time, sample in timed_samples if t_start <= sample_time <= t_end]
    if windowed:
        return windowed
    peak = select_peak_sample(timed_samples, t_peak)
    return [peak] if peak is not None else []


def select_peak_sample(
    timed_samples: Sequence[Tuple[float, Dict[str, Any]]],
    t_peak: float,
) -> Optional[Dict[str, Any]]:
    if not timed_samples:
        return None
    return min(timed_samples, key=lambda item: abs(item[0] - t_peak))[1]


def select_peak_window_samples(
    timed_samples: Sequence[Tuple[float, Dict[str, Any]]],
    t_peak: float,
    peak_window_seconds: float,
    fallback_sample: Dict[str, Any],
) -> List[Dict[str, Any]]:
    selected = [
        sample for sample_time, sample in timed_samples if abs(sample_time - t_peak) <= peak_window_seconds
    ]
    return selected if selected else [fallback_sample]


def get_nested(mapping: Mapping[str, Any], *keys: str) -> Optional[Any]:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def format_xyz(point: Mapping[str, Any]) -> str:
    try:
        x_value = float(point["x"])
        y_value = float(point["y"])
        z_value = float(point["z"])
    except (KeyError, TypeError, ValueError):
        return "unknown"
    return f"({x_value:.3f}, {y_value:.3f}, {z_value:.3f})"


def point_to_dict(point: Optional[Mapping[str, Any]]) -> Optional[Dict[str, float]]:
    if not isinstance(point, Mapping):
        return None
    try:
        return {
            "x": float(point["x"]),
            "y": float(point["y"]),
            "z": float(point["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def sample_representative_indices(length: int, max_points: int) -> List[int]:
    if length <= 0:
        return []
    if length <= max_points:
        return list(range(length))
    if max_points <= 1:
        return [length // 2]
    step = (length - 1) / float(max_points - 1)
    indices = sorted({int(round(step * index)) for index in range(max_points)})
    while len(indices) < max_points:
        indices.append(indices[-1])
    return indices[:max_points]


def build_gaze_summary(samples: Sequence[Mapping[str, Any]], max_points: int) -> str:
    eye_open_count = 0
    hit_points: List[Mapping[str, Any]] = []
    gaze_points: List[Mapping[str, Any]] = []
    camera_positions: List[Mapping[str, Any]] = []

    for sample in samples:
        eye_gaze = sample.get("eyeGaze")
        if not isinstance(eye_gaze, Mapping):
            continue
        if eye_gaze.get("isEyeOpen") is True:
            eye_open_count += 1
        camera_hit_point = get_nested(sample, "eyeGaze", "cameraHitPoint")
        if isinstance(camera_hit_point, Mapping):
            hit_points.append(camera_hit_point)
        gaze_point = get_nested(sample, "eyeGaze", "gazePoint")
        if isinstance(gaze_point, Mapping):
            gaze_points.append(gaze_point)
        camera_position = sample.get("cameraPosition")
        if isinstance(camera_position, Mapping):
            camera_positions.append(camera_position)

    representative_lines: List[str] = []
    for index in sample_representative_indices(len(gaze_points), max_points):
        representative_lines.append(f"gazePoint={format_xyz(gaze_points[index])}")
    if hit_points:
        representative_lines.append("cameraHitPoint_examples=" + "; ".join(
            format_xyz(hit_points[index]) for index in sample_representative_indices(len(hit_points), min(max_points, 2))
        ))

    position_text = ""
    valid_camera_positions = [
        item for item in camera_positions if all(axis in item for axis in ("x", "y", "z"))
    ]
    if valid_camera_positions:
        avg_x = mean(float(item["x"]) for item in valid_camera_positions)
        avg_y = mean(float(item["y"]) for item in valid_camera_positions)
        avg_z = mean(float(item["z"]) for item in valid_camera_positions)
        position_text = f" average_camera_position=({avg_x:.3f}, {avg_y:.3f}, {avg_z:.3f})."

    return (
        f"{len(samples)} multimodal samples in the event window; eye_open_count={eye_open_count}. "
        f"Representative gaze targets: {'; '.join(representative_lines) if representative_lines else 'none available.'}"
        f"{position_text}"
    ).strip()


def build_hand_summary(samples: Sequence[Mapping[str, Any]]) -> str:
    left_tracked = 0
    right_tracked = 0
    right_hit_points: List[Mapping[str, Any]] = []
    left_joint_counts: List[int] = []
    right_joint_counts: List[int] = []

    for sample in samples:
        hand_data = sample.get("handData")
        if not isinstance(hand_data, Mapping):
            continue
        if hand_data.get("isLeftHandTracked") is True:
            left_tracked += 1
        if hand_data.get("isRightHandTracked") is True:
            right_tracked += 1

        left_hand = hand_data.get("leftHand")
        if isinstance(left_hand, Mapping):
            joints = left_hand.get("joints")
            if isinstance(joints, list):
                left_joint_counts.append(len(joints))

        right_hand = hand_data.get("rightHand")
        if isinstance(right_hand, Mapping):
            joints = right_hand.get("joints")
            if isinstance(joints, list):
                right_joint_counts.append(len(joints))

        right_hit_point = hand_data.get("rightIndexFingerRayHitPoint")
        if isinstance(right_hit_point, Mapping):
            right_hit_points.append(right_hit_point)

    tracked_text = (
        f"left_hand_tracked_frames={left_tracked}/{len(samples)}, "
        f"right_hand_tracked_frames={right_tracked}/{len(samples)}."
    )

    joint_text = ""
    if left_joint_counts or right_joint_counts:
        left_avg = mean(left_joint_counts) if left_joint_counts else 0.0
        right_avg = mean(right_joint_counts) if right_joint_counts else 0.0
        joint_text = f" average_joint_counts=(left={left_avg:.1f}, right={right_avg:.1f})."

    ray_text = ""
    non_zero_hits = [
        point
        for point in right_hit_points
        if any(abs(float(point.get(axis, 0.0))) > 1e-6 for axis in ("x", "y", "z"))
    ]
    if non_zero_hits:
        representative = non_zero_hits[len(non_zero_hits) // 2]
        ray_text = f" representative_right_index_ray_hit={format_xyz(representative)}."
    else:
        ray_text = " representative_right_index_ray_hit=none available."

    return (tracked_text + joint_text + ray_text).strip()


def vector_subtract(a: Mapping[str, Any], b: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(a["x"]) - float(b["x"]),
        float(a["y"]) - float(b["y"]),
        float(a["z"]) - float(b["z"]),
    )


def quaternion_conjugate(q: Mapping[str, Any]) -> Tuple[float, float, float, float]:
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
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (vx, vy, vz, 0.0)),
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
    )
    return rotated[0], rotated[1], rotated[2]


def world_to_camera(sample: Mapping[str, Any], point: Mapping[str, Any]) -> Tuple[float, float, float]:
    camera_position = sample.get("cameraPosition")
    camera_rotation = sample.get("cameraRotation")
    if not isinstance(camera_position, Mapping) or not isinstance(camera_rotation, Mapping):
        raise ManifestConversionError("Missing cameraPosition or cameraRotation in multimodal sample.")
    translated = vector_subtract(point, camera_position)
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
    if abs(x_ndc) > 1.2 or abs(y_ndc) > 1.2:
        return None
    u_norm = (x_ndc + 1.0) * 0.5
    v_norm = (1.0 - y_ndc) * 0.5
    if u_norm < 0.0 or u_norm > 1.0 or v_norm < 0.0 or v_norm > 1.0:
        return None
    return u_norm, v_norm


def project_sample_point(
    sample: Mapping[str, Any],
    point_source: str,
    width: int,
    height: int,
) -> Optional[Tuple[float, float]]:
    eye_gaze = sample.get("eyeGaze")
    camera_fov = sample.get("cameraFOV")
    if not isinstance(eye_gaze, Mapping) or camera_fov is None:
        return None
    if point_source == "gazePoint":
        gaze_point = eye_gaze.get("gazePoint")
        if not isinstance(gaze_point, Mapping):
            return None
        camera_point = world_to_camera(sample, gaze_point)
    elif point_source == "cameraHitPoint":
        camera_hit = eye_gaze.get("cameraHitPoint")
        if not isinstance(camera_hit, Mapping):
            return None
        camera_point = (float(camera_hit["x"]), float(camera_hit["y"]), float(camera_hit["z"]))
    elif point_source == "rightIndexFingerRayHitPoint":
        hand_data = sample.get("handData")
        hit_point = hand_data.get("rightIndexFingerRayHitPoint") if isinstance(hand_data, Mapping) else None
        if not isinstance(hit_point, Mapping):
            return None
        camera_point = world_to_camera(sample, hit_point)
    else:
        return None
    try:
        return project_camera_point(camera_point, width, height, float(camera_fov))
    except (TypeError, ValueError, ManifestConversionError):
        return None


def compute_spatial_prior(
    samples: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
    source_order: Sequence[str],
) -> Dict[str, Any]:
    source_results: Dict[str, Dict[str, Any]] = {}
    for source in source_order:
        projected_points: List[Tuple[float, float]] = []
        world_points: List[Dict[str, float]] = []
        for sample in samples:
            projected = project_sample_point(sample, source, width, height)
            if projected is None:
                continue
            projected_points.append(projected)
            if source == "gazePoint":
                point = point_to_dict(get_nested(sample, "eyeGaze", "gazePoint"))
            elif source == "cameraHitPoint":
                point = point_to_dict(get_nested(sample, "eyeGaze", "cameraHitPoint"))
            elif source == "rightIndexFingerRayHitPoint":
                point = point_to_dict(get_nested(sample, "handData", "rightIndexFingerRayHitPoint"))
            else:
                point = None
            if point is not None:
                world_points.append(point)
        if projected_points:
            u_values = [point[0] for point in projected_points]
            v_values = [point[1] for point in projected_points]
            source_results[source] = {
                "source": source,
                "u_norm": median(u_values),
                "v_norm": median(v_values),
                "world_point": world_points[len(world_points) // 2] if world_points else None,
                "sample_count": len(projected_points),
            }

    for source in source_order:
        result = source_results.get(source)
        if result is not None:
            return result
    return {"source": "none", "u_norm": None, "v_norm": None, "world_point": None, "sample_count": 0}


def compact_sample(sample: Mapping[str, Any]) -> Dict[str, Any]:
    hand_data = sample.get("handData") if isinstance(sample.get("handData"), Mapping) else {}
    eye_gaze = sample.get("eyeGaze") if isinstance(sample.get("eyeGaze"), Mapping) else {}
    return {
        "timestamp": sample.get("timestamp"),
        "camera_position": point_to_dict(sample.get("cameraPosition") if isinstance(sample.get("cameraPosition"), Mapping) else None),
        "camera_rotation": sample.get("cameraRotation") if isinstance(sample.get("cameraRotation"), Mapping) else None,
        "camera_fov": sample.get("cameraFOV"),
        "gaze_point": point_to_dict(eye_gaze.get("gazePoint") if isinstance(eye_gaze.get("gazePoint"), Mapping) else None),
        "camera_hit_point": point_to_dict(eye_gaze.get("cameraHitPoint") if isinstance(eye_gaze.get("cameraHitPoint"), Mapping) else None),
        "gaze_origin": point_to_dict(eye_gaze.get("gazeOrigin") if isinstance(eye_gaze.get("gazeOrigin"), Mapping) else None),
        "gaze_vector": point_to_dict(eye_gaze.get("gazeVector") if isinstance(eye_gaze.get("gazeVector"), Mapping) else None),
        "left_hand_tracked": bool(hand_data.get("isLeftHandTracked")) if isinstance(hand_data, Mapping) else False,
        "right_hand_tracked": bool(hand_data.get("isRightHandTracked")) if isinstance(hand_data, Mapping) else False,
        "right_index_ray_hit_point": point_to_dict(hand_data.get("rightIndexFingerRayHitPoint") if isinstance(hand_data.get("rightIndexFingerRayHitPoint"), Mapping) else None),
    }


def build_event_json_payload(
    event_id: str,
    json_path: Path,
    t_start: float,
    t_peak: float,
    t_end: float,
    peak_window_seconds: float,
    window_samples: Sequence[Mapping[str, Any]],
    peak_window_samples: Sequence[Mapping[str, Any]],
    peak_sample: Mapping[str, Any],
    image_width: int,
    image_height: int,
    spatial_prior: Mapping[str, Any],
    include_prior_metadata: bool,
) -> Dict[str, Any]:
    representative_window = [compact_sample(window_samples[index]) for index in sample_representative_indices(len(window_samples), 3)]
    peak_context_window = [compact_sample(sample) for sample in peak_window_samples[:5]]
    payload = {
        "event_id": event_id,
        "source_multimodal_json": str(json_path),
        "time_window": {
            "t_start": t_start,
            "t_peak": t_peak,
            "t_end": t_end,
            "peak_window_seconds": peak_window_seconds,
        },
        "image_size": {"width": image_width, "height": image_height},
        "window_sample_count": len(window_samples),
        "peak_spatial": compact_sample(peak_sample),
        "peak_window_samples": peak_context_window[:5],
        "representative_window_samples": representative_window,
    }
    if include_prior_metadata:
        payload["spatial_prior"] = spatial_prior
    return payload


def build_spatial_context_text(payload: Mapping[str, Any], output_profile: str = "gaze_only_api") -> str:
    prior = payload.get("spatial_prior") if isinstance(payload.get("spatial_prior"), Mapping) else {}
    peak_spatial = payload.get("peak_spatial") if isinstance(payload.get("peak_spatial"), Mapping) else {}
    if output_profile == "legacy":
        lines = [
            f"window_sample_count={payload.get('window_sample_count', 'unknown')}",
            f"t_peak={get_nested(payload, 'time_window', 't_peak')}",
            f"image_size={payload.get('image_size')}",
            f"spatial_prior_source={prior.get('source', 'none')}",
            f"spatial_prior_u_norm={format_optional_float(prior.get('u_norm'))}",
            f"spatial_prior_v_norm={format_optional_float(prior.get('v_norm'))}",
            f"peak_camera_hit_point={format_xyz(peak_spatial.get('camera_hit_point')) if isinstance(peak_spatial.get('camera_hit_point'), Mapping) else 'unknown'}",
            f"peak_gaze_point={format_xyz(peak_spatial.get('gaze_point')) if isinstance(peak_spatial.get('gaze_point'), Mapping) else 'unknown'}",
            f"peak_right_index_ray_hit_point={format_xyz(peak_spatial.get('right_index_ray_hit_point')) if isinstance(peak_spatial.get('right_index_ray_hit_point'), Mapping) else 'unknown'}",
            f"peak_camera_position={format_xyz(peak_spatial.get('camera_position')) if isinstance(peak_spatial.get('camera_position'), Mapping) else 'unknown'}",
            f"peak_camera_fov={peak_spatial.get('camera_fov', 'unknown')}",
        ]
    else:
        lines = [
            f"window_sample_count={payload.get('window_sample_count', 'unknown')}",
            f"t_peak={get_nested(payload, 'time_window', 't_peak')}",
            f"peak_gaze_point={format_xyz(peak_spatial.get('gaze_point')) if isinstance(peak_spatial.get('gaze_point'), Mapping) else 'unknown'}",
            f"peak_gaze_vector={format_xyz(peak_spatial.get('gaze_vector')) if isinstance(peak_spatial.get('gaze_vector'), Mapping) else 'unknown'}",
            f"peak_gaze_origin={format_xyz(peak_spatial.get('gaze_origin')) if isinstance(peak_spatial.get('gaze_origin'), Mapping) else 'unknown'}",
            f"peak_camera_position={format_xyz(peak_spatial.get('camera_position')) if isinstance(peak_spatial.get('camera_position'), Mapping) else 'unknown'}",
            f"peak_camera_rotation={format_xyz(peak_spatial.get('camera_rotation')) if isinstance(peak_spatial.get('camera_rotation'), Mapping) else 'unknown'}",
            f"peak_camera_fov={peak_spatial.get('camera_fov', 'unknown')}",
            f"right_hand_tracked={peak_spatial.get('right_hand_tracked', False)}",
        ]
    return "\n".join(lines)


def write_event_json(event_json_dir: Path, event_id: str, payload: Mapping[str, Any]) -> Path:
    event_json_dir.mkdir(parents=True, exist_ok=True)
    output_path = event_json_dir / f"{sanitize_filename(event_id)}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "event"


def format_optional_float(value: Any) -> str:
    if value in (None, "", "null", "None"):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.6f}"


def format_time_seconds(value: float) -> str:
    return f"{value:.3f}"


def build_instruction_text(existing_text: str, target_description: str) -> str:
    if existing_text.strip():
        return existing_text.strip()
    if target_description.strip():
        return (
            "Ground the intended referent for this event. "
            "Use the visual scene, gaze cues, and structured event context to localize the action-consistent target implied by: "
            f"{target_description.strip()}"
        )
    return (
        "Ground the intended referent for this event using the visual scene, gaze cues, and structured event context."
    )


def column_letters_to_index(column_letters: str) -> int:
    text = column_letters.strip().upper()
    if not text or not text.isalpha():
        raise ManifestConversionError(f"Invalid Excel column reference: {column_letters}")
    value = 0
    for char in text:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def parse_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    shared_strings_path = "xl/sharedStrings.xml"
    if shared_strings_path not in archive.namelist():
        return []
    root = ET.fromstring(archive.read(shared_strings_path))
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: List[str] = []
    for item in root.findall("main:si", namespace):
        text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
        values.append("".join(text_parts))
    return values


def resolve_sheet_path(archive: zipfile.ZipFile, sheet_name: Optional[str]) -> str:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
        "docrel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    relationship_map = {
        rel.attrib.get("Id"): rel.attrib.get("Target", "") for rel in rel_root.findall("rel:Relationship", namespace)
    }
    sheets = workbook_root.findall("main:sheets/main:sheet", namespace)
    if not sheets:
        raise ManifestConversionError("The XLSX workbook does not contain any worksheets.")
    selected = None
    if sheet_name:
        for sheet in sheets:
            if sheet.attrib.get("name") == sheet_name:
                selected = sheet
                break
        if selected is None:
            raise ManifestConversionError(f"Worksheet not found in XLSX: {sheet_name}")
    else:
        selected = sheets[0]
    relation_id = selected.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
    target = relationship_map.get(relation_id, "")
    if not target:
        raise ManifestConversionError("Failed to resolve worksheet path inside the XLSX file.")
    return "xl/" + target.replace('\\', '/').lstrip('/')


def read_sheet_rows(xlsx_path: Path, sheet_name: Optional[str]) -> List[List[str]]:
    if not xlsx_path.exists() or not xlsx_path.is_file():
        raise ManifestConversionError(f"Instruction XLSX does not exist or is not a file: {xlsx_path}")
    try:
        with zipfile.ZipFile(xlsx_path, "r") as archive:
            sheet_path = resolve_sheet_path(archive, sheet_name)
            shared_strings = parse_shared_strings(archive)
            root = ET.fromstring(archive.read(sheet_path))
    except zipfile.BadZipFile as exc:
        raise ManifestConversionError(f"Invalid XLSX file: {xlsx_path}") from exc
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: List[List[str]] = []
    for row in root.findall("main:sheetData/main:row", namespace):
        values: List[str] = []
        current_index = 0
        for cell in row.findall("main:c", namespace):
            ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)(\d+)", ref)
            cell_index = column_letters_to_index(match.group(1)) if match else current_index
            while len(values) < cell_index:
                values.append("")
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                text_value = "".join(node.text or "" for node in cell.findall(".//main:t", namespace))
            else:
                value_node = cell.find("main:v", namespace)
                raw_value = value_node.text if value_node is not None and value_node.text is not None else ""
                if cell_type == "s":
                    try:
                        text_value = shared_strings[int(raw_value)]
                    except (ValueError, IndexError):
                        text_value = ""
                else:
                    text_value = raw_value
            values.append(text_value.strip())
            current_index = len(values)
        rows.append(values)
    return rows


def build_instruction_lookup(args: argparse.Namespace, manifest_rows: Sequence[Mapping[str, str]]) -> Dict[str, str]:
    if not args.instruction_xlsx:
        return {}
    if args.instruction_start_row <= 0:
        raise ManifestConversionError("instruction-start-row must be at least 1.")
    instruction_rows = read_sheet_rows(Path(args.instruction_xlsx).resolve(), args.instruction_sheet)
    text_column_index = column_letters_to_index(args.instruction_column)
    lookup: Dict[str, str] = {}
    if args.instruction_key_mode == "row_order":
        manifest_index = 0
        for row_index in range(args.instruction_start_row - 1, len(instruction_rows)):
            if manifest_index >= len(manifest_rows):
                break
            row = instruction_rows[row_index]
            instruction_text = row[text_column_index].strip() if text_column_index < len(row) else ""
            if instruction_text:
                event_id = str(manifest_rows[manifest_index].get("event_id", "")).strip()
                if event_id:
                    lookup[event_id] = instruction_text
            manifest_index += 1
        return lookup
    if not args.instruction_key_column:
        raise ManifestConversionError("instruction-key-column is required when instruction-key-mode=scene_id.")
    key_column_index = column_letters_to_index(args.instruction_key_column)
    key_to_event: Dict[str, str] = {}
    for row in manifest_rows:
        scene_id = str(row.get("scene_id", "")).strip()
        event_id = str(row.get("event_id", "")).strip()
        if scene_id and event_id:
            key_to_event[scene_id] = event_id
    if not key_to_event:
        raise ManifestConversionError("instruction-key-mode=scene_id requires scene_id to exist in event_manifest.csv.")
    for row_index in range(args.instruction_start_row - 1, len(instruction_rows)):
        row = instruction_rows[row_index]
        key_value = row[key_column_index].strip() if key_column_index < len(row) else ""
        instruction_text = row[text_column_index].strip() if text_column_index < len(row) else ""
        if not key_value or not instruction_text:
            continue
        event_id = key_to_event.get(key_value)
        if event_id:
            lookup[event_id] = instruction_text
    return lookup


def merge_instruction_fields(
    preserved_fields: Dict[str, Dict[str, str]],
    instruction_lookup: Mapping[str, str],
) -> Dict[str, Dict[str, str]]:
    merged = {event_id: dict(values) for event_id, values in preserved_fields.items()}
    for event_id, instruction_text in instruction_lookup.items():
        entry = merged.setdefault(event_id, {column: "" for column in PRESERVED_OPTIONAL_COLUMNS})
        if instruction_text.strip():
            entry["instruction_text"] = instruction_text.strip()
    return merged


def load_assignment_fields(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if csv_path is None or not csv_path.exists() or not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "event_id" not in reader.fieldnames:
            raise ManifestConversionError(f"Instruction assignment CSV is missing event_id: {csv_path}")
        assigned: Dict[str, Dict[str, str]] = {}
        for row in reader:
            event_id = (row.get("event_id") or "").strip()
            if not event_id:
                continue
            instruction_text = (row.get("instruction_text") or "").strip()
            target_description = (row.get("target_description") or "").strip()
            if instruction_text and (not target_description or target_description != instruction_text):
                target_description = instruction_text
            assigned[event_id] = {
                "instruction_text": instruction_text,
                "utterance_text": (row.get("utterance_text") or "").strip(),
                "target_description": target_description,
            }
        return assigned


def merge_preserved_field_maps(
    base_fields: Dict[str, Dict[str, str]],
    override_fields: Mapping[str, Mapping[str, str]],
) -> Dict[str, Dict[str, str]]:
    merged = {event_id: dict(values) for event_id, values in base_fields.items()}
    for event_id, values in override_fields.items():
        entry = merged.setdefault(event_id, {column: "" for column in PRESERVED_OPTIONAL_COLUMNS})
        for column in PRESERVED_OPTIONAL_COLUMNS:
            value = str(values.get(column, "")).strip()
            if value:
                entry[column] = value
    return merged


def read_manifest_rows(input_csv: Path) -> List[Dict[str, str]]:
    if not input_csv.exists():
        raise ManifestConversionError(f"Input CSV does not exist: {input_csv}")
    if not input_csv.is_file():
        raise ManifestConversionError(f"Input CSV is not a file: {input_csv}")

    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ManifestConversionError(f"Input CSV has no header row: {input_csv}")
        split_columns(reader.fieldnames, MANIFEST_REQUIRED_COLUMNS, "event_manifest.csv")

        rows: List[Dict[str, str]] = []
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items()}
            if not any(normalized.values()):
                continue
            rows.append(normalized)

    if not rows:
        raise ManifestConversionError(f"Input CSV contains no valid rows: {input_csv}")
    return rows


def load_preserved_fields(csv_paths: Sequence[Path]) -> Dict[str, Dict[str, str]]:
    preserved: Dict[str, Dict[str, str]] = {}
    for csv_path in csv_paths:
        if not csv_path.exists() or not csv_path.is_file():
            continue
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "event_id" not in reader.fieldnames:
                continue
            for row in reader:
                event_id = (row.get("event_id") or "").strip()
                if not event_id:
                    continue
                preserved[event_id] = {
                    column: (row.get(column) or "").strip() for column in PRESERVED_OPTIONAL_COLUMNS
                }
    return preserved


def write_rows(output_csv: Path, rows: Iterable[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def get_image_size(image_path: Path) -> Tuple[int, int]:
    if not image_path.exists():
        raise ManifestConversionError(f"Missing keyframe image: {image_path}")
    if not image_path.is_file():
        raise ManifestConversionError(f"Expected keyframe image to be a file: {image_path}")
    with image_path.open("rb") as handle:
        header = handle.read(26)
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = struct.unpack(">II", header[16:24])
            return int(width), int(height)
        if header.startswith(b"\xff\xd8"):
            handle.seek(2)
            while True:
                marker_prefix = handle.read(1)
                if not marker_prefix:
                    break
                if marker_prefix != b"\xff":
                    continue
                marker = handle.read(1)
                while marker == b"\xff":
                    marker = handle.read(1)
                if marker in {b"\xc0", b"\xc1", b"\xc2", b"\xc3", b"\xc5", b"\xc6", b"\xc7", b"\xc9", b"\xca", b"\xcb", b"\xcd", b"\xce", b"\xcf"}:
                    segment = handle.read(7)
                    height, width = struct.unpack(">HH", segment[1:5])
                    return int(width), int(height)
                length_bytes = handle.read(2)
                if len(length_bytes) != 2:
                    break
                segment_length = struct.unpack(">H", length_bytes)[0]
                handle.seek(segment_length - 2, 1)
    raise ManifestConversionError(f"Unsupported or unreadable keyframe image format: {image_path}")


def main() -> int:
    args = parse_args()

    try:
        input_csv = Path(args.input_csv).resolve()
        output_csv = Path(args.output_csv).resolve()
        with_video_output_csv = (
            Path(args.with_video_output_csv).resolve()
            if args.with_video_output_csv
            else output_csv.with_name(f"{output_csv.stem}_with_video{output_csv.suffix}")
        )
        if output_csv.exists() and not args.overwrite:
            raise ManifestConversionError(
                f"Output CSV already exists. Use --overwrite to replace it: {output_csv}"
            )
        if with_video_output_csv.exists() and not args.overwrite:
            raise ManifestConversionError(
                f"Video output CSV already exists. Use --overwrite to replace it: {with_video_output_csv}"
            )
        if args.max_gaze_points <= 0:
            raise ManifestConversionError("max-gaze-points must be positive.")
        if args.peak_window_seconds < 0:
            raise ManifestConversionError("peak-window-seconds must be non-negative.")

        event_json_dir = (
            Path(args.event_json_dir).resolve()
            if args.event_json_dir
            else output_csv.parent / f"{output_csv.stem}_event_json"
        )

        rows = read_manifest_rows(input_csv)
        preserved_fields = load_preserved_fields((output_csv, with_video_output_csv))
        assignment_fields = load_assignment_fields(Path(args.instruction_assignment_csv).resolve() if args.instruction_assignment_csv else None)
        preserved_fields = merge_preserved_field_maps(preserved_fields, assignment_fields)
        instruction_lookup = build_instruction_lookup(args, rows)
        preserved_fields = merge_instruction_fields(preserved_fields, instruction_lookup)
        prior_source_order = [item.strip() for item in args.prior_source_order.split(",") if item.strip()]
        if not prior_source_order:
            raise ManifestConversionError("prior-source-order must contain at least one source name.")
        allowed_sources = {"gazePoint", "cameraHitPoint", "rightIndexFingerRayHitPoint"}
        invalid_sources = [item for item in prior_source_order if item not in allowed_sources]
        if invalid_sources:
            raise ManifestConversionError(
                "prior-source-order contains unsupported source names: " + ", ".join(invalid_sources)
            )
        builder = SummaryBuilder(
            max_points=args.max_gaze_points,
            peak_window_seconds=args.peak_window_seconds,
            prior_source_order=prior_source_order,
            output_profile=args.output_profile,
        )

        output_rows: List[Dict[str, str]] = []
        for row_index, row in enumerate(rows, start=1):
            event_id = row["event_id"]
            keyframe_path = Path(row["keyframe_path"]).expanduser().resolve()
            video_path = Path(row["video_path"]).expanduser().resolve()
            json_path = Path(row["json_path"]).expanduser().resolve()
            t_start = parse_float(row["t_start"], "t_start", row_index)
            t_peak = parse_float(row["t_peak"], "t_peak", row_index)
            t_end = parse_float(row["t_end"], "t_end", row_index)

            if t_end < t_start:
                raise ManifestConversionError(f"Row {row_index} has t_end earlier than t_start.")

            output_rows.append(
                builder.build_record(
                    event_id=event_id,
                    keyframe_path=keyframe_path,
                    video_path=video_path,
                    json_path=json_path,
                    t_start=t_start,
                    t_peak=t_peak,
                    t_end=t_end,
                    preserved_fields=preserved_fields.get(event_id, {}),
                    event_json_dir=event_json_dir,
                )
            )
            print(f"Prepared grounding input row {row_index}/{len(rows)} | event_id={event_id}", flush=True)

        write_rows(output_csv, output_rows, OUTPUT_COLUMNS)
        write_rows(with_video_output_csv, output_rows, OUTPUT_WITH_VIDEO_COLUMNS)
        print(f"Saved grounding input CSV to: {output_csv}")
        print(f"Saved grounding input CSV with video fields to: {with_video_output_csv}")
        print(f"Saved event JSON snippets to: {event_json_dir}")
        return 0
    except ManifestConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
