#!/usr/bin/env python3
"""Shared utilities for candidate-free 3D anchor-point grounding."""

from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCENES: Tuple[str, ...] = (
    "scene1",
    "scene2",
    "scene3",
    "scene4_room1",
    "scene4_room2",
    "scene4_room3",
    "scene4_room4",
    "scene5",
)

LEAKY_FIELD_NAMES = {
    "anchor_x",
    "anchor_y",
    "anchor_z",
    "candidate_anchor_count",
    "candidate_anchors",
    "gt_anchor_ids",
    "gt_anchor_points_json",
    "gt_boxes_json",
    "gt_referents_mapped",
    "gt_referents_raw",
    "gt_u_norm",
    "gt_v_norm",
    "gt_x",
    "gt_y",
    "projection_valid",
    "referent_index",
    "referent_name",
    "target_description",
    "true_positive_referents",
    "false_positive_referents",
    "false_negative_referents",
}


class PointGroundingError(Exception):
    """Raised when point-grounding preparation cannot proceed."""


@dataclass(frozen=True)
class Anchor:
    anchor_id: str
    point: Tuple[float, float, float]


@dataclass(frozen=True)
class EvidenceFrame:
    panel_id: str
    frame_path: str
    video_time: float
    sample_time: float
    selection_score: float
    selection_reason: str
    cue_gaze_valid: bool
    cue_hand_valid: bool
    sample: Mapping[str, Any]


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_float(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:5])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise PointGroundingError(f"Missing CSV: {path}")
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(text.splitlines(), delimiter=detect_delimiter(text))
    if reader.fieldnames is None:
        raise PointGroundingError(f"CSV has no header: {path}")
    return [dict(row) for row in reader]


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def point3(mapping: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(mapping, Mapping):
        return None
    values = [parse_float(mapping.get(axis)) for axis in ("x", "y", "z")]
    if any(value is None for value in values):
        return None
    return float(values[0]), float(values[1]), float(values[2])  # type: ignore[arg-type]


def quat4(mapping: Any) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(mapping, Mapping):
        return None
    values = [parse_float(mapping.get(axis)) for axis in ("x", "y", "z", "w")]
    if any(value is None for value in values):
        return None
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])  # type: ignore[arg-type]


def vector_add(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return left[0] + right[0], left[1] + right[1], left[2] + right[2]


def vector_sub(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return left[0] - right[0], left[1] - right[1], left[2] - right[2]


def vector_scale(vector: Tuple[float, float, float], scale: float) -> Tuple[float, float, float]:
    return vector[0] * scale, vector[1] * scale, vector[2] * scale


def vector_norm(vector: Tuple[float, float, float]) -> float:
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)


def vector_distance(left: Tuple[float, float, float], right: Tuple[float, float, float]) -> float:
    return vector_norm(vector_sub(left, right))


def normalize_vector(vector: Tuple[float, float, float], eps: float = 1e-9) -> Optional[Tuple[float, float, float]]:
    norm = vector_norm(vector)
    if norm <= eps or not math.isfinite(norm):
        return None
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def is_nonzero_point(point: Optional[Tuple[float, float, float]], eps: float = 1e-6) -> bool:
    return point is not None and any(abs(value) > eps for value in point)


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


def camera_basis(sample: Mapping[str, Any]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    rotation = quat4(sample.get("cameraRotation"))
    if rotation is None:
        return None, None, None
    right = normalize_vector(rotate_vector_by_quaternion((1.0, 0.0, 0.0), rotation))
    up = normalize_vector(rotate_vector_by_quaternion((0.0, 1.0, 0.0), rotation))
    forward = normalize_vector(rotate_vector_by_quaternion((0.0, 0.0, 1.0), rotation))
    return forward, right, up


def normalize_iso_timestamp(value: Any) -> str:
    text = normalize_text(value)
    if "." not in text:
        return text
    timezone_start = max(text.rfind("+"), text.rfind("-"))
    if timezone_start <= text.find("."):
        timezone_start = len(text)
    fraction_start = text.find(".")
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


def repair_json_array_text(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("["):
        # Some captured multimodal files are array-like logs where adjacent
        # objects are written as "}\n{" instead of "},\n{".
        stripped = stripped.replace("}\n{", "},\n{")
        stripped = stripped.replace("}\r\n{", "},\r\n{")
        if stripped.count("[") == stripped.count("]") + 1 and stripped.endswith("}"):
            stripped += "\n]"
        return stripped
    pieces = []
    for line in stripped.splitlines():
        cleaned = line.strip().rstrip(",")
        if cleaned:
            pieces.append(cleaned)
    return "[" + ",".join(pieces) + "]"


def load_multimodal_samples(path: Path) -> List[Mapping[str, Any]]:
    raw_text = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = json.loads(repair_json_array_text(raw_text))
    if not isinstance(payload, list):
        raise PointGroundingError(f"Multimodal JSON is not a list: {path}")
    return [item for item in payload if isinstance(item, Mapping)]


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


def nearest_sample(timed_samples: Sequence[Tuple[float, Mapping[str, Any]]], target_time: float) -> Tuple[float, Mapping[str, Any]]:
    if not timed_samples:
        raise PointGroundingError("No timed samples available.")
    return min(timed_samples, key=lambda item: abs(item[0] - target_time))


def dense_candidate_times(sample_start: float, sample_end: float, step_seconds: float) -> List[float]:
    if sample_end <= sample_start:
        return [sample_start]
    step = max(0.05, float(step_seconds))
    times: List[float] = []
    current = sample_start
    while current <= sample_end + 1e-6:
        times.append(current)
        current += step
    midpoint = (sample_start + sample_end) / 2.0
    if all(abs(midpoint - item) > step * 0.25 for item in times):
        times.append(midpoint)
    return sorted(times)


def load_anchor_table(repo_root: Path, scene: str) -> List[Anchor]:
    path = repo_root / "data" / f"{scene}_anchor_table.tsv"
    rows = read_csv_rows(path)
    anchors: List[Anchor] = []
    for row in rows:
        name = normalize_text(row.get("object_name") or row.get("anchor_id") or row.get("物体名称"))
        x = parse_float(row.get("x_world") or row.get("location_x") or row.get("x"))
        y = parse_float(row.get("y_world") or row.get("location_y") or row.get("y"))
        z = parse_float(row.get("z_world") or row.get("location_z") or row.get("z"))
        if name and x is not None and y is not None and z is not None:
            anchors.append(Anchor(name, (x, y, z)))
    if not anchors:
        raise PointGroundingError(f"No anchors loaded from {path}")
    return anchors


def split_names(value: Any) -> List[str]:
    text = normalize_text(value)
    if not text:
        return []
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def scene_api_csv(repo_root: Path, scene: str) -> Path:
    return repo_root / "data" / f"{scene}_api_input.csv"


def match_eval_csv(repo_root: Path, scene: str, eval_dir: Optional[Path] = None) -> Path:
    if eval_dir is not None:
        return eval_dir / f"{scene}_match_eval.csv"
    candidates = [
        repo_root / "data" / "match_eval_qwen3vl30b_mention_first_v3" / f"{scene}_match_eval.csv",
        repo_root / "data" / "match_eval_qwen3vl30b_mention_first_v2" / f"{scene}_match_eval.csv",
        repo_root / "data" / "match_eval" / f"{scene}_match_eval.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise PointGroundingError(f"Missing match-eval CSV for {scene}; tried: {', '.join(str(p) for p in candidates)}")


def build_gt_from_match_eval(repo_root: Path, scene: str, eval_dir: Optional[Path] = None) -> Dict[int, List[Anchor]]:
    anchors_by_id = {anchor.anchor_id: anchor for anchor in load_anchor_table(repo_root, scene)}
    rows = read_csv_rows(match_eval_csv(repo_root, scene, eval_dir))
    result: Dict[int, List[Anchor]] = {}
    for row in rows:
        row_index = parse_int(row.get("row_index"))
        if row_index is None:
            continue
        names = split_names(row.get("gt_referents_mapped") or row.get("gt_referents_raw"))
        anchors: List[Anchor] = []
        seen = set()
        for name in names:
            if name in anchors_by_id and name not in seen:
                anchors.append(anchors_by_id[name])
                seen.add(name)
        result[row_index] = anchors
    return result


def robust_bounds(anchors: Sequence[Anchor]) -> Dict[str, Any]:
    points = [anchor.point for anchor in anchors]
    xs = sorted(point[0] for point in points)
    ys = sorted(point[1] for point in points)
    zs = sorted(point[2] for point in points)

    def percentile(values: Sequence[float], ratio: float) -> float:
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = ratio * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    q05 = (percentile(xs, 0.05), percentile(ys, 0.05), percentile(zs, 0.05))
    q95 = (percentile(xs, 0.95), percentile(ys, 0.95), percentile(zs, 0.95))
    diagonal = vector_distance(q05, q95)
    return {
        "x_q05": q05[0],
        "x_q95": q95[0],
        "y_q05": q05[1],
        "y_q95": q95[1],
        "z_q05": q05[2],
        "z_q95": q95[2],
        "robust_diagonal": diagonal,
    }


def nearest_anchor(point: Tuple[float, float, float], anchors: Sequence[Anchor]) -> Tuple[Optional[Anchor], Optional[float]]:
    if not anchors:
        return None, None
    anchor = min(anchors, key=lambda item: vector_distance(point, item.point))
    return anchor, vector_distance(point, anchor.point)


def nearest_negative_distance(gt_anchor: Anchor, all_anchors: Sequence[Anchor], gt_ids: Iterable[str]) -> Optional[float]:
    gt_id_set = set(gt_ids)
    negatives = [anchor for anchor in all_anchors if anchor.anchor_id not in gt_id_set]
    if not negatives:
        return None
    return min(vector_distance(gt_anchor.point, anchor.point) for anchor in negatives)


def format_point(point: Optional[Tuple[float, float, float]], ndigits: int = 6) -> str:
    if point is None:
        return ""
    return "[" + ", ".join(f"{value:.{ndigits}f}" for value in point) + "]"


def compact_point(point: Optional[Tuple[float, float, float]], ndigits: int = 3) -> str:
    if point is None:
        return "unavailable"
    return "(" + ", ".join(f"{value:.{ndigits}f}" for value in point) + ")"


def first_sample_datetime(samples: Sequence[Mapping[str, Any]]) -> Optional[dt.datetime]:
    values = [parse_sample_datetime(sample.get("timestamp")) for sample in samples]
    valid = [value for value in values if value is not None]
    return min(valid) if valid else None


def read_metadata_datetime(media_dir: Path) -> Tuple[Optional[dt.datetime], str]:
    metadata_paths = sorted(media_dir.glob("metadata_*.json"))
    if not metadata_paths:
        return None, "auto_unavailable:no_metadata_json"
    for metadata_path in metadata_paths:
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if isinstance(payload, Mapping):
            metadata_dt = parse_sample_datetime(payload.get("timestamp"))
            if metadata_dt is not None:
                return metadata_dt, f"metadata:{metadata_path.name}"
    return None, "auto_unavailable:no_metadata_timestamp"


def read_video_filename_datetime(video_path: Path, reference_dt: Optional[dt.datetime]) -> Tuple[Optional[dt.datetime], str]:
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
    threshold_seconds: float,
    video_bias_seconds: float,
) -> Tuple[Optional[dt.datetime], str]:
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
    manual_adjustment: float = 0.0,
    threshold_seconds: float = 1.0,
    video_bias_seconds: float = 0.5,
) -> Tuple[float, str]:
    metadata_dt, metadata_source = read_metadata_datetime(json_path.parent)
    first_dt = first_sample_datetime(samples)
    if first_dt is None:
        return manual_adjustment, "auto_unavailable:no_multimodal_timestamp+manual_adjustment"
    video_dt, video_source = read_video_filename_datetime(video_path, first_dt) if video_path is not None else (None, "auto_unavailable:no_video_path")
    start_dt, start_source = choose_auto_start_datetime(
        metadata_dt,
        metadata_source,
        video_dt,
        video_source,
        threshold_seconds,
        video_bias_seconds,
    )
    if start_dt is None:
        return manual_adjustment, start_source + "+manual_adjustment"
    return (first_dt - start_dt).total_seconds() + manual_adjustment, start_source + f"+manual_adjustment:{manual_adjustment:.3f}"


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


def load_script_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise PointGroundingError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_qwen_module(repo_root: Path) -> Any:
    return load_script_module(repo_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py", "qwen3vl_local_grounding_exam3_point")
