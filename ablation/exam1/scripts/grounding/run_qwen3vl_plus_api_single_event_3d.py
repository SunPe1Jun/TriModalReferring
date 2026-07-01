#!/usr/bin/env python3
"""Call Qwen3-VL-Plus API for a single event row and select 3D referent objects from scene anchors."""

from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REGION_TO_BASE_URL = {
    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "us-east-1": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
}
DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_API_KEY_ENV = "DASHSCOPE_API_KEY"
ALLOWED_PRIMARY_SOURCES = {"gazePoint", "visual_only", "language", "none"}
ALLOWED_ABLATION_MODALITIES = {"visual", "gaze", "hand", "structured_geometry", "timeline"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call Qwen3-VL-Plus API for one CSV row and one local video, then select a 3D referent object.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--row_index", type=int, default=0, help="Zero-based row index to test. Default: 0.")
    parser.add_argument("--output_json", required=True, help="Path to save the raw API response and metadata.")
    parser.add_argument("--scene_anchor_csv", required=True, help="Path to the scene anchor candidate table.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name. Default: qwen3-vl-plus")
    parser.add_argument("--region", choices=tuple(REGION_TO_BASE_URL.keys()), default="cn-beijing", help="DashScope region. Default: cn-beijing")
    parser.add_argument("--base_url", help="Optional explicit OpenAI-compatible DashScope base URL. Overrides --region.")
    parser.add_argument("--api_key_env", default=DEFAULT_API_KEY_ENV, help="Environment variable holding the API key.")
    parser.add_argument("--api_key", help="Optional API key. Overrides environment variables.")
    parser.add_argument("--video_path", help="Optional explicit local video path. Overrides CSV video_path.")
    parser.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for the API video input.")
    parser.add_argument("--max_tokens", type=int, default=896, help="Max completion tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--prompt_style", choices=("world_only", "full"), default="full", help="Prompt style. Default: full")
    parser.add_argument("--prompt_strategy", choices=("standard", "mention_first"), default="standard", help="Prompt strategy. Default: standard")
    parser.add_argument("--max_evidence_segments", type=int, default=4, help="Maximum sparse timeline evidence segments to include in the prompt. Default: 4")
    parser.add_argument("--evidence_segment_duration", type=float, default=0.5, help="Duration in seconds for each sparse evidence segment. Default: 0.5")
    parser.add_argument(
        "--ablate_modalities",
        default="",
        help=(
            "Comma-separated modalities to hide from the prompt: "
            "visual,gaze,hand,structured_geometry,timeline. Default: none."
        ),
    )
    parser.add_argument("--timeout_seconds", type=int, default=600, help="HTTP timeout in seconds.")
    return parser.parse_args()


def normalize_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def compress_summary(text: Any, max_chars: int = 220) -> str:
    clean = " ".join(normalize_text(text).split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."




def parse_ablation_modalities(raw_value: Any) -> Set[str]:
    if isinstance(raw_value, set):
        return {str(item) for item in raw_value}
    if isinstance(raw_value, (list, tuple)):
        return {str(item) for item in raw_value if str(item).strip()}
    text = normalize_text(raw_value).lower()
    if not text or text in {"none", "full", "baseline"}:
        return set()
    aliases = {
        "vision": "visual",
        "image": "visual",
        "video": "visual",
        "visual_evidence": "visual",
        "eye": "gaze",
        "eye_cue": "gaze",
        "eye_cues": "gaze",
        "gaze_text": "gaze",
        "gaze_prior": "gaze",
        "hands": "hand",
        "gesture": "hand",
        "geometry": "structured_geometry",
        "spatial": "structured_geometry",
        "spatial_context": "structured_geometry",
        "camera": "structured_geometry",
        "sparse_timeline": "timeline",
        "eye_timeline": "timeline",
    }
    result: Set[str] = set()
    for item in re.split(r"[,;\s]+", text):
        if not item:
            continue
        normalized = aliases.get(item, item)
        if normalized not in ALLOWED_ABLATION_MODALITIES:
            allowed = ", ".join(sorted(ALLOWED_ABLATION_MODALITIES))
            raise ValueError(f"Unsupported ablation modality: {item}. Allowed: {allowed}")
        result.add(normalized)
    return result


def modality_disabled(modalities: Sequence[str] | Set[str], *names: str) -> bool:
    modality_set = set(modalities)
    return any(name in modality_set for name in names)


def build_ablation_protocol_note(modalities: Sequence[str] | Set[str]) -> str:
    modality_set = set(modalities)
    if not modality_set:
        return ""
    lines = [
        "Ablation protocol:",
        "- These rules override any generic cue description below.",
    ]
    if "visual" in modality_set:
        lines.append("- Visual video/image evidence is hidden; use language and remaining non-visual fields only.")
    if "gaze" in modality_set:
        lines.append(
            "- Gaze is hidden: do not use gaze_summary, gazePoint, gazeVector, gazeOrigin, "
            "sparse eye-cue proposals, or any visible green gaze marker if it appears."
        )
    if "hand" in modality_set:
        lines.append("- Hand/gesture evidence is hidden: do not use hand_summary or hand/ray cues.")
    if "structured_geometry" in modality_set:
        lines.append("- Structured world/camera geometry is hidden except for the required candidate anchor list.")
    if "timeline" in modality_set:
        lines.append("- Sparse timeline evidence segments are hidden.")
    return "\n".join(lines) + "\n"


def mask_peak_spatial_for_ablation(peak_data: Dict[str, Any], modalities: Sequence[str] | Set[str]) -> Dict[str, Any]:
    if modality_disabled(modalities, "structured_geometry"):
        return {}
    result = dict(peak_data)
    if modality_disabled(modalities, "gaze"):
        for key in (
            "gaze_point",
            "gaze_vector",
            "gaze_origin",
            "camera_gaze_origin",
            "camera_gaze_direction",
        ):
            result.pop(key, None)
    if modality_disabled(modalities, "hand"):
        for key in (
            "right_index_ray_hit_point",
            "rightIndexFingerRayHitPoint",
            "right_hand_hit_point",
        ):
            result.pop(key, None)
    return result


def sanitize_spatial_context_text(text: Any) -> str:
    raw = normalize_text(text)
    if not raw:
        return "No spatial context was provided."
    blocked_terms = (
        "spatial_prior_source",
        "spatial_prior_u_norm",
        "spatial_prior_v_norm",
        "u_norm_prior",
        "v_norm_prior",
        "image_size=",
        "camera_hit_point",
        "peak_camera_hit_point",
        "right_index_ray_hit_point",
        "peak_right_index_ray_hit_point",
        "prior:",
        "prior ",
    )
    kept_lines: List[str] = []
    for line in raw.splitlines():
        lowered = line.lower()
        if any(term.lower() in lowered for term in blocked_terms):
            continue
        kept_lines.append(line)
    sanitized = " ".join(" ".join(kept_lines).split())
    return sanitized or "Spatial context was provided, but prior-related text was removed."


def read_row(input_csv: Path, row_index: int) -> Dict[str, str]:
    if not input_csv.exists() or not input_csv.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f"row_index {row_index} out of range. CSV rows: {len(rows)}")
    return rows[row_index]


def encode_video_as_data_url(video_path: Path) -> str:
    suffix = video_path.suffix.lower()
    mime_type = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }.get(suffix, "video/mp4")
    with video_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def first_nonempty(*values: Any, fallback: str = "") -> str:
    for value in values:
        text = normalize_text(value)
        if text:
            return text
    return fallback


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_iso_timestamp(raw_value: Any) -> str:
    text = normalize_text(raw_value)
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


def repair_json_array_if_needed(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("[") and stripped.count("[") == stripped.count("]") + 1 and stripped.endswith("}"):
        return stripped + "\n]"
    return stripped


def load_multimodal_samples(json_path: Path) -> List[Dict[str, Any]]:
    raw_text = json_path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        payload = json.loads(repair_json_array_if_needed(raw_text))
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def collect_timed_samples(samples: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    dated_samples: List[Tuple[dt.datetime, Dict[str, Any]]] = []
    for sample in samples:
        sample_datetime = parse_sample_datetime(sample.get("timestamp"))
        if sample_datetime is not None:
            dated_samples.append((sample_datetime, sample))
    if not dated_samples:
        return []
    dated_samples.sort(key=lambda item: item[0])
    base_time = dated_samples[0][0]
    return [((sample_datetime - base_time).total_seconds(), sample) for sample_datetime, sample in dated_samples]


def point3_from_mapping(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, dict):
        return None
    try:
        point = (float(value["x"]), float(value["y"]), float(value["z"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in point):
        return None
    if sum(abs(item) for item in point) <= 1e-6:
        return None
    return point


def distance3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3)))


def sample_gaze_point(sample: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    eye_gaze = sample.get("eyeGaze") if isinstance(sample.get("eyeGaze"), dict) else {}
    if eye_gaze.get("isEyeOpen") is False:
        return None
    return point3_from_mapping(eye_gaze.get("gazePoint"))


def sample_eye_cue(sample_time: float, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    eye_gaze = sample.get("eyeGaze") if isinstance(sample.get("eyeGaze"), dict) else {}
    if eye_gaze.get("isEyeOpen") is False:
        return None
    gaze_point = point3_from_mapping(eye_gaze.get("gazePoint"))
    if gaze_point is None:
        return None
    return {
        "time": sample_time,
        "gaze_point": gaze_point,
        "gaze_origin": point3_from_mapping(eye_gaze.get("gazeOrigin")),
        "gaze_vector": point3_from_mapping(eye_gaze.get("gazeVector")),
    }


def sample_right_hand_hit(sample: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    hand_data = sample.get("handData") if isinstance(sample.get("handData"), dict) else {}
    if hand_data.get("isRightHandTracked") is False:
        return None
    return point3_from_mapping(hand_data.get("rightIndexFingerRayHitPoint"))


def build_sparse_timeline_evidence(
    row: Dict[str, str],
    anchor_rows: List[Dict[str, Any]],
    max_segments: int,
    segment_duration: float,
    top_anchors_per_segment: int = 3,
) -> List[Dict[str, Any]]:
    if max_segments <= 0 or segment_duration <= 0:
        return []
    json_path_text = normalize_text(row.get("json_path"))
    if not json_path_text:
        return []
    json_path = Path(json_path_text).expanduser()
    if not json_path.exists() or not json_path.is_file():
        return []

    try:
        timed_samples = collect_timed_samples(load_multimodal_samples(json_path))
    except Exception:
        return []
    if not timed_samples:
        return []

    t_start = parse_float(row.get("t_start"))
    t_end = parse_float(row.get("t_end"))
    if t_start is None:
        t_start = timed_samples[0][0]
    if t_end is None:
        t_end = timed_samples[-1][0]
    if t_end <= t_start:
        return []

    anchor_points = [
        (
            item["object_name"],
            (float(item["x_world"]), float(item["y_world"]), float(item["z_world"])),
        )
        for item in anchor_rows
        if all(key in item for key in ("object_name", "x_world", "y_world", "z_world"))
    ]
    if not anchor_points:
        return []

    segments: List[Dict[str, Any]] = []
    segment_index = 0
    current_start = t_start
    while current_start < t_end:
        current_end = min(current_start + segment_duration, t_end)
        segment_samples = [
            (sample_time, sample)
            for sample_time, sample in timed_samples
            if current_start <= sample_time < current_end
        ]
        valid_gaze_points = [
            (sample_time, gaze_point)
            for sample_time, sample in segment_samples
            for gaze_point in [sample_gaze_point(sample)]
            if gaze_point is not None
        ]
        eye_cues = [
            cue
            for sample_time, sample in segment_samples
            for cue in [sample_eye_cue(sample_time, sample)]
            if cue is not None
        ]
        hand_hits = [
            (sample_time, hand_hit)
            for sample_time, sample in segment_samples
            for hand_hit in [sample_right_hand_hit(sample)]
            if hand_hit is not None
        ]
        if len(valid_gaze_points) >= 2:
            anchor_stats: List[Dict[str, Any]] = []
            for object_name, anchor_point in anchor_points:
                distances = [distance3(gaze_point, anchor_point) for _, gaze_point in valid_gaze_points]
                min_distance = min(distances)
                nearest_index = distances.index(min_distance)
                close_count = sum(1 for item in distances if item <= min_distance + 0.75)
                anchor_stats.append(
                    {
                        "object_name": object_name,
                        "min_gaze_distance": min_distance,
                        "nearest_time": valid_gaze_points[nearest_index][0],
                        "close_gaze_frames": close_count,
                    }
                )
            anchor_stats.sort(key=lambda item: (item["min_gaze_distance"], -item["close_gaze_frames"], item["object_name"]))
            top_anchor_stats = anchor_stats[:top_anchors_per_segment]
            best = top_anchor_stats[0]
            representative_eye_cues: List[Dict[str, Any]] = []
            if eye_cues:
                nearest_time = float(best["nearest_time"])
                nearest_eye_cue = min(eye_cues, key=lambda item: abs(float(item["time"]) - nearest_time))
                representative_eye_cues.append(nearest_eye_cue)
                middle_eye_cue = eye_cues[len(eye_cues) // 2]
                if abs(float(middle_eye_cue["time"]) - float(nearest_eye_cue["time"])) > 1e-3:
                    representative_eye_cues.append(middle_eye_cue)
            score = len(valid_gaze_points) / (1.0 + float(best["min_gaze_distance"]))
            segments.append(
                {
                    "segment_id": f"E{segment_index + 1}",
                    "start_time": current_start,
                    "end_time": current_end,
                    "representative_time": float(best["nearest_time"]),
                    "valid_gaze_frames": len(valid_gaze_points),
                    "right_hand_hit_frames": len(hand_hits),
                    "nearest_anchors": top_anchor_stats,
                    "representative_eye_cues": representative_eye_cues[:2],
                    "cue_confidence": "proposal_only",
                    "score": score,
                    "reason_for_inclusion": "valid eye-cue samples with nearby candidate anchors",
                }
            )
        segment_index += 1
        current_start += segment_duration

    segments.sort(key=lambda item: item["score"], reverse=True)
    selected: List[Dict[str, Any]] = []
    min_gap = max(0.25, segment_duration * 0.75)
    for segment in segments:
        if any(abs(float(segment["representative_time"]) - float(existing["representative_time"])) < min_gap for existing in selected):
            continue
        selected.append(segment)
        if len(selected) >= max_segments:
            break
    selected.sort(key=lambda item: item["representative_time"])
    for index, segment in enumerate(selected, start=1):
        segment["segment_id"] = f"E{index}"
    return selected


def format_sparse_timeline_evidence(evidence_segments: List[Dict[str, Any]]) -> str:
    if not evidence_segments:
        return (
            "No sparse eye-cue proposal segments passed the validity filter. "
            "Use the uploaded visual input, language, and candidate list without forcing extra referents."
        )
    lines: List[str] = []
    lines.append(
        "These are sparse eye-cue proposals generated by a pre-filter. "
        "They may contain useful evidence, but may also include incidental gaze or background. "
        "Nearest anchors are geometric candidates, not ground-truth labels."
    )
    for segment in evidence_segments:
        lines.append(
            f"{segment['segment_id']}: {float(segment['start_time']):.2f}-{float(segment['end_time']):.2f}s "
            f"(representative_time={float(segment['representative_time']):.2f}s, "
            f"valid_gaze_frames={segment['valid_gaze_frames']}, "
            f"right_hand_hit_frames={segment['right_hand_hit_frames']}, "
            f"cue_confidence={segment.get('cue_confidence', 'proposal_only')})"
        )
        lines.append(f"  inclusion_reason: {segment['reason_for_inclusion']}")
        eye_cues = segment.get("representative_eye_cues", [])
        if eye_cues:
            lines.append("  representative_eye_cues:")
            for cue in eye_cues:
                gaze_point = cue.get("gaze_point")
                gaze_origin = cue.get("gaze_origin")
                gaze_vector = cue.get("gaze_vector")
                lines.append(
                    f"    - t={float(cue['time']):.2f}s "
                    f"gazePoint={format_point_tuple(gaze_point)} "
                    f"gazeOrigin={format_point_tuple(gaze_origin)} "
                    f"gazeVector={format_point_tuple(gaze_vector)}"
                )
        lines.append("  nearest_anchor_candidates:")
        for anchor in segment.get("nearest_anchors", []):
            lines.append(
                f"    - {anchor['object_name']}: min_gaze_distance={float(anchor['min_gaze_distance']):.3f}, "
                f"nearest_time={float(anchor['nearest_time']):.2f}s, "
                f"close_gaze_frames={anchor['close_gaze_frames']}"
            )
        lines.append("  interpretation_warning: proposal only; do not select objects unless supported by instruction semantics and visual evidence.")
    return "\n".join(lines)


def format_point_tuple(point: Any) -> str:
    if not isinstance(point, tuple) or len(point) != 3:
        return "unknown"
    return f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"


def parse_peak_spatial(row: Dict[str, str]) -> Dict[str, Any]:
    spatial_context_json = normalize_text(row.get("spatial_context_json"), "{}")
    try:
        payload = json.loads(spatial_context_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    peak_data = payload.get("peak_spatial")
    return peak_data if isinstance(peak_data, dict) else {}


def format_xyz(value: Any) -> str:
    if not isinstance(value, dict):
        return "unknown"
    try:
        if {"x", "y", "z", "w"} <= set(value.keys()):
            return f"({float(value['x']):.3f}, {float(value['y']):.3f}, {float(value['z']):.3f}, {float(value['w']):.3f})"
        if {"x", "y", "z"} <= set(value.keys()):
            return f"({float(value['x']):.3f}, {float(value['y']):.3f}, {float(value['z']):.3f})"
    except Exception:
        return "unknown"
    return json.dumps(value, ensure_ascii=False)


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        return dialect.delimiter
    except csv.Error:
        return "\t" if "\t" in sample else ","


def load_scene_anchor_table(path: Path) -> List[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Scene anchor CSV does not exist: {path}")
    raw_text = path.read_text(encoding="utf-8-sig")
    delimiter = detect_delimiter(raw_text)
    reader = csv.DictReader(raw_text.splitlines(), delimiter=delimiter)
    if reader.fieldnames is None:
        raise ValueError(f"Scene anchor table has no header: {path}")
    normalized_headers = {name.strip(): name for name in reader.fieldnames}
    name_key = normalized_headers.get("物体名称") or normalized_headers.get("object_name")
    x_key = normalized_headers.get("location_x") or normalized_headers.get("x_world")
    y_key = normalized_headers.get("location_y") or normalized_headers.get("y_world")
    z_key = normalized_headers.get("location_z") or normalized_headers.get("z_world")
    aliases_key = normalized_headers.get("aliases") or normalized_headers.get("alias") or normalized_headers.get("object_aliases")
    missing = [label for label, key in (("物体名称/object_name", name_key), ("location_x/x_world", x_key), ("location_y/y_world", y_key), ("location_z/z_world", z_key)) if key is None]
    if missing:
        raise ValueError(f"Scene anchor table is missing required columns: {', '.join(missing)}")

    anchor_rows: List[Dict[str, Any]] = []
    for row in reader:
        object_name = normalize_text(row.get(name_key))
        if not object_name:
            continue
        x_world = parse_float(row.get(x_key))
        y_world = parse_float(row.get(y_key))
        z_world = parse_float(row.get(z_key))
        if x_world is None or y_world is None or z_world is None:
            continue
        aliases = []
        if aliases_key is not None:
            aliases = [
                part.strip()
                for part in re.split(r"[,;/|]+", normalize_text(row.get(aliases_key)))
                if part and part.strip()
            ]
        anchor_rows.append(
            {
                "object_name": object_name,
                "aliases": aliases,
                "x_world": x_world,
                "y_world": y_world,
                "z_world": z_world,
            }
        )
    if not anchor_rows:
        raise ValueError(f"Scene anchor table contains no valid candidate rows: {path}")
    return sorted(anchor_rows, key=lambda item: item["object_name"])


def format_scene_anchor_table_for_prompt(anchor_rows: List[Dict[str, Any]]) -> str:
    lines = []
    for row in sorted(anchor_rows, key=lambda item: item["object_name"]):
        alias_text = ""
        aliases = row.get("aliases")
        if isinstance(aliases, list) and aliases:
            alias_text = f" | aliases: {', '.join(str(alias) for alias in aliases)}"
        lines.append(f"- {row['object_name']}: ({row['x_world']:.3f}, {row['y_world']:.3f}, {row['z_world']:.3f}){alias_text}")
    return "\n".join(lines)


def build_mention_first_3d_object_prompt(
    row: Dict[str, str],
    anchor_rows: List[Dict[str, Any]],
    prompt_style: str,
    ablate_modalities: Any = "",
) -> str:
    disabled_modalities = parse_ablation_modalities(ablate_modalities)
    instruction_text = first_nonempty(row.get("instruction_text"), row.get("target_description"), fallback="Describe and identify the intended referent object for this VR event.")
    utterance_text = normalize_text(row.get("utterance_text"), "No utterance text was provided.")
    target_description = first_nonempty(row.get("target_description"), row.get("instruction_text"), fallback="No target description was provided.")
    gaze_summary = (
        "Hidden by ablation."
        if modality_disabled(disabled_modalities, "gaze")
        else compress_summary(row.get("gaze_summary") or "No gaze summary was provided.")
    )
    hand_summary = (
        "Hidden by ablation."
        if modality_disabled(disabled_modalities, "hand")
        else compress_summary(row.get("hand_summary") or "No hand summary was provided.")
    )
    spatial_context_text = (
        "Hidden by ablation except for the candidate anchor list."
        if modality_disabled(disabled_modalities, "structured_geometry")
        else compress_summary(sanitize_spatial_context_text(row.get("spatial_context_text")), max_chars=260 if prompt_style == "full" else 220)
    )
    peak_data = mask_peak_spatial_for_ablation(parse_peak_spatial(row), disabled_modalities)
    candidate_table = format_scene_anchor_table_for_prompt(anchor_rows)
    if modality_disabled(disabled_modalities, "visual"):
        style_note = "The visual video/storyboard is hidden; do not infer from the placeholder image."
    else:
        style_note = (
            "Use both the uploaded video/storyboard and the world-coordinate cues below."
            if prompt_style == "full"
            else "Use the uploaded video/storyboard as the main evidence and treat world-coordinate cues as lightweight support."
        )
    ablation_note = build_ablation_protocol_note(disabled_modalities)
    if modality_disabled(disabled_modalities, "gaze"):
        gaze_evidence_lines = [
            "- Gaze cues are hidden for this ablation; ignore gaze marker descriptions and do not cite gaze or eye_cue evidence.",
            "- When several candidates share the same class/name prefix, use language, visible video/storyboard evidence, and available non-gaze context; return no match if the instance remains ambiguous.",
        ]
    else:
        gaze_evidence_lines = [
            "- The green gaze marker helps disambiguate which candidate instance a mention refers to, especially among same-class candidates.",
            "- A single gaze marker must not suppress additional explicit language referents.",
            "- Use gazePoint, gazeVector, gazeOrigin, cameraPosition, and spatial_context_text as eye_cue evidence to compare candidate instances.",
            "- When several candidates share the same class/name prefix, do not default to the first candidate, a random numbered candidate, or the visually largest candidate. Choose the instance best supported by eye_cue evidence, video/storyboard evidence, and language constraints.",
            '- For actor/control-subject mentions such as "this drone", "that drone", "that colleague", or "the person", only output a specific numbered instance when eye_cue, visual/storyboard evidence, or direct language evidence identifies it.',
            "- gazePoint is a useful 3D cue, but do not treat raw coordinates as a final label.",
        ]
    if modality_disabled(disabled_modalities, "hand"):
        hand_evidence_line = "- Hand cues are hidden for this ablation; do not cite hand evidence."
    else:
        hand_evidence_line = "- hand_summary is weak auxiliary context only."
    evidence_use_text = "\n".join(
        [
            f"- {style_note}",
            "- Language determines which referent mentions must be considered.",
            *gaze_evidence_lines,
            "- If same-class candidates cannot be disambiguated, return no match for that ambiguous mention instead of guessing a numbered instance.",
            "- Visual evidence can reject a candidate if it is clearly inconsistent with the scene.",
            hand_evidence_line,
            "- rightIndexFingerRayHitPoint and cameraHitPoint are noisy and must not be primary sources.",
        ]
    )

    return f'''You are doing mention-first multimodal 3D referent object selection for one VR interaction event.

{ablation_note}
Task:
Complete the decision in this exact order:
1. Read instruction_text and target_description.
2. Extract every explicit referent mention from the language.
3. Classify each mention as an actor/control subject, action target, spatial anchor/context object, category/group, or unsupported/vague location.
4. For each mention, decide whether it refers to one object, multiple object instances, a group/category, or no reliable candidate.
5. Match each mention to exact canonical labels from the candidate object list.
6. Merge all mention-level selections into selected_object_names.

Core goal:
- The main failure mode to avoid is selecting only one object when the instruction contains multiple referents.
- Include the controlled actor/control subject when it is explicitly mentioned and can be matched reliably, for example "this drone", "that colleague", "the person".
- Include concrete spatial anchor/context objects when they are explicit candidate objects, for example "table", "door", "window", "warehouse", "truck".
- Do not include vague unsupported areas such as "there", "here", "road area", "open yard", or "gap" unless an exact candidate label clearly represents that place.
- Do not invent object names.
- Do not predict free-form coordinates or a 2D point.
- Every selected object must come from the candidate object list exactly.
- If a candidate line lists aliases, use them only for recognition; output the canonical label before the colon.
- Do not expand a category/group phrase into every visible instance by default.
- Treat phrases like "rear row of tanks", "front row of chairs", or "near the warehouses" as locating context unless the instruction clearly commands the whole group.
- Include multiple instances only when the language explicitly asks for all/both/several objects, or the action target is genuinely plural.
- If the instruction contains several separate referent mentions, process each mention independently before merging the final set.
- If a mention is uncertain, keep it in referent_mentions with an empty matched_object_names list and explain why.

Evidence use:
{evidence_use_text}

Candidate matching rules:
- For each mention, candidate_reasoning must compare plausible candidates before giving matched_object_names.
- selection_reason must name the evidence actually used: language, visual/storyboard, eye_cue/gaze, 3D spatial context, or hand cue.
- For category/group mentions, candidate_reasoning must state whether the phrase is the actual target group or only a locator for another target.
- For actor/control-subject mentions with multiple same-class candidates, candidate_reasoning must state why the selected instance is identified; otherwise matched_object_names must be [].
- selected_object_names must equal the union of all non-empty matched_object_names from referent_mentions.
- selected_objects must contain the same objects as selected_object_names, with rank 1 for the most central or first-mentioned referent.
- selected_object_name is a compatibility field: set it to rank 1, or "" if no referent exists.
- expected_referent_count is the number of mentions that should map to candidate objects, not just the number of final selected labels.
- If no reliable candidate can be matched for any mention, output referent_type="none", primary_source="none", selected_object_name="", selected_object_names=[], and selected_objects=[].
- Return one strict JSON object only. Do not use markdown, code fences, comments, or any text before/after JSON.

Language input:
instruction_text: {instruction_text}
utterance_text: {utterance_text}
target_description: {target_description}

Compressed event summaries:
gaze_summary: {gaze_summary}
hand_summary: {hand_summary} (weak auxiliary description only)

Reference-moment world-coordinate cues:
gazePoint: {format_xyz(peak_data.get("gaze_point"))}
gazeVector: {format_xyz(peak_data.get("gaze_vector"))}
gazeOrigin: {format_xyz(peak_data.get("gaze_origin"))}
cameraPosition: {format_xyz(peak_data.get("camera_position"))}
cameraRotation: {format_xyz(peak_data.get("camera_rotation"))}
cameraFOV: {peak_data.get("camera_fov", "unknown")}

Compressed structured context:
spatial_context_text: {spatial_context_text}

Candidate object list:
{candidate_table}

Return JSON with this schema:
{{
  "best_timestamp_seconds": 0.0,
  "referent_type": "entity|spatial|none",
  "primary_source": "gazePoint|visual_only|language|none",
  "expected_referent_count": 1,
  "referent_mentions": [
    {{
      "mention_text": "exact phrase or normalized phrase from the instruction",
      "mention_role": "actor|target|spatial_anchor|category|unsupported_location|unknown",
      "mention_type": "single|multiple|category|spatial|unknown",
      "candidate_reasoning": "brief comparison of plausible candidate labels, especially same-class instances",
      "matched_object_names": ["candidate_label"],
      "selection_reason": "brief reason naming language, visual/storyboard, eye_cue/gaze, 3D spatial context, or hand cue evidence"
    }}
  ],
  "selected_object_name": "rank_1_candidate_or_empty",
  "selected_object_names": ["one_or_more_candidate_labels_when_present"],
  "selected_object_rank": 1,
  "selected_objects": [
    {{"object_name": "candidate_label", "rank": 1, "selection_reason": "brief reason"}}
  ],
  "x_world": 0.0,
  "y_world": 0.0,
  "z_world": 0.0,
  "referent_text": "",
  "reasoning_summary": "",
  "validation_note": "",
  "confidence": 0.0
}}'''


def build_3d_object_prompt(
    row: Dict[str, str],
    anchor_rows: List[Dict[str, Any]],
    prompt_style: str,
    prompt_strategy: str = "standard",
    max_evidence_segments: int = 4,
    evidence_segment_duration: float = 0.5,
    ablate_modalities: Any = "",
) -> str:
    disabled_modalities = parse_ablation_modalities(ablate_modalities)
    if prompt_strategy == "mention_first":
        return build_mention_first_3d_object_prompt(row, anchor_rows, prompt_style, disabled_modalities)

    instruction_text = first_nonempty(row.get("instruction_text"), row.get("target_description"), fallback="Describe and identify the intended referent object for this VR event.")
    utterance_text = normalize_text(row.get("utterance_text"), "No utterance text was provided.")
    target_description = first_nonempty(row.get("target_description"), row.get("instruction_text"), fallback="No target description was provided.")
    gaze_summary = (
        "Hidden by ablation."
        if modality_disabled(disabled_modalities, "gaze")
        else compress_summary(row.get("gaze_summary") or "No gaze summary was provided.")
    )
    hand_summary = (
        "Hidden by ablation."
        if modality_disabled(disabled_modalities, "hand")
        else compress_summary(row.get("hand_summary") or "No hand summary was provided.")
    )
    spatial_context_text = (
        "Hidden by ablation except for the candidate anchor list."
        if modality_disabled(disabled_modalities, "structured_geometry")
        else compress_summary(sanitize_spatial_context_text(row.get("spatial_context_text")), max_chars=260 if prompt_style == "full" else 220)
    )
    peak_data = mask_peak_spatial_for_ablation(parse_peak_spatial(row), disabled_modalities)
    candidate_table = format_scene_anchor_table_for_prompt(anchor_rows)
    if modality_disabled(disabled_modalities, "gaze", "timeline", "structured_geometry"):
        timeline_evidence_text = "Hidden by ablation."
    else:
        timeline_evidence = build_sparse_timeline_evidence(
            row=row,
            anchor_rows=anchor_rows,
            max_segments=max_evidence_segments,
            segment_duration=evidence_segment_duration,
        )
        timeline_evidence_text = format_sparse_timeline_evidence(timeline_evidence)
    if modality_disabled(disabled_modalities, "visual"):
        style_note = "The visual video is hidden; do not infer from the placeholder image."
    else:
        style_note = (
            "Use both the uploaded video and the world-coordinate cues below."
            if prompt_style == "full"
            else "Use the uploaded video as the main evidence and treat the world-coordinate cues below as lightweight support."
        )
    ablation_note = build_ablation_protocol_note(disabled_modalities)
    if modality_disabled(disabled_modalities, "gaze"):
        marker_definition_text = "- Gaze marker cues are hidden for this ablation; ignore any gaze marker text and do not cite gaze evidence."
        geometry_source_text = "- Gaze geometry is hidden for this ablation; use language, visual evidence, candidate anchors, and non-gaze context only."
        entity_selection_text = "- For entity referents, choose candidate objects whose visible objects match language semantics or explicit plural/multi-object instruction."
        hard_gaze_text = "- Gaze is hidden for this ablation; do not choose primary_source = \"gazePoint\" and do not rely on the green marker."
    else:
        marker_definition_text = "- If the green gaze marker is clearly visible and consistent with the intended referent, prioritize it over all other cues."
        geometry_source_text = "{geometry_source_text}"
        entity_selection_text = "{entity_selection_text}"
        hard_gaze_text = "- When the green gaze marker is clearly visible, prioritize it over all other cues.\n- Select the object indicated by the green gaze marker, not a merely salient object unsupported by the gaze marker.\n- If the green marker falls slightly off the object but clearly refers to a nearby object, still choose the nearby object supported by the marker.\n- If gaze is usable, choose primary_source = \"gazePoint\"."
    if modality_disabled(disabled_modalities, "hand"):
        hand_marker_text = "- Hand/gesture cues are hidden for this ablation."
        hand_summary_text = "- hand_summary is hidden for this ablation."
        hand_aux_text = "- Do not use the virtual hand or hand_summary as evidence in this ablation."
    else:
        hand_marker_text = "- A translucent virtual right hand is the visualized hand/gesture cue.\n- It is only a weak auxiliary disambiguation cue.\n- Do not use the hand as the primary source."
        hand_summary_text = "{hand_summary_text}"
        hand_aux_text = "{hand_aux_text}"

    return f'''You are doing multimodal 3D referent object selection for one VR interaction event.

{ablation_note}
Task:
Given the uploaded video and the structured cues below, complete the decision in this exact order:
1. choose best_timestamp_seconds
2. identify referent_type
3. choose primary_source
4. parse all explicit referent mentions in the instruction
5. inspect the sparse evidence segments and decide which segments are useful or ignorable
6. select all candidate objects from the candidate list that are explicitly intended referents

Core goal:
- Do not predict a free-form 2D point.
- Do not invent a new object name.
- Select every intended candidate object name from the provided candidate list when reliable referents exist.
- If the instruction clearly refers to multiple objects, output all of them.
- If the instruction refers to a category/plural group, include every candidate instance that is part of the intended group.
- If the instruction refers to one object, output a one-element list.
- The number of selected objects is variable: it can be 1 to N, usually not more than 4, but do not force it to be 2 or any fixed number.
- Never add extra objects just to fill the list.
- Sparse eye-cue proposal segments are proposals, not required selections.
- Do not select an object merely because it appears in an evidence segment.
- Select an object only when it is supported by the instruction language and at least one useful visual or geometry cue.
- Ignore weak or unrelated evidence segments instead of turning every segment into a referent.
- The program will resolve x_world, y_world, z_world for each selected object by looking up its name in the candidate table.

Visual marker definitions in the uploaded video:
- A small green dot is the visualized gaze marker.
- It indicates the current gaze-based attention location in the image.
{hand_marker_text}
{marker_definition_text}

Geometric source rules:
- gazePoint is the only trusted geometric source
- gazePoint is a 3D world cue
- rightIndexFingerRayHitPoint is noisy and must not be used as the primary source
- cameraHitPoint must not be used as a grounding source
- no trusted explicit 2D prior is provided

Candidate object rules:
- Every final object must be selected from the provided candidate list.
- Do not invent new object names outside the list.
- selected_object_names must contain exact candidate labels from the list.
- selected_object_names is a variable-length list. It may contain 1 object, several objects, or [] when no reliable referent exists.
- selected_objects must contain the same variable-length set of objects as selected_object_names, with rank 1 for the most central/primary referent.
- selected_object_name is a compatibility field: set it to the rank-1 object, or "" when no referent exists.
- If a candidate line lists aliases, those aliases are only alternative names for the same object; output the canonical label before the colon, not the alias.
- For entity referents, choose all candidate objects whose visible objects match the green gaze marker, language semantics, or explicit plural/multi-object instruction.
- hand_summary only provides weak auxiliary context.
- The virtual hand may help disambiguate between multiple nearby gaze-consistent objects, but it cannot become the primary source.
- Sparse eye-cue proposals are computed before this prompt from valid gazePoint samples, gaze vectors, and scene-anchor proximity over the full event window.
- Use sparse eye-cue proposals to recover additional referents that may appear at different times, but do not treat nearest anchor candidates as automatic labels.

Hard rules:
- {style_note}
{hard_gaze_text}
- If the language names multiple referents, do not collapse them into only one object unless the visual evidence makes the others clearly impossible.
- If the instruction contains multiple acceptable referents, output every clearly intended candidate rather than only the first one.
- If the instruction names only one referent, output exactly one object.
- If you are uncertain about an additional object, do not include it unless there is clear language or visual support.

- Choose visual_only only if the gaze marker is clearly unreliable, invisible, or inconsistent with the visible referent.
- If no reliable referent object can be chosen from the candidate list, output referent_type = "none", primary_source = "none", selected_object_name = "", selected_object_names = [], and selected_objects = [].
- Return strict JSON only.

Language input:
instruction_text: {instruction_text}
utterance_text: {utterance_text}
target_description: {target_description}

Compressed event summaries:
gaze_summary: {gaze_summary}
hand_summary: {hand_summary} (weak auxiliary description only, not a valid primary source)

Reference-moment world-coordinate cues:
gazePoint: {format_xyz(peak_data.get("gaze_point"))}
gazeVector: {format_xyz(peak_data.get("gaze_vector"))}
gazeOrigin: {format_xyz(peak_data.get("gaze_origin"))}
cameraPosition: {format_xyz(peak_data.get("camera_position"))}
cameraRotation: {format_xyz(peak_data.get("camera_rotation"))}
cameraFOV: {peak_data.get("camera_fov", "unknown")}

Compressed structured context:
spatial_context_text: {spatial_context_text}

Sparse eye-cue proposals:
{timeline_evidence_text}

Candidate object list:
{candidate_table}

Validation reminders:
- If primary_source == "gazePoint", validation_note should mention whether the green marker was clearly visible.
- If primary_source == "visual_only", validation_note should mention that the gaze marker was unreliable, invisible, or inconsistent.
- selected_object_name and every item in selected_object_names must come from the candidate object list exactly.
- Do not output a descriptive nickname instead of the exact candidate label.

Return JSON with this schema:
{{
  "best_timestamp_seconds": 0.0,
  "referent_type": "entity|spatial|none",
  "primary_source": "gazePoint|visual_only|none",
  "expected_referent_count": 1,
  "referent_mentions": [
    {{
      "mention_text": "referent phrase from the instruction",
      "evidence_segment_ids": ["E1"],
      "matched_object_names": ["candidate_label"],
      "selection_reason": "brief reason"
    }}
  ],
  "ignored_evidence_segments": [
    {{"segment_id": "E2", "reason": "weak, background, or not linked to any instruction mention"}}
  ],
  "selected_object_name": "rank_1_candidate_or_empty",
  "selected_object_names": ["one_or_more_candidate_labels_when_present"],
  "selected_object_rank": 1,
  "selected_objects": [
    {{"object_name": "candidate_label", "rank": 1, "selection_reason": "brief reason"}}
  ],
  "x_world": 0.0,
  "y_world": 0.0,
  "z_world": 0.0,
  "referent_text": "",
  "reasoning_summary": "",
  "validation_note": "",
  "confidence": 0.0
}}'''


def call_api(*, base_url: str, api_key: str, model: str, prompt_text: str, video_data_url: str, fps: float, max_tokens: int, temperature: float, timeout_seconds: int) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_data_url},
                        "fps": fps,
                    },
                    {
                        "type": "text",
                        "text": prompt_text,
                    },
                ],
            }
        ],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        region_hint = ""
        if exc.code == 401:
            region_hint = (
                " Authentication failed. Please confirm you are using a 百炼 Model Studio API Key for the selected region. "
                "For 北京地域 use --region cn-beijing and base_url=https://dashscope.aliyuncs.com/compatible-mode/v1."
            )
        raise RuntimeError(f"HTTP {exc.code}: {error_body}{region_hint}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc}") from exc


def extract_text_response(payload: Dict[str, Any]) -> str:
    try:
        return payload["choices"][0]["message"]["content"]
    except Exception:
        return ""


def extract_json_object_text(raw_text: str) -> str:
    text = normalize_text(raw_text)
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def parse_3d_response_text(raw_text: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    text = extract_json_object_text(raw_text)
    if not text:
        warnings.append("empty_response_text")
        return None, warnings
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        warnings.append("response_text_not_valid_json")
        return None, warnings
    if not isinstance(payload, dict):
        warnings.append("response_json_not_object")
        return None, warnings
    for key in (
        "best_timestamp_seconds",
        "referent_type",
        "primary_source",
        "expected_referent_count",
        "referent_mentions",
        "selected_object_name",
        "selected_object_names",
        "selected_object_rank",
        "selected_objects",
        "x_world",
        "y_world",
        "z_world",
        "referent_text",
        "reasoning_summary",
        "validation_note",
        "confidence",
    ):
        if key not in payload:
            warnings.append(f"missing_field:{key}")
    return payload, warnings


def resolve_selected_object_to_world(anchor_rows: List[Dict[str, Any]], selected_object_name: str) -> Optional[Dict[str, Any]]:
    target = normalize_text(selected_object_name)
    target_normalized = target.lower()
    if not target:
        return None
    for row in anchor_rows:
        aliases = row.get("aliases") if isinstance(row.get("aliases"), list) else []
        candidates = [row["object_name"], *[str(alias) for alias in aliases]]
        if any(candidate.lower() == target_normalized for candidate in candidates):
            return dict(row)
    return None


def iter_response_object_names(parsed_response: Dict[str, Any]) -> List[str]:
    """Collect selected object names from the new multi-object schema and old single-object schema."""
    names: List[str] = []

    selected_objects = parsed_response.get("selected_objects")
    if isinstance(selected_objects, list):
        for item in selected_objects:
            if isinstance(item, dict):
                names.append(normalize_text(item.get("object_name") or item.get("selected_object_name")))
            else:
                names.append(normalize_text(item))

    selected_object_names = parsed_response.get("selected_object_names")
    if isinstance(selected_object_names, list):
        names.extend(normalize_text(item) for item in selected_object_names)
    elif isinstance(selected_object_names, str):
        names.extend(part.strip() for part in re.split(r"[,;/|]+", selected_object_names) if part.strip())

    legacy_name = normalize_text(parsed_response.get("selected_object_name"))
    if legacy_name:
        names.append(legacy_name)

    referent_mentions = parsed_response.get("referent_mentions")
    if isinstance(referent_mentions, list):
        for mention in referent_mentions:
            if isinstance(mention, dict):
                names.extend(split_prediction_like_names(mention.get("matched_object_names")))

    deduped: List[str] = []
    seen = set()
    for name in names:
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def split_prediction_like_names(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_text(item.get("object_name") if isinstance(item, dict) else item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[,;/|]+", value) if part.strip()]
    return []


def resolve_selected_objects_to_world(anchor_rows: List[Dict[str, Any]], selected_object_names: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    resolved_rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    seen = set()
    for selected_name in selected_object_names:
        resolved = resolve_selected_object_to_world(anchor_rows, selected_name)
        if resolved is None:
            warnings.append(f"invalid_object_name:{selected_name}")
            continue
        key = resolved["object_name"].lower()
        if key in seen:
            continue
        seen.add(key)
        resolved_rows.append(resolved)
    return resolved_rows, warnings


def quaternion_conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    return (-x, -y, -z, w)


def quaternion_multiply(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate_vector_by_quaternion(vector: Tuple[float, float, float], quaternion: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    qvec = (vector[0], vector[1], vector[2], 0.0)
    rotated = quaternion_multiply(quaternion_multiply(quaternion, qvec), quaternion_conjugate(quaternion))
    return (rotated[0], rotated[1], rotated[2])


def project_world_to_image(peak_data: Dict[str, Any], resolved_object_row: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float], bool]:
    if not resolved_object_row:
        return None, None, False
    camera_position = peak_data.get("camera_position")
    camera_rotation = peak_data.get("camera_rotation")
    camera_fov = parse_float(peak_data.get("camera_fov"))
    if not isinstance(camera_position, dict) or not isinstance(camera_rotation, dict) or camera_fov is None:
        return None, None, False
    try:
        position = (
            float(resolved_object_row["x_world"]) - float(camera_position["x"]),
            float(resolved_object_row["y_world"]) - float(camera_position["y"]),
            float(resolved_object_row["z_world"]) - float(camera_position["z"]),
        )
        rotation = (
            float(camera_rotation["x"]),
            float(camera_rotation["y"]),
            float(camera_rotation["z"]),
            float(camera_rotation["w"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, None, False

    inv_rotation = quaternion_conjugate(rotation)
    local = rotate_vector_by_quaternion(position, inv_rotation)
    x_local, y_local, z_local = local
    if z_local <= 1e-6:
        return None, None, False

    fov_rad = math.radians(camera_fov)
    tan_half = math.tan(fov_rad / 2.0)
    if tan_half <= 1e-6:
        return None, None, False

    u_norm = 0.5 + (x_local / (z_local * tan_half * 2.0))
    v_norm = 0.5 - (y_local / (z_local * tan_half * 2.0))
    if not math.isfinite(u_norm) or not math.isfinite(v_norm):
        return None, None, False
    return u_norm, v_norm, True


def validate_and_adjust_3d_response(parsed_response: Optional[Dict[str, Any]], anchor_rows: List[Dict[str, Any]], prompt_style: str) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], str]:
    if parsed_response is None:
        return None, ["parsed_response_missing"], None, "parsed_response_missing"

    warnings: List[str] = []
    adjusted = dict(parsed_response)
    resolved_object_row: Optional[Dict[str, Any]] = None
    resolved_object_rows: List[Dict[str, Any]] = []

    referent_type = normalize_text(adjusted.get("referent_type"))
    primary_source = normalize_text(adjusted.get("primary_source"))
    selected_object_names = iter_response_object_names(adjusted)
    validation_note = normalize_text(adjusted.get("validation_note"))
    response_status = "ok" if referent_type else "incomplete"

    if prompt_style == "world_only":
        warnings.append("world_only_has_less_structured_context_for_3d_selection")

    if primary_source and primary_source not in ALLOWED_PRIMARY_SOURCES:
        warnings.append(f"invalid_primary_source:{primary_source}")
        response_status = "inconsistent_source"

    if referent_type == "none" or primary_source == "none":
        adjusted["selected_object_name"] = ""
        adjusted["selected_object_names"] = []
        adjusted["selected_objects"] = []
        adjusted["x_world"] = None
        adjusted["y_world"] = None
        adjusted["z_world"] = None
        adjusted["resolved_object_rows"] = []
        if not validation_note:
            adjusted["validation_note"] = "no_reliable_referent"
        return adjusted, warnings, None, response_status

    if primary_source == "gazePoint" and not selected_object_names:
        warnings.append("gazePoint_requires_selected_object_names")
        response_status = "incomplete"

    if selected_object_names:
        resolved_object_rows, invalid_warnings = resolve_selected_objects_to_world(anchor_rows, selected_object_names)
        warnings.extend(invalid_warnings)
        if invalid_warnings and not resolved_object_rows:
            response_status = "invalid_object_name"
        elif invalid_warnings:
            response_status = "partial_invalid_object_name"

        if resolved_object_rows:
            resolved_object_row = resolved_object_rows[0]
            adjusted["selected_object_name"] = resolved_object_row["object_name"]
            adjusted["selected_object_names"] = [row["object_name"] for row in resolved_object_rows]
            adjusted["selected_objects"] = [
                {
                    "object_name": row["object_name"],
                    "rank": index + 1,
                    "x_world": row["x_world"],
                    "y_world": row["y_world"],
                    "z_world": row["z_world"],
                }
                for index, row in enumerate(resolved_object_rows)
            ]
            adjusted["x_world"] = resolved_object_row["x_world"]
            adjusted["y_world"] = resolved_object_row["y_world"]
            adjusted["z_world"] = resolved_object_row["z_world"]
            adjusted["resolved_object_rows"] = resolved_object_rows
        else:
            adjusted["selected_object_name"] = ""
            adjusted["selected_object_names"] = []
            adjusted["selected_objects"] = []
            adjusted["x_world"] = None
            adjusted["y_world"] = None
            adjusted["z_world"] = None
            adjusted["resolved_object_rows"] = []
    else:
        adjusted["selected_object_name"] = ""
        adjusted["selected_object_names"] = []
        adjusted["selected_objects"] = []
        adjusted["x_world"] = None
        adjusted["y_world"] = None
        adjusted["z_world"] = None
        adjusted["resolved_object_rows"] = []

    if primary_source == "gazePoint":
        if not normalize_text(adjusted.get("validation_note")):
            adjusted["validation_note"] = "gaze_guided_object_selection"
    elif primary_source == "visual_only":
        note_text = normalize_text(adjusted.get("validation_note")).lower()
        if "gaze" not in note_text and "unreliable" not in note_text and "invisible" not in note_text and "inconsistent" not in note_text:
            warnings.append("visual_only_missing_gaze_unreliability_note")
            if not normalize_text(adjusted.get("validation_note")):
                adjusted["validation_note"] = "visual_only_due_to_gaze_unreliability_or_inconsistency"

    adjusted["selected_object_rank"] = parse_int(adjusted.get("selected_object_rank")) or 1
    adjusted["best_timestamp_seconds"] = parse_float(adjusted.get("best_timestamp_seconds"))
    adjusted["confidence"] = parse_float(adjusted.get("confidence"))
    adjusted["response_status"] = response_status
    return adjusted, warnings, resolved_object_row, response_status


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.getenv(args.api_key_env) or os.getenv("BAILIAN_API_KEY")
    if not api_key:
        raise SystemExit(f"Missing API key. Set --api_key or environment variable {args.api_key_env} / BAILIAN_API_KEY.")
    base_url = args.base_url or REGION_TO_BASE_URL[args.region]

    row = read_row(Path(args.input_csv), args.row_index)
    anchor_rows = load_scene_anchor_table(Path(args.scene_anchor_csv))
    video_path = Path(args.video_path) if args.video_path else Path(normalize_text(row.get("video_path")))
    if not video_path.exists() or not video_path.is_file():
        raise SystemExit(f"Video file does not exist: {video_path}")

    sparse_timeline_evidence = build_sparse_timeline_evidence(
        row=row,
        anchor_rows=anchor_rows,
        max_segments=args.max_evidence_segments,
        segment_duration=args.evidence_segment_duration,
    )
    prompt_text = build_3d_object_prompt(
        row,
        anchor_rows,
        args.prompt_style,
        prompt_strategy=args.prompt_strategy,
        max_evidence_segments=args.max_evidence_segments,
        evidence_segment_duration=args.evidence_segment_duration,
        ablate_modalities=args.ablate_modalities,
    )
    video_data_url = encode_video_as_data_url(video_path)
    response_payload = call_api(
        base_url=base_url,
        api_key=api_key,
        model=args.model,
        prompt_text=prompt_text,
        video_data_url=video_data_url,
        fps=args.fps,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_seconds=args.timeout_seconds,
    )

    response_text = extract_text_response(response_payload)
    parsed_response, parse_warnings = parse_3d_response_text(response_text)
    adjusted_response, validation_warnings, resolved_object_row, response_status = validate_and_adjust_3d_response(
        parsed_response,
        anchor_rows,
        args.prompt_style,
    )

    peak_data = parse_peak_spatial(row)
    projected_u_norm, projected_v_norm, projection_valid = project_world_to_image(peak_data, resolved_object_row)

    output = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_index": args.row_index,
        "event_id": normalize_text(row.get("event_id")),
        "video_path": str(video_path.resolve()),
        "scene_anchor_csv": str(Path(args.scene_anchor_csv).resolve()),
        "scene_anchor_candidates": anchor_rows,
        "model": args.model,
        "region": args.region,
        "base_url": base_url,
        "fps": args.fps,
        "prompt_strategy": args.prompt_strategy,
        "max_evidence_segments": args.max_evidence_segments,
        "evidence_segment_duration": args.evidence_segment_duration,
        "sparse_timeline_evidence": sparse_timeline_evidence,
        "prompt_style": args.prompt_style,
        "prompt_text": prompt_text,
        "raw_spatial_prior_source": normalize_text(row.get("spatial_prior_source"), "none"),
        "raw_spatial_prior_u_norm": parse_float(row.get("spatial_prior_u_norm")),
        "raw_spatial_prior_v_norm": parse_float(row.get("spatial_prior_v_norm")),
        "response_text": response_text,
        "parsed_response": parsed_response,
        "resolved_object_row": resolved_object_row,
        "resolved_object_rows": adjusted_response.get("resolved_object_rows") if isinstance(adjusted_response, dict) else [],
        "validation_warnings": parse_warnings + validation_warnings,
        "adjusted_response": adjusted_response,
        "response_status": response_status,
        "projected_u_norm": projected_u_norm,
        "projected_v_norm": projected_v_norm,
        "projection_valid": projection_valid,
        "response_source_recommendation": "adjusted_response",
        "response": response_payload,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved API response to: {output_path}")


if __name__ == "__main__":
    main()
