#!/usr/bin/env python3
"""Target-free evidence-frame selection for point-supervised 3D grounding."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import (  # noqa: E402
    EvidenceFrame,
    collect_timed_samples,
    dense_candidate_times,
    effective_video_time_offset,
    extract_frame,
    is_nonzero_point,
    load_multimodal_samples,
    nearest_sample,
    normalize_text,
    parse_float,
    point3,
    vector_distance,
)


def gaze_hit(sample: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    eye = sample.get("eyeGaze")
    if not isinstance(eye, Mapping) or eye.get("isEyeOpen") is False:
        return None
    point = point3(eye.get("gazePoint"))
    return point if is_nonzero_point(point) else None


def gaze_direction(sample: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    eye = sample.get("eyeGaze")
    if not isinstance(eye, Mapping) or eye.get("isEyeOpen") is False:
        return None
    origin = point3(eye.get("gazeOrigin"))
    hit = gaze_hit(sample)
    if origin is not None and hit is not None:
        dx = hit[0] - origin[0]
        dy = hit[1] - origin[1]
        dz = hit[2] - origin[2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm > 1e-9:
            return dx / norm, dy / norm, dz / norm
    vector = point3(eye.get("gazeVector"))
    if vector is None:
        return None
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if norm <= 1e-9:
        return None
    return vector[0] / norm, vector[1] / norm, vector[2] / norm


def hand_hit(sample: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    hand = sample.get("handData")
    if not isinstance(hand, Mapping):
        return None
    tracked = bool(hand.get("isRightHandTracked") or hand.get("isLeftHandTracked"))
    if not tracked:
        return None
    point = point3(hand.get("rightIndexFingerRayHitPoint"))
    return point if is_nonzero_point(point) else None


def hand_origin(sample: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    hand = sample.get("handData")
    if not isinstance(hand, Mapping):
        return None
    right = hand.get("rightHand")
    if not isinstance(right, Mapping):
        return None
    joints = right.get("joints")
    if not isinstance(joints, list) or not joints:
        return None
    candidates = [joint for joint in joints if isinstance(joint, Mapping) and joint.get("id") in (8, 9, 10, 11, 12)]
    if not candidates:
        candidates = [joint for joint in joints if isinstance(joint, Mapping)]
    points = [point3(joint.get("position")) for joint in candidates]
    points = [point for point in points if point is not None]
    if not points:
        return None
    count = float(len(points))
    return (
        sum(point[0] for point in points) / count,
        sum(point[1] for point in points) / count,
        sum(point[2] for point in points) / count,
    )


def valid_camera(sample: Mapping[str, Any]) -> bool:
    return point3(sample.get("cameraPosition")) is not None and isinstance(sample.get("cameraRotation"), Mapping)


def local_stability(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    sample_time: float,
    sample: Mapping[str, Any],
    point_getter,
    window_seconds: float,
    scale: float,
) -> Tuple[float, int]:
    center = point_getter(sample)
    if center is None:
        return 0.0, 0
    distances: List[float] = []
    for other_time, other in timed_samples:
        if abs(other_time - sample_time) <= 1e-6 or abs(other_time - sample_time) > window_seconds:
            continue
        point = point_getter(other)
        if point is not None:
            distances.append(vector_distance(center, point))
    if not distances:
        return 0.35, 0
    avg = sum(distances) / len(distances)
    return max(0.0, 1.0 - avg / scale), len(distances)


def camera_stability(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    sample_time: float,
    sample: Mapping[str, Any],
    window_seconds: float,
) -> Tuple[float, int]:
    return local_stability(timed_samples, sample_time, sample, lambda item: point3(item.get("cameraPosition")), window_seconds, 0.35)


def evidence_score(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    sample_time: float,
    sample: Mapping[str, Any],
    window_seconds: float,
) -> Tuple[float, str, bool, bool]:
    gaze_valid = gaze_hit(sample) is not None or gaze_direction(sample) is not None
    hand_valid = hand_hit(sample) is not None
    cam_valid = valid_camera(sample)
    gaze_stability, gaze_neighbors = local_stability(timed_samples, sample_time, sample, gaze_hit, window_seconds, 4.0)
    hand_stability, hand_neighbors = local_stability(timed_samples, sample_time, sample, hand_hit, window_seconds, 4.0)
    cam_stability, cam_neighbors = camera_stability(timed_samples, sample_time, sample, window_seconds)
    score = 0.0
    reasons: List[str] = []
    if gaze_valid:
        score += 1.5 + 0.8 * gaze_stability
        reasons.append(f"gaze_valid:gaze_stability={gaze_stability:.2f}:neighbors={gaze_neighbors}")
    else:
        reasons.append("gaze_invalid")
    if hand_valid:
        score += 1.1 + 0.7 * hand_stability
        reasons.append(f"hand_valid:hand_stability={hand_stability:.2f}:neighbors={hand_neighbors}")
    else:
        reasons.append("hand_invalid")
    if cam_valid:
        score += 0.3 + 0.4 * cam_stability
        reasons.append(f"camera_valid:camera_stability={cam_stability:.2f}:neighbors={cam_neighbors}")
    else:
        reasons.append("camera_invalid")
    return score, ";".join(reasons), gaze_valid, hand_valid


def choose_diverse_candidates(
    candidates: Sequence[Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]],
    max_frames: int,
    min_sep_seconds: float,
) -> List[Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]]:
    selected: List[Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]] = []

    def can_add(candidate: Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]) -> bool:
        return all(abs(candidate[1] - used[1]) >= min_sep_seconds for used in selected)

    for preferred in ("gaze", "hand"):
        ranked = sorted(candidates, key=lambda item: (-item[4], item[1]))
        for candidate in ranked:
            has_cue = candidate[6] if preferred == "gaze" else candidate[7]
            if has_cue and can_add(candidate):
                selected.append(candidate)
                break
        if len(selected) >= max_frames:
            return sorted(selected, key=lambda item: item[1])

    for candidate in sorted(candidates, key=lambda item: (-item[4], item[1])):
        if len(selected) >= max_frames:
            break
        if can_add(candidate):
            selected.append(candidate)
    return sorted(selected[:max_frames], key=lambda item: item[1])


def fallback_candidates(
    timed_samples: Sequence[Tuple[float, Mapping[str, Any]]],
    sample_start: float,
    sample_end: float,
    video_offset: float,
    max_frames: int,
) -> List[Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]]:
    ratios = [0.25, 0.5, 0.75][:max_frames]
    result = []
    duration = max(0.0, sample_end - sample_start)
    for idx, ratio in enumerate(ratios, start=1):
        target_sample_time = sample_start + duration * ratio
        sample_time, sample = nearest_sample(timed_samples, target_sample_time)
        video_time = sample_time + video_offset
        gaze_valid = gaze_hit(sample) is not None or gaze_direction(sample) is not None
        hand_valid = hand_hit(sample) is not None
        result.append((f"P{idx}", video_time, sample_time, sample, 0.0, f"fallback_ratio:{ratio:.2f}", gaze_valid, hand_valid))
    return result


def select_evidence_frames(
    api_row: Mapping[str, str],
    output_frame_dir: Path,
    max_frames: int = 3,
    candidate_step_seconds: float = 0.5,
    stability_window_seconds: float = 0.3,
    min_sep_seconds: float = 0.85,
    ffmpeg_path: str = "ffmpeg",
    no_extract_frames: bool = False,
    overwrite_frames: bool = False,
) -> Tuple[List[EvidenceFrame], Dict[str, Any]]:
    json_path = Path(normalize_text(api_row.get("json_path")))
    video_path_text = normalize_text(api_row.get("video_path"))
    video_path = Path(video_path_text) if video_path_text else None
    if not json_path.exists():
        raise FileNotFoundError(f"Missing multimodal JSON: {json_path}")
    if video_path is not None and not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")

    samples = load_multimodal_samples(json_path)
    timed_samples = collect_timed_samples(samples)
    if not timed_samples:
        raise RuntimeError(f"No timed samples in {json_path}")

    t_start = parse_float(api_row.get("t_start"))
    t_end = parse_float(api_row.get("t_end"))
    if t_start is None:
        t_start = timed_samples[0][0]
    if t_end is None:
        t_end = timed_samples[-1][0]
    sample_start = max(timed_samples[0][0], t_start)
    sample_end = min(timed_samples[-1][0], t_end)
    video_offset, video_offset_source = effective_video_time_offset(json_path, video_path, samples)

    raw_candidates: List[Tuple[str, float, float, Mapping[str, Any], float, str, bool, bool]] = []
    for idx, target_sample_time in enumerate(dense_candidate_times(sample_start, sample_end, candidate_step_seconds), start=1):
        sample_time, sample = nearest_sample(timed_samples, target_sample_time)
        video_time = sample_time + video_offset
        score, reason, gaze_valid, hand_valid = evidence_score(timed_samples, sample_time, sample, stability_window_seconds)
        raw_candidates.append((f"C{idx}", video_time, sample_time, sample, score, reason, gaze_valid, hand_valid))

    positive_candidates = [item for item in raw_candidates if item[4] > 0.0]
    if positive_candidates:
        selected = choose_diverse_candidates(positive_candidates, max_frames, min_sep_seconds)
    else:
        selected = []
    if not selected:
        selected = fallback_candidates(timed_samples, sample_start, sample_end, video_offset, max_frames)

    frames: List[EvidenceFrame] = []
    extraction_fail_count = 0
    for _candidate_id, video_time, sample_time, sample, score, reason, gaze_valid, hand_valid in selected:
        panel_id = f"P{len(frames) + 1}"
        frame_path = output_frame_dir / f"{panel_id}.jpg"
        extracted = False
        if no_extract_frames:
            extracted = frame_path.exists()
        elif video_path is not None:
            extracted = extract_frame(ffmpeg_path, video_path, video_time, frame_path, overwrite_frames)
        reason_with_extract = reason + f";frame_extracted={extracted}"
        if not extracted or not frame_path.exists():
            extraction_fail_count += 1
            continue
        frames.append(
            EvidenceFrame(
                panel_id=panel_id,
                frame_path=str(frame_path),
                video_time=video_time,
                sample_time=sample_time,
                selection_score=score,
                selection_reason=reason_with_extract,
                cue_gaze_valid=gaze_valid,
                cue_hand_valid=hand_valid,
                sample=sample,
            )
        )
    if not frames:
        raise RuntimeError("No evidence frames extracted for selected candidates.")

    stats = {
        "json_path": str(json_path),
        "video_path": str(video_path) if video_path is not None else "",
        "video_time_offset_seconds": video_offset,
        "video_time_offset_source": video_offset_source,
        "candidate_count": len(raw_candidates),
        "positive_candidate_count": len(positive_candidates),
        "sample_start": sample_start,
        "sample_end": sample_end,
        "requested_selected_count": len(selected),
        "selected_count": len(frames),
        "extraction_fail_count": extraction_fail_count,
    }
    return frames, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select target-free evidence frames for one event row.")
    parser.add_argument("--repo_root", default=".", help="Repository root.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--row_index", type=int, required=True)
    parser.add_argument("--output_dir", default="exam3_point_grounding/outputs/evidence_frames")
    parser.add_argument("--no_extract_frames", action="store_true")
    parser.add_argument("--overwrite_frames", action="store_true")
    parser.add_argument("--ffmpeg_path", default="ffmpeg")
    return parser.parse_args()


def main() -> None:
    from point_grounding_common import read_csv_rows, scene_api_csv, write_json

    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    rows = read_csv_rows(scene_api_csv(repo_root, args.scene))
    if args.row_index < 0 or args.row_index >= len(rows):
        raise RuntimeError(f"row_index out of range: {args.row_index}")
    api_row = rows[args.row_index]
    out_dir = repo_root / args.output_dir / args.scene / f"row_{args.row_index}"
    frames, stats = select_evidence_frames(
        api_row,
        out_dir,
        ffmpeg_path=args.ffmpeg_path,
        no_extract_frames=args.no_extract_frames,
        overwrite_frames=args.overwrite_frames,
    )
    write_json(
        out_dir / "selection_debug.json",
        {
            "scene": args.scene,
            "row_index": args.row_index,
            "stats": stats,
            "frames": [
                {
                    "panel_id": frame.panel_id,
                    "frame_path": frame.frame_path,
                    "video_time": frame.video_time,
                    "sample_time": frame.sample_time,
                    "selection_score": frame.selection_score,
                    "selection_reason": frame.selection_reason,
                    "cue_gaze_valid": frame.cue_gaze_valid,
                    "cue_hand_valid": frame.cue_hand_valid,
                }
                for frame in frames
            ],
        },
    )


if __name__ == "__main__":
    main()
