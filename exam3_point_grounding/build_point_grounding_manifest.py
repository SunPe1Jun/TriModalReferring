#!/usr/bin/env python3
"""Build the candidate-free point-supervised 3D grounding manifest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import (  # noqa: E402
    Anchor,
    SCENES,
    build_gt_from_match_eval,
    camera_basis,
    compact_point,
    format_point,
    is_nonzero_point,
    load_anchor_table,
    nearest_negative_distance,
    normalize_text,
    parse_float,
    parse_int,
    point3,
    read_csv_rows,
    robust_bounds,
    scene_api_csv,
    split_names,
    vector_distance,
    write_csv,
    write_json,
)
from select_target_free_evidence import (  # noqa: E402
    gaze_direction,
    gaze_hit,
    hand_hit,
    hand_origin,
    select_evidence_frames,
)


MANIFEST_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "utterance_text",
    "frame_paths_json",
    "evidence_json",
    "prompt_text",
    "json_path",
    "video_path",
    "t_start",
    "t_end",
    "scene_bounds_json",
    "selection_debug_json",
    "status",
    "status_detail",
)

GT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "gt_anchor_ids",
    "gt_points_json",
    "gt_count",
    "gt_mapping_status",
    "gt_mapping_detail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-supervised 3D grounding manifest.")
    parser.add_argument("--repo_root", default=".", help="Repository root.")
    parser.add_argument("--output_dir", default="exam3_point_grounding/outputs", help="Output directory.")
    parser.add_argument("--scenes", nargs="*", default=list(SCENES), help="Scenes to include.")
    parser.add_argument("--eval_dir", help="Optional match-eval directory.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max_frames", type=int, default=3)
    parser.add_argument("--candidate_step_seconds", type=float, default=0.5)
    parser.add_argument("--stability_window_seconds", type=float, default=0.3)
    parser.add_argument("--min_sep_seconds", type=float, default=0.85)
    parser.add_argument("--ffmpeg_path", default="ffmpeg")
    parser.add_argument("--no_extract_frames", action="store_true")
    parser.add_argument("--overwrite_frames", action="store_true")
    parser.add_argument("--prompt_template", default="exam3_point_grounding/prompts/qwen3vl_point_grounding.md")
    return parser.parse_args()


def choose_rows(rows: Sequence[Mapping[str, str]], start_index: int, limit: Optional[int]) -> Iterable[Tuple[int, Mapping[str, str]]]:
    end_index = len(rows) if limit is None else min(len(rows), start_index + limit)
    for row_index in range(start_index, end_index):
        yield row_index, rows[row_index]


def point_json(point: Optional[Tuple[float, float, float]]) -> Optional[List[float]]:
    if point is None:
        return None
    return [float(point[0]), float(point[1]), float(point[2])]


def distance_json(left: Optional[Tuple[float, float, float]], right: Optional[Tuple[float, float, float]]) -> Optional[float]:
    if left is None or right is None:
        return None
    return round(vector_distance(left, right), 6)


def unit_direction(origin: Optional[Tuple[float, float, float]], hit: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    if origin is None or hit is None:
        return None
    dx = hit[0] - origin[0]
    dy = hit[1] - origin[1]
    dz = hit[2] - origin[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-9:
        return None
    return dx / norm, dy / norm, dz / norm


def sample_telemetry(sample: Mapping[str, Any]) -> Dict[str, Any]:
    camera_position = point3(sample.get("cameraPosition"))
    forward, right, up = camera_basis(sample)
    fov = parse_float(sample.get("cameraFOV"))

    eye = sample.get("eyeGaze") if isinstance(sample.get("eyeGaze"), Mapping) else {}
    gaze_origin = point3(eye.get("gazeOrigin")) if isinstance(eye, Mapping) else None
    g_hit = gaze_hit(sample)
    g_dir = gaze_direction(sample)
    if g_dir is None:
        g_dir = unit_direction(gaze_origin, g_hit)

    h_origin = hand_origin(sample)
    h_hit = hand_hit(sample)
    h_dir = unit_direction(h_origin, h_hit)

    return {
        "camera_position": point_json(camera_position),
        "camera_forward_world": point_json(forward),
        "camera_right_world": point_json(right),
        "camera_up_world": point_json(up),
        "camera_fov_degrees": fov,
        "gaze_valid": g_dir is not None or g_hit is not None,
        "gaze_origin": point_json(gaze_origin),
        "gaze_direction_world": point_json(g_dir),
        "gaze_hit": point_json(g_hit),
        "hand_valid": h_hit is not None,
        "hand_origin": point_json(h_origin),
        "hand_direction_world": point_json(h_dir),
        "hand_hit": point_json(h_hit),
        "camera_to_gaze_hit_distance_world_units": distance_json(camera_position, g_hit),
        "camera_to_hand_hit_distance_world_units": distance_json(camera_position, h_hit),
        "gaze_to_hand_hit_distance_world_units": distance_json(g_hit, h_hit),
    }


def build_evidence_payload(frames: Sequence[Any]) -> List[Dict[str, Any]]:
    payload = []
    for frame in frames:
        telemetry = sample_telemetry(frame.sample)
        payload.append(
            {
                "panel_id": frame.panel_id,
                "frame_path": frame.frame_path,
                "relative_sample_time_seconds": round(frame.sample_time, 6),
                "video_time_seconds": round(frame.video_time, 6),
                "selection_score": round(frame.selection_score, 6),
                "selection_reason": frame.selection_reason,
                "cue_gaze_valid": frame.cue_gaze_valid,
                "cue_hand_valid": frame.cue_hand_valid,
                "telemetry": telemetry,
            }
        )
    return payload


def json_compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def xyz_text(value: Any) -> str:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return "None"
    try:
        x, y, z = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return "None"
    return f"x={x:.6f}, y={y:.6f}, z={z:.6f}"


def build_scene_bounds_block(bounds: Mapping[str, Any]) -> str:
    center = (
        (bounds["x_q05"] + bounds["x_q95"]) / 2.0,
        (bounds["y_q05"] + bounds["y_q95"]) / 2.0,
        (bounds["z_q05"] + bounds["z_q95"]) / 2.0,
    )
    return "\n".join(
        [
            "world_coordinate_format: [x_world, y_world, z_world]",
            f"x_robust_5_95_world_units: [{bounds['x_q05']:.3f}, {bounds['x_q95']:.3f}]",
            f"y_robust_5_95_world_units: [{bounds['y_q05']:.3f}, {bounds['y_q95']:.3f}]",
            f"z_robust_5_95_world_units: [{bounds['z_q05']:.3f}, {bounds['z_q95']:.3f}]",
            f"scene_robust_center_world: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]",
            f"robust_scene_diagonal_world_units: {bounds['robust_diagonal']:.3f}",
            "scale_note: these ranges and distances are broad sanity checks, not target coordinates.",
        ]
    )


def build_event_block(api_row: Mapping[str, str]) -> str:
    instruction = normalize_text(api_row.get("instruction_text"))
    utterance = normalize_text(api_row.get("utterance_text"))
    lines = [
        f"event_id: {normalize_text(api_row.get('event_id'))}",
        f"instruction_text: {instruction}",
    ]
    if utterance:
        lines.append(f"utterance_text: {utterance}")
    return "\n".join(lines)


def format_telemetry_block(item: Mapping[str, Any]) -> str:
    telemetry = item["telemetry"]
    lines = [
        f"- {item['panel_id']}:",
        f"  image: {item['frame_path']}",
        f"  relative_sample_time_seconds: {item['relative_sample_time_seconds']}",
        f"  selection_score: {item['selection_score']}",
        f"  selection_reason: {item['selection_reason']}",
        "  camera_context_world:",
        f"    camera_position_world: {xyz_text(telemetry.get('camera_position'))}",
        f"    camera_forward_world_unit_vector_do_not_copy: {xyz_text(telemetry.get('camera_forward_world'))}",
        f"    camera_right_world_unit_vector_do_not_copy: {xyz_text(telemetry.get('camera_right_world'))}",
        f"    camera_up_world_unit_vector_do_not_copy: {xyz_text(telemetry.get('camera_up_world'))}",
        f"    camera_fov_degrees: {telemetry.get('camera_fov_degrees')}",
        "  ray_cues_not_ground_truth:",
        f"    gaze_valid: {telemetry.get('gaze_valid')}",
        f"    gaze_origin_world: {xyz_text(telemetry.get('gaze_origin'))}",
        f"    gaze_direction_world_unit_vector_do_not_copy: {xyz_text(telemetry.get('gaze_direction_world'))}",
        f"    gaze_hit_world_ray_endpoint: {xyz_text(telemetry.get('gaze_hit'))}",
        f"    hand_valid: {telemetry.get('hand_valid')}",
        f"    hand_origin_world: {xyz_text(telemetry.get('hand_origin'))}",
        f"    hand_direction_world_unit_vector_do_not_copy: {xyz_text(telemetry.get('hand_direction_world'))}",
        f"    hand_hit_world_ray_endpoint: {xyz_text(telemetry.get('hand_hit'))}",
        "  primary_copyable_gaze_point_hypotheses:",
        f"    {item['panel_id']}_GAZE: {telemetry.get('gaze_hit')}",
        "  secondary_copyable_hand_point_hypotheses:",
        f"    {item['panel_id']}_HAND: {telemetry.get('hand_hit')}",
        "  scale_cues_world_units:",
        f"    camera_to_gaze_hit_distance: {telemetry.get('camera_to_gaze_hit_distance_world_units')}",
        f"    camera_to_hand_hit_distance: {telemetry.get('camera_to_hand_hit_distance_world_units')}",
        f"    gaze_to_hand_hit_distance: {telemetry.get('gaze_to_hand_hit_distance_world_units')}",
        "  interpretation_guard: ray endpoints may land on floor, wall, or background behind the referent; use them as pointing evidence together with image content and language, not as guaranteed object centers.",
        "  output_guard: output points must be exactly one [x_world, y_world, z_world] triple; copy each selected gaze/hand hypothesis as a separate JSON entry and never concatenate them into one point.",
    ]
    return "\n".join(lines)


def build_prompt(template: str, api_row: Mapping[str, str], bounds: Mapping[str, Any], evidence_payload: Sequence[Mapping[str, Any]]) -> str:
    prompt = template
    replacements = {
        "[EVENT_BLOCK]": build_event_block(api_row),
        "[SCENE_BOUNDS_BLOCK]": build_scene_bounds_block(bounds),
        "[EVIDENCE_BLOCK]": "\n".join(format_telemetry_block(item) for item in evidence_payload),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def gt_payload(anchors: Sequence[Anchor]) -> List[Dict[str, Any]]:
    return [{"id": anchor.anchor_id, "point": list(anchor.point)} for anchor in anchors]


def unresolved_gt_names(repo_root: Path, scene: str, api_row: Mapping[str, str], mapped_anchors: Sequence[Anchor]) -> List[str]:
    # Prefer match-eval mappings for evaluation. This only reports if the API target text
    # looks like an exact anchor id but was not mapped.
    known = {anchor.anchor_id for anchor in load_anchor_table(repo_root, scene)}
    target_tokens = split_names(api_row.get("target_description"))
    mapped = {anchor.anchor_id for anchor in mapped_anchors}
    return [name for name in target_tokens if name in known and name not in mapped]


def percentile(values: Sequence[float], ratio: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = ratio * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def nearest_anchor_distance_stats(anchors: Sequence[Anchor]) -> Dict[str, Any]:
    distances: List[float] = []
    for anchor in anchors:
        others = [item for item in anchors if item.anchor_id != anchor.anchor_id]
        if others:
            distances.append(min(vector_distance(anchor.point, other.point) for other in others))
    return {
        "count": len(distances),
        "mean": mean(distances) if distances else None,
        "median": median(distances) if distances else None,
        "p05": percentile(distances, 0.05),
        "p95": percentile(distances, 0.95),
    }


def audit_scene(repo_root: Path, scene: str, gt_by_row: Mapping[int, Sequence[Anchor]]) -> Dict[str, Any]:
    anchors = load_anchor_table(repo_root, scene)
    bounds = robust_bounds(anchors)
    xs = [anchor.point[0] for anchor in anchors]
    ys = [anchor.point[1] for anchor in anchors]
    zs = [anchor.point[2] for anchor in anchors]
    unresolved_rows = [row for row, gt in gt_by_row.items() if not gt]
    return {
        "anchor_count": len(anchors),
        "x_range": [min(xs), max(xs)],
        "y_range": [min(ys), max(ys)],
        "z_range": [min(zs), max(zs)],
        "robust_bounds": bounds,
        "nearest_anchor_distance": nearest_anchor_distance_stats(anchors),
        "unresolved_gt_rows_in_match_eval": unresolved_rows[:50],
        "unresolved_gt_row_count": len(unresolved_rows),
    }


def write_data_audit(path: Path, summary: Mapping[str, Any]) -> None:
    scene_lines = []
    for scene, item in summary["scene_audit"].items():
        nearest = item["nearest_anchor_distance"]
        bounds = item["robust_bounds"]
        scene_lines.append(
            f"| {scene} | {item['anchor_count']} | "
            f"{item['x_range'][0]:.3f}..{item['x_range'][1]:.3f} | "
            f"{item['y_range'][0]:.3f}..{item['y_range'][1]:.3f} | "
            f"{item['z_range'][0]:.3f}..{item['z_range'][1]:.3f} | "
            f"{bounds['robust_diagonal']:.3f} | "
            f"{nearest['median'] if nearest['median'] is not None else 'n/a'} |"
        )
    invalid_counts = summary.get("status_counts", {})
    content = f"""# Experiment 3 Point-Grounding Data Audit

This audit is generated by `exam3_point_grounding/build_point_grounding_manifest.py`.

## Coordinate Fields

- Anchor tables: `data/*_anchor_table.tsv`, fields `object_name`, `location_x`, `location_y`, `location_z`.
- Camera telemetry: raw multimodal JSON fields `cameraPosition`, `cameraRotation`, `cameraFOV`.
- Gaze telemetry: `eyeGaze.gazeOrigin`, `eyeGaze.gazeVector`, `eyeGaze.gazePoint`.
- Hand telemetry: `handData.rightIndexFingerRayHitPoint` and right-hand joints for an approximate ray origin.
- Camera basis vectors are precomputed from `cameraRotation` using local +Z forward, +X right, and +Y up.

## Unit Convention

Unity world units are not independently verified as meters in this audit. Results and prompts must use `world units` unless a later external calibration verifies meter scale.

## Leakage Boundary

The model-facing manifest excludes candidate anchors, GT coordinates, GT anchor ids, `target_description`, Exam2 projected GT fields, and old Experiment 3 candidate-anchor prompts. GT anchors are written only to `gt_manifest_eval.csv` for evaluator use after inference.

## Scene Anchor Ranges

| scene | anchors | x range | y range | z range | robust diagonal | nearest-anchor median |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(scene_lines)}

## Manifest Coverage

- model-facing samples: {summary.get('model_sample_count', 0)}
- evaluator GT rows: {summary.get('gt_sample_count', 0)}
- status counts: {dict(invalid_counts)}
- gaze-valid selected frames: {summary.get('selected_gaze_valid_count', 0)} / {summary.get('selected_frame_count', 0)}
- hand-valid selected frames: {summary.get('selected_hand_valid_count', 0)} / {summary.get('selected_frame_count', 0)}

## Same-Coordinate Assumption

Anchor coordinates, camera positions, gaze hits, and hand hits have overlapping Unity-world numeric ranges and are consumed without an extra transform. This is an implementation assumption inherited from the released telemetry and the existing projection code. No unverified coordinate transform is introduced.

## Hard-Stop Checks

- Full scene API CSVs were required for every requested scene.
- Raw multimodal JSON paths were required for every selected event.
- Events without mapped GT anchors were excluded from the model-facing/evaluable manifest and counted in status counts.
- Dynamic object event-time anchors are not present in the released supervision; this implementation treats released scene-level anchor points as static and reports this as a limitation rather than inventing event-time anchors.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    prompt_template = (repo_root / args.prompt_template).read_text(encoding="utf-8")
    eval_dir = Path(args.eval_dir).resolve() if args.eval_dir else None

    model_rows: List[Dict[str, Any]] = []
    gt_rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    scene_audit: Dict[str, Any] = {}
    selected_frame_count = 0
    selected_gaze_valid_count = 0
    selected_hand_valid_count = 0

    for scene in args.scenes:
        api_rows = read_csv_rows(scene_api_csv(repo_root, scene))
        anchors = load_anchor_table(repo_root, scene)
        bounds = robust_bounds(anchors)
        gt_by_row = build_gt_from_match_eval(repo_root, scene, eval_dir)
        scene_audit[scene] = audit_scene(repo_root, scene, gt_by_row)
        for row_index, api_row in choose_rows(api_rows, args.start_index, args.limit):
            event_id = normalize_text(api_row.get("event_id")) or f"{scene}_{row_index}"
            gt_anchors = list(gt_by_row.get(row_index, []))
            if not gt_anchors:
                status_counts["missing_gt_anchor"] += 1
                continue
            frame_dir = output_dir / "evidence_frames" / scene / f"row_{row_index}"
            try:
                frames, selection_stats = select_evidence_frames(
                    api_row,
                    frame_dir,
                    max_frames=args.max_frames,
                    candidate_step_seconds=args.candidate_step_seconds,
                    stability_window_seconds=args.stability_window_seconds,
                    min_sep_seconds=args.min_sep_seconds,
                    ffmpeg_path=args.ffmpeg_path,
                    no_extract_frames=args.no_extract_frames,
                    overwrite_frames=args.overwrite_frames,
                )
            except Exception as exc:
                status_counts["evidence_selection_error"] += 1
                gt_rows.append(
                    {
                        "scene": scene,
                        "row_index": row_index,
                        "event_id": event_id,
                        "gt_anchor_ids": ",".join(anchor.anchor_id for anchor in gt_anchors),
                        "gt_points_json": json_compact(gt_payload(gt_anchors)),
                        "gt_count": len(gt_anchors),
                        "gt_mapping_status": "valid_gt_but_no_evidence",
                        "gt_mapping_detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            evidence_payload = build_evidence_payload(frames)
            selected_frame_count += len(evidence_payload)
            selected_gaze_valid_count += sum(1 for item in evidence_payload if item["telemetry"].get("gaze_valid"))
            selected_hand_valid_count += sum(1 for item in evidence_payload if item["telemetry"].get("hand_valid"))
            prompt_text = build_prompt(prompt_template, api_row, bounds, evidence_payload)
            frame_paths = [frame.frame_path for frame in frames]
            selection_debug_path = frame_dir / "selection_debug.json"
            write_json(
                selection_debug_path,
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "selection_stats": selection_stats,
                    "evidence": evidence_payload,
                },
            )

            model_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "instruction": normalize_text(api_row.get("instruction_text")),
                    "utterance_text": normalize_text(api_row.get("utterance_text")),
                    "frame_paths_json": json_compact(frame_paths),
                    "evidence_json": json_compact(evidence_payload),
                    "prompt_text": prompt_text,
                    "json_path": normalize_text(api_row.get("json_path")),
                    "video_path": normalize_text(api_row.get("video_path")),
                    "t_start": normalize_text(api_row.get("t_start")),
                    "t_end": normalize_text(api_row.get("t_end")),
                    "scene_bounds_json": json_compact(bounds),
                    "selection_debug_json": str(selection_debug_path),
                    "status": "ok",
                    "status_detail": "",
                }
            )
            unresolved = unresolved_gt_names(repo_root, scene, api_row, gt_anchors)
            gt_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "gt_anchor_ids": ",".join(anchor.anchor_id for anchor in gt_anchors),
                    "gt_points_json": json_compact(gt_payload(gt_anchors)),
                    "gt_count": len(gt_anchors),
                    "gt_mapping_status": "valid",
                    "gt_mapping_detail": ",".join(unresolved),
                }
            )
            status_counts["ok"] += 1

    manifest_path = output_dir / "manifest.csv"
    gt_path = output_dir / "gt_manifest_eval.csv"
    write_csv(manifest_path, MANIFEST_COLUMNS, model_rows)
    write_csv(gt_path, GT_COLUMNS, gt_rows)

    summary = {
        "model_sample_count": len(model_rows),
        "gt_sample_count": len(gt_rows),
        "status_counts": dict(status_counts),
        "selected_frame_count": selected_frame_count,
        "selected_gaze_valid_count": selected_gaze_valid_count,
        "selected_hand_valid_count": selected_hand_valid_count,
        "scene_audit": scene_audit,
        "config": {
            "max_frames": args.max_frames,
            "candidate_step_seconds": args.candidate_step_seconds,
            "stability_window_seconds": args.stability_window_seconds,
            "min_sep_seconds": args.min_sep_seconds,
            "no_extract_frames": args.no_extract_frames,
        },
        "outputs": {
            "manifest": str(manifest_path),
            "gt_manifest_eval": str(gt_path),
        },
    }
    write_json(output_dir / "manifest_summary.json", summary)
    write_data_audit(repo_root / "exam3_point_grounding" / "DATA_AUDIT.md", summary)
    print(json.dumps({"manifest": str(manifest_path), "gt_manifest_eval": str(gt_path), "model_sample_count": len(model_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
