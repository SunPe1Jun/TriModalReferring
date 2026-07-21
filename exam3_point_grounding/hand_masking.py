#!/usr/bin/env python3
"""Create image-level hand-removed panels from frozen VR telemetry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from point_grounding_common import collect_timed_samples, load_multimodal_samples, nearest_sample


MASK_VERSION = "hand_mask_v1"
WORLD_MARGIN = 0.10
MIN_MARGIN_PIXELS = 48.0
MAX_MARGIN_PIXELS = 360.0
MASK_COLOR = (127, 127, 127)
JPEG_QUALITY = 85


def _quaternion_multiply(
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


def _rotate_by_inverse_camera(
    vector: Tuple[float, float, float], rotation: Mapping[str, Any]
) -> Tuple[float, float, float]:
    q = tuple(float(rotation[key]) for key in ("x", "y", "z", "w"))
    inverse = (-q[0], -q[1], -q[2], q[3])
    rotated = _quaternion_multiply(
        _quaternion_multiply(inverse, (vector[0], vector[1], vector[2], 0.0)), q
    )
    return rotated[0], rotated[1], rotated[2]


def _project_joint(
    sample: Mapping[str, Any], position: Mapping[str, Any], width: int, height: int
) -> Tuple[float, float, float] | None:
    camera_position = sample.get("cameraPosition")
    camera_rotation = sample.get("cameraRotation")
    camera_fov = sample.get("cameraFOV")
    if not isinstance(camera_position, Mapping) or not isinstance(camera_rotation, Mapping):
        return None
    try:
        translated = (
            float(position["x"]) - float(camera_position["x"]),
            float(position["y"]) - float(camera_position["y"]),
            float(position["z"]) - float(camera_position["z"]),
        )
        local = _rotate_by_inverse_camera(translated, camera_rotation)
        fov = math.radians(float(camera_fov))
    except (KeyError, TypeError, ValueError):
        return None
    if local[2] <= 1e-6 or not 0.0 < fov < math.pi:
        return None
    tan_vertical = math.tan(fov / 2.0)
    tan_horizontal = tan_vertical * width / float(height)
    x_ndc = local[0] / (local[2] * tan_horizontal)
    y_ndc = local[1] / (local[2] * tan_vertical)
    u = (x_ndc + 1.0) * 0.5
    v = (1.0 - y_ndc) * 0.5
    if not all(math.isfinite(value) for value in (u, v, local[2])):
        return None
    # Slightly off-screen joints are retained so edge masks cover the visible
    # part of the rendered hand.
    if not -0.5 <= u <= 1.5 or not -0.5 <= v <= 1.5:
        return None
    return u * width, v * height, local[2]


def _tracked_hand_boxes(
    sample: Mapping[str, Any], width: int, height: int
) -> Tuple[List[Dict[str, Any]], List[str]]:
    hand_data = sample.get("handData")
    if not isinstance(hand_data, Mapping):
        return [], []
    try:
        focal_pixels = height / (
            2.0 * math.tan(math.radians(float(sample.get("cameraFOV", 90.0))) / 2.0)
        )
    except (TypeError, ValueError, ZeroDivisionError):
        focal_pixels = height / 2.0
    boxes: List[Dict[str, Any]] = []
    tracked_sides: List[str] = []
    for side, flag in (("leftHand", "isLeftHandTracked"), ("rightHand", "isRightHandTracked")):
        if not bool(hand_data.get(flag)):
            continue
        tracked_sides.append(side)
        hand = hand_data.get(side)
        joints = hand.get("joints") if isinstance(hand, Mapping) else None
        projected: List[Tuple[float, float, float]] = []
        if isinstance(joints, list):
            for joint in joints:
                if not isinstance(joint, Mapping) or not isinstance(joint.get("position"), Mapping):
                    continue
                item = _project_joint(sample, joint["position"], width, height)
                if item is not None:
                    projected.append(item)
        if not projected:
            boxes.append({"side": side, "status": "tracked_offscreen_or_unprojectable", "joint_count": 0})
            continue
        min_depth = min(item[2] for item in projected)
        margin = focal_pixels * WORLD_MARGIN / max(min_depth, 0.05)
        margin = max(MIN_MARGIN_PIXELS, min(MAX_MARGIN_PIXELS, margin))
        left = max(0.0, min(item[0] for item in projected) - margin)
        top = max(0.0, min(item[1] for item in projected) - margin)
        right = min(float(width), max(item[0] for item in projected) + margin)
        bottom = min(float(height), max(item[1] for item in projected) + margin)
        if right <= left or bottom <= top:
            boxes.append({
                "side": side,
                "status": "tracked_offscreen_or_unprojectable",
                "joint_count": len(projected),
                "min_depth_world": min_depth,
            })
            continue
        boxes.append({
            "side": side,
            "status": "masked",
            "joint_count": len(projected),
            "min_depth_world": min_depth,
            "margin_pixels": margin,
            "bbox_xyxy": [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)],
        })
    return boxes, tracked_sides


def _mask_image(source_path: Path, output_path: Path, sample: Mapping[str, Any]) -> Dict[str, Any]:
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    boxes, tracked_sides = _tracked_hand_boxes(sample, width, height)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        coords = box.get("bbox_xyxy")
        if coords:
            draw.rectangle(tuple(int(round(value)) for value in coords), fill=255)
    histogram = mask.histogram()
    mask_pixels = sum(histogram[1:])
    mask_fraction = mask_pixels / float(width * height)
    if mask_pixels:
        image.paste(MASK_COLOR, (0, 0, width, height), mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return {
        "mask_version": MASK_VERSION,
        "source_path": str(source_path),
        "masked_path": str(output_path),
        "image_size": [width, height],
        "tracked_sides": tracked_sides,
        "boxes": boxes,
        "mask_pixels": mask_pixels,
        "mask_fraction": mask_fraction,
        "mask_mode": "expanded_joint_bbox_neutral_fill",
        "world_margin": WORLD_MARGIN,
        "mask_color_rgb": list(MASK_COLOR),
        "jpeg_quality": JPEG_QUALITY,
        "status": "masked" if mask_pixels else ("no_tracked_hand" if not tracked_sides else "tracked_offscreen"),
    }


def prepare_row_hand_masks(
    row: Mapping[str, str], output_root: Path, overwrite: bool = False
) -> Tuple[List[Path], List[Dict[str, Any]]]:
    """Return one hand-masked image path per frozen panel in panel order."""
    evidence = json.loads(row.get("evidence_json") or "[]")
    if not isinstance(evidence, list) or not evidence:
        raise RuntimeError("row has no evidence panels")
    samples = collect_timed_samples(load_multimodal_samples(Path(row["json_path"])))
    paths: List[Path] = []
    audits: List[Dict[str, Any]] = []
    row_dir = output_root / str(row["scene"]) / f"row_{int(row['row_index'])}"
    existing_audit: Dict[str, Any] = {}
    audit_path = row_dir / "mask_audit.json"
    if audit_path.exists() and not overwrite:
        existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for item in evidence:
        source_path = Path(str(item["frame_path"]))
        target_time = float(item["relative_sample_time_seconds"])
        sample_time, sample = nearest_sample(samples, target_time)
        output_path = row_dir / f"{item['panel_id']}.jpg"
        if output_path.exists() and not overwrite:
            audit = next(
                (entry for entry in existing_audit.get("panels", []) if entry.get("panel_id") == item["panel_id"]),
                {"status": "reused_without_audit", "masked_path": str(output_path)},
            )
        else:
            audit = _mask_image(source_path, output_path, sample)
        audit = dict(audit)
        audit.update({
            "panel_id": item["panel_id"],
            "target_sample_time": target_time,
            "nearest_sample_time": sample_time,
            "sample_time_delta": sample_time - target_time,
        })
        paths.append(output_path)
        audits.append(audit)
    row_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps({
            "mask_version": MASK_VERSION,
            "scene": row["scene"],
            "row_index": int(row["row_index"]),
            "panels": audits,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths, audits


def prepare_manifest_hand_masks(
    manifest_path: Path, output_root: Path, audit_path: Path, overwrite: bool = False
) -> Dict[str, Any]:
    import csv

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    panel_audits: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        paths, audits = prepare_row_hand_masks(row, output_root, overwrite=overwrite)
        for path, audit in zip(paths, audits):
            panel_audits.append({"scene": row["scene"], "row_index": row["row_index"], **audit})
        if index % 100 == 0:
            print(f"[mask] {index}/{len(rows)}", flush=True)
    masked = sum(item.get("status") == "masked" for item in panel_audits)
    tracked_offscreen = sum(item.get("status") == "tracked_offscreen" for item in panel_audits)
    no_hand = sum(item.get("status") == "no_tracked_hand" for item in panel_audits)
    summary = {
        "mask_version": MASK_VERSION,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "sample_count": len(rows),
        "panel_count": len(panel_audits),
        "masked_panel_count": masked,
        "tracked_offscreen_panel_count": tracked_offscreen,
        "no_tracked_hand_panel_count": no_hand,
        "mean_mask_fraction": (
            sum(float(item.get("mask_fraction", 0.0)) for item in panel_audits) / len(panel_audits)
            if panel_audits else 0.0
        ),
        "panels": panel_audits,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
