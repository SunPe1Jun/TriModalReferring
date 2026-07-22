#!/usr/bin/env python3
"""Export path-free per-panel evidence from the full strict-hand mask audit."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_audit", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_summary", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input_audit).read_text(encoding="utf-8"))
    rows = []
    for item in payload["panels"]:
        boxes = item.get("boxes") or []
        rows.append({
            "scene": item["scene"],
            "row_index": int(item["row_index"]),
            "unified_id": f"{item['scene']}::{int(item['row_index'])}",
            "panel_id": item["panel_id"],
            "status": item["status"],
            "tracked_sides": json.dumps(item.get("tracked_sides") or [], separators=(",", ":")),
            "masked_side_count": sum(box.get("status") == "masked" for box in boxes),
            "projected_joint_count": sum(int(box.get("joint_count") or 0) for box in boxes),
            "mask_pixels": int(item.get("mask_pixels") or 0),
            "mask_fraction": float(item.get("mask_fraction") or 0.0),
            "image_width": int((item.get("image_size") or [0, 0])[0]),
            "image_height": int((item.get("image_size") or [0, 0])[1]),
            "boxes_json": json.dumps(boxes, separators=(",", ":")),
            "sample_time_delta": float(item.get("sample_time_delta") or 0.0),
            "mask_version": item.get("mask_version") or payload.get("mask_version"),
            "mask_mode": item.get("mask_mode", ""),
            "world_margin": item.get("world_margin", ""),
            "mask_color_rgb": json.dumps(item.get("mask_color_rgb") or [], separators=(",", ":")),
            "jpeg_quality": item.get("jpeg_quality", ""),
        })
    fields = list(rows[0])
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "mask_version": payload["mask_version"],
        "sample_count": payload["sample_count"],
        "panel_count": len(rows),
        "status_counts": dict(status_counts),
        "masked_panel_count": status_counts["masked"],
        "tracked_offscreen_panel_count": status_counts["tracked_offscreen"],
        "no_tracked_hand_panel_count": status_counts["no_tracked_hand"],
        "mean_mask_fraction": sum(row["mask_fraction"] for row in rows) / len(rows),
        "max_abs_sample_time_delta": max(abs(row["sample_time_delta"]) for row in rows),
        "unique_sample_count": len({row["unified_id"] for row in rows}),
        "validation_pass": (
            len(rows) == int(payload["panel_count"])
            and len({row["unified_id"] for row in rows}) == int(payload["sample_count"])
            and set(status_counts) <= {"masked", "tracked_offscreen", "no_tracked_hand"}
        ),
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not summary["validation_pass"]:
        raise RuntimeError(f"strict hand mask evidence failed validation: {summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
