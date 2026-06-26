#!/usr/bin/env python3
"""Visualize two-stage grounding predictions on selected keyframes."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class VisualizationError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render standalone visualization images from two-stage grounding output CSV."
    )
    parser.add_argument("--pred-csv", required=True, help="Path to qwen3vl stage2 output CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered visualization images.")
    parser.add_argument(
        "--fallback-source-csv",
        help="Optional CSV for event_id -> keyframe_path fallback lookup when selected_keyframe_path is missing.",
    )
    parser.add_argument("--event-id-column", default="event_id", help="Event id column name.")
    parser.add_argument(
        "--keyframe-column",
        default="selected_keyframe_path",
        help="Keyframe image path column name in pred CSV.",
    )
    parser.add_argument(
        "--source-keyframe-column",
        default="keyframe_path",
        help="Fallback keyframe path column name in fallback-source-csv.",
    )
    parser.add_argument(
        "--draw-prior",
        action="store_true",
        help="Also draw spatial prior point when spatial_prior_u_norm and spatial_prior_v_norm are available.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing rendered images.")
    return parser.parse_args()


def ensure_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise VisualizationError("Missing dependency: Pillow. Please install pillow before rendering visualizations.") from exc
    return Image, ImageDraw, ImageFont


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or not csv_path.is_file():
        raise VisualizationError(f"CSV file does not exist or is not a file: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise VisualizationError(f"CSV file has no header row: {csv_path}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader if any(row.values())]


def build_lookup(rows: List[Dict[str, str]], event_id_column: str, keyframe_column: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for row in rows:
        event_id = row.get(event_id_column, "")
        keyframe_path = row.get(keyframe_column, "")
        if event_id and keyframe_path:
            lookup[event_id] = keyframe_path
    return lookup


def parse_float(value: str) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "event"


def clamp_pixel(value: float, limit: int) -> int:
    return max(0, min(limit - 1, int(round(value))))


def select_keyframe_path(
    row: Dict[str, str],
    event_id_column: str,
    keyframe_column: str,
    fallback_lookup: Dict[str, str],
) -> Optional[Path]:
    event_id = row.get(event_id_column, "")
    direct = row.get(keyframe_column, "")
    fallback = fallback_lookup.get(event_id, "")
    candidate = direct or fallback
    if not candidate:
        return None
    return Path(candidate).expanduser().resolve()


def build_label_lines(row: Dict[str, str]) -> List[str]:
    lines = [
        f"event_id: {row.get('event_id', '')}",
        f"referent_type: {row.get('referent_type', '') or 'unknown'}",
        f"parse_ok: {row.get('parse_ok', '') or 'unknown'}",
    ]
    referent_text = row.get("referent_text", "")
    if referent_text:
        lines.append(f"referent: {referent_text}")
    confidence = row.get("confidence", "")
    if confidence:
        lines.append(f"confidence: {confidence}")
    peak_time = row.get("predicted_peak_time_seconds", "")
    if peak_time:
        lines.append(f"peak_time: {peak_time}")
    prior_source = row.get("spatial_prior_source", "")
    if prior_source:
        lines.append(f"prior_source: {prior_source}")
    error_message = row.get("error_message", "")
    if error_message:
        lines.append(f"error: {error_message}")
    return lines


def draw_point(draw, x: int, y: int, radius: int, outline: Tuple[int, int, int], fill: Tuple[int, int, int]) -> None:
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=outline, width=3, fill=fill)


def main() -> int:
    args = parse_args()
    try:
        Image, ImageDraw, ImageFont = ensure_pillow()
        pred_csv = Path(args.pred_csv).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_rows = load_rows(pred_csv)
        fallback_lookup: Dict[str, str] = {}
        if args.fallback_source_csv:
            source_rows = load_rows(Path(args.fallback_source_csv).expanduser().resolve())
            fallback_lookup = build_lookup(source_rows, args.event_id_column, args.source_keyframe_column)

        rendered = 0
        skipped = 0
        for row in pred_rows:
            event_id = row.get(args.event_id_column, "")
            if not event_id:
                skipped += 1
                continue

            image_path = select_keyframe_path(row, args.event_id_column, args.keyframe_column, fallback_lookup)
            if image_path is None:
                skipped += 1
                print(f"Skip {event_id}: missing keyframe path", file=sys.stderr)
                continue
            if not image_path.exists() or not image_path.is_file():
                skipped += 1
                print(f"Skip {event_id}: keyframe not found at {image_path}", file=sys.stderr)
                continue

            output_path = output_dir / f"{sanitize_filename(event_id)}.png"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"Skip {event_id}: visualization already exists", file=sys.stderr)
                continue

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            font = ImageFont.load_default()
            radius = max(6, int(min(width, height) * 0.015))

            u_norm = parse_float(row.get("u_norm", ""))
            v_norm = parse_float(row.get("v_norm", ""))
            prior_u = parse_float(row.get("spatial_prior_u_norm", ""))
            prior_v = parse_float(row.get("spatial_prior_v_norm", ""))

            if args.draw_prior and prior_u is not None and prior_v is not None:
                px = clamp_pixel(prior_u * width, width)
                py = clamp_pixel(prior_v * height, height)
                draw_point(draw, px, py, max(4, radius - 2), outline=(0, 170, 255), fill=(100, 220, 255))
                draw.text(
                    (min(px + radius + 4, width - 120), max(py - radius - 14, 8)),
                    "prior",
                    fill=(0, 170, 255),
                    font=font,
                    stroke_width=2,
                    stroke_fill=(0, 0, 0),
                )

            text_x = 8
            text_y = 8
            if u_norm is not None and v_norm is not None:
                x = clamp_pixel(u_norm * width, width)
                y = clamp_pixel(v_norm * height, height)
                draw_point(draw, x, y, radius, outline=(255, 0, 0), fill=(255, 220, 0))
                draw.line((x - radius - 4, y, x + radius + 4, y), fill=(255, 0, 0), width=2)
                draw.line((x, y - radius - 4, x, y + radius + 4), fill=(255, 0, 0), width=2)
                text_x = min(max(8, x + radius + 6), max(8, width - 320))
                text_y = min(max(8, y - radius - 40), max(8, height - 120))

            label_text = "\n".join(build_label_lines(row))
            draw.multiline_text(
                (text_x, text_y),
                label_text,
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                spacing=4,
            )

            image.save(output_path)
            rendered += 1
            print(f"Rendered visualization: {output_path}")

        print(f"Rendered: {rendered}")
        print(f"Skipped: {skipped}")
        return 0
    except VisualizationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
