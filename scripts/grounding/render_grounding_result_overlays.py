#!/usr/bin/env python3
"""Render grounding result overlays on keyframes."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


class OverlayRenderError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render grounding predictions onto keyframe images.")
    parser.add_argument("--pred-csv", required=True, help="Path to grounding output CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered overlay images.")
    parser.add_argument("--source-csv", help="Optional CSV that contains event_id -> keyframe_path mapping.")
    parser.add_argument("--event-id-column", default="event_id", help="Event id column name.")
    parser.add_argument("--keyframe-column", default="keyframe_path", help="Keyframe path column name in source CSV.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing overlay images.")
    return parser.parse_args()


def ensure_pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise OverlayRenderError("Missing dependency: Pillow. Please install pillow before rendering overlays.") from exc
    return Image, ImageDraw, ImageFont


def sanitize_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized or "event"


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or not csv_path.is_file():
        raise OverlayRenderError(f"CSV file does not exist or is not a file: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise OverlayRenderError(f"CSV file has no header row: {csv_path}")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader if any(row.values())]


def build_keyframe_lookup(rows: List[Dict[str, str]], event_id_column: str, keyframe_column: str) -> Dict[str, str]:
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


def main() -> int:
    args = parse_args()

    try:
        Image, ImageDraw, ImageFont = ensure_pillow()
        pred_csv = Path(args.pred_csv).expanduser().resolve()
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        pred_rows = load_csv_rows(pred_csv)
        keyframe_lookup: Dict[str, str] = {}
        if args.source_csv:
            source_rows = load_csv_rows(Path(args.source_csv).expanduser().resolve())
            keyframe_lookup = build_keyframe_lookup(source_rows, args.event_id_column, args.keyframe_column)

        rendered = 0
        skipped = 0
        for row in pred_rows:
            event_id = row.get(args.event_id_column, "")
            if not event_id:
                continue
            keyframe_path_text = row.get(args.keyframe_column, "") or keyframe_lookup.get(event_id, "")
            if not keyframe_path_text:
                skipped += 1
                print(f"Skip {event_id}: missing keyframe_path mapping", file=sys.stderr)
                continue

            image_path = Path(keyframe_path_text).expanduser().resolve()
            if not image_path.exists() or not image_path.is_file():
                skipped += 1
                print(f"Skip {event_id}: keyframe not found at {image_path}", file=sys.stderr)
                continue

            output_path = output_dir / f"{sanitize_filename(event_id)}.png"
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"Skip {event_id}: overlay already exists", file=sys.stderr)
                continue

            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            font = ImageFont.load_default()

            u_norm = parse_float(row.get("u_norm", ""))
            v_norm = parse_float(row.get("v_norm", ""))
            confidence = row.get("confidence", "")
            referent_text = row.get("referent_text", "")
            parse_ok = row.get("parse_ok", "")
            error_message = row.get("error_message", "")

            label_lines = [
                f"event_id: {event_id}",
                f"parse_ok: {parse_ok or 'unknown'}",
            ]
            if referent_text:
                label_lines.append(f"referent: {referent_text}")
            if confidence:
                label_lines.append(f"confidence: {confidence}")
            if error_message:
                label_lines.append(f"error: {error_message}")

            text = "\n".join(label_lines)
            if u_norm is not None and v_norm is not None:
                x_pixel = int(round(u_norm * width))
                y_pixel = int(round(v_norm * height))
                radius = max(6, int(min(width, height) * 0.015))
                draw.ellipse(
                    (x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius),
                    outline=(255, 0, 0),
                    width=3,
                    fill=(255, 220, 0),
                )
                text_x = min(max(8, x_pixel + radius + 4), max(8, width - 260))
                text_y = min(max(8, y_pixel - radius - 28), max(8, height - 80))
            else:
                text_x = 8
                text_y = 8

            draw.multiline_text(
                (text_x, text_y),
                text,
                fill=(255, 255, 255),
                font=font,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                spacing=4,
            )

            image.save(output_path)
            rendered += 1
            print(f"Rendered overlay: {output_path}")

        print(f"Rendered: {rendered}")
        print(f"Skipped: {skipped}")
        return 0
    except OverlayRenderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
