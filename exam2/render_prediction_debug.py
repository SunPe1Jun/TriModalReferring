#!/usr/bin/env python3
"""Render visual debug sheets for exam2 2D point predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prediction/GT debug sheets for exam2.")
    parser.add_argument("--manifest", required=True, help="manifest_all.csv")
    parser.add_argument("--pred_csv", required=True, help="qwen3vl_2d_predictions.csv")
    parser.add_argument("--output_dir", required=True, help="Output debug image directory.")
    parser.add_argument("--max_events", type=int, default=200, help="Maximum events to render. Default: 200.")
    parser.add_argument("--scenes", nargs="*", help="Optional scene filter.")
    parser.add_argument("--start_index", type=int, default=0, help="Minimum row_index. Default: 0.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--panel_width", type=int, default=360, help="Rendered panel width. Default: 360.")
    parser.add_argument("--panel_height", type=int, default=360, help="Rendered panel height. Default: 360.")
    parser.add_argument("--columns", type=int, default=5, help="Panel columns. Default: 5.")
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="Rewrite path prefixes for frame paths. Can be repeated.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_float(value: Any) -> Optional[float]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_rewrites(items: Sequence[str]) -> List[Tuple[str, str]]:
    rewrites: List[Tuple[str, str]] = []
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --path-rewrite value: {item}")
        old, new = item.split("=", 1)
        rewrites.append((old, new))
    return rewrites


def rewrite_path(text: str, rewrites: Sequence[Tuple[str, str]]) -> Path:
    result = normalize_text(text)
    for old, new in rewrites:
        if result.startswith(old):
            result = new + result[len(old) :]
    return Path(result)


def group_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            grouped[(scene, row_index)].append(row)
    return grouped


def prediction_map(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], Mapping[str, str]]:
    result: Dict[Tuple[str, int], Mapping[str, str]] = {}
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            result[(scene, row_index)] = row
    return result


def parse_predictions(row: Mapping[str, str]) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(normalize_text(row.get("parsed_json")) or "{}")
    except json.JSONDecodeError:
        return []
    raw = payload.get("referents") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list):
        return []
    predictions: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        panel_id = normalize_text(item.get("panel_id")).upper()
        if panel_id and not panel_id.startswith("P"):
            panel_id = "P" + panel_id
        x_norm = parse_float(item.get("x_norm"))
        y_norm = parse_float(item.get("y_norm"))
        if not panel_id or x_norm is None or y_norm is None:
            continue
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            continue
        predictions.append(
            {
                "mention": normalize_text(item.get("mention")),
                "panel_id": panel_id,
                "x_norm": x_norm,
                "y_norm": y_norm,
                "confidence": parse_float(item.get("confidence")),
            }
        )
    return predictions


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def panel_sort_key(panel_id: str) -> int:
    number = parse_int(panel_id.upper().replace("P", ""))
    return number if number is not None else 999


def resize_panel(image: Image.Image, panel_size: Tuple[int, int]) -> Tuple[Image.Image, float, int, int]:
    target_w, target_h = panel_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(src_w * scale))
    new_h = max(1, int(src_h * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", panel_size, (18, 18, 18))
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def point_to_panel(norm_x: float, norm_y: float, image_size: Tuple[int, int], scale: float, pad_x: int, pad_y: int) -> Tuple[float, float]:
    image_w, image_h = image_size
    return pad_x + norm_x * image_w * scale, pad_y + norm_y * image_h * scale


def short_text(text: str, max_len: int = 42) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "~"


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: Tuple[int, int, int], radius: int = 7, width: int = 3) -> None:
    draw.line((x - radius, y, x + radius, y), fill=color, width=width)
    draw.line((x, y - radius, x, y + radius), fill=color, width=width)
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=width)


def draw_event(
    key: Tuple[str, int],
    rows: Sequence[Mapping[str, str]],
    pred_row: Mapping[str, str],
    output_path: Path,
    rewrites: Sequence[Tuple[str, str]],
    panel_size: Tuple[int, int],
    columns: int,
) -> bool:
    panel_to_row: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        panel_id = normalize_text(row.get("panel_id")).upper()
        frame_path = rewrite_path(normalize_text(row.get("frame_path")), rewrites)
        if panel_id and frame_path.exists() and panel_id not in panel_to_row:
            panel_to_row[panel_id] = row
    panel_ids = sorted(panel_to_row, key=panel_sort_key)
    if not panel_ids:
        return False

    predictions_by_panel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for prediction in parse_predictions(pred_row):
        predictions_by_panel[normalize_text(prediction.get("panel_id")).upper()].append(prediction)

    gt_by_panel: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    gaze_by_panel: Dict[str, Mapping[str, str]] = {}
    for row in rows:
        panel_id = normalize_text(row.get("panel_id")).upper()
        if normalize_text(row.get("projection_valid")) == "True":
            gt_by_panel[panel_id].append(row)
        if normalize_text(row.get("gaze_projection_valid")) == "True" and panel_id not in gaze_by_panel:
            gaze_by_panel[panel_id] = row

    scene, row_index = key
    first = rows[0]
    instruction = normalize_text(first.get("instruction"))
    columns = max(1, columns)
    label_h = 12
    header_h = 92
    gutter = 14
    panel_w, panel_h = panel_size
    sheet_rows = int(math.ceil(len(panel_ids) / columns))
    sheet_w = columns * panel_w + (columns + 1) * gutter
    sheet_h = header_h + sheet_rows * (panel_h + label_h) + (sheet_rows + 1) * gutter
    sheet = Image.new("RGB", (sheet_w, sheet_h), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = load_font(18)
    small_font = load_font(14)
    draw.text((gutter, 12), f"{scene} row_{row_index}  {normalize_text(first.get('event_id'))}", fill=(255, 255, 255), font=font)
    draw.text((gutter, 40), short_text(instruction, 120), fill=(235, 235, 235), font=small_font)
    draw.text((gutter, 64), "red=prediction  green=acceptable GT  blue=other GT  yellow=gaze", fill=(230, 230, 230), font=small_font)

    for idx, panel_id in enumerate(panel_ids):
        row = panel_to_row[panel_id]
        frame_path = rewrite_path(normalize_text(row.get("frame_path")), rewrites)
        image = Image.open(frame_path).convert("RGB")
        src_size = image.size
        panel_image, scale, pad_x, pad_y = resize_panel(image, panel_size)
        row_pos = idx // columns
        col_pos = idx % columns
        x0 = gutter + col_pos * (panel_w + gutter)
        y0 = header_h + gutter + row_pos * (panel_h + label_h + gutter)
        sheet.paste(panel_image, (x0, y0))
        acceptable = any(normalize_text(gt.get("evidence_acceptable")) == "True" for gt in gt_by_panel.get(panel_id, []))
        border = (72, 220, 110) if acceptable else (190, 190, 190)
        draw.rectangle((x0, y0, x0 + panel_w, y0 + panel_h), outline=border, width=4 if acceptable else 2)
        video_time = normalize_text(row.get("video_frame_time") or row.get("frame_time"))
        draw.text((x0 + 8, y0 + 8), f"{panel_id}  t={video_time}s", fill=(255, 255, 255), font=font)
        if acceptable:
            draw.text((x0 + 8, y0 + 32), "acceptable evidence", fill=(120, 255, 150), font=small_font)

        if panel_id in gaze_by_panel:
            gaze_row = gaze_by_panel[panel_id]
            gaze_u = parse_float(gaze_row.get("gaze_u_norm"))
            gaze_v = parse_float(gaze_row.get("gaze_v_norm"))
            if gaze_u is not None and gaze_v is not None:
                gx, gy = point_to_panel(gaze_u, gaze_v, src_size, scale, pad_x, pad_y)
                draw_cross(draw, x0 + gx, y0 + gy, (255, 220, 60), radius=5, width=2)
                draw.text((x0 + gx + 8, y0 + gy - 8), "gaze", fill=(255, 220, 60), font=small_font)

        for gt in gt_by_panel.get(panel_id, []):
            gt_u = parse_float(gt.get("gt_u_norm"))
            gt_v = parse_float(gt.get("gt_v_norm"))
            if gt_u is None or gt_v is None:
                continue
            gx, gy = point_to_panel(gt_u, gt_v, src_size, scale, pad_x, pad_y)
            color = (80, 255, 140) if normalize_text(gt.get("evidence_acceptable")) == "True" else (70, 190, 255)
            draw_cross(draw, x0 + gx, y0 + gy, color, radius=6, width=2)
            draw.text((x0 + gx + 8, y0 + gy + 4), short_text(normalize_text(gt.get("referent_name")), 20), fill=color, font=small_font)

        for prediction in predictions_by_panel.get(panel_id, []):
            px, py = point_to_panel(float(prediction["x_norm"]), float(prediction["y_norm"]), src_size, scale, pad_x, pad_y)
            draw_cross(draw, x0 + px, y0 + py, (255, 70, 90), radius=9, width=3)
            label = "pred"
            mention = normalize_text(prediction.get("mention"))
            if mention:
                label += f": {short_text(mention, 18)}"
            draw.text((x0 + px + 10, y0 + py - 18), label, fill=(255, 100, 120), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def selected(key: Tuple[str, int], args: argparse.Namespace) -> bool:
    scene, row_index = key
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def main() -> None:
    args = parse_args()
    rewrites = parse_rewrites(args.path_rewrite)
    manifest = group_manifest(read_csv_rows(Path(args.manifest)))
    predictions = prediction_map(read_csv_rows(Path(args.pred_csv)))
    output_dir = Path(args.output_dir)
    rendered = 0
    for key in sorted(manifest, key=lambda item: (item[0], item[1])):
        if not selected(key, args):
            continue
        if rendered >= args.max_events:
            break
        scene, row_index = key
        output_path = output_dir / scene / f"row_{row_index}.jpg"
        if draw_event(key, manifest[key], predictions.get(key, {}), output_path, rewrites, (args.panel_width, args.panel_height), args.columns):
            rendered += 1
    print(f"Rendered debug sheets: {rendered}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
