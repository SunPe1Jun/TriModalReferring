#!/usr/bin/env python3
"""Refine v10 2D point predictions with GroundingDINO boxes.

This script keeps the Qwen/VL result as the temporal and referent-selection
stage. For each predicted referent, it runs GroundingDINO on the selected full
panel image with the predicted mention phrase, then converts the detected box
into a point that can be evaluated by evaluate_2d_point_grounding.py.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image


OUTPUT_EXTRA_COLUMNS = (
    "dino_model",
    "dino_point_mode",
    "dino_box_threshold",
    "dino_text_threshold",
    "dino_refined_count",
    "dino_detected_count",
    "dino_fallback_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine v10 2D predictions with GroundingDINO.")
    parser.add_argument("--manifest", required=True, help="v10 manifest_all.csv")
    parser.add_argument("--pred_csv", required=True, help="v10 qwen3vl_2d_predictions.csv")
    parser.add_argument("--output_csv", required=True, help="Refined prediction CSV")
    parser.add_argument("--output_json_dir", required=True, help="Per-event refined JSON output directory")
    parser.add_argument("--detection_json_dir", help="Per-referent DINO cache directory. Default: <output_json_dir>/dino_cache")
    parser.add_argument("--model_name", required=True, help="GroundingDINO model path or HF repo id")
    parser.add_argument("--local_files_only", action="store_true", help="Load DINO model from local files only")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu. Default: cuda")
    parser.add_argument("--box_threshold", type=float, default=0.30, help="GroundingDINO box threshold. Default: 0.30")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="GroundingDINO text threshold. Default: 0.25")
    parser.add_argument(
        "--box_select_mode",
        choices=("highest_score", "vl_point_nearest", "gaze_nearest"),
        default="vl_point_nearest",
        help="How to choose one box when DINO returns multiple boxes. Default: vl_point_nearest.",
    )
    parser.add_argument(
        "--point_mode",
        choices=("box_center", "gaze_nearest", "vl_point_nearest"),
        default="box_center",
        help="How to convert a DINO box into the final point. Default: box_center.",
    )
    parser.add_argument("--proximity_weight", type=float, default=0.25, help="Penalty weight for box selection distance. Default: 0.25")
    parser.add_argument("--fallback_to_vl", action="store_true", help="Use original v10 point when DINO returns no box.")
    parser.add_argument("--strip_deictic", action="store_true", default=True, help="Strip words such as this/that from DINO prompt. Default: true.")
    parser.add_argument("--no_strip_deictic", action="store_false", dest="strip_deictic", help="Do not strip deictic words.")
    parser.add_argument("--start_index", type=int, default=0, help="Minimum row_index per scene. Default: 0.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional scenes to run.")
    parser.add_argument("--max_events", type=int, help="Optional maximum number of events after filters.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-event refined JSON")
    parser.add_argument("--overwrite_detections", action="store_true", help="Re-run DINO even when per-referent cache exists")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue after one event/refinement error")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(float(text))
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


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def manifest_groups(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            grouped[(scene, row_index)].append(row)
    return grouped


def selected_key(scene: str, row_index: int, args: argparse.Namespace) -> bool:
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def parse_prediction_payload(row: Mapping[str, str]) -> Dict[str, Any]:
    try:
        payload = json.loads(normalize_text(row.get("parsed_json")) or "{}")
    except json.JSONDecodeError:
        payload = {}
    refs = payload.get("referents") if isinstance(payload, Mapping) else None
    if not isinstance(refs, list):
        refs = []
    return {"referents": [dict(ref) for ref in refs if isinstance(ref, Mapping)]}


DEICTIC_PATTERN = re.compile(
    r"\b(this|that|these|those|there|here|target|selected|current|intended|one|it)\b",
    flags=re.IGNORECASE,
)


def clean_phrase(mention: str, instruction: str, strip_deictic: bool) -> str:
    phrase = normalize_text(mention)
    if not phrase:
        phrase = normalize_text(instruction)
    phrase = phrase.replace("_", " ").replace("-", " ")
    phrase = re.sub(r"\s+", " ", phrase).strip(" .,:;\"'")
    if strip_deictic:
        stripped = DEICTIC_PATTERN.sub(" ", phrase)
        stripped = re.sub(r"\s+", " ", stripped).strip(" .,:;\"'")
        if stripped:
            phrase = stripped
    if not phrase:
        phrase = normalize_text(mention) or normalize_text(instruction) or "object"
    phrase = phrase.lower()
    return phrase if phrase.endswith(".") else phrase + "."


def panel_row_for_prediction(event_rows: Sequence[Mapping[str, str]], panel_id: str) -> Optional[Mapping[str, str]]:
    panel = normalize_text(panel_id).upper()
    candidates = [
        row
        for row in event_rows
        if normalize_text(row.get("panel_id")).upper() == panel
        and normalize_text(row.get("frame_path"))
        and normalize_text(row.get("frame_extracted")) != "False"
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda row: 0 if normalize_text(row.get("projection_valid")) == "True" else 1)
    return candidates[0]


def box_center(box: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_distance_to_box_center(box: Sequence[float], point: Tuple[float, float], image_size: Tuple[int, int]) -> float:
    cx, cy = box_center(box)
    px, py = point
    diag = max(1.0, math.hypot(float(image_size[0]), float(image_size[1])))
    return math.hypot(cx - px, cy - py) / diag


def clamp_point_to_box(box: Sequence[float], point: Tuple[float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    px, py = point
    return clamp(px, x1, x2), clamp(py, y1, y2)


def normalized_point_to_pixels(x_norm: Optional[float], y_norm: Optional[float], image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    if x_norm is None or y_norm is None:
        return None
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return None
    width, height = image_size
    return x_norm * width, y_norm * height


def gaze_point_from_panel(row: Mapping[str, str], image_size: Tuple[int, int]) -> Optional[Tuple[float, float]]:
    if normalize_text(row.get("gaze_projection_valid")) != "True":
        return None
    return normalized_point_to_pixels(parse_float(row.get("gaze_u_norm")), parse_float(row.get("gaze_v_norm")), image_size)


def choose_box(
    detections: Sequence[Mapping[str, Any]],
    mode: str,
    reference_point: Optional[Tuple[float, float]],
    image_size: Tuple[int, int],
    proximity_weight: float,
) -> Optional[Mapping[str, Any]]:
    if not detections:
        return None
    if mode == "highest_score" or reference_point is None:
        return max(detections, key=lambda det: float(det.get("score") or 0.0))
    return max(
        detections,
        key=lambda det: float(det.get("score") or 0.0)
        - proximity_weight * point_distance_to_box_center(det["box_xyxy"], reference_point, image_size),
    )


class GroundingDINORunner:
    def __init__(self, model_name: str, device_name: str, local_files_only: bool, box_threshold: float, text_threshold: float):
        self.model_name = model_name
        self.device_name = device_name
        self.local_files_only = local_files_only
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = None
        self.model = None
        self.torch = None
        self.device = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.device = torch.device(self.device_name if self.device_name == "cpu" or torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        ).to(self.device)
        self.model.eval()

    def _post_process_grounded(self, outputs: Any, input_ids: Any, target_sizes: Any) -> Mapping[str, Any]:
        assert self.processor is not None
        post_process = self.processor.post_process_grounded_object_detection
        try:
            parameters = inspect.signature(post_process).parameters
        except (TypeError, ValueError):
            parameters = {}

        kwargs: Dict[str, Any] = {}
        if "input_ids" in parameters:
            kwargs["input_ids"] = input_ids
        if "target_sizes" in parameters:
            kwargs["target_sizes"] = target_sizes
        if "box_threshold" in parameters:
            kwargs["box_threshold"] = self.box_threshold
        elif "threshold" in parameters:
            kwargs["threshold"] = self.box_threshold
        if "text_threshold" in parameters:
            kwargs["text_threshold"] = self.text_threshold

        if kwargs:
            return post_process(outputs, **kwargs)[0]

        call_patterns = (
            (outputs, input_ids, {"box_threshold": self.box_threshold, "text_threshold": self.text_threshold, "target_sizes": target_sizes}),
            (outputs, input_ids, {"threshold": self.box_threshold, "text_threshold": self.text_threshold, "target_sizes": target_sizes}),
            (outputs, {"box_threshold": self.box_threshold, "text_threshold": self.text_threshold, "target_sizes": target_sizes}),
            (outputs, {"threshold": self.box_threshold, "text_threshold": self.text_threshold, "target_sizes": target_sizes}),
            (outputs, {"target_sizes": target_sizes}),
        )
        last_error: Optional[TypeError] = None
        for pattern in call_patterns:
            *positional, call_kwargs = pattern
            try:
                return post_process(*positional, **call_kwargs)[0]
            except TypeError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return post_process(outputs)[0]

    def detect(self, image_path: Path, text: str) -> List[Dict[str, Any]]:
        assert self.processor is not None and self.model is not None and self.torch is not None and self.device is not None
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=text, return_tensors="pt")
        inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([image.size[::-1]], device=self.device)
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            result = self._post_process_grounded(outputs, inputs.get("input_ids"), target_sizes)
        else:
            result = self.processor.post_process_object_detection(
                outputs,
                threshold=self.box_threshold,
                target_sizes=target_sizes,
            )[0]
        boxes = result.get("boxes", [])
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        detections: List[Dict[str, Any]] = []
        for idx, box in enumerate(boxes):
            score = float(scores[idx].detach().cpu().item()) if idx < len(scores) else 0.0
            label = labels[idx] if idx < len(labels) else ""
            if hasattr(label, "detach"):
                label = str(int(label.detach().cpu().item()))
            box_values = [float(value) for value in box.detach().cpu().tolist()]
            x1, y1, x2, y2 = box_values
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"score": score, "label": str(label), "box_xyxy": [x1, y1, x2, y2]})
        return detections


def load_or_run_detection(
    runner: GroundingDINORunner,
    cache_path: Path,
    image_path: Path,
    phrase: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if cache_path.exists() and not args.overwrite_detections:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    detections = runner.detect(image_path, phrase)
    payload = {
        "image_path": str(image_path),
        "phrase": phrase,
        "model_name": args.model_name,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "detections": detections,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def refine_event(
    event_key: Tuple[str, int],
    pred_row: Mapping[str, str],
    event_rows: Sequence[Mapping[str, str]],
    runner: GroundingDINORunner,
    detection_json_dir: Path,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    scene, row_index = event_key
    instruction = normalize_text(pred_row.get("instruction"))
    parsed = parse_prediction_payload(pred_row)
    refined_refs: List[Dict[str, Any]] = []
    counts = {"refined": 0, "detected": 0, "fallback": 0}
    for ref_index, ref in enumerate(parsed.get("referents", []), start=1):
        item = dict(ref)
        panel_id = normalize_text(item.get("panel_id")).upper()
        panel_row = panel_row_for_prediction(event_rows, panel_id)
        vl_x = parse_float(item.get("x_norm"))
        vl_y = parse_float(item.get("y_norm"))
        item["dino_status"] = "not_run"
        item["dino_point_mode"] = args.point_mode
        item["dino_box_select_mode"] = args.box_select_mode
        item["dino_model"] = args.model_name
        item["vl_x_norm"] = vl_x
        item["vl_y_norm"] = vl_y
        if panel_row is None:
            item["dino_status"] = "missing_panel_frame"
            if not args.fallback_to_vl:
                item["x_norm"] = None
                item["y_norm"] = None
            else:
                counts["fallback"] += 1
            refined_refs.append(item)
            continue
        image_path = Path(normalize_text(panel_row.get("frame_path")))
        if not image_path.exists():
            item["dino_status"] = "missing_image"
            if not args.fallback_to_vl:
                item["x_norm"] = None
                item["y_norm"] = None
            else:
                counts["fallback"] += 1
            refined_refs.append(item)
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        image_size = (width, height)
        phrase = clean_phrase(normalize_text(item.get("mention")), instruction, args.strip_deictic)
        cache_path = detection_json_dir / scene / f"row_{row_index}" / f"ref_{ref_index}_{panel_id}.json"
        detection_payload = load_or_run_detection(runner, cache_path, image_path, phrase, args)
        detections = detection_payload.get("detections") if isinstance(detection_payload, Mapping) else []
        if not isinstance(detections, list):
            detections = []
        vl_point = normalized_point_to_pixels(vl_x, vl_y, image_size)
        gaze_point = gaze_point_from_panel(panel_row, image_size)
        select_ref = None
        if args.box_select_mode == "vl_point_nearest":
            select_ref = vl_point
        elif args.box_select_mode == "gaze_nearest":
            select_ref = gaze_point
        chosen = choose_box(detections, args.box_select_mode, select_ref, image_size, float(args.proximity_weight))
        item["dino_phrase"] = phrase
        item["dino_detection_count"] = len(detections)
        if chosen is None:
            item["dino_status"] = "no_box"
            if args.fallback_to_vl:
                counts["fallback"] += 1
            else:
                item["x_norm"] = None
                item["y_norm"] = None
            refined_refs.append(item)
            continue
        counts["detected"] += 1
        box = [float(value) for value in chosen["box_xyxy"]]
        if args.point_mode == "gaze_nearest" and gaze_point is not None:
            point_x, point_y = clamp_point_to_box(box, gaze_point)
            point_source = "dino_box_gaze_nearest"
        elif args.point_mode == "vl_point_nearest" and vl_point is not None:
            point_x, point_y = clamp_point_to_box(box, vl_point)
            point_source = "dino_box_vl_point_nearest"
        else:
            point_x, point_y = box_center(box)
            point_source = "dino_box_center"
        item["x_norm"] = clamp(point_x / width, 0.0, 1.0)
        item["y_norm"] = clamp(point_y / height, 0.0, 1.0)
        item["dino_status"] = "detected"
        item["dino_point_source"] = point_source
        item["dino_box_xyxy"] = [round(value, 3) for value in box]
        item["dino_box_norm"] = [
            round(clamp(box[0] / width, 0.0, 1.0), 6),
            round(clamp(box[1] / height, 0.0, 1.0), 6),
            round(clamp(box[2] / width, 0.0, 1.0), 6),
            round(clamp(box[3] / height, 0.0, 1.0), 6),
        ]
        item["dino_score"] = float(chosen.get("score") or 0.0)
        item["dino_label"] = normalize_text(chosen.get("label"))
        counts["refined"] += 1
        refined_refs.append(item)
    return {"referents": refined_refs}, counts


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    pred_csv = Path(args.pred_csv)
    output_csv = Path(args.output_csv)
    output_json_dir = Path(args.output_json_dir)
    detection_json_dir = Path(args.detection_json_dir) if args.detection_json_dir else output_json_dir / "dino_cache"
    output_json_dir.mkdir(parents=True, exist_ok=True)
    detection_json_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(manifest_path)
    pred_rows = read_csv_rows(pred_csv)
    groups = manifest_groups(manifest_rows)
    selected_rows: List[Mapping[str, str]] = []
    for row in pred_rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if not scene or row_index is None or not selected_key(scene, row_index, args):
            continue
        selected_rows.append(row)
        if args.max_events is not None and len(selected_rows) >= args.max_events:
            break

    runner = GroundingDINORunner(
        model_name=args.model_name,
        device_name=args.device,
        local_files_only=args.local_files_only,
        box_threshold=float(args.box_threshold),
        text_threshold=float(args.text_threshold),
    )
    runner.load()

    output_rows: List[Dict[str, Any]] = []
    input_fieldnames = list(pred_rows[0].keys()) if pred_rows else []
    fieldnames = list(dict.fromkeys([*input_fieldnames, *OUTPUT_EXTRA_COLUMNS]))
    total_counts = defaultdict(int)
    for row in selected_rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if row_index is None:
            continue
        key = (scene, row_index)
        per_event_json = output_json_dir / f"{scene}_row_{row_index}.json"
        try:
            if per_event_json.exists() and not args.overwrite:
                payload = json.loads(per_event_json.read_text(encoding="utf-8"))
                refined = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else {"referents": []}
                counts = payload.get("counts") if isinstance(payload.get("counts"), Mapping) else {}
            else:
                refined, counts = refine_event(
                    key,
                    row,
                    groups.get(key, []),
                    runner,
                    detection_json_dir,
                    args,
                )
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(row.get("event_id")),
                    "instruction": normalize_text(row.get("instruction")),
                    "source_prediction_csv": str(pred_csv),
                    "parsed_json": refined,
                    "counts": counts,
                    "settings": {
                        "model_name": args.model_name,
                        "point_mode": args.point_mode,
                        "box_select_mode": args.box_select_mode,
                        "box_threshold": args.box_threshold,
                        "text_threshold": args.text_threshold,
                        "fallback_to_vl": args.fallback_to_vl,
                        "strip_deictic": args.strip_deictic,
                    },
                }
                per_event_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            refined_count = int(counts.get("refined", 0))
            detected_count = int(counts.get("detected", 0))
            fallback_count = int(counts.get("fallback", 0))
            total_counts["refined"] += refined_count
            total_counts["detected"] += detected_count
            total_counts["fallback"] += fallback_count
            out_row = dict(row)
            out_row.update(
                {
                    "parsed_json": json.dumps(refined, ensure_ascii=False),
                    "prediction_count": len(refined.get("referents", [])) if isinstance(refined, Mapping) else 0,
                    "dino_model": args.model_name,
                    "dino_point_mode": args.point_mode,
                    "dino_box_threshold": args.box_threshold,
                    "dino_text_threshold": args.text_threshold,
                    "dino_refined_count": refined_count,
                    "dino_detected_count": detected_count,
                    "dino_fallback_count": fallback_count,
                }
            )
            output_rows.append(out_row)
            print(f"[ok] {scene} row_{row_index} refs={out_row['prediction_count']} detected={detected_count} fallback={fallback_count}", flush=True)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            out_row = dict(row)
            out_row.update(
                {
                    "parsed_json": "{\"referents\": []}",
                    "prediction_count": 0,
                    "parse_ok": "False",
                    "error_message": f"{type(exc).__name__}: {exc}",
                    "dino_model": args.model_name,
                    "dino_point_mode": args.point_mode,
                    "dino_box_threshold": args.box_threshold,
                    "dino_text_threshold": args.text_threshold,
                    "dino_refined_count": 0,
                    "dino_detected_count": 0,
                    "dino_fallback_count": 0,
                }
            )
            output_rows.append(out_row)
            print(f"[error] {scene} row_{row_index}: {exc}", file=sys.stderr, flush=True)
    write_csv(output_csv, output_rows, fieldnames)
    summary = {
        "events": len(output_rows),
        "model_name": args.model_name,
        "point_mode": args.point_mode,
        "box_select_mode": args.box_select_mode,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "fallback_to_vl": args.fallback_to_vl,
        "counts": dict(total_counts),
        "output_csv": str(output_csv),
    }
    (output_csv.parent / f"dino_refine_summary_{args.point_mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote refined predictions: {output_csv}")


if __name__ == "__main__":
    main()
