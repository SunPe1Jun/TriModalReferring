#!/usr/bin/env python3
"""Run Qwen3-VL for camera-centered 3D directional point diagnostic."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from PIL import Image


OUTPUT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "evidence_panel_id",
    "evidence_selection_strategy",
    "evidence_frame_path",
    "video_frame_time",
    "json_sample_time",
    "json_path",
    "camera_x",
    "camera_y",
    "camera_z",
    "camera_rotation_xyzw",
    "camera_fov",
    "gt_anchor_ids",
    "gt_anchor_points_json",
    "candidate_anchor_count",
    "model_input_image",
    "raw_json_path",
    "model_raw_output",
    "parsed_json",
    "parse_ok",
    "point_x",
    "point_y",
    "point_z",
    "invalid_reason",
    "error_message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run camera-centered 3D directional point diagnostic.")
    parser.add_argument("--repo_root", default=".", help="Repository root.")
    parser.add_argument(
        "--manifest",
        default="exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv",
        help="Exam2 v10 manifest_all.csv.",
    )
    parser.add_argument("--output_csv", required=True, help="Parsed prediction CSV.")
    parser.add_argument("--output_json_dir", required=True, help="Per-event raw output JSON directory.")
    parser.add_argument(
        "--prompt_template",
        default="exam3/prompts/camera_centered_3d_directional_prompt.md",
        help="Prompt template file.",
    )
    parser.add_argument("--model_name", required=True, help="Local Qwen3-VL model path.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model/processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum new tokens.")
    parser.add_argument("--offload_folder", help="Optional Accelerate disk offload folder.")
    parser.add_argument("--start_index", type=int, default=0, help="Minimum row_index per scene.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional scenes/partitions.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite per-event raw output JSON.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue if one sample fails.")
    parser.add_argument(
        "--evidence_panel_strategy",
        choices=("highest_score", "first", "middle"),
        default="highest_score",
        help="How to choose one evidence panel from the exam2 panel set.",
    )
    parser.add_argument(
        "--max_anchor_lines",
        type=int,
        default=260,
        help="Maximum candidate anchor rows included in the prompt. Default keeps all current scenes.",
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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def group_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            grouped[(scene, row_index)].append(row)
    return grouped


def selected_key(key: Tuple[str, int], args: argparse.Namespace) -> bool:
    scene, row_index = key
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def load_script_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_qwen_module(repo_root: Path) -> Any:
    return load_script_module(repo_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py", "qwen3vl_local_grounding_exam3")


def load_exam2_manifest_module(repo_root: Path) -> Any:
    return load_script_module(repo_root / "exam2" / "build_2d_eval_manifest.py", "exam2_manifest_helpers")


def scene_api_rows(repo_root: Path, scene: str) -> Dict[str, str]:
    path = repo_root / "data" / f"{scene}_api_input.csv"
    rows = read_csv_rows(path)
    return {str(index): row for index, row in enumerate(rows)}


def load_anchor_table(repo_root: Path, scene: str) -> List[Dict[str, Any]]:
    path = repo_root / "data" / f"{scene}_anchor_table.tsv"
    text = path.read_text(encoding="utf-8-sig")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    anchors: List[Dict[str, Any]] = []
    for row in reader:
        name = normalize_text(row.get("object_name") or row.get("物体名称"))
        x = parse_float(row.get("x_world") or row.get("location_x") or row.get("x"))
        y = parse_float(row.get("y_world") or row.get("location_y") or row.get("y"))
        z = parse_float(row.get("z_world") or row.get("location_z") or row.get("z"))
        if name and x is not None and y is not None and z is not None:
            anchors.append({"id": name, "x": x, "y": y, "z": z})
    if not anchors:
        raise RuntimeError(f"No anchors loaded from {path}")
    return anchors


def panel_sort_key(row: Mapping[str, str]) -> Tuple[int, float]:
    panel_id = normalize_text(row.get("panel_id")).upper()
    panel_index = parse_int(row.get("panel_index"))
    if panel_index is None:
        panel_index = parse_int(panel_id.replace("P", "")) or 999
    video_time = parse_float(row.get("video_frame_time")) or parse_float(row.get("frame_time")) or 0.0
    return panel_index, video_time


def unique_panel_rows(rows: Sequence[Mapping[str, str]]) -> List[Mapping[str, str]]:
    result: List[Mapping[str, str]] = []
    seen = set()
    for row in sorted(rows, key=panel_sort_key):
        panel_id = normalize_text(row.get("panel_id")).upper()
        if not panel_id or panel_id in seen:
            continue
        if normalize_text(row.get("frame_extracted")) == "False":
            continue
        frame_path = Path(normalize_text(row.get("frame_path")))
        if not frame_path.exists():
            continue
        seen.add(panel_id)
        result.append(row)
    return result


def choose_evidence_panel(rows: Sequence[Mapping[str, str]], strategy: str) -> Mapping[str, str]:
    panels = unique_panel_rows(rows)
    if not panels:
        raise RuntimeError("No extracted evidence panels found in manifest group.")
    if strategy == "first":
        return panels[0]
    if strategy == "middle":
        return panels[len(panels) // 2]
    return min(
        panels,
        key=lambda row: (
            -(parse_float(row.get("panel_selection_score")) or 0.0),
            panel_sort_key(row)[0],
            panel_sort_key(row)[1],
        ),
    )


def gt_anchor_points(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    anchors: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        name = normalize_text(row.get("referent_name"))
        x = parse_float(row.get("anchor_x"))
        y = parse_float(row.get("anchor_y"))
        z = parse_float(row.get("anchor_z"))
        if not name or x is None or y is None or z is None:
            continue
        key = (name, round(x, 6), round(y, 6), round(z, 6))
        if key in seen:
            continue
        seen.add(key)
        anchors.append({"id": name, "x": x, "y": y, "z": z})
    return anchors


def nearest_camera_sample(exam2_module: Any, json_path: Path, json_sample_time: float) -> Mapping[str, Any]:
    samples = exam2_module.load_multimodal_samples(json_path)
    timed_samples = exam2_module.collect_timed_samples(samples)
    if not timed_samples:
        raise RuntimeError(f"No timed samples in {json_path}")
    _sample_time, sample = exam2_module.nearest_sample(timed_samples, json_sample_time)
    return sample


def point3_from_mapping(exam2_module: Any, mapping: Any) -> Optional[Tuple[float, float, float]]:
    point = exam2_module.point3(mapping)
    if point is None:
        return None
    return float(point[0]), float(point[1]), float(point[2])


def quat_text(sample: Mapping[str, Any]) -> str:
    rotation = sample.get("cameraRotation")
    if not isinstance(rotation, Mapping):
        return ""
    parts = [parse_float(rotation.get(key)) for key in ("x", "y", "z", "w")]
    if any(part is None for part in parts):
        return ""
    return ",".join(f"{float(part):.6f}" for part in parts if part is not None)


def compact_mapping(mapping: Any) -> str:
    if not isinstance(mapping, Mapping):
        return "unavailable"
    parts = []
    for key in ("x", "y", "z"):
        value = parse_float(mapping.get(key))
        if value is None:
            return "unavailable"
        parts.append(f"{key}={value:.3f}")
    return "(" + ", ".join(parts) + ")"


def build_instruction_block(api_row: Mapping[str, str], manifest_row: Mapping[str, str]) -> str:
    lines = [
        f"instruction_text: {normalize_text(api_row.get('instruction_text') or manifest_row.get('instruction'))}",
    ]
    utterance = normalize_text(api_row.get("utterance_text"))
    target = normalize_text(api_row.get("target_description"))
    spatial = normalize_text(api_row.get("spatial_context_text"))
    if utterance:
        lines.append(f"utterance_text: {utterance}")
    if target and target != lines[0].split(": ", 1)[-1]:
        lines.append(f"target_description: {target}")
    if spatial:
        lines.append("spatial_context_text:")
        lines.append(spatial[:1800])
    return "\n".join(lines)


def build_evidence_block(panel_row: Mapping[str, str]) -> str:
    return "\n".join(
        [
            f"panel_id: {normalize_text(panel_row.get('panel_id'))}",
            f"selection_strategy: exam2 evidence panel with highest panel_selection_score",
            f"panel_selection_score: {normalize_text(panel_row.get('panel_selection_score'))}",
            f"panel_selection_reason: {normalize_text(panel_row.get('panel_selection_reason'))}",
            f"video_frame_time_seconds: {normalize_text(panel_row.get('video_frame_time'))}",
            f"json_sample_time_seconds: {normalize_text(panel_row.get('json_sample_time'))}",
        ]
    )


def build_camera_block(camera: Tuple[float, float, float], sample: Mapping[str, Any]) -> str:
    fov = parse_float(sample.get("cameraFOV"))
    lines = [
        f"camera_position: [{camera[0]:.6f}, {camera[1]:.6f}, {camera[2]:.6f}]",
        f"camera_rotation_xyzw: [{quat_text(sample)}]",
    ]
    if fov is not None:
        lines.append(f"camera_fov_degrees: {fov:.6f}")
    return "\n".join(lines)


def build_cue_block(api_row: Mapping[str, str], sample: Mapping[str, Any]) -> str:
    lines = []
    gaze_summary = normalize_text(api_row.get("gaze_summary"))
    hand_summary = normalize_text(api_row.get("hand_summary"))
    if gaze_summary:
        lines.append("event_gaze_summary:")
        lines.append(gaze_summary[:1400])
    if hand_summary:
        lines.append("event_hand_summary:")
        lines.append(hand_summary[:1000])
    eye_gaze = sample.get("eyeGaze") if isinstance(sample.get("eyeGaze"), Mapping) else {}
    lines.append(f"evidence_sample_gazePoint: {compact_mapping(eye_gaze.get('gazePoint'))}")
    lines.append(f"evidence_sample_cameraHitPoint: {compact_mapping(eye_gaze.get('cameraHitPoint'))}")
    return "\n".join(lines)


def build_anchor_block(anchors: Sequence[Mapping[str, Any]], max_lines: int) -> str:
    lines = []
    for anchor in anchors[:max_lines]:
        lines.append(f"- {anchor['id']}: [{anchor['x']:.3f}, {anchor['y']:.3f}, {anchor['z']:.3f}]")
    if len(anchors) > max_lines:
        lines.append(f"... {len(anchors) - max_lines} additional anchors omitted")
    return "\n".join(lines)


def build_prompt(
    template: str,
    api_row: Mapping[str, str],
    panel_row: Mapping[str, str],
    camera: Tuple[float, float, float],
    sample: Mapping[str, Any],
    anchors: Sequence[Mapping[str, Any]],
    max_anchor_lines: int,
) -> str:
    replacements = {
        "[INSTRUCTION_BLOCK]": build_instruction_block(api_row, panel_row),
        "[EVIDENCE_BLOCK]": build_evidence_block(panel_row),
        "[CAMERA_BLOCK]": build_camera_block(camera, sample),
        "[CUE_BLOCK]": build_cue_block(api_row, sample),
        "[ANCHOR_BLOCK]": build_anchor_block(anchors, max_anchor_lines),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def build_model_inputs(processor: Any, image: Image.Image, prompt_text: str, system_prompt: str) -> Mapping[str, Any]:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[chat_text], images=[image], return_tensors="pt")
    return processor(text=[system_prompt + "\n\n" + prompt_text], images=[image], return_tensors="pt")


def extract_json_text(raw_response: str) -> Optional[str]:
    candidates: List[str] = []
    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL))
    candidates.extend(re.findall(r"\{.*\}", raw_response, flags=re.DOTALL))
    for candidate in candidates:
        cleaned = re.sub(r"^```(?:json)?", "", candidate.strip()).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return json.dumps(payload, ensure_ascii=False)
    return None


def parse_point_payload(raw_response: str) -> Tuple[Optional[Dict[str, Any]], bool, Optional[Tuple[float, float, float]], str]:
    json_text = extract_json_text(raw_response)
    if json_text is None:
        return None, False, None, "no_json_object"
    payload = json.loads(json_text)
    raw_point = payload.get("point_3d")
    if isinstance(raw_point, Mapping):
        values = [raw_point.get(key) for key in ("x", "y", "z")]
    elif isinstance(raw_point, list):
        values = raw_point
    else:
        return payload, False, None, "missing_point_3d"
    if len(values) != 3:
        return payload, False, None, "point_3d_wrong_dimension"
    parsed_values: List[float] = []
    for value in values:
        parsed = parse_float(value)
        if parsed is None:
            return payload, False, None, "point_3d_nonfinite_or_not_number"
        parsed_values.append(parsed)
    payload["point_3d"] = parsed_values
    return payload, True, (parsed_values[0], parsed_values[1], parsed_values[2]), ""


def run_one_event(
    runner: Any,
    qwen_module: Any,
    image_path: Path,
    prompt_text: str,
    max_new_tokens: int,
) -> Tuple[str, Optional[Dict[str, Any]], bool, Optional[Tuple[float, float, float]], str]:
    image = Image.open(image_path).convert("RGB")
    system_prompt = "Return strict JSON only for the camera-centered 3D directional point diagnostic."
    model_inputs = build_model_inputs(runner.processor, image, prompt_text, system_prompt)
    model_inputs = qwen_module.move_batch_to_device(model_inputs, runner.device)
    with runner.runtime.torch.inference_mode():
        generated = runner.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated_ids = qwen_module.trim_generated_ids(model_inputs, generated)
    raw_response = qwen_module.decode_response(runner.processor, generated_ids).strip()
    parsed, parse_ok, point, invalid_reason = parse_point_payload(raw_response)
    return raw_response, parsed, parse_ok, point, invalid_reason


def load_scene_cache(repo_root: Path, scenes: Iterable[str]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, List[Dict[str, Any]]]]:
    api_cache: Dict[str, Dict[str, str]] = {}
    anchor_cache: Dict[str, List[Dict[str, Any]]] = {}
    for scene in sorted(set(scenes)):
        api_cache[scene] = scene_api_rows(repo_root, scene)
        anchor_cache[scene] = load_anchor_table(repo_root, scene)
    return api_cache, anchor_cache


def make_error_row(scene: str, row_index: int, rows: Sequence[Mapping[str, str]], error: Exception) -> Dict[str, Any]:
    first = rows[0] if rows else {}
    return {
        "scene": scene,
        "row_index": row_index,
        "event_id": normalize_text(first.get("event_id")),
        "instruction": normalize_text(first.get("instruction")),
        "parse_ok": "False",
        "invalid_reason": "runtime_error",
        "error_message": f"{type(error).__name__}: {error}",
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json_dir = Path(args.output_json_dir).resolve()
    prompt_template_path = Path(args.prompt_template).resolve()
    output_json_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = read_csv_rows(manifest_path)
    grouped = group_manifest(manifest_rows)
    selected = [(key, grouped[key]) for key in sorted(grouped, key=lambda item: (item[0], item[1])) if selected_key(key, args)]
    if not selected:
        raise RuntimeError("No samples selected.")

    api_cache, anchor_cache = load_scene_cache(repo_root, [key[0] for key, _rows in selected])
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    exam2_module = load_exam2_manifest_module(repo_root)
    qwen_module = load_qwen_module(repo_root)
    runner = qwen_module.LocalQwen3VLRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
        input_mode="image",
        max_video_frames=1,
        ffmpeg_path=None,
        ffprobe_path=None,
        prompt_variant="debug",
        offload_folder=args.offload_folder,
    )
    runner.load()

    output_rows: List[Dict[str, Any]] = []
    for key, rows in selected:
        scene, row_index = key
        first = rows[0]
        raw_path = output_json_dir / f"{scene}_row_{row_index}.json"
        try:
            panel_row = choose_evidence_panel(rows, args.evidence_panel_strategy)
            json_path = Path(normalize_text(panel_row.get("json_path")))
            json_sample_time = parse_float(panel_row.get("json_sample_time"))
            if json_sample_time is None:
                raise RuntimeError("Missing json_sample_time for selected evidence panel.")
            sample = nearest_camera_sample(exam2_module, json_path, json_sample_time)
            camera = point3_from_mapping(exam2_module, sample.get("cameraPosition"))
            if camera is None:
                raise RuntimeError("Missing cameraPosition in evidence sample.")
            api_row = api_cache[scene].get(str(row_index), {})
            gt_points = gt_anchor_points(rows)
            anchors = anchor_cache[scene]
            prompt_text = build_prompt(
                prompt_template,
                api_row,
                panel_row,
                camera,
                sample,
                anchors,
                int(args.max_anchor_lines),
            )
            image_path = Path(normalize_text(panel_row.get("frame_path")))

            if raw_path.exists() and not args.overwrite:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
                raw_output = normalize_text(payload.get("model_raw_output"))
                parsed = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else None
                parse_ok = bool(payload.get("parse_ok"))
                point_payload = parsed.get("point_3d") if isinstance(parsed, Mapping) else None
                point = None
                invalid_reason = normalize_text(payload.get("invalid_reason"))
                if parse_ok and isinstance(point_payload, list) and len(point_payload) == 3:
                    values = [parse_float(value) for value in point_payload]
                    if all(value is not None for value in values):
                        point = (float(values[0]), float(values[1]), float(values[2]))  # type: ignore[arg-type]
            else:
                raw_output, parsed, parse_ok, point, invalid_reason = run_one_event(
                    runner,
                    qwen_module,
                    image_path,
                    prompt_text,
                    args.max_new_tokens,
                )
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(first.get("event_id")),
                    "instruction": normalize_text(first.get("instruction")),
                    "evidence_panel_strategy": args.evidence_panel_strategy,
                    "evidence_panel": {
                        "panel_id": normalize_text(panel_row.get("panel_id")),
                        "frame_path": str(image_path),
                        "video_frame_time": normalize_text(panel_row.get("video_frame_time")),
                        "json_sample_time": normalize_text(panel_row.get("json_sample_time")),
                        "panel_selection_score": normalize_text(panel_row.get("panel_selection_score")),
                        "panel_selection_reason": normalize_text(panel_row.get("panel_selection_reason")),
                    },
                    "camera_position": list(camera),
                    "camera_rotation_xyzw": quat_text(sample),
                    "camera_fov": parse_float(sample.get("cameraFOV")),
                    "gt_anchor_points": gt_points,
                    "candidate_anchor_count": len(anchors),
                    "prompt_template_path": str(prompt_template_path),
                    "prompt_text": prompt_text,
                    "model_input_image": str(image_path),
                    "model_raw_output": raw_output,
                    "parsed_json": parsed or {},
                    "parse_ok": parse_ok,
                    "invalid_reason": invalid_reason,
                }
                raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            output_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(first.get("event_id")),
                    "instruction": normalize_text(first.get("instruction")),
                    "evidence_panel_id": normalize_text(panel_row.get("panel_id")),
                    "evidence_selection_strategy": args.evidence_panel_strategy,
                    "evidence_frame_path": str(image_path),
                    "video_frame_time": normalize_text(panel_row.get("video_frame_time")),
                    "json_sample_time": normalize_text(panel_row.get("json_sample_time")),
                    "json_path": str(json_path),
                    "camera_x": f"{camera[0]:.9f}",
                    "camera_y": f"{camera[1]:.9f}",
                    "camera_z": f"{camera[2]:.9f}",
                    "camera_rotation_xyzw": quat_text(sample),
                    "camera_fov": parse_float(sample.get("cameraFOV")) or "",
                    "gt_anchor_ids": ";".join(item["id"] for item in gt_points),
                    "gt_anchor_points_json": json.dumps(gt_points, ensure_ascii=False),
                    "candidate_anchor_count": len(anchors),
                    "model_input_image": str(image_path),
                    "raw_json_path": str(raw_path),
                    "model_raw_output": raw_output,
                    "parsed_json": json.dumps(parsed or {}, ensure_ascii=False),
                    "parse_ok": str(bool(parse_ok)),
                    "point_x": f"{point[0]:.9f}" if point is not None else "",
                    "point_y": f"{point[1]:.9f}" if point is not None else "",
                    "point_z": f"{point[2]:.9f}" if point is not None else "",
                    "invalid_reason": invalid_reason,
                    "error_message": "",
                }
            )
            print(f"[ok] {scene} row_{row_index} parse_ok={parse_ok}")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            output_rows.append(make_error_row(scene, row_index, rows, exc))
            print(f"[error] {scene} row_{row_index}: {exc}", file=sys.stderr)

    write_csv(output_csv, output_rows)
    print(f"Wrote predictions: {output_csv}")


if __name__ == "__main__":
    main()
