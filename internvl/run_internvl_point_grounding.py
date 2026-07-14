#!/usr/bin/env python3
"""Run InternVL for candidate-free point-supervised 3D grounding."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
EXAM3_DIR = REPO_ROOT / "exam3_point_grounding"
sys.path.insert(0, str(EXAM3_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from internvl_utils import InternVLChatRunner  # noqa: E402
from build_point_grounding_manifest import build_prompt  # noqa: E402
from point_grounding_common import normalize_text, parse_int, read_csv_rows, write_csv, write_json  # noqa: E402
from point_parser import parse_points_3d_output  # noqa: E402


OUTPUT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "raw_json_path",
    "model_raw_output",
    "parsed_json",
    "parse_ok",
    "invalid_reason",
    "pred_point_count",
    "pred_points_json",
    "error_message",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InternVL point-supervised 3D grounding.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", default="exam3_point_grounding/outputs/manifest.csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_images", type=int, default=0, help="Maximum evidence frames to send to InternVL; <=0 keeps all frames.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--prompt_template", default="exam3_point_grounding/prompts/qwen3vl_point_grounding.md")
    parser.add_argument("--offload_folder")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample_keys", nargs="*", help="Exact scene:row_index keys.")
    parser.add_argument("--sample_keys_file", help="Text file with one scene:row_index key per line.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--flush_every", type=int, default=25, help="Write the prediction CSV every N processed samples; <=0 only writes at the end.")
    return parser.parse_args()


def parse_sample_keys(items: Optional[Sequence[str]], file_path: Optional[str] = None) -> Optional[set[Tuple[str, int]]]:
    merged_items = list(items or [])
    if file_path:
        for line in Path(file_path).read_text(encoding="utf-8").splitlines():
            item = normalize_text(line)
            if item and not item.startswith("#"):
                merged_items.append(item)
    if not merged_items:
        return None
    result = set()
    for item in merged_items:
        text = normalize_text(item)
        if ":" not in text:
            raise ValueError(f"sample key must be scene:row_index, got {text!r}")
        scene, row_text = text.rsplit(":", 1)
        row_index = parse_int(row_text)
        if not scene or row_index is None:
            raise ValueError(f"sample key must be scene:row_index, got {text!r}")
        result.add((scene, row_index))
    return result


def selected(row: Mapping[str, str], args: argparse.Namespace, exact_keys: Optional[set[Tuple[str, int]]]) -> bool:
    scene = normalize_text(row.get("scene"))
    row_index = parse_int(row.get("row_index"))
    if row_index is None:
        return False
    if exact_keys is not None:
        return (scene, row_index) in exact_keys
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return normalize_text(row.get("status")) == "ok"


def frame_paths_from_row(row: Mapping[str, str], max_images: int = 0) -> List[Path]:
    try:
        payload = json.loads(normalize_text(row.get("frame_paths_json")) or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    paths = [Path(str(item)) for item in payload if str(item)]
    if max_images and max_images > 0:
        return paths[:max_images]
    return paths


def load_images_from_row(row: Mapping[str, str], max_images: int = 0) -> List[Image.Image]:
    images = []
    for path in frame_paths_from_row(row, max_images=max_images):
        if not path.exists():
            raise FileNotFoundError(f"Missing evidence image: {path}")
        image = Image.open(path).convert("RGB")
        image.load()
        images.append(image)
    if not images:
        raise RuntimeError("No frame paths in manifest row.")
    return images


def prompt_text_from_row(row: Mapping[str, str], prompt_template: str, max_images: int = 0) -> str:
    if max_images <= 0:
        return normalize_text(row.get("prompt_text"))
    try:
        evidence_payload = json.loads(normalize_text(row.get("evidence_json")) or "[]")
        scene_bounds = json.loads(normalize_text(row.get("scene_bounds_json")) or "{}")
    except json.JSONDecodeError:
        return normalize_text(row.get("prompt_text"))
    if not isinstance(evidence_payload, list) or not isinstance(scene_bounds, Mapping):
        return normalize_text(row.get("prompt_text"))
    trimmed_evidence = evidence_payload[:max_images]
    api_row = {
        "event_id": normalize_text(row.get("event_id")),
        "instruction_text": normalize_text(row.get("instruction")),
        "utterance_text": normalize_text(row.get("utterance_text")),
    }
    return build_prompt(prompt_template, api_row, scene_bounds, trimmed_evidence)


def cleanup_after_sample(runner: InternVLChatRunner) -> None:
    gc.collect()
    runtime = getattr(runner, "runtime", None)
    torch = getattr(runtime, "torch", None) if runtime is not None else None
    if torch is not None and hasattr(torch, "cuda"):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def run_one(runner: InternVLChatRunner, row: Mapping[str, str], max_images: int, prompt_text: str) -> Tuple[str, bool, Dict[str, Any], str]:
    images = load_images_from_row(row, max_images=max_images)
    system_prompt = "Return strict JSON only for candidate-free point-supervised 3D referent grounding."
    try:
        raw_output = runner.chat_video_frames(images, system_prompt, prompt_text)
    finally:
        del images
        cleanup_after_sample(runner)
    parse_ok, parsed, invalid_reason = parse_points_3d_output(raw_output)
    return raw_output, parse_ok, parsed, invalid_reason


def output_row_from_payload(row: Mapping[str, str], raw_path: Path, payload: Mapping[str, Any]) -> Dict[str, Any]:
    parsed = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else {"points_3d": []}
    points = parsed.get("points_3d") if isinstance(parsed, Mapping) else []
    if not isinstance(points, list):
        points = []
    return {
        "scene": normalize_text(row.get("scene")),
        "row_index": normalize_text(row.get("row_index")),
        "event_id": normalize_text(row.get("event_id")),
        "instruction": normalize_text(row.get("instruction")),
        "raw_json_path": str(raw_path),
        "model_raw_output": normalize_text(payload.get("model_raw_output")),
        "parsed_json": json.dumps(parsed, ensure_ascii=False),
        "parse_ok": str(bool(payload.get("parse_ok"))),
        "invalid_reason": normalize_text(payload.get("invalid_reason")),
        "pred_point_count": len(points),
        "pred_points_json": json.dumps(points, ensure_ascii=False),
        "error_message": normalize_text(payload.get("error_message")),
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_rows = read_csv_rows((repo_root / args.manifest).resolve())
    exact_keys = parse_sample_keys(args.sample_keys, args.sample_keys_file)
    rows = [row for row in manifest_rows if selected(row, args, exact_keys)]
    if not rows:
        raise RuntimeError("No manifest rows selected.")

    output_json_dir = Path(args.output_json_dir).resolve()
    output_json_dir.mkdir(parents=True, exist_ok=True)
    prompt_template = (repo_root / args.prompt_template).read_text(encoding="utf-8")

    load_in_8bit = bool(args.load_in_8bit and not args.no_load_in_8bit)
    runner = InternVLChatRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        load_in_8bit=load_in_8bit,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size,
        offload_folder=args.offload_folder,
    )
    runner.load()

    output_rows: List[Dict[str, Any]] = []
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = normalize_text(row.get("row_index"))
        raw_path = output_json_dir / f"{scene}_row_{row_index}.json"
        try:
            if raw_path.exists() and not args.overwrite:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                prompt_text = prompt_text_from_row(row, prompt_template, args.max_images)
                raw_output, parse_ok, parsed, invalid_reason = run_one(runner, row, args.max_images, prompt_text)
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(row.get("event_id")),
                    "prompt_text": prompt_text,
                    "frame_paths": [str(path) for path in frame_paths_from_row(row)],
                    "used_frame_paths": [str(path) for path in frame_paths_from_row(row, max_images=args.max_images)],
                    "max_images": args.max_images,
                    "model_raw_output": raw_output,
                    "parsed_json": parsed,
                    "parse_ok": parse_ok,
                    "invalid_reason": invalid_reason,
                    "error_message": "",
                }
                write_json(raw_path, payload)
            output_rows.append(output_row_from_payload(row, raw_path, payload))
            point_count = len(payload.get("parsed_json", {}).get("points_3d", [])) if isinstance(payload.get("parsed_json"), Mapping) else 0
            print(f"[ok] {scene}:{row_index} parse_ok={payload.get('parse_ok')} points={point_count}", flush=True)
        except Exception as exc:
            payload = {
                "scene": scene,
                "row_index": row_index,
                "event_id": normalize_text(row.get("event_id")),
                "prompt_text": normalize_text(row.get("prompt_text")),
                "frame_paths": [str(path) for path in frame_paths_from_row(row)],
                "used_frame_paths": [str(path) for path in frame_paths_from_row(row, max_images=args.max_images)],
                "max_images": args.max_images,
                "model_raw_output": "",
                "parsed_json": {"points_3d": []},
                "parse_ok": False,
                "invalid_reason": "runtime_error",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-12:],
            }
            write_json(raw_path, payload)
            output_rows.append(output_row_from_payload(row, raw_path, payload))
            print(f"[error] {scene}:{row_index} {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
        if args.flush_every > 0 and len(output_rows) % args.flush_every == 0:
            write_csv(Path(args.output_csv).resolve(), OUTPUT_COLUMNS, output_rows)

    write_csv(Path(args.output_csv).resolve(), OUTPUT_COLUMNS, output_rows)


if __name__ == "__main__":
    main()
