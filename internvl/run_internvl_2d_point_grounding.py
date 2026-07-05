#!/usr/bin/env python3
"""Run InternVL for experiment-2 projected-2D point grounding."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from internvl_utils import InternVLChatRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local InternVL on exam2 multi-panel 2D point-grounding inputs.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json_dir", required=True)
    parser.add_argument("--model_input_dir")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--offload_folder")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--panel_width", type=int, default=512)
    parser.add_argument("--panel_height", type=int, default=384)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--input_mode", choices=("multi_image", "contact_sheet"), default="multi_image")
    parser.add_argument("--panel_caption_mode", choices=("none", "text"), default="none")
    parser.add_argument("--panel_context_mode", choices=("full", "full_crop", "paired_crop"), default="paired_crop")
    parser.add_argument("--gaze_crop_ratio", type=float, default=0.35)
    parser.add_argument("--crop_output_size", type=int, default=768)
    parser.add_argument(
        "--paired_crop_coordinate_policy",
        choices=("none", "source_or_canvas_map", "paired_canvas_map"),
        default="paired_canvas_map",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=("expected_count", "gt_referents", "instruction_only"),
        default="expected_count",
    )
    parser.add_argument("--ablate_modalities", default="")
    parser.add_argument("--gaze_mask_radius_ratio", type=float, default=0.035)
    return parser.parse_args()


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_qwen2d_module(repo_root: Path) -> Any:
    return load_module(
        "internvl_reused_qwen2d_grounding",
        repo_root / "ablation" / "exam2" / "scripts" / "run_qwen3vl_2d_point_grounding.py",
    )


def run_one_event(
    runner: InternVLChatRunner,
    image_paths: Sequence[Path],
    prompt_text: str,
    system_prompt: str,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image.load()
        images.append(image)
    raw_response = runner.chat_images(images, system_prompt, prompt_text)
    qwen2d = load_qwen2d_module(REPO_ROOT)
    return raw_response, qwen2d.extract_json(raw_response)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    qwen2d = load_qwen2d_module(repo_root)
    manifest_path = Path(args.manifest).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json_dir = Path(args.output_json_dir).resolve()
    model_input_dir = Path(args.model_input_dir).resolve() if args.model_input_dir else output_json_dir / "model_inputs"
    output_json_dir.mkdir(parents=True, exist_ok=True)
    model_input_dir.mkdir(parents=True, exist_ok=True)

    disabled_modalities: Set[str] = qwen2d.parse_ablation_modalities(args.ablate_modalities)
    system_prompt = qwen2d.build_system_prompt(disabled_modalities)
    manifest_rows = qwen2d.read_csv_rows(manifest_path)
    grouped = qwen2d.group_manifest(manifest_rows)
    selected = [
        (key, grouped[key])
        for key in sorted(grouped, key=lambda item: (item[0], item[1]))
        if qwen2d.selected_key(key, args)
    ]

    runner = InternVLChatRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        local_files_only=args.local_files_only,
        load_in_8bit=bool(args.load_in_8bit and not args.no_load_in_8bit),
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        image_size=args.image_size,
        offload_folder=args.offload_folder,
    )
    print(f"Loading InternVL model for {len(selected)} exam2 events...", flush=True)
    runner.load()
    print("Model loaded.", flush=True)

    output_rows: List[Dict[str, Any]] = []
    for key, rows in selected:
        scene, row_index = key
        first = rows[0]
        event_id = qwen2d.normalize_text(first.get("event_id"))
        per_event_json = output_json_dir / f"{scene}_row_{row_index}.json"
        try:
            if args.input_mode == "contact_sheet":
                image_path, panel_meta = qwen2d.build_model_contact_sheet(
                    key,
                    rows,
                    model_input_dir,
                    (args.panel_width, args.panel_height),
                    args.columns,
                    args.overwrite,
                )
                image_paths = [image_path]
            else:
                image_paths, panel_meta = qwen2d.collect_model_panels(key, rows)
                image_paths, panel_meta = qwen2d.prepare_ablation_panel_images(
                    key,
                    image_paths,
                    panel_meta,
                    model_input_dir,
                    args,
                    disabled_modalities,
                )
                image_paths, panel_meta = qwen2d.expand_panel_context_images(key, image_paths, panel_meta, model_input_dir, args)
                image_path = image_paths[0]
            prompt_text = qwen2d.build_prompt(rows, panel_meta, args.prompt_mode, disabled_modalities)
            coordinate_policy = args.paired_crop_coordinate_policy if args.panel_context_mode == "paired_crop" else "none"
            if per_event_json.exists() and not args.overwrite:
                payload = json.loads(per_event_json.read_text(encoding="utf-8"))
                raw_output = qwen2d.normalize_text(payload.get("model_raw_output"))
                raw_parsed = payload.get("raw_parsed_json")
                if not isinstance(raw_parsed, Mapping):
                    raw_parsed = payload.get("parsed_json") if isinstance(payload.get("parsed_json"), Mapping) else None
                parsed = qwen2d.map_prediction_coordinates(raw_parsed, panel_meta, coordinate_policy)
            else:
                raw_output, raw_parsed = run_one_event(runner, image_paths, prompt_text, system_prompt)
                parsed = qwen2d.map_prediction_coordinates(raw_parsed, panel_meta, coordinate_policy)
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "instruction": qwen2d.normalize_text(first.get("instruction")),
                    "model_family": "InternVL",
                    "model_name": args.model_name,
                    "input_mode": args.input_mode,
                    "panel_context_mode": args.panel_context_mode,
                    "panel_caption_mode": args.panel_caption_mode,
                    "ablate_modalities": sorted(disabled_modalities),
                    "paired_crop_coordinate_policy": coordinate_policy,
                    "model_input_image": ";".join(str(path) for path in image_paths),
                    "model_input_images": [str(path) for path in image_paths],
                    "prompt_text": prompt_text,
                    "panel_meta": list(panel_meta),
                    "model_raw_output": raw_output,
                    "raw_parsed_json": raw_parsed or {},
                    "parsed_json": parsed or {},
                    "parse_ok": raw_parsed is not None,
                }
                per_event_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            prediction_count = len(parsed.get("referents", [])) if isinstance(parsed, Mapping) else 0
            output_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "instruction": qwen2d.normalize_text(first.get("instruction")),
                    "model_input_image": ";".join(str(path) for path in image_paths),
                    "model_raw_output": raw_output,
                    "parsed_json": json.dumps(parsed or {}, ensure_ascii=False),
                    "parse_ok": str(parsed is not None),
                    "prediction_count": prediction_count,
                    "error_message": "",
                }
            )
            print(f"[ok] {scene} row_{row_index} predictions={prediction_count}", flush=True)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            output_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": event_id,
                    "instruction": qwen2d.normalize_text(first.get("instruction")),
                    "model_input_image": "",
                    "model_raw_output": "",
                    "parsed_json": "",
                    "parse_ok": "False",
                    "prediction_count": 0,
                    "error_message": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[error] {scene} row_{row_index}: {exc}", file=sys.stderr, flush=True)
        finally:
            runner.empty_cache()
    qwen2d.write_csv(output_csv, output_rows)
    print(f"Wrote predictions: {output_csv}", flush=True)


if __name__ == "__main__":
    main()
