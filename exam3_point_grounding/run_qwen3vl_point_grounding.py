#!/usr/bin/env python3
"""Run Qwen3-VL for candidate-free point-supervised 3D grounding."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import load_qwen_module, normalize_text, parse_int, read_csv_rows, write_csv, write_json  # noqa: E402
from point_parser import parse_points_3d_output  # noqa: E402
from ablation_inputs import ABLATION_VARIANTS, audit_prompt, frame_paths as ablation_frame_paths, render_prompt  # noqa: E402
from hand_masking import prepare_row_hand_masks  # noqa: E402


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
    "ablation_variant",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL point-supervised 3D grounding.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", default="exam3_point_grounding/outputs/manifest.csv")
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--offload_folder")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample_keys", nargs="*", help="Exact scene:row_index keys.")
    parser.add_argument("--sample_keys_file", help="Text file with one scene:row_index key per line.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--flush_every", type=int, default=25, help="Write the prediction CSV every N processed samples; <=0 only writes at the end.")
    parser.add_argument("--ablation_variant", choices=("full",) + ABLATION_VARIANTS, default="full")
    parser.add_argument("--hand_mask_root", default="ablation/exam3/hand_masked_frames_v1")
    parser.add_argument("--overwrite_hand_masks", action="store_true")
    parser.add_argument("--prompt_template", default="exam3_point_grounding/prompts/qwen3vl_point_grounding.md")
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


def frame_paths_from_row(row: Mapping[str, str]) -> List[Path]:
    try:
        payload = json.loads(normalize_text(row.get("frame_paths_json")) or "[]")
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [Path(str(item)) for item in payload if str(item)]


def build_model_inputs(processor: Any, images: Sequence[Image.Image], prompt_text: str) -> Mapping[str, Any]:
    system_prompt = "Return strict JSON only for candidate-free point-supervised 3D referent grounding."
    content = [{"type": "image"} for _image in images]
    content.append({"type": "text", "text": prompt_text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if images:
            return processor(text=[chat_text], images=list(images), return_tensors="pt")
        return processor(text=[chat_text], return_tensors="pt")
    if images:
        return processor(text=[system_prompt + "\n\n" + prompt_text], images=list(images), return_tensors="pt")
    return processor(text=[system_prompt + "\n\n" + prompt_text], return_tensors="pt")


def run_one(
    runner: Any,
    qwen_module: Any,
    row: Mapping[str, str],
    max_new_tokens: int,
    prompt_text: str,
    selected_frame_paths: Sequence[Path],
) -> Tuple[str, bool, Dict[str, Any], str]:
    frame_paths = list(selected_frame_paths)
    images = []
    for path in frame_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing evidence image: {path}")
        images.append(Image.open(path).convert("RGB"))
    model_inputs = build_model_inputs(runner.processor, images, prompt_text)
    model_inputs = qwen_module.move_batch_to_device(model_inputs, runner.device)
    with runner.runtime.torch.inference_mode():
        generated = runner.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated_ids = qwen_module.trim_generated_ids(model_inputs, generated)
    raw_output = qwen_module.decode_response(runner.processor, generated_ids).strip()
    parse_ok, parsed, invalid_reason = parse_points_3d_output(raw_output)
    return raw_output, parse_ok, parsed, invalid_reason


def output_row_from_payload(row: Mapping[str, str], raw_path: Path, payload: Mapping[str, Any], variant: str) -> Dict[str, Any]:
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
        "ablation_variant": variant,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_rows = read_csv_rows((repo_root / args.manifest).resolve())
    prompt_template = (repo_root / args.prompt_template).read_text(encoding="utf-8")
    exact_keys = parse_sample_keys(args.sample_keys, args.sample_keys_file)
    rows = [row for row in manifest_rows if selected(row, args, exact_keys)]
    if not rows:
        raise RuntimeError("No manifest rows selected.")

    output_json_dir = Path(args.output_json_dir).resolve()
    output_json_dir.mkdir(parents=True, exist_ok=True)
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
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = normalize_text(row.get("row_index"))
        raw_path = output_json_dir / f"{scene}_row_{row_index}.json"
        variant = args.ablation_variant
        prompt_text = normalize_text(row.get("prompt_text")) if variant == "full" else render_prompt(prompt_template, row, variant)
        source_frame_paths = frame_paths_from_row(row) if variant == "full" else ablation_frame_paths(row, variant)
        hand_mask_audit: List[Dict[str, Any]] = []
        if variant == "no_hand_strict":
            selected_frame_paths, hand_mask_audit = prepare_row_hand_masks(
                row, repo_root / args.hand_mask_root, overwrite=args.overwrite_hand_masks
            )
        else:
            selected_frame_paths = source_frame_paths
        try:
            if raw_path.exists() and not args.overwrite:
                payload = json.loads(raw_path.read_text(encoding="utf-8"))
            else:
                raw_output, parse_ok, parsed, invalid_reason = run_one(
                    runner, qwen_module, row, args.max_new_tokens, prompt_text, selected_frame_paths
                )
                payload = {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(row.get("event_id")),
                    "prompt_text": prompt_text,
                    "frame_paths": [str(path) for path in selected_frame_paths],
                    "source_frame_paths": [str(path) for path in source_frame_paths],
                    "ablation_variant": variant,
                    "prompt_audit": audit_prompt(prompt_text, variant),
                    "hand_mask_audit": hand_mask_audit,
                    "model_raw_output": raw_output,
                    "parsed_json": parsed,
                    "parse_ok": parse_ok,
                    "invalid_reason": invalid_reason,
                    "error_message": "",
                }
                write_json(raw_path, payload)
            output_rows.append(output_row_from_payload(row, raw_path, payload, variant))
            print(f"[ok] {scene}:{row_index} parse_ok={payload.get('parse_ok')} points={len(payload.get('parsed_json', {}).get('points_3d', [])) if isinstance(payload.get('parsed_json'), Mapping) else 0}", flush=True)
        except Exception as exc:
            payload = {
                "scene": scene,
                "row_index": row_index,
                "event_id": normalize_text(row.get("event_id")),
                "prompt_text": prompt_text,
                "frame_paths": [str(path) for path in selected_frame_paths],
                "source_frame_paths": [str(path) for path in source_frame_paths],
                "ablation_variant": variant,
                "prompt_audit": audit_prompt(prompt_text, variant),
                "hand_mask_audit": hand_mask_audit,
                "model_raw_output": "",
                "parsed_json": {"points_3d": []},
                "parse_ok": False,
                "invalid_reason": "runtime_error",
                "error_message": f"{type(exc).__name__}: {exc}",
            }
            write_json(raw_path, payload)
            output_rows.append(output_row_from_payload(row, raw_path, payload, variant))
            print(f"[error] {scene}:{row_index} {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
        if args.flush_every > 0 and len(output_rows) % args.flush_every == 0:
            write_csv(Path(args.output_csv).resolve(), OUTPUT_COLUMNS, output_rows)

    write_csv(Path(args.output_csv).resolve(), OUTPUT_COLUMNS, output_rows)


if __name__ == "__main__":
    main()
