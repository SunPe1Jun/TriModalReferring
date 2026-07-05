#!/usr/bin/env python3
"""Run InternVL for closed-set 3D anchor selection with existing VR-TriRef prompts/parsers."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from internvl_utils import InternVLChatRunner, extract_video_frames, parse_float


class InternVL3DError(Exception):
    """Raised when InternVL 3D inference cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run InternVL closed-set 3D anchor selection for a row range.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--scene_anchor_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--no_load_in_8bit", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--max_new_tokens", type=int, default=1536)
    parser.add_argument("--input_mode", choices=("video", "image", "auto"), default="video")
    parser.add_argument("--max_video_frames", type=int, default=16)
    parser.add_argument("--max_evidence_segments", type=int, default=0)
    parser.add_argument("--evidence_segment_duration", type=float, default=0.5)
    parser.add_argument("--ffmpeg_path")
    parser.add_argument("--ffprobe_path")
    parser.add_argument("--prompt_style", choices=("world_only", "full"), default="full")
    parser.add_argument("--prompt_strategy", choices=("standard", "mention_first"), default="mention_first")
    parser.add_argument("--ablate_modalities", default="")
    parser.add_argument("--offload_folder")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise InternVL3DError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_single_event_module() -> Any:
    return load_module(
        "internvl_exam1_single_event_3d",
        REPO_ROOT / "ablation" / "exam1" / "scripts" / "grounding" / "run_qwen3vl_local_single_event_3d.py",
    )


def load_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise InternVL3DError(f"Input CSV has no header: {input_csv}")
        return [dict(row) for row in reader]


def resolve_media_mode(row: Mapping[str, str], requested_mode: str) -> str:
    if requested_mode == "auto":
        return "video" if str(row.get("video_path", "")).strip() else "image"
    return requested_mode


def build_error_payload(
    row_index: int,
    row: Dict[str, str],
    args: argparse.Namespace,
    api3d: Any,
    anchor_rows: List[Dict[str, Any]],
    exc: Exception,
) -> Dict[str, Any]:
    return {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_index": row_index,
        "event_id": api3d.normalize_text(row.get("event_id")),
        "video_path": api3d.normalize_text(row.get("video_path")),
        "keyframe_path": api3d.normalize_text(row.get("keyframe_path")),
        "scene_anchor_csv": str(Path(args.scene_anchor_csv).resolve()),
        "scene_anchor_candidates": anchor_rows,
        "model_name": args.model_name,
        "model_family": "InternVL",
        "dtype": args.dtype,
        "load_in_8bit": bool(args.load_in_8bit and not args.no_load_in_8bit),
        "input_mode": args.input_mode,
        "max_video_frames": args.max_video_frames,
        "max_evidence_segments": args.max_evidence_segments,
        "evidence_segment_duration": args.evidence_segment_duration,
        "prompt_style": args.prompt_style,
        "prompt_strategy": args.prompt_strategy,
        "ablate_modalities": sorted(api3d.parse_ablation_modalities(args.ablate_modalities)),
        "prompt_text": "",
        "response_text": "",
        "parsed_response": None,
        "resolved_object_row": None,
        "resolved_object_rows": [],
        "validation_warnings": ["batch_row_exception"],
        "adjusted_response": None,
        "response_status": "error",
        "error_message": f"{type(exc).__name__}: {exc}",
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
        "used_storyboard_fallback": False,
        "storyboard_fallback_reason": "",
        "projected_u_norm": None,
        "projected_v_norm": None,
        "projection_valid": False,
        "raw_spatial_prior_source": api3d.normalize_text(row.get("spatial_prior_source"), "none"),
        "raw_spatial_prior_u_norm": api3d.parse_float(row.get("spatial_prior_u_norm")),
        "raw_spatial_prior_v_norm": api3d.parse_float(row.get("spatial_prior_v_norm")),
    }


def load_images_for_row(row: Mapping[str, str], args: argparse.Namespace, api3d: Any) -> List[Image.Image]:
    media_mode = resolve_media_mode(row, args.input_mode)
    if media_mode == "video":
        video_path_raw = api3d.normalize_text(row.get("video_path"))
        if not video_path_raw:
            raise InternVL3DError("input_mode=video requires video_path")
        return extract_video_frames(
            Path(video_path_raw).expanduser().resolve(),
            max_video_frames=args.max_video_frames,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            t_start=parse_float(row.get("t_start")),
            t_end=parse_float(row.get("t_end")),
        )
    keyframe_path_raw = api3d.normalize_text(row.get("keyframe_path"))
    if not keyframe_path_raw:
        raise InternVL3DError("input_mode=image requires keyframe_path")
    image = Image.open(Path(keyframe_path_raw).expanduser().resolve()).convert("RGB")
    image.load()
    return [image]


def run_one_row(
    row_index: int,
    row: Dict[str, str],
    args: argparse.Namespace,
    single_event: Any,
    api3d: Any,
    runner: InternVLChatRunner,
    anchor_rows: List[Dict[str, Any]],
    ablation_modalities: Sequence[str],
    output_path: Path,
) -> None:
    if api3d.modality_disabled(ablation_modalities, "gaze", "timeline", "structured_geometry"):
        sparse_timeline_evidence: List[Dict[str, Any]] = []
    else:
        sparse_timeline_evidence = api3d.build_sparse_timeline_evidence(
            row=row,
            anchor_rows=anchor_rows,
            max_segments=args.max_evidence_segments,
            segment_duration=args.evidence_segment_duration,
        )
    prompt_text = single_event.build_local_prompt(
        row,
        anchor_rows,
        args.prompt_style,
        args.prompt_strategy,
        api3d,
        max_evidence_segments=args.max_evidence_segments,
        evidence_segment_duration=args.evidence_segment_duration,
        ablate_modalities=args.ablate_modalities,
    )
    system_prompt = single_event.build_system_prompt()

    if api3d.modality_disabled(ablation_modalities, "visual"):
        images = [Image.new("RGB", (64, 64), color=(18, 18, 18))]
        prompt_text = (
            prompt_text
            + "\n\nVisual ablation note: the attached blank image is a placeholder only; "
            + "do not treat it as scene evidence."
        )
    else:
        images = load_images_for_row(row, args, api3d)

    response_text = runner.chat_video_frames(images, system_prompt, prompt_text)
    parsed_response, parse_warnings = api3d.parse_3d_response_text(response_text)
    adjusted_response, validation_warnings, resolved_object_row, response_status = api3d.validate_and_adjust_3d_response(
        parsed_response,
        anchor_rows,
        args.prompt_style,
    )
    peak_data = api3d.parse_peak_spatial(row)
    projected_u_norm, projected_v_norm, projection_valid = api3d.project_world_to_image(peak_data, resolved_object_row)

    video_path_raw = api3d.normalize_text(row.get("video_path"))
    keyframe_path_raw = api3d.normalize_text(row.get("keyframe_path"))
    output = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_index": row_index,
        "event_id": api3d.normalize_text(row.get("event_id")),
        "video_path": str(Path(video_path_raw).expanduser().resolve()) if video_path_raw else "",
        "keyframe_path": str(Path(keyframe_path_raw).expanduser().resolve()) if keyframe_path_raw else "",
        "scene_anchor_csv": str(Path(args.scene_anchor_csv).resolve()),
        "scene_anchor_candidates": anchor_rows,
        "model_name": args.model_name,
        "model_family": "InternVL",
        "dtype": args.dtype,
        "load_in_8bit": bool(args.load_in_8bit and not args.no_load_in_8bit),
        "input_mode": args.input_mode,
        "max_video_frames": args.max_video_frames,
        "max_evidence_segments": args.max_evidence_segments,
        "evidence_segment_duration": args.evidence_segment_duration,
        "sparse_timeline_evidence": sparse_timeline_evidence,
        "prompt_style": args.prompt_style,
        "prompt_strategy": args.prompt_strategy,
        "ablate_modalities": sorted(ablation_modalities),
        "prompt_text": prompt_text,
        "response_text": response_text,
        "parsed_response": parsed_response,
        "resolved_object_row": resolved_object_row,
        "resolved_object_rows": adjusted_response.get("resolved_object_rows") if isinstance(adjusted_response, dict) else [],
        "validation_warnings": parse_warnings + validation_warnings,
        "adjusted_response": adjusted_response,
        "response_status": response_status,
        "used_storyboard_fallback": False,
        "storyboard_fallback_reason": "",
        "projected_u_norm": projected_u_norm,
        "projected_v_norm": projected_v_norm,
        "projection_valid": projection_valid,
        "raw_spatial_prior_source": api3d.normalize_text(row.get("spatial_prior_source"), "none"),
        "raw_spatial_prior_u_norm": api3d.parse_float(row.get("spatial_prior_u_norm")),
        "raw_spatial_prior_v_norm": api3d.parse_float(row.get("spatial_prior_v_norm")),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.start_index < 0:
        raise SystemExit("--start_index must be >= 0")
    if args.limit is not None and args.limit < 0:
        raise SystemExit("--limit must be >= 0")

    single_event = load_single_event_module()
    api3d = single_event.load_api3d_module()

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = load_rows(input_csv)
    anchor_rows = api3d.load_scene_anchor_table(Path(args.scene_anchor_csv).resolve())
    ablation_modalities = api3d.parse_ablation_modalities(args.ablate_modalities)

    end_index = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    if args.start_index >= len(rows):
        raise SystemExit(f"start_index={args.start_index} is outside input rows ({len(rows)})")

    print(
        f"InternVL 3D inference rows {args.start_index}..{end_index - 1} "
        f"({end_index - args.start_index} rows), ablate_modalities={sorted(ablation_modalities)}",
        flush=True,
    )
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
    print("Loading InternVL model once for this scene...", flush=True)
    runner.load()
    print("Model loaded.", flush=True)

    ok_count = 0
    error_count = 0
    for row_index in range(args.start_index, end_index):
        row = rows[row_index]
        output_path = output_dir / f"row_{row_index}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] row {row_index}: {output_path}", flush=True)
            continue
        print(f"[row {row_index}] -> {output_path}", flush=True)
        try:
            run_one_row(
                row_index=row_index,
                row=row,
                args=args,
                single_event=single_event,
                api3d=api3d,
                runner=runner,
                anchor_rows=anchor_rows,
                ablation_modalities=ablation_modalities,
                output_path=output_path,
            )
            ok_count += 1
            print(f"[ok] row {row_index}", flush=True)
        except Exception as exc:
            error_count += 1
            print(f"[error] row {row_index}: {type(exc).__name__}: {exc}", flush=True)
            if not args.continue_on_error:
                raise
            error_payload = build_error_payload(row_index, row, args, api3d, anchor_rows, exc)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            runner.empty_cache()

    print(f"Finished InternVL 3D inference: ok={ok_count}, errors={error_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
