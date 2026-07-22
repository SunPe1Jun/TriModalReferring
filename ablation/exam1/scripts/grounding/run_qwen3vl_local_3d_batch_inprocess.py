#!/usr/bin/env python3
"""Run local Qwen3-VL 3D anchor selection for many rows with one model load."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class Batch3DError(Exception):
    """Raised when batch 3D inference cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Qwen3-VL 3D referent object selection for a row range with one model load."
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--scene_anchor_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--input_mode", choices=("video", "image", "auto"), default="video")
    parser.add_argument("--max_video_frames", type=int, default=16)
    parser.add_argument("--max_evidence_segments", type=int, default=4)
    parser.add_argument("--evidence_segment_duration", type=float, default=0.5)
    parser.add_argument("--ffmpeg_path")
    parser.add_argument("--ffprobe_path")
    parser.add_argument("--prompt_style", choices=("world_only", "full"), default="full")
    parser.add_argument("--prompt_strategy", choices=("standard", "mention_first"), default="standard")
    parser.add_argument("--ablate_modalities", default="")
    parser.add_argument("--offload_folder")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--strict_hand_visual",
        action="store_true",
        help="Mask projected tracked hand joints in every video frame before processor input.",
    )
    parser.add_argument(
        "--video_time_offset_manifest",
        help="Optional Exp2 manifest used to map video timestamps to telemetry sample timestamps.",
    )
    return parser.parse_args()


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise Batch3DError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_single_event_module() -> Any:
    return load_module(
        "ablation_exam1_single_event_3d",
        Path(__file__).resolve().with_name("run_qwen3vl_local_single_event_3d.py"),
    )


def load_rows(input_csv: Path) -> List[Dict[str, str]]:
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise Batch3DError(f"Input CSV has no header: {input_csv}")
        return [dict(row) for row in reader]


def move_to_device(single_event: Any, model_inputs: Mapping[str, Any], device: Optional[Any]) -> Mapping[str, Any]:
    return single_event.move_batch_to_device(model_inputs, device)


def maybe_empty_cuda_cache(runner: Any) -> None:
    runtime = getattr(runner, "runtime", None)
    torch = getattr(runtime, "torch", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and hasattr(cuda, "empty_cache"):
        try:
            cuda.empty_cache()
        except Exception:
            pass


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
        "dtype": args.dtype,
        "input_mode": args.input_mode,
        "max_video_frames": args.max_video_frames,
        "max_evidence_segments": args.max_evidence_segments,
        "evidence_segment_duration": args.evidence_segment_duration,
        "prompt_style": args.prompt_style,
        "prompt_strategy": args.prompt_strategy,
        "ablate_modalities": sorted(api3d.parse_ablation_modalities(args.ablate_modalities)),
        "strict_hand_visual": bool(args.strict_hand_visual),
        "video_time_offset_manifest": args.video_time_offset_manifest or "",
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


def run_one_row(
    row_index: int,
    row: Dict[str, str],
    args: argparse.Namespace,
    single_event: Any,
    api3d: Any,
    local_grounding: Any,
    runner: Any,
    anchor_rows: List[Dict[str, Any]],
    ablation_modalities: Sequence[str],
    output_path: Path,
    hand_masking: Optional[Any] = None,
    video_offset_map: Optional[Mapping[str, float]] = None,
) -> None:
    video_path_raw = api3d.normalize_text(row.get("video_path"))
    video_path = Path(video_path_raw).expanduser().resolve() if video_path_raw else None
    keyframe_path_raw = api3d.normalize_text(row.get("keyframe_path"))
    keyframe_path = Path(keyframe_path_raw).expanduser().resolve() if keyframe_path_raw else Path()
    if video_path is None and not keyframe_path_raw:
        raise Batch3DError("Input row must contain at least video_path or keyframe_path.")

    hand_mask_context: Optional[Dict[str, Any]] = None
    if args.strict_hand_visual:
        if hand_masking is None or video_path is None:
            raise Batch3DError("--strict_hand_visual requires a video and hand masking module.")
        json_path_raw = api3d.normalize_text(row.get("json_path"))
        if not json_path_raw:
            raise Batch3DError("--strict_hand_visual requires json_path in the input row.")
        timed_samples = hand_masking.collect_timed_samples(
            hand_masking.load_multimodal_samples(Path(json_path_raw))
        )
        video_offset, offset_source = single_event.resolve_video_time_offset(
            video_path, video_offset_map or {}
        )
        hand_mask_context = {
            "module": hand_masking,
            "timed_samples": timed_samples,
            "video_time_offset": video_offset,
            "video_time_offset_source": offset_source,
        }

    if api3d.modality_disabled(ablation_modalities, "gaze", "timeline", "structured_geometry"):
        sparse_timeline_evidence: List[Dict[str, Any]] = []
    else:
        sparse_timeline_evidence = api3d.build_sparse_timeline_evidence(
            row=row,
            anchor_rows=anchor_rows,
            max_segments=args.max_evidence_segments,
            segment_duration=args.evidence_segment_duration,
        )
    evidence_times = [float(item["representative_time"]) for item in sparse_timeline_evidence if "representative_time" in item]
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

    sample = SimpleNamespace(
        event_id=api3d.normalize_text(row.get("event_id")),
        keyframe_path=keyframe_path,
        video_path=video_path,
        t_start=api3d.normalize_text(row.get("t_start")),
        t_end=api3d.normalize_text(row.get("t_end")),
    )
    media_mode = single_event.resolve_media_mode(row, args.input_mode)
    hand_mask_audit: List[Dict[str, Any]] = []
    if api3d.modality_disabled(ablation_modalities, "visual"):
        media_mode = "image"
        placeholder = runner.runtime.image_module.new("RGB", (64, 64), color=(18, 18, 18))
        placeholder_prompt = (
            prompt_text
            + "\n\nVisual ablation note: the attached blank image is a placeholder only; "
            + "do not treat it as scene evidence."
        )
        model_inputs = single_event._build_processor_image_input(
            runner.processor,
            runner.runtime,
            system_prompt,
            placeholder_prompt,
            placeholder,
        )
    else:
        if args.strict_hand_visual:
            model_inputs, hand_mask_audit = single_event.build_strict_hand_video_model_inputs(
                processor=runner.processor,
                runtime=runner.runtime,
                sample=sample,
                system_prompt=system_prompt,
                prompt_text=prompt_text,
                local_grounding=local_grounding,
                hand_mask_context=hand_mask_context or {},
                ffmpeg_path=args.ffmpeg_path,
                ffprobe_path=args.ffprobe_path,
                max_video_frames=args.max_video_frames,
            )
        else:
            model_inputs = local_grounding.build_model_inputs(
                processor=runner.processor,
                runtime=runner.runtime,
                sample=sample,
                system_prompt=system_prompt,
                prompt_text=prompt_text,
                media_mode=media_mode,
                ffmpeg_path=args.ffmpeg_path,
                ffprobe_path=args.ffprobe_path,
                max_video_frames=args.max_video_frames,
            )

    used_storyboard_fallback = False
    fallback_reason = ""
    try:
        model_inputs = move_to_device(single_event, model_inputs, runner.device)
        with runner.runtime.torch.inference_mode():
            generated = runner.model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    except Exception as exc:
        if media_mode != "video" or not single_event.should_retry_with_storyboard(exc):
            raise
        used_storyboard_fallback = True
        fallback_reason = str(exc)
        storyboard_inputs = single_event.build_storyboard_model_inputs(
            processor=runner.processor,
            runtime=runner.runtime,
            sample=sample,
            system_prompt=system_prompt,
            prompt_text=prompt_text,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            max_video_frames=args.max_video_frames,
            evidence_times=evidence_times,
            hand_mask_context=hand_mask_context,
        )
        model_inputs = move_to_device(single_event, storyboard_inputs, runner.device)
        with runner.runtime.torch.inference_mode():
            generated = runner.model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

    generated_ids = single_event.trim_generated_ids(model_inputs, generated)
    response_text = single_event.decode_response(runner.processor, generated_ids).strip()

    parsed_response, parse_warnings = api3d.parse_3d_response_text(response_text)
    adjusted_response, validation_warnings, resolved_object_row, response_status = api3d.validate_and_adjust_3d_response(
        parsed_response,
        anchor_rows,
        args.prompt_style,
    )
    peak_data = api3d.parse_peak_spatial(row)
    projected_u_norm, projected_v_norm, projection_valid = api3d.project_world_to_image(peak_data, resolved_object_row)

    output = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_index": row_index,
        "event_id": api3d.normalize_text(row.get("event_id")),
        "video_path": str(video_path.resolve()) if video_path else "",
        "keyframe_path": str(keyframe_path.resolve()) if keyframe_path_raw else "",
        "scene_anchor_csv": str(Path(args.scene_anchor_csv).resolve()),
        "scene_anchor_candidates": anchor_rows,
        "model_name": args.model_name,
        "dtype": args.dtype,
        "input_mode": args.input_mode,
        "max_video_frames": args.max_video_frames,
        "max_evidence_segments": args.max_evidence_segments,
        "evidence_segment_duration": args.evidence_segment_duration,
        "sparse_timeline_evidence": sparse_timeline_evidence,
        "prompt_style": args.prompt_style,
        "prompt_strategy": args.prompt_strategy,
        "ablate_modalities": sorted(ablation_modalities),
        "strict_hand_visual": bool(args.strict_hand_visual),
        "video_time_offset": hand_mask_context.get("video_time_offset", 0.0) if hand_mask_context else 0.0,
        "video_time_offset_source": hand_mask_context.get("video_time_offset_source", "") if hand_mask_context else "",
        "hand_mask_audit": hand_mask_audit if args.strict_hand_visual else [],
        "prompt_text": prompt_text,
        "response_text": response_text,
        "parsed_response": parsed_response,
        "resolved_object_row": resolved_object_row,
        "resolved_object_rows": adjusted_response.get("resolved_object_rows") if isinstance(adjusted_response, dict) else [],
        "validation_warnings": parse_warnings + validation_warnings,
        "adjusted_response": adjusted_response,
        "response_status": response_status,
        "used_storyboard_fallback": used_storyboard_fallback,
        "storyboard_fallback_reason": fallback_reason,
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
    local_grounding = single_event.load_local_grounding_module()
    hand_masking = single_event.load_hand_masking_module() if args.strict_hand_visual else None
    single_event.local_grounding_module_ref["module"] = local_grounding

    input_csv = Path(args.input_csv).resolve()
    output_dir = Path(args.output_dir).resolve()
    rows = load_rows(input_csv)
    anchor_rows = api3d.load_scene_anchor_table(Path(args.scene_anchor_csv).resolve())
    ablation_modalities = api3d.parse_ablation_modalities(args.ablate_modalities)
    video_offset_map = (
        single_event.load_video_offset_map(args.video_time_offset_manifest)
        if args.strict_hand_visual
        else {}
    )

    end_index = len(rows) if args.limit is None else min(len(rows), args.start_index + args.limit)
    if args.start_index >= len(rows):
        raise SystemExit(f"start_index={args.start_index} is outside input rows ({len(rows)})")

    print(
        f"Batch local 3D inference rows {args.start_index}..{end_index - 1} "
        f"({end_index - args.start_index} rows), ablate_modalities={sorted(ablation_modalities)}",
        flush=True,
    )
    print("Loading model once for this scene/variant...", flush=True)
    runner = local_grounding.LocalQwen3VLRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
        input_mode=args.input_mode,
        max_video_frames=args.max_video_frames,
        ffmpeg_path=args.ffmpeg_path,
        ffprobe_path=args.ffprobe_path,
        prompt_variant="debug",
        offload_folder=args.offload_folder,
    )
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
                local_grounding=local_grounding,
                runner=runner,
                anchor_rows=anchor_rows,
                ablation_modalities=ablation_modalities,
                output_path=output_path,
                hand_masking=hand_masking,
                video_offset_map=video_offset_map,
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
            maybe_empty_cuda_cache(runner)

    print(f"Finished batch local 3D inference: ok={ok_count}, errors={error_count}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
