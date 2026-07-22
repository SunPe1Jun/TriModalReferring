#!/usr/bin/env python3
"""Run local Qwen3-VL for single-event 3D referent object selection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ALLOWED_PRIMARY_SOURCES = {"gazePoint", "visual_only", "language", "none"}


class Local3DError(Exception):
    """Raised when local 3D single-event inference fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen3-VL on one CSV row and select a 3D referent object from scene anchors.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--row_index", type=int, default=0, help="Zero-based row index to test. Default: 0.")
    parser.add_argument("--output_json", required=True, help="Path to save the local model response JSON.")
    parser.add_argument("--scene_anchor_csv", required=True, help="Path to the scene anchor candidate table.")
    parser.add_argument("--model_name", required=True, help="Local Qwen3-VL model path, e.g. /workspace/usr3/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first, then fall back automatically.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model and processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens for generation.")
    parser.add_argument("--input_mode", choices=("video", "image", "auto"), default="video", help="Default: video")
    parser.add_argument("--max_video_frames", type=int, default=16, help="Maximum number of sampled frames when input_mode uses video.")
    parser.add_argument("--max_evidence_segments", type=int, default=4, help="Maximum sparse timeline evidence segments to include. Default: 4.")
    parser.add_argument("--evidence_segment_duration", type=float, default=0.5, help="Duration in seconds for sparse evidence segments. Default: 0.5.")
    parser.add_argument("--ffmpeg_path", help="Optional path to ffmpeg executable.")
    parser.add_argument("--ffprobe_path", help="Optional path to ffprobe executable.")
    parser.add_argument("--prompt_style", choices=("world_only", "full"), default="full", help="Prompt style. Default: full")
    parser.add_argument("--prompt_strategy", choices=("standard", "mention_first"), default="standard", help="Prompt strategy. Default: standard")
    parser.add_argument(
        "--ablate_modalities",
        default="",
        help=(
            "Comma-separated modalities to hide: visual,gaze,hand,structured_geometry,timeline. "
            "Default: none."
        ),
    )
    parser.add_argument("--offload_folder", help="Optional directory for Accelerate disk offload when loading very large models or MoE checkpoints.")
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
        raise Local3DError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_api3d_module() -> Any:
    project_root = Path(__file__).resolve().parents[2]
    return load_module("api3d_module", project_root / "scripts" / "grounding" / "run_qwen3vl_plus_api_single_event_3d.py")


def load_local_grounding_module() -> Any:
    project_root = Path(__file__).resolve().parents[2]
    return load_module("local_grounding_module", project_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py")


def load_hand_masking_module() -> Any:
    repo_root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(repo_root / "exam3_point_grounding"))
    return load_module("strict_hand_masking_module", repo_root / "exam3_point_grounding" / "hand_masking.py")


def decode_response(processor: Any, generated_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0] if decoded else ""
    if hasattr(processor, "decode"):
        if isinstance(generated_ids, (list, tuple)) and generated_ids and isinstance(generated_ids[0], (list, tuple)):
            return processor.decode(generated_ids[0], skip_special_tokens=True)
        return processor.decode(generated_ids, skip_special_tokens=True)
    raise Local3DError("Processor does not support decode or batch_decode.")


def trim_generated_ids(model_inputs: Mapping[str, Any], generated_ids: Any) -> Any:
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        return generated_ids
    prompt_length = input_ids.shape[-1]
    return generated_ids[:, prompt_length:]


def resolve_media_mode(row: Mapping[str, str], requested_mode: str) -> str:
    if requested_mode == "auto":
        return "video" if str(row.get("video_path", "")).strip() else "image"
    return requested_mode


def build_local_prompt(
    row: Dict[str, str],
    anchor_rows: List[Dict[str, Any]],
    prompt_style: str,
    prompt_strategy: str,
    api3d: Any,
    max_evidence_segments: int,
    evidence_segment_duration: float,
    ablate_modalities: str,
) -> str:
    return api3d.build_3d_object_prompt(
        row,
        anchor_rows,
        prompt_style,
        prompt_strategy=prompt_strategy,
        max_evidence_segments=max_evidence_segments,
        evidence_segment_duration=evidence_segment_duration,
        ablate_modalities=ablate_modalities,
    )


def build_system_prompt() -> str:
    return (
        "You are a precise multimodal 3D referent selection model for egocentric VR interaction. "
        "You must use the uploaded visual evidence together with language and structured world cues to choose all valid candidate object labels from the provided list. "
        "Do not invent object names. Return strict JSON only."
    )


def move_batch_to_device(batch: Mapping[str, Any], device: Optional[Any]) -> Mapping[str, Any]:
    if device is None:
        return batch
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def should_retry_with_storyboard(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc!r} {exc}"
    return any(
        marker in text
        for marker in (
            "StopIteration",
            "video metadata was provided",
            "video metadata",
            "get_rope_index",
            "grid_thw",
        )
    )


def build_storyboard_image(frames: Sequence[Any], runtime: Any) -> Any:
    if not frames:
        raise Local3DError("Cannot build storyboard image because no frames were provided.")

    image_module = runtime.image_module
    draw_module = runtime.image_draw_module
    font_module = runtime.image_font_module

    max_panels = min(len(frames), 9)
    selected_frames = list(frames[:max_panels])
    panel_width = 448
    panel_height = 448
    columns = 3 if max_panels > 4 else 2 if max_panels > 1 else 1
    rows = int(math.ceil(max_panels / columns))
    gutter = 12
    label_band = 32

    canvas_width = columns * panel_width + (columns + 1) * gutter
    canvas_height = rows * (panel_height + label_band) + (rows + 1) * gutter
    canvas = image_module.new("RGB", (canvas_width, canvas_height), color=(18, 18, 18))
    draw = draw_module.Draw(canvas)
    font = font_module.load_default()

    for index, frame in enumerate(selected_frames):
        row = index // columns
        col = index % columns
        x0 = gutter + col * (panel_width + gutter)
        y0 = gutter + row * (panel_height + label_band + gutter)
        resized = frame.copy()
        resized.thumbnail((panel_width, panel_height))

        paste_x = x0 + (panel_width - resized.size[0]) // 2
        paste_y = y0 + (panel_height - resized.size[1]) // 2
        canvas.paste(resized, (paste_x, paste_y))
        draw.rectangle((x0, y0, x0 + panel_width, y0 + panel_height), outline=(120, 120, 120), width=2)
        label = f"P{index + 1}"
        draw.text((x0 + 8, y0 + panel_height + 6), label, fill=(255, 255, 255), font=font)

    return canvas


def extract_video_frames_at_times(
    video_path: Path,
    runtime: Any,
    ffmpeg_path: Optional[str],
    times: Sequence[float],
) -> List[Any]:
    if not times:
        return []
    ffmpeg_binary = local_grounding_module_ref["module"].resolve_binary_path(ffmpeg_path, "ffmpeg")
    frames: List[Any] = []
    with tempfile.TemporaryDirectory(prefix="qwen3vl_sparse_evidence_") as temp_dir:
        for index, timestamp in enumerate(times, start=1):
            if timestamp < 0:
                continue
            output_path = Path(temp_dir) / f"evidence_{index:03d}.jpg"
            command = [
                ffmpeg_binary,
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                str(output_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0 or not output_path.exists():
                continue
            image = runtime.image_module.open(output_path).convert("RGB")
            image.load()
            frames.append(image.copy())
    return frames


def build_storyboard_prompt(prompt_text: str, evidence_times: Optional[Sequence[float]] = None) -> str:
    if evidence_times:
        time_text = ", ".join(f"P{index + 1}≈{timestamp:.2f}s" for index, timestamp in enumerate(evidence_times))
        extra = (
            "\n\nLocal fallback note:\n"
            "The original event video was converted into a sparse evidence storyboard. "
            "Panels P1, P2, P3, ... correspond to the selected evidence times from the timeline evidence table: "
            f"{time_text}.\n"
            "Treat these panels as proposed evidence, not mandatory referents. Ignore panels that are not linked to the instruction language."
        )
    else:
        extra = (
            "\n\nLocal fallback note:\n"
            "The original event video was converted into a storyboard image made of chronologically ordered panels "
            "P1, P2, P3, ... from left to right and top to bottom.\n"
            "Treat later panels as later times in the same event.\n"
            "Base your object selection on the panel where the green gaze marker most clearly supports the intended referent."
        )
    return prompt_text + extra


def build_storyboard_model_inputs(
    processor: Any,
    runtime: Any,
    sample: Any,
    system_prompt: str,
    prompt_text: str,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
    evidence_times: Optional[Sequence[float]] = None,
    hand_mask_context: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    frames = extract_video_frames_at_times(
        video_path=sample.video_path or Path(),
        runtime=runtime,
        ffmpeg_path=ffmpeg_path,
        times=evidence_times or [],
    )
    if not frames:
        frames = local_grounding_module_ref["module"].load_video_frames(
            video_path=sample.video_path or Path(),
            runtime=runtime,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            max_video_frames=max_video_frames,
            t_start=local_grounding_module_ref["module"].parse_time_value(sample.t_start, "t_start"),
            t_end=local_grounding_module_ref["module"].parse_time_value(sample.t_end, "t_end"),
        )
    if hand_mask_context:
        frames, _ = mask_frame_sequence(
            frames,
            source_times=list(evidence_times or []),
            sample=sample,
            hand_mask_context=hand_mask_context,
        )
    storyboard = build_storyboard_image(frames, runtime)
    storyboard_prompt = build_storyboard_prompt(prompt_text, evidence_times=evidence_times)
    return _build_processor_image_input(processor, runtime, system_prompt, storyboard_prompt, storyboard)


def _build_processor_image_input(processor: Any, runtime: Any, system_prompt: str, prompt_text: str, image: Any) -> Mapping[str, Any]:
    messages = local_grounding_module_ref["module"].build_messages(system_prompt, prompt_text, "image")
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[chat_text], images=[image], return_tensors="pt")
    return processor(text=[system_prompt + "\n\n" + prompt_text], images=[image], return_tensors="pt")


local_grounding_module_ref: Dict[str, Any] = {}


def load_video_offset_map(manifest_path: Optional[str]) -> Dict[str, float]:
    if not manifest_path:
        return {}
    import csv

    path = Path(manifest_path).expanduser().resolve()
    if not path.exists():
        raise Local3DError(f"Missing video-time offset manifest: {path}")
    offsets: Dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            video = str(row.get("video_path") or "").strip()
            raw_offset = str(row.get("video_time_offset_seconds") or "").strip()
            if not video or not raw_offset:
                continue
            try:
                offset = float(raw_offset)
            except ValueError:
                continue
            if math.isfinite(offset):
                offsets.setdefault(str(Path(video).expanduser().resolve()), offset)
    return offsets


def resolve_video_time_offset(video_path: Optional[Path], offset_map: Mapping[str, float]) -> Tuple[float, str]:
    if video_path is None:
        return 0.0, "none:no_video"
    key = str(video_path.expanduser().resolve())
    if key in offset_map:
        return float(offset_map[key]), "exp2_manifest"
    return 0.0, "default_zero:no_manifest_match"


def video_frame_times(
    video_path: Path,
    local_grounding: Any,
    ffprobe_path: Optional[str],
    max_video_frames: int,
    t_start: Optional[float],
    t_end: Optional[float],
    frame_count: int,
) -> List[float]:
    duration = local_grounding.probe_video_duration(video_path, ffprobe_path)
    clip_start = t_start if t_start is not None else 0.0
    clip_end = t_end
    if clip_end is None and duration is not None:
        clip_end = duration
    clip_duration = None if clip_end is None else max(clip_end - clip_start, 1e-3)
    fps = min(8.0, max(1.0, max_video_frames / clip_duration)) if clip_duration is not None else 1.0
    return [clip_start + index / fps for index in range(frame_count)]


def mask_frame_sequence(
    frames: Sequence[Any],
    source_times: Sequence[float],
    sample: Any,
    hand_mask_context: Mapping[str, Any],
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    hand_masking = hand_mask_context["module"]
    timed_samples = hand_mask_context["timed_samples"]
    video_offset = float(hand_mask_context.get("video_time_offset", 0.0))
    masked_frames: List[Any] = []
    audits: List[Dict[str, Any]] = []
    for index, frame in enumerate(frames):
        video_time = float(source_times[index]) if index < len(source_times) else float(source_times[-1] if source_times else 0.0)
        target_sample_time = video_time - video_offset
        nearest_time, telemetry_sample = hand_masking.nearest_sample(timed_samples, target_sample_time)
        masked, audit = hand_masking.mask_pil_image(frame, telemetry_sample)
        audit.update({
            "frame_index": index,
            "video_time": video_time,
            "target_sample_time": target_sample_time,
            "nearest_sample_time": nearest_time,
            "sample_time_delta": nearest_time - target_sample_time,
        })
        masked_frames.append(masked)
        audits.append(audit)
    return masked_frames, audits


def build_strict_hand_video_model_inputs(
    processor: Any,
    runtime: Any,
    sample: Any,
    system_prompt: str,
    prompt_text: str,
    local_grounding: Any,
    hand_mask_context: Mapping[str, Any],
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
) -> Tuple[Mapping[str, Any], List[Dict[str, Any]]]:
    frames = local_grounding.load_video_frames(
        video_path=sample.video_path or Path(),
        runtime=runtime,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        max_video_frames=max_video_frames,
        t_start=local_grounding.parse_time_value(sample.t_start, "t_start"),
        t_end=local_grounding.parse_time_value(sample.t_end, "t_end"),
    )
    source_times = video_frame_times(
        sample.video_path or Path(),
        local_grounding,
        ffprobe_path,
        max_video_frames,
        local_grounding.parse_time_value(sample.t_start, "t_start"),
        local_grounding.parse_time_value(sample.t_end, "t_end"),
        len(frames),
    )
    masked_frames, audits = mask_frame_sequence(frames, source_times, sample, hand_mask_context)
    messages = local_grounding.build_messages(system_prompt, prompt_text, "video")
    if hasattr(processor, "apply_chat_template"):
        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return processor(text=[chat_text], videos=[masked_frames], return_tensors="pt"), audits
    return processor(text=[system_prompt + "\n\n" + prompt_text], videos=[masked_frames], return_tensors="pt"), audits


def validate_and_adjust_local_3d_response(parsed_response: Optional[Dict[str, Any]], anchor_rows: List[Dict[str, Any]], prompt_style: str, api3d: Any) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], str]:
    return api3d.validate_and_adjust_3d_response(parsed_response, anchor_rows, prompt_style)


def main() -> None:
    args = parse_args()
    api3d = load_api3d_module()
    local_grounding = load_local_grounding_module()
    hand_masking = load_hand_masking_module() if args.strict_hand_visual else None
    local_grounding_module_ref["module"] = local_grounding

    row = api3d.read_row(Path(args.input_csv), args.row_index)
    anchor_rows = api3d.load_scene_anchor_table(Path(args.scene_anchor_csv))
    ablation_modalities = api3d.parse_ablation_modalities(args.ablate_modalities)
    video_path = Path(args.video_path).expanduser().resolve() if getattr(args, "video_path", None) else None
    if video_path is None:
        video_path_raw = api3d.normalize_text(row.get("video_path"))
        video_path = Path(video_path_raw).expanduser().resolve() if video_path_raw else None

    keyframe_path_raw = api3d.normalize_text(row.get("keyframe_path"))
    keyframe_path = Path(keyframe_path_raw).expanduser().resolve() if keyframe_path_raw else Path()
    if video_path is None and not keyframe_path_raw:
        raise SystemExit("Input row must contain at least video_path or keyframe_path.")

    hand_mask_context: Optional[Dict[str, Any]] = None
    if args.strict_hand_visual:
        if video_path is None:
            raise SystemExit("--strict_hand_visual requires video input.")
        timed_samples = hand_masking.collect_timed_samples(hand_masking.load_multimodal_samples(Path(row["json_path"])))
        offset_map = load_video_offset_map(args.video_time_offset_manifest)
        video_offset, offset_source = resolve_video_time_offset(video_path, offset_map)
        hand_mask_context = {
            "module": hand_masking,
            "timed_samples": timed_samples,
            "video_time_offset": video_offset,
            "video_time_offset_source": offset_source,
        }

    if api3d.modality_disabled(ablation_modalities, "gaze", "timeline", "structured_geometry"):
        sparse_timeline_evidence = []
    else:
        sparse_timeline_evidence = api3d.build_sparse_timeline_evidence(
            row=row,
            anchor_rows=anchor_rows,
            max_segments=args.max_evidence_segments,
            segment_duration=args.evidence_segment_duration,
        )
    evidence_times = [float(item["representative_time"]) for item in sparse_timeline_evidence if "representative_time" in item]
    prompt_text = build_local_prompt(
        row,
        anchor_rows,
        args.prompt_style,
        args.prompt_strategy,
        api3d,
        max_evidence_segments=args.max_evidence_segments,
        evidence_segment_duration=args.evidence_segment_duration,
        ablate_modalities=args.ablate_modalities,
    )
    system_prompt = build_system_prompt()

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

    sample = SimpleNamespace(
        event_id=api3d.normalize_text(row.get("event_id")),
        keyframe_path=keyframe_path,
        video_path=video_path,
        t_start=api3d.normalize_text(row.get("t_start")),
        t_end=api3d.normalize_text(row.get("t_end")),
    )
    media_mode = resolve_media_mode(row, args.input_mode)
    hand_mask_audit: List[Dict[str, Any]] = []
    if api3d.modality_disabled(ablation_modalities, "visual"):
        media_mode = "image"
        placeholder = runner.runtime.image_module.new("RGB", (64, 64), color=(18, 18, 18))
        placeholder_prompt = (
            prompt_text
            + "\n\nVisual ablation note: the attached blank image is a placeholder only; "
            + "do not treat it as scene evidence."
        )
        model_inputs = _build_processor_image_input(
            runner.processor,
            runner.runtime,
            system_prompt,
            placeholder_prompt,
            placeholder,
        )
    else:
        if args.strict_hand_visual:
            model_inputs, hand_mask_audit = build_strict_hand_video_model_inputs(
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
        model_inputs = move_batch_to_device(model_inputs, runner.device)
        with runner.runtime.torch.inference_mode():
            generated = runner.model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    except Exception as exc:
        if media_mode != "video" or not should_retry_with_storyboard(exc):
            raise
        used_storyboard_fallback = True
        fallback_reason = str(exc)
        storyboard_inputs = build_storyboard_model_inputs(
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
        model_inputs = move_batch_to_device(storyboard_inputs, runner.device)
        with runner.runtime.torch.inference_mode():
            generated = runner.model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    generated_ids = trim_generated_ids(model_inputs, generated)
    response_text = decode_response(runner.processor, generated_ids).strip()

    parsed_response, parse_warnings = api3d.parse_3d_response_text(response_text)
    adjusted_response, validation_warnings, resolved_object_row, response_status = validate_and_adjust_local_3d_response(
        parsed_response,
        anchor_rows,
        args.prompt_style,
        api3d,
    )

    peak_data = api3d.parse_peak_spatial(row)
    projected_u_norm, projected_v_norm, projection_valid = api3d.project_world_to_image(peak_data, resolved_object_row)

    output = {
        "input_csv": str(Path(args.input_csv).resolve()),
        "row_index": args.row_index,
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
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved local 3D response to: {output_path}")


if __name__ == "__main__":
    main()
