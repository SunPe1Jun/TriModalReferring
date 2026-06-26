#!/usr/bin/env python3
"""Two-stage Qwen3-VL grounding: first select peak time, then ground with peak-aligned spatial context."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class TwoStageError(Exception):
    pass


@dataclass
class TwoStageSample:
    event_id: str
    video_path: Path
    json_path: Path
    t_start: float
    t_end: float
    instruction_text: str
    utterance_text: str
    target_description: str
    gaze_summary: str
    hand_summary: str
    keyframe_path: Optional[Path]
    preset_peak_time_seconds: str = ""


@dataclass
class PeakResult:
    peak_time_seconds: str
    confidence: str
    reasoning_summary: str
    parse_ok: str
    raw_output: str
    error_message: str
    panel_id: str = ""
    candidate_panel_ids: Tuple[str, ...] = ()


OUTPUT_COLUMNS = (
    "event_id",
    "predicted_peak_time_seconds",
    "predicted_peak_confidence",
    "peak_reasoning_summary",
    "peak_parse_ok",
    "peak_raw_output",
    "peak_error_message",
    "selected_keyframe_path",
    "prompt_text",
    "model_raw_output",
    "parsed_json",
    "referent_type",
    "u_norm",
    "v_norm",
    "x_world",
    "y_world",
    "z_world",
    "referent_text",
    "reasoning_summary",
    "confidence",
    "parse_ok",
    "error_message",
    "spatial_prior_u_norm",
    "spatial_prior_v_norm",
    "spatial_prior_source",
)

DEFAULT_SUMMARY_MAX_POINTS = 5
STAGE1_OUTPUT_COLUMNS = (
    "event_id",
    "video_path",
    "json_path",
    "t_start",
    "t_end",
    "predicted_peak_time_seconds",
    "predicted_peak_confidence",
    "peak_parse_ok",
    "selected_keyframe_path",
    "instruction_text",
    "utterance_text",
    "target_description",
    "gaze_summary",
    "hand_summary",
    "event_json_path",
    "spatial_context_text",
    "spatial_context_json",
    "spatial_prior_u_norm",
    "spatial_prior_v_norm",
    "spatial_prior_source",
)

PEAK_SYSTEM_PROMPT = (
    "You are a storyboard panel selector for event-level multimodal referent grounding. "
    "Your job is to choose exactly one visible storyboard panel that will give the best chance of correct final grounding in stage 2. "
    "Do not estimate a free-form timestamp first. First choose the best visible panel among the labeled candidates such as P1, P2, P3, and so on. Then copy that panel's displayed timestamp. "
    "The intended referent may be entity, spatial, or none if not reliably inferable, but your job here is still to choose the panel that best supports the final grounding decision. "
    "The best panel is the one where the intended referent becomes most visually resolvable and most consistent with the language, gaze, hand, and scene context. "
    "Do not choose a panel only because motion is large. Do not choose a panel only because an object is visually salient. Do not default to the temporal middle. "
    "Return strict JSON only."
)

PEAK_USER_PROMPT_TEMPLATE = """Event ID: {event_id}

Stage-1 objective:
Select the storyboard panel or small set of panels that best support final referent grounding.

Important task interpretation:
- The best grounding panel is not necessarily the final action-completion frame.
- The correct panel is the moment when the intended referent becomes most identifiable and most grounded by the combined evidence.
- In many events, this disambiguation happens in the middle portion of the video.
- Some events may contain more than one plausible peak moment.

What the visual input shows:
- The visual input is a storyboard composed of sampled frames from the event clip.
- Each panel has a visible ID such as P1, P2, P3, ...
- Each panel also shows its timestamp in seconds.

Possible referent types:
- entity: a concrete visible object or object part
- spatial: a point, placement area, contact location, destination, or local region in space
- none: no reliable referent can be inferred

Language inputs:
instruction_text:
{instruction_text}

utterance_text:
{utterance_text}

target_description:
{target_description}

Available event summaries:
gaze_summary:
{gaze_summary}

hand_summary:
{hand_summary}

Required decision procedure:
Step 1. Infer the likely referent type.
- Decide whether the event is mainly about an entity, a spatial target, or no reliable referent.

Step 2. Compare the visible storyboard panels.
- Look for the panel where the referent is most disambiguated.
- Prefer the panel where language, gaze, hand, and scene layout align most clearly.
- Do not assume that the final panel is best.
- Do not assume that the action-completion frame is best.
- Do not assume that the visually calmest frame is best.

Step 3. Allow multiple plausible peaks.
- If several nearby panels are all plausible, choose one primary panel and optionally list a small set of candidate panels.
- Candidate panels should be visually and temporally plausible alternatives, not arbitrary distant frames.

Step 4. Avoid weak selection behavior.
- Do not default to the final segment of the video.
- Do not choose a panel only because motion is large.
- Do not choose a panel only because an object is most salient.
- Do not output a timestamp that does not correspond to a visible panel.

Output requirements:
- primary_panel_id must be one visible panel label.
- candidate_panel_ids should contain 1 to 3 visible panel labels, including the primary panel.
- primary_panel_timestamp_seconds must match the timestamp shown on the selected primary panel.
- reasoning_summary should be brief and concrete.
- Output strict JSON only.

Required JSON schema:
{{
  "primary_panel_id": "P6",
  "candidate_panel_ids": ["P5", "P6", "P7"],
  "primary_panel_timestamp_seconds": 0.0,
  "confidence": 0.0,
  "reasoning_summary": "brief explanation"
}}
"""



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run two-stage Qwen3-VL grounding with model-selected peak time.")
    parser.add_argument("--input_csv", required=True, help="CSV with event_id, video_path, json_path, t_start, t_end and optional language fields.")
    parser.add_argument("--output_csv", required=True, help="Output CSV path.")
    parser.add_argument("--stage1_output_csv", help="Optional CSV path for saving the stage-1 selected intermediate input used by stage 2.")
    parser.add_argument("--event_json_dir", help="Optional directory to save peak-aligned compact event JSON files.")
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", help="Local model directory or Hugging Face model name.")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first, then fall back automatically.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model and processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens for grounding stage.")
    parser.add_argument("--peak_max_new_tokens", type=int, default=512, help="Maximum new tokens for peak selection stage.")
    parser.add_argument("--input_mode", choices=("video",), default="video", help="Two-stage script always uses video for peak selection.")
    parser.add_argument("--max_video_frames", type=int, default=16, help="Maximum sampled frames for the event clip.")
    parser.add_argument("--ffmpeg_path", help="Optional path to ffmpeg executable.")
    parser.add_argument("--ffprobe_path", help="Optional path to ffprobe executable.")
    parser.add_argument("--peak_window_seconds", type=float, default=0.25, help="Half window size around predicted peak used for compact event JSON.")
    parser.add_argument("--peak_fallback", choices=("midpoint", "t_start", "t_end", "disable"), default="midpoint", help="Fallback peak choice when stage-1 peak parsing fails. Default: midpoint.")
    parser.add_argument("--stage", choices=("full", "stage1_only", "stage2_only"), default="full", help="Run full two-stage flow, only stage 1, or only stage 2 using a stage-1 selected CSV.")
    parser.add_argument("--prior_source_order", default="gazePoint,cameraHitPoint", help="Comma-separated source priority for spatial prior selection.")
    parser.add_argument("--prompt_variant", choices=("debug", "minimal"), default="debug", help="Grounding-stage prompt variant.")
    parser.add_argument("--offload_folder", help="Optional directory for Accelerate disk offload when loading very large models or MoE checkpoints.")
    parser.add_argument("--vis_dir", help="Optional directory for saving stage-2 keyframe overlays with predicted 2D points.")
    parser.add_argument("--stage1_vis_dir", help="Optional directory for saving stage-1 selected frames and storyboards.")
    parser.add_argument("--summary_max_points", type=int, default=DEFAULT_SUMMARY_MAX_POINTS, help="Maximum representative gaze points when rebuilding summaries from JSON.")
    parser.add_argument("--uv_decision_mode", choices=("model", "prefer_gaze_prior", "gaze_prior_only"), default="gaze_prior_only", help="How to determine final u,v in stage 2. Default forces gazePoint prior when available.")
    parser.add_argument("--preferred_panel_start", type=int, default=5, help="Preferred panel range start for stage-1 panel biasing.")
    parser.add_argument("--preferred_panel_end", type=int, default=10, help="Preferred panel range end for stage-1 panel biasing.")
    parser.add_argument("--preferred_panel_bonus", type=float, default=2.0, help="Extra score bonus for panels inside the preferred stage-1 range.")
    parser.add_argument("--continue_on_error", action="store_true", help="Continue processing later rows when one sample fails.")
    return parser.parse_args()


def load_module(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise TwoStageError(f"Failed to load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_float(raw_value: str, label: str, row_index: int) -> float:
    try:
        return float(raw_value)
    except ValueError as exc:
        raise TwoStageError(f"Row {row_index} has invalid {label}: {raw_value}") from exc


def normalize_text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def read_samples(input_csv: Path) -> List[TwoStageSample]:
    if not input_csv.exists() or not input_csv.is_file():
        raise TwoStageError(f"Input CSV does not exist or is not a file: {input_csv}")
    with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise TwoStageError(f"Input CSV has no header row: {input_csv}")
        required = ("event_id", "video_path", "json_path", "t_start", "t_end")
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise TwoStageError("Input CSV is missing required columns: " + ", ".join(missing))
        rows: List[TwoStageSample] = []
        for row_index, row in enumerate(reader, start=1):
            if not any(row.values()):
                continue
            rows.append(
                TwoStageSample(
                    event_id=normalize_text(row.get("event_id")),
                    video_path=Path(normalize_text(row.get("video_path"))).expanduser().resolve(),
                    json_path=Path(normalize_text(row.get("json_path"))).expanduser().resolve(),
                    t_start=parse_float(normalize_text(row.get("t_start")), "t_start", row_index),
                    t_end=parse_float(normalize_text(row.get("t_end")), "t_end", row_index),
                    instruction_text=normalize_text(row.get("instruction_text")),
                    utterance_text=normalize_text(row.get("utterance_text")),
                    target_description=normalize_text(row.get("target_description")),
                    gaze_summary=normalize_text(row.get("gaze_summary")),
                    hand_summary=normalize_text(row.get("hand_summary")),
                    keyframe_path=(Path(normalize_text(row.get("selected_keyframe_path"))).expanduser().resolve() if normalize_text(row.get("selected_keyframe_path")) else (Path(normalize_text(row.get("keyframe_path"))).expanduser().resolve() if normalize_text(row.get("keyframe_path")) else None)),
                    preset_peak_time_seconds=normalize_text(row.get("predicted_peak_time_seconds")),
                )
            )
    return rows


def collect_json_candidates(raw_response: str) -> List[str]:
    candidates: List[str] = []
    stripped = raw_response.strip()
    if stripped:
        candidates.append(stripped)
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw_response, flags=re.DOTALL))
    candidates.extend(re.findall(r"\{.*?\}", raw_response, flags=re.DOTALL))
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        text = candidate.strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def choose_text(preferred: str, fallback: str) -> str:
    return preferred if preferred.strip() else fallback


def parse_panel_ids_from_text(raw_response: str) -> Tuple[str, ...]:
    seen: List[str] = []
    for match in re.findall(r'P\d+', raw_response, flags=re.IGNORECASE):
        panel_id = match.upper()
        if panel_id not in seen:
            seen.append(panel_id)
    return tuple(seen)


def panel_number(panel_id: str) -> Optional[int]:
    match = re.fullmatch(r'P(\d+)', panel_id.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def choose_biased_panel_id(panel_ids: Sequence[str], primary_panel_id: str, preferred_start: int, preferred_end: int, preferred_bonus: float) -> str:
    if not panel_ids:
        return primary_panel_id
    center = (preferred_start + preferred_end) / 2.0
    spread = max((preferred_end - preferred_start) / 2.0, 1.0)

    def score(panel_id: str) -> float:
        number = panel_number(panel_id)
        score_value = 0.0
        if panel_id == primary_panel_id:
            score_value += 1.0
        if number is not None and preferred_start <= number <= preferred_end:
            score_value += preferred_bonus
            score_value += max(0.0, 1.0 - abs(number - center) / (spread + 1.0))
        elif number is not None:
            score_value += 0.1
        return score_value

    return max(panel_ids, key=score)


def build_right_hand_only_summary(samples: Sequence[Mapping[str, Any]]) -> str:
    right_tracked = 0
    right_hit_points: List[Mapping[str, Any]] = []
    right_joint_counts: List[int] = []
    for sample in samples:
        hand_data = sample.get('handData')
        if not isinstance(hand_data, Mapping):
            continue
        if hand_data.get('isRightHandTracked') is True:
            right_tracked += 1
        right_hand = hand_data.get('rightHand')
        if isinstance(right_hand, Mapping):
            joints = right_hand.get('joints')
            if isinstance(joints, list):
                right_joint_counts.append(len(joints))
        right_hit_point = hand_data.get('rightIndexFingerRayHitPoint')
        if isinstance(right_hit_point, Mapping):
            right_hit_points.append(right_hit_point)
    tracked_text = f"right_hand_tracked_frames={right_tracked}/{len(samples)}."
    joint_text = ""
    if right_joint_counts:
        right_avg = sum(right_joint_counts) / len(right_joint_counts)
        joint_text = f" average_right_joint_count={right_avg:.1f}."
    ray_text = ""
    non_zero_hits = [
        point for point in right_hit_points
        if any(abs(float(point.get(axis, 0.0))) > 1e-6 for axis in ('x', 'y', 'z'))
    ]
    if non_zero_hits:
        representative = non_zero_hits[min(len(non_zero_hits) - 1, len(non_zero_hits) // 2)]
        ray_text = f" representative_right_index_ray_hit=({float(representative.get('x', 0.0)):.3f}, {float(representative.get('y', 0.0)):.3f}, {float(representative.get('z', 0.0)):.3f})."
    return (tracked_text + joint_text + ray_text).strip()


def normalize_peak_response_text(raw_response: str) -> str:
    text = raw_response.replace('\r', ' ')
    text = text.replace('\"', '"')
    text = text.replace('?', '"').replace('?', '"')
    return text


def apply_uv_decision_mode(result: Any, sample: Any, mode: str) -> Any:
    prior_source = (sample.spatial_prior_source or '').strip()
    prior_u = (sample.spatial_prior_u_norm or '').strip()
    prior_v = (sample.spatial_prior_v_norm or '').strip()
    if prior_source != 'gazePoint' or not prior_u or not prior_v:
        return result
    if mode == 'model':
        return result
    if getattr(result, 'referent_type', '') == 'none':
        return result
    use_prior = mode == 'gaze_prior_only' or getattr(result, 'parse_ok', 'false') != 'true' or prior_source == 'gazePoint'
    if not use_prior:
        return result
    result.u_norm = prior_u
    result.v_norm = prior_v
    suffix = 'uv overridden from gazePoint prior' if mode == 'gaze_prior_only' else 'uv aligned to gazePoint prior'
    if getattr(result, 'reasoning_summary', ''):
        result.reasoning_summary = f"{result.reasoning_summary} | {suffix}"
    else:
        result.reasoning_summary = suffix
    if getattr(result, 'error_message', ''):
        result.error_message = f"{result.error_message} | {suffix}"
    return result


def parse_peak_response(raw_response: str, t_start: float, t_end: float) -> PeakResult:
    normalized_response = normalize_peak_response_text(raw_response)
    for candidate in collect_json_candidates(normalized_response):
        cleaned = re.sub(r"^```(?:json)?", "", candidate.strip()).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        try:
            peak_time_raw = payload.get("primary_panel_timestamp_seconds", payload.get("panel_timestamp_seconds", payload.get("peak_time_seconds")))
            peak_time = float(peak_time_raw)
            confidence = float(payload["confidence"])
        except (KeyError, TypeError, ValueError):
            continue
        panel_id = str(payload.get("primary_panel_id", payload.get("panel_id", ""))).strip()
        candidate_panel_ids = tuple(
            str(item).strip().upper()
            for item in payload.get("candidate_panel_ids", [])
            if str(item).strip()
        )
        peak_time = max(t_start, min(t_end, peak_time))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(payload.get("reasoning_summary", "")).strip()
        message = ""
        if panel_id:
            message = f"Selected panel_id={panel_id}."
        return PeakResult(
            peak_time_seconds=f"{peak_time:.3f}",
            confidence=f"{confidence:.6f}",
            reasoning_summary=reasoning,
            parse_ok="true",
            raw_output=normalized_response,
            error_message=message,
            panel_id=panel_id,
            candidate_panel_ids=candidate_panel_ids,
        )

    time_match = re.search(r'(?:primary[_\s-]*panel[_\s-]*timestamp[_\s-]*seconds|panel[_\s-]*timestamp[_\s-]*seconds|peak[_\s-]*time[_\s-]*seconds)\s*[:=]\s*"?([0-9]+(?:\.[0-9]+)?)', normalized_response, flags=re.IGNORECASE)
    if time_match:
        peak_time = max(t_start, min(t_end, float(time_match.group(1))))
        confidence_match = re.search(r'confidence\s*[:=]\s*"?([0-9]+(?:\.[0-9]+)?)', normalized_response, flags=re.IGNORECASE)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        panel_match = re.search(r'(?:primary[_\s-]*panel[_\s-]*id|panel[_\s-]*id)\s*[:=]\s*"?(P\d+)"?', normalized_response, flags=re.IGNORECASE)
        panel_note = f" Parsed panel_id={panel_match.group(1).upper()}." if panel_match else ""
        recovered_panel_ids = parse_panel_ids_from_text(normalized_response)
        return PeakResult(
            peak_time_seconds=f"{peak_time:.3f}",
            confidence=f"{confidence:.6f}",
            reasoning_summary="",
            parse_ok="true",
            raw_output=normalized_response,
            error_message="Parsed peak timestamp from non-JSON text." + panel_note,
            panel_id=panel_match.group(1).upper() if panel_match else "",
            candidate_panel_ids=recovered_panel_ids,
        )

    recovered_panel_ids = parse_panel_ids_from_text(normalized_response)
    fallback_panel_id = recovered_panel_ids[0] if recovered_panel_ids else ""
    return PeakResult(
        peak_time_seconds="",
        confidence="",
        reasoning_summary="",
        parse_ok="false",
        raw_output=normalized_response,
        error_message="Failed to parse peak selection JSON.",
        panel_id=fallback_panel_id,
        candidate_panel_ids=recovered_panel_ids,
    )


def build_fallback_peak_result(sample: TwoStageSample, raw_output: str, parse_error: str, fallback_mode: str) -> PeakResult:
    if fallback_mode == "disable":
        return PeakResult(
            peak_time_seconds="",
            confidence="",
            reasoning_summary="",
            parse_ok="false",
            raw_output=raw_output,
            error_message=parse_error or "Peak fallback disabled.",
            panel_id="",
            candidate_panel_ids=(),
        )
    if fallback_mode == "t_start":
        fallback_time = sample.t_start
    elif fallback_mode == "t_end":
        fallback_time = sample.t_end
    else:
        fallback_time = sample.t_start + max(0.0, sample.t_end - sample.t_start) / 2.0
    return PeakResult(
        peak_time_seconds=f"{fallback_time:.3f}",
        confidence="0.000000",
        reasoning_summary="",
        parse_ok="false",
        raw_output=raw_output,
        error_message=(parse_error or "Failed to parse peak selection JSON.") + f" Using fallback peak: {fallback_mode}.",
        panel_id="",
        candidate_panel_ids=(),
    )


def estimate_frame_timestamps(t_start: float, t_end: float, frame_count: int) -> List[float]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [t_start + max(0.0, t_end - t_start) / 2.0]
    clip_duration = max(t_end - t_start, 1e-3)
    step = clip_duration / frame_count
    return [t_start + (index + 0.5) * step for index in range(frame_count)]


def panel_id_to_timestamp(panel_id: str, timestamps: Sequence[float]) -> Optional[float]:
    match = re.fullmatch(r"P(\d+)", panel_id.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    index = int(match.group(1)) - 1
    if 0 <= index < len(timestamps):
        return float(timestamps[index])
    return None


def build_storyboard_image(runtime: Any, frames: Sequence[Any], timestamps: Sequence[float], event_id: str) -> Path:
    if not frames:
        raise TwoStageError(f"No frames available for storyboard: {event_id}")
    image_module = runtime.image_module
    draw_module = runtime.image_draw_module
    font_module = runtime.image_font_module
    font = font_module.load_default()
    cell_width = 384
    cell_height = 384
    columns = 4 if len(frames) > 4 else max(1, len(frames))
    rows = (len(frames) + columns - 1) // columns
    header_height = 72
    canvas = image_module.new("RGB", (columns * cell_width, rows * cell_height + header_height), (18, 18, 18))
    header_draw = draw_module.Draw(canvas)
    header_draw.text((16, 12), f"Event: {event_id}", fill=(255, 255, 255), font=font)
    header_draw.text((16, 38), "Choose the panel whose timestamp best reveals the final spatial referent.", fill=(220, 220, 220), font=font)
    for index, frame in enumerate(frames):
        row = index // columns
        col = index % columns
        x0 = col * cell_width
        y0 = row * cell_height + header_height
        tile = frame.copy()
        tile.thumbnail((cell_width, cell_height))
        paste_x = x0 + (cell_width - tile.width) // 2
        paste_y = y0 + (cell_height - tile.height) // 2
        canvas.paste(tile, (paste_x, paste_y))
        draw = draw_module.Draw(canvas)
        label = f"P{index + 1} | t={timestamps[index]:.3f}s"
        draw.rectangle((x0 + 8, y0 + 8, x0 + cell_width - 8, y0 + 36), fill=(0, 0, 0))
        draw.text((x0 + 14, y0 + 14), label, fill=(255, 255, 0), font=font)
        draw.rectangle((x0, y0, x0 + cell_width - 1, y0 + cell_height - 1), outline=(90, 90, 90), width=2)
    output_dir = Path(tempfile.mkdtemp(prefix="two_stage_storyboard_"))
    output_path = output_dir / "storyboard.jpg"
    canvas.save(output_path, format="JPEG", quality=95)
    return output_path


def extract_peak_frame(ffmpeg_path: str, video_path: Path, peak_time_seconds: float) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="two_stage_peak_"))
    output_path = temp_dir / "peak.jpg"
    import subprocess
    command = [ffmpeg_path, "-y", "-ss", f"{peak_time_seconds:.3f}", "-i", str(video_path), "-frames:v", "1", str(output_path)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not output_path.exists():
        raise TwoStageError(f"Failed to extract peak frame from {video_path}: {completed.stderr.strip() or completed.stdout.strip()}")
    return output_path


def build_peak_prompt(sample: TwoStageSample) -> str:
    return PEAK_USER_PROMPT_TEMPLATE.format(
        event_id=sample.event_id,
        instruction_text=sample.instruction_text or "No instruction_text was provided.",
        utterance_text=sample.utterance_text or "No utterance_text was provided.",
        target_description=sample.target_description or "No target_description was provided.",
        gaze_summary=sample.gaze_summary or "No gaze summary was provided.",
        hand_summary=sample.hand_summary or "No hand summary was provided.",
    )


def save_stage1_visuals(event_id: str, peak_frame_path: Path, storyboard_path: Optional[Path], stage1_vis_dir: Path) -> Tuple[Path, Optional[Path]]:
    stage1_vis_dir.mkdir(parents=True, exist_ok=True)
    safe_event_id = re.sub(r'[^A-Za-z0-9._-]+', '_', event_id)
    saved_peak_path = stage1_vis_dir / f"{safe_event_id}_peak.jpg"
    shutil.copy2(peak_frame_path, saved_peak_path)
    saved_storyboard_path: Optional[Path] = None
    if storyboard_path and storyboard_path.exists():
        saved_storyboard_path = stage1_vis_dir / f"{safe_event_id}_storyboard.jpg"
        shutil.copy2(storyboard_path, saved_storyboard_path)
    return saved_peak_path, saved_storyboard_path


def main() -> int:
    args = parse_args()
    try:
        project_root = Path(__file__).resolve().parents[2]
        grounding_module = load_module("grounding_module", project_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py")
        prep_module = load_module("prep_module", project_root / "scripts" / "data_prep" / "build_keyframe_grounding_input.py")
        input_csv = Path(args.input_csv).resolve()
        output_csv = Path(args.output_csv).resolve()
        stage1_output_csv = Path(args.stage1_output_csv).resolve() if args.stage1_output_csv else output_csv.with_name(f"{output_csv.stem}_stage1_selected.csv")
        vis_dir = Path(args.vis_dir).resolve() if args.vis_dir else None
        stage1_vis_dir = Path(args.stage1_vis_dir).resolve() if args.stage1_vis_dir else output_csv.parent / f"{output_csv.stem}_stage1_frames"
        event_json_dir = Path(args.event_json_dir).resolve() if args.event_json_dir else output_csv.parent / f"{output_csv.stem}_event_json"
        samples = read_samples(input_csv)
        runtime = grounding_module.ensure_runtime_dependencies()
        model_kwargs = {
            "device_map": "auto",
            "dtype": grounding_module.resolve_dtype(args.dtype, runtime.torch),
            "trust_remote_code": True,
            "local_files_only": args.local_files_only,
        }
        processor = runtime.auto_processor_cls.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=args.local_files_only)
        model = grounding_module.load_model_with_fallbacks(args.model_name, model_kwargs, runtime, args.use_flash_attn)
        device = grounding_module.resolve_model_device(model)
        ffmpeg_path = grounding_module.resolve_binary_path(args.ffmpeg_path, "ffmpeg")
        results: List[Dict[str, str]] = []
        stage1_rows: List[Dict[str, str]] = []
        for index, sample in enumerate(samples, start=1):
            try:
                timed_samples = prep_module.collect_timed_samples(prep_module.load_multimodal_samples(sample.json_path))
                if not timed_samples:
                    raise TwoStageError(f"No timed multimodal samples found for event_id={sample.event_id}")
                full_window_samples = prep_module.select_window_samples(timed_samples, sample.t_start, sample.t_start, sample.t_end)
                stage1_gaze_summary = choose_text(
                    prep_module.build_gaze_summary(full_window_samples, args.summary_max_points),
                    sample.gaze_summary,
                )
                stage1_hand_summary = choose_text(
                    build_right_hand_only_summary(full_window_samples),
                    sample.hand_summary,
                )
                stage1_sample = TwoStageSample(
                    event_id=sample.event_id,
                    video_path=sample.video_path,
                    json_path=sample.json_path,
                    t_start=sample.t_start,
                    t_end=sample.t_end,
                    instruction_text=sample.instruction_text,
                    utterance_text=sample.utterance_text,
                    target_description=sample.target_description,
                    gaze_summary=stage1_gaze_summary,
                    hand_summary=stage1_hand_summary,
                    keyframe_path=sample.keyframe_path,
                    preset_peak_time_seconds=sample.preset_peak_time_seconds,
                )
                peak_prompt = ""
                peak_raw_output = ""
                storyboard_path: Optional[Path] = None
                if args.stage == "stage2_only":
                    peak_result = PeakResult(
                        peak_time_seconds=sample.preset_peak_time_seconds,
                        confidence="",
                        reasoning_summary="",
                        parse_ok="true" if sample.preset_peak_time_seconds else "false",
                        raw_output="",
                        error_message="Loaded peak from stage1-selected CSV." if sample.preset_peak_time_seconds else "Missing predicted_peak_time_seconds in stage2_only mode.",
                    )
                else:
                    peak_prompt = build_peak_prompt(stage1_sample)
                    storyboard_frames = grounding_module.load_video_frames(
                        video_path=sample.video_path,
                        runtime=runtime,
                        ffmpeg_path=args.ffmpeg_path,
                        ffprobe_path=args.ffprobe_path,
                        max_video_frames=args.max_video_frames,
                        t_start=sample.t_start,
                        t_end=sample.t_end,
                    )
                    storyboard_timestamps = estimate_frame_timestamps(sample.t_start, sample.t_end, len(storyboard_frames))
                    storyboard_path = build_storyboard_image(runtime, storyboard_frames, storyboard_timestamps, sample.event_id)
                    media_sample = type("MediaSample", (), {
                        "video_path": sample.video_path,
                        "t_start": f"{sample.t_start:.3f}",
                        "t_end": f"{sample.t_end:.3f}",
                        "keyframe_path": storyboard_path,
                    })()
                    peak_inputs = grounding_module.build_model_inputs(
                        processor=processor,
                        runtime=runtime,
                        sample=media_sample,
                        system_prompt=PEAK_SYSTEM_PROMPT,
                        prompt_text=peak_prompt,
                        media_mode="image",
                        ffmpeg_path=args.ffmpeg_path,
                        ffprobe_path=args.ffprobe_path,
                        max_video_frames=args.max_video_frames,
                    )
                    peak_inputs = grounding_module.move_batch_to_device(peak_inputs, device)
                    with runtime.torch.inference_mode():
                        peak_generated = model.generate(**peak_inputs, max_new_tokens=args.peak_max_new_tokens, do_sample=False, use_cache=True)
                    peak_generated_ids = grounding_module.trim_generated_ids(peak_inputs, peak_generated)
                    peak_raw_output = grounding_module.decode_response(processor, peak_generated_ids).strip()
                    peak_result = parse_peak_response(peak_raw_output, sample.t_start, sample.t_end)
                    candidate_panel_ids = tuple(dict.fromkeys(tuple([peak_result.panel_id]) + tuple(peak_result.candidate_panel_ids)))
                    biased_panel_id = choose_biased_panel_id(
                        [panel_id for panel_id in candidate_panel_ids if panel_id],
                        peak_result.panel_id,
                        args.preferred_panel_start,
                        args.preferred_panel_end,
                        args.preferred_panel_bonus,
                    ) if candidate_panel_ids else peak_result.panel_id
                    if biased_panel_id and biased_panel_id != peak_result.panel_id:
                        peak_result.error_message = (peak_result.error_message + ' ' if peak_result.error_message else '') + f"Bias-adjusted to {biased_panel_id}."
                        peak_result.panel_id = biased_panel_id
                    if peak_result.panel_id:
                        mapped_timestamp = panel_id_to_timestamp(peak_result.panel_id, storyboard_timestamps)
                        if mapped_timestamp is not None:
                            peak_result.peak_time_seconds = f"{max(sample.t_start, min(sample.t_end, mapped_timestamp)):.3f}"
                            peak_result.parse_ok = "true"
                            if not peak_result.confidence:
                                peak_result.confidence = "0.500000"
                            if not peak_result.error_message:
                                peak_result.error_message = f"Recovered timestamp from panel_id={peak_result.panel_id}."
                    if peak_result.parse_ok != "true":
                        peak_result = build_fallback_peak_result(sample, peak_raw_output, peak_result.error_message, args.peak_fallback)
                if not peak_result.peak_time_seconds:
                    raise TwoStageError(peak_result.error_message)
                peak_time = float(peak_result.peak_time_seconds)
                peak_frame_path = sample.keyframe_path if args.stage == "stage2_only" and sample.keyframe_path and sample.keyframe_path.exists() else extract_peak_frame(ffmpeg_path, sample.video_path, peak_time)
                saved_peak_frame_path, _ = save_stage1_visuals(sample.event_id, peak_frame_path, storyboard_path, stage1_vis_dir)
                peak_frame_path = saved_peak_frame_path

                peak_sample = prep_module.select_peak_sample(timed_samples, peak_time)
                if peak_sample is None:
                    raise TwoStageError(f"No peak sample found for event_id={sample.event_id}")
                peak_window_samples = prep_module.select_peak_window_samples(
                    timed_samples,
                    t_peak=peak_time,
                    peak_window_seconds=args.peak_window_seconds,
                    fallback_sample=peak_sample,
                )
                window_samples = prep_module.select_window_samples(timed_samples, sample.t_start, peak_time, sample.t_end)
                stage2_gaze_summary = choose_text(
                    prep_module.build_gaze_summary(peak_window_samples, args.summary_max_points),
                    stage1_gaze_summary,
                )
                stage2_hand_summary = choose_text(
                    build_right_hand_only_summary(peak_window_samples),
                    stage1_hand_summary,
                )
                image_width, image_height = prep_module.get_image_size(peak_frame_path)
                prior_source_order = [item.strip() for item in args.prior_source_order.split(',') if item.strip()]
                spatial_prior = prep_module.compute_spatial_prior(peak_window_samples, image_width, image_height, prior_source_order)
                spatial_payload = prep_module.build_event_json_payload(
                    event_id=sample.event_id,
                    json_path=sample.json_path,
                    t_start=sample.t_start,
                    t_peak=peak_time,
                    t_end=sample.t_end,
                    peak_window_seconds=args.peak_window_seconds,
                    window_samples=window_samples,
                    peak_window_samples=peak_window_samples,
                    peak_sample=peak_sample,
                    image_width=image_width,
                    image_height=image_height,
                    spatial_prior=spatial_prior,
                )
                peak_event_json_path = prep_module.write_event_json(event_json_dir, sample.event_id, spatial_payload)
                spatial_context = grounding_module.summarize_event_json(spatial_payload)
                grounding_sample = grounding_module.Sample(
                    event_id=sample.event_id,
                    keyframe_path=peak_frame_path,
                    video_path=sample.video_path,
                    t_start=f"{sample.t_start:.3f}",
                    t_end=f"{sample.t_end:.3f}",
                    gaze_summary=stage2_gaze_summary,
                    hand_summary=stage2_hand_summary,
                    instruction_text=sample.instruction_text,
                    utterance_text=sample.utterance_text,
                    target_description=sample.target_description,
                    event_json_path=peak_event_json_path,
                    spatial_context_text=prep_module.build_spatial_context_text(spatial_payload),
                    spatial_context_json=json.dumps(spatial_payload, ensure_ascii=False),
                    spatial_prior_u_norm=prep_module.format_optional_float(spatial_prior.get("u_norm")),
                    spatial_prior_v_norm=prep_module.format_optional_float(spatial_prior.get("v_norm")),
                    spatial_prior_source=str(spatial_prior.get("source", "none")),
                )
                stage1_rows.append(
                    {
                        "event_id": sample.event_id,
                        "video_path": str(sample.video_path),
                        "json_path": str(sample.json_path),
                        "t_start": f"{sample.t_start:.3f}",
                        "t_end": f"{sample.t_end:.3f}",
                        "predicted_peak_time_seconds": peak_result.peak_time_seconds,
                        "predicted_peak_confidence": peak_result.confidence,
                        "peak_parse_ok": peak_result.parse_ok,
                        "selected_keyframe_path": str(peak_frame_path) if peak_frame_path else "",
                        "instruction_text": sample.instruction_text,
                        "utterance_text": sample.utterance_text,
                        "target_description": sample.target_description,
                        "gaze_summary": stage2_gaze_summary,
                        "hand_summary": stage2_hand_summary,
                        "event_json_path": str(peak_event_json_path),
                        "spatial_context_text": grounding_sample.spatial_context_text,
                        "spatial_context_json": grounding_sample.spatial_context_json,
                        "spatial_prior_u_norm": grounding_sample.spatial_prior_u_norm,
                        "spatial_prior_v_norm": grounding_sample.spatial_prior_v_norm,
                        "spatial_prior_source": grounding_sample.spatial_prior_source,
                    }
                )
                if args.stage == "stage1_only":
                    results.append(
                        {
                            "event_id": sample.event_id,
                            "predicted_peak_time_seconds": peak_result.peak_time_seconds,
                            "predicted_peak_confidence": peak_result.confidence,
                            "peak_reasoning_summary": peak_result.reasoning_summary,
                            "peak_parse_ok": peak_result.parse_ok,
                            "peak_raw_output": peak_result.raw_output,
                            "peak_error_message": peak_result.error_message,
                            "selected_keyframe_path": str(peak_frame_path) if peak_frame_path else "",
                            "prompt_text": peak_prompt,
                            "model_raw_output": "",
                            "parsed_json": "",
                            "referent_type": "",
                            "u_norm": "",
                            "v_norm": "",
                            "x_world": "",
                            "y_world": "",
                            "z_world": "",
                            "referent_text": "",
                            "reasoning_summary": "",
                            "confidence": "",
                            "parse_ok": "",
                            "error_message": "",
                            "spatial_prior_u_norm": grounding_sample.spatial_prior_u_norm,
                            "spatial_prior_v_norm": grounding_sample.spatial_prior_v_norm,
                            "spatial_prior_source": grounding_sample.spatial_prior_source,
                        }
                    )
                    print(f"Processed {index}/{len(samples)} | event_id={sample.event_id}", flush=True)
                    continue
                runner = grounding_module.LocalQwen3VLRunner(
                    args.model_name,
                    args.dtype,
                    args.use_flash_attn,
                    args.max_new_tokens,
                    args.local_files_only,
                    "image",
                    args.max_video_frames,
                    args.ffmpeg_path,
                    args.ffprobe_path,
                    args.prompt_variant,
                    args.offload_folder,
                )
                runner.runtime = runtime
                runner.processor = processor
                runner.model = model
                runner.device = device
                result = runner.predict(grounding_sample, spatial_context)
                result = apply_uv_decision_mode(result, grounding_sample, args.uv_decision_mode)
                if vis_dir is not None and runner.runtime is not None:
                    grounding_module.render_overlay(runner.runtime, grounding_sample, result, vis_dir)
                results.append(
                    {
                        "event_id": sample.event_id,
                        "predicted_peak_time_seconds": peak_result.peak_time_seconds,
                        "predicted_peak_confidence": peak_result.confidence,
                        "peak_reasoning_summary": peak_result.reasoning_summary,
                        "peak_parse_ok": peak_result.parse_ok,
                        "peak_raw_output": peak_result.raw_output,
                        "peak_error_message": peak_result.error_message,
                        "selected_keyframe_path": str(peak_frame_path),
                        "prompt_text": result.prompt_text,
                        "model_raw_output": result.model_raw_output,
                        "parsed_json": result.parsed_json,
                        "referent_type": result.referent_type,
                        "u_norm": result.u_norm,
                        "v_norm": result.v_norm,
                        "x_world": result.x_world,
                        "y_world": result.y_world,
                        "z_world": result.z_world,
                        "referent_text": result.referent_text,
                        "reasoning_summary": result.reasoning_summary,
                        "confidence": result.confidence,
                        "parse_ok": result.parse_ok,
                        "error_message": result.error_message,
                        "spatial_prior_u_norm": result.spatial_prior_u_norm,
                        "spatial_prior_v_norm": result.spatial_prior_v_norm,
                        "spatial_prior_source": result.spatial_prior_source,
                    }
                )
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                error_text = f"{type(exc).__name__}: {exc}".strip()
                if error_text.endswith(":"):
                    error_text = f"{type(exc).__name__}: {repr(exc)}"
                traceback_text = traceback.format_exc(limit=2).strip().replace("\r", " ").replace("\n", " | ")
                combined_error = error_text if error_text and error_text != type(exc).__name__ + ':' else f"{type(exc).__name__}: {repr(exc)}"
                if traceback_text:
                    combined_error = f"{combined_error} | {traceback_text}"
                stage1_rows.append(
                    {
                        "event_id": sample.event_id,
                        "video_path": str(sample.video_path),
                        "json_path": str(sample.json_path),
                        "t_start": f"{sample.t_start:.3f}",
                        "t_end": f"{sample.t_end:.3f}",
                        "predicted_peak_time_seconds": "",
                        "predicted_peak_confidence": "",
                        "peak_parse_ok": "false",
                        "selected_keyframe_path": "",
                        "instruction_text": sample.instruction_text,
                        "utterance_text": sample.utterance_text,
                        "target_description": sample.target_description,
                        "gaze_summary": sample.gaze_summary,
                        "hand_summary": sample.hand_summary,
                        "event_json_path": "",
                        "spatial_context_text": "",
                        "spatial_context_json": "",
                        "spatial_prior_u_norm": "",
                        "spatial_prior_v_norm": "",
                        "spatial_prior_source": "",
                    }
                )
                results.append(
                    {
                        "event_id": sample.event_id,
                        "predicted_peak_time_seconds": "",
                        "predicted_peak_confidence": "",
                        "peak_reasoning_summary": "",
                        "peak_parse_ok": "false",
                        "peak_raw_output": "",
                        "peak_error_message": combined_error,
                        "selected_keyframe_path": "",
                        "prompt_text": "",
                        "model_raw_output": "",
                        "parsed_json": "",
                        "referent_type": "",
                        "u_norm": "",
                        "v_norm": "",
                        "x_world": "",
                        "y_world": "",
                        "z_world": "",
                        "referent_text": "",
                        "reasoning_summary": "",
                        "confidence": "",
                        "parse_ok": "false",
                        "error_message": combined_error,
                        "spatial_prior_u_norm": "",
                        "spatial_prior_v_norm": "",
                        "spatial_prior_source": "",
                    }
                )
            print(f"Processed {index}/{len(samples)} | event_id={sample.event_id}", flush=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        stage1_output_csv.parent.mkdir(parents=True, exist_ok=True)
        with stage1_output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(STAGE1_OUTPUT_COLUMNS))
            writer.writeheader()
            for row in stage1_rows:
                writer.writerow(row)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Saved stage-1 selected CSV to: {stage1_output_csv}")
        print(f"Saved stage-1 visualization frames to: {stage1_vis_dir}")
        print(f"Saved two-stage grounding CSV to: {output_csv}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
