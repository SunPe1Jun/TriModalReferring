#!/usr/bin/env python3
"""Run local InternVL3 for single-event 3D referent object selection."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class Local3DError(Exception):
    """Raised when local 3D single-event inference fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local InternVL3 on one CSV row and select a 3D referent object from scene anchors.")
    parser.add_argument("--input_csv", required=True, help="Input CSV path.")
    parser.add_argument("--row_index", type=int, default=0, help="Zero-based row index to test. Default: 0.")
    parser.add_argument("--output_json", required=True, help="Path to save the local model response JSON.")
    parser.add_argument("--scene_anchor_csv", required=True, help="Path to the scene anchor candidate table.")
    parser.add_argument("--model_name", required=True, help="Local InternVL3 model path, e.g. /ai/data/InternVL3-38B")
    parser.add_argument("--dtype", default="auto", help="Torch dtype: auto, float16, bfloat16, float32.")
    parser.add_argument("--use_flash_attn", action="store_true", help="Try flash_attention_2 first, then fall back automatically.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model and processor from local files only.")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens for generation.")
    parser.add_argument("--input_mode", choices=("video", "image", "auto"), default="video", help="Default: video")
    parser.add_argument("--max_video_frames", type=int, default=16, help="Maximum number of sampled frames when input_mode uses video.")
    parser.add_argument("--ffmpeg_path", help="Optional path to ffmpeg executable.")
    parser.add_argument("--ffprobe_path", help="Optional path to ffprobe executable.")
    parser.add_argument("--prompt_style", choices=("world_only", "full"), default="full", help="Prompt style. Default: full")
    parser.add_argument("--offload_folder", help="Optional directory for Accelerate disk offload.")
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
    return load_module("api3d_module_internvl", project_root / "scripts" / "grounding" / "run_qwen3vl_plus_api_single_event_3d.py")


def load_local_grounding_module() -> Any:
    project_root = Path(__file__).resolve().parents[2]
    return load_module("local_grounding_module_internvl", project_root / "scripts" / "grounding" / "run_qwen3vl_local_keyframe_grounding.py")


def resolve_media_mode(row: Mapping[str, str], requested_mode: str) -> str:
    if requested_mode == "auto":
        return "video" if str(row.get("video_path", "")).strip() else "image"
    return requested_mode


def build_local_prompt(row: Dict[str, str], anchor_rows: List[Dict[str, Any]], prompt_style: str, api3d: Any) -> str:
    return api3d.build_3d_object_prompt(row, anchor_rows, prompt_style)


def build_system_prompt() -> str:
    return (
        "You are a precise multimodal 3D referent selection model for egocentric VR interaction. "
        "You must use the uploaded visual evidence together with language and structured world cues to choose one valid candidate object label from the provided list. "
        "Do not invent object names. Return strict JSON only."
    )


def move_batch_to_device(batch: Mapping[str, Any], device: Optional[Any]) -> Mapping[str, Any]:
    if device is None:
        return batch
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


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


def build_storyboard_prompt(prompt_text: str) -> str:
    extra = (
        "\n\nLocal fallback note:\n"
        "The original event video was converted into a storyboard image made of chronologically ordered panels "
        "P1, P2, P3, ... from left to right and top to bottom.\n"
        "Treat later panels as later times in the same event.\n"
        "Base your object selection on the panel where the green gaze marker most clearly supports the intended referent."
    )
    return prompt_text + extra


def build_storyboard_frames(
    runtime: Any,
    sample: Any,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
) -> Any:
    frames = local_grounding_module_ref["module"].load_video_frames(
        video_path=sample.video_path or Path(),
        runtime=runtime,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        max_video_frames=max_video_frames,
        t_start=local_grounding_module_ref["module"].parse_time_value(sample.t_start, "t_start"),
        t_end=local_grounding_module_ref["module"].parse_time_value(sample.t_end, "t_end"),
    )
    storyboard = build_storyboard_image(frames, runtime)
    return storyboard


local_grounding_module_ref: Dict[str, Any] = {}


class LocalInternVLRunner:
    def __init__(
        self,
        model_name: str,
        dtype_name: str,
        use_flash_attn: bool,
        max_new_tokens: int,
        local_files_only: bool,
        offload_folder: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.dtype_name = dtype_name
        self.use_flash_attn = use_flash_attn
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self.offload_folder = offload_folder
        self.runtime = None
        self.tokenizer = None
        self.model = None
        self.device = None

    def _build_official_split_device_map(self, auto_config_cls: Any) -> Optional[Dict[str, int]]:
        if self.runtime is None:
            return None
        world_size = self.runtime.torch.cuda.device_count()
        if world_size < 2:
            return None
        config = auto_config_cls.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        llm_config = getattr(config, "llm_config", None)
        num_layers = getattr(llm_config, "num_hidden_layers", None)
        if not isinstance(num_layers, int) or num_layers <= 0:
            return None

        device_map: Dict[str, int] = {}
        num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
        allocations = [num_layers_per_gpu] * world_size
        allocations[0] = math.ceil(allocations[0] * 0.5)

        layer_cnt = 0
        for device_index, num_layer in enumerate(allocations):
            for _ in range(num_layer):
                if layer_cnt >= num_layers:
                    break
                device_map[f"language_model.model.layers.{layer_cnt}"] = device_index
                layer_cnt += 1

        device_map["vision_model"] = 0
        device_map["mlp1"] = 0
        device_map["language_model.model.tok_embeddings"] = 0
        device_map["language_model.model.embed_tokens"] = 0
        device_map["language_model.output"] = 0
        device_map["language_model.model.norm"] = 0
        device_map["language_model.model.rotary_emb"] = 0
        device_map["language_model.lm_head"] = 0
        device_map[f"language_model.model.layers.{num_layers - 1}"] = 0
        return device_map

    def load(self) -> None:
        local_grounding = local_grounding_module_ref["module"]
        self.runtime = local_grounding.ensure_runtime_dependencies()
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise Local3DError("Missing dependency: transformers AutoModel/AutoTokenizer/AutoConfig are required for InternVL3.") from exc

        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "use_fast": False,
        }
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                fix_mistral_regex=True,
                **tokenizer_kwargs,
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "torch_dtype": local_grounding.resolve_dtype(self.dtype_name, self.runtime.torch),
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": True,
        }
        if self.offload_folder:
            offload_path = Path(self.offload_folder).expanduser().resolve()
            offload_path.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_path)
        attempted: List[str] = []
        load_errors: List[str] = []

        split_device_map = self._build_official_split_device_map(AutoConfig)
        if split_device_map is not None:
            for use_flash in ([True, False] if self.use_flash_attn else [False]):
                try_kwargs = dict(model_kwargs)
                try_kwargs["device_map"] = split_device_map
                if use_flash:
                    try_kwargs["use_flash_attn"] = True
                try:
                    self.model = AutoModel.from_pretrained(self.model_name, **try_kwargs).eval()
                    self.device = local_grounding.resolve_model_device(self.model)
                    return
                except Exception as exc:
                    attempted.append(f"AutoModel(split_map,flash={use_flash})")
                    load_errors.append(f"AutoModel split_map (flash={use_flash}): {exc}")
        else:
            for use_flash in ([True, False] if self.use_flash_attn else [False]):
                try_kwargs = dict(model_kwargs)
                try_kwargs.pop("offload_folder", None)
                try_kwargs["low_cpu_mem_usage"] = False
                try_kwargs.pop("device_map", None)
                if use_flash:
                    try_kwargs["use_flash_attn"] = True
                try:
                    self.model = AutoModel.from_pretrained(self.model_name, **try_kwargs).eval()
                    if self.runtime.torch.cuda.is_available():
                        self.model = self.model.cuda()
                    self.device = local_grounding.resolve_model_device(self.model)
                    return
                except Exception as exc:
                    attempted.append(f"AutoModel(single_gpu,flash={use_flash})")
                    load_errors.append(f"AutoModel single_gpu (flash={use_flash}): {exc}")

        raise Local3DError(
            f"Failed to load InternVL model {self.model_name}. Tried: {', '.join(attempted)}. "
            f"Recent errors: {' || '.join(load_errors[-4:])}. "
            "If you are on a single 60GB GPU, this model may simply exceed the supported local inference budget. "
            "The official model card recommends multiple large GPUs for bf16 inference."
        )


def build_transform(runtime: Any) -> Any:
    try:
        import torchvision.transforms as T  # type: ignore
        from torchvision.transforms.functional import InterpolationMode  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise Local3DError("Missing dependency: torchvision is required for InternVL3 image preprocessing.") from exc

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio: float, target_ratios: Sequence[Tuple[int, int]], width: int, height: int, image_size: int) -> Tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image: Any, image_size: int = 448, max_num: int = 1, use_thumbnail: bool = True) -> List[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        set(
            (i, j)
            for n in range(1, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= 1
        ),
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def build_pixel_values(runtime: Any, images: Sequence[Any], max_num: int = 1) -> Tuple[Any, List[int]]:
    transform = build_transform(runtime)
    pixel_values_list = []
    num_patches_list = []
    for image in images:
        tiles = dynamic_preprocess(image, image_size=448, max_num=max_num, use_thumbnail=True)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = runtime.torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = runtime.torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


def prepare_internvl_inputs(
    runner: LocalInternVLRunner,
    media_mode: str,
    sample: Any,
    keyframe_path: Path,
    prompt_text: str,
    ffmpeg_path: Optional[str],
    ffprobe_path: Optional[str],
    max_video_frames: int,
) -> Tuple[Any, str, List[int]]:
    runtime = runner.runtime
    if runtime is None:
        raise Local3DError("Runner is not loaded.")

    if media_mode == "video":
        frames = local_grounding_module_ref["module"].load_video_frames(
            video_path=sample.video_path or Path(),
            runtime=runtime,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            max_video_frames=max_video_frames,
            t_start=local_grounding_module_ref["module"].parse_time_value(sample.t_start, "t_start"),
            t_end=local_grounding_module_ref["module"].parse_time_value(sample.t_end, "t_end"),
        )
        pixel_values, num_patches_list = build_pixel_values(runtime, frames, max_num=1)
        prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
        question = prefix + prompt_text
        return pixel_values, question, num_patches_list

    image = runtime.image_module.open(keyframe_path).convert("RGB")
    pixel_values, num_patches_list = build_pixel_values(runtime, [image], max_num=12)
    question = "<image>\n" + prompt_text
    return pixel_values, question, num_patches_list


def validate_and_adjust_local_3d_response(
    parsed_response: Optional[Dict[str, Any]],
    anchor_rows: List[Dict[str, Any]],
    prompt_style: str,
    api3d: Any,
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], str]:
    return api3d.validate_and_adjust_3d_response(parsed_response, anchor_rows, prompt_style)


def main() -> None:
    args = parse_args()
    api3d = load_api3d_module()
    local_grounding = load_local_grounding_module()
    local_grounding_module_ref["module"] = local_grounding

    row = api3d.read_row(Path(args.input_csv), args.row_index)
    anchor_rows = api3d.load_scene_anchor_table(Path(args.scene_anchor_csv))
    video_path = None
    video_path_raw = api3d.normalize_text(row.get("video_path"))
    if video_path_raw:
        video_path = Path(video_path_raw).expanduser().resolve()

    keyframe_path_raw = api3d.normalize_text(row.get("keyframe_path"))
    keyframe_path = Path(keyframe_path_raw).expanduser().resolve() if keyframe_path_raw else Path()
    if video_path is None and not keyframe_path_raw:
        raise SystemExit("Input row must contain at least video_path or keyframe_path.")

    prompt_text = build_local_prompt(row, anchor_rows, args.prompt_style, api3d)
    system_prompt = build_system_prompt()

    runner = LocalInternVLRunner(
        model_name=args.model_name,
        dtype_name=args.dtype,
        use_flash_attn=args.use_flash_attn,
        max_new_tokens=args.max_new_tokens,
        local_files_only=args.local_files_only,
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

    used_storyboard_fallback = False
    fallback_reason = ""

    generation_config = {"max_new_tokens": args.max_new_tokens, "do_sample": False}
    try:
        pixel_values, question, num_patches_list = prepare_internvl_inputs(
            runner=runner,
            media_mode=media_mode,
            sample=sample,
            keyframe_path=keyframe_path,
            prompt_text=prompt_text,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            max_video_frames=args.max_video_frames,
        )
        pixel_values = pixel_values.to(dtype=local_grounding.resolve_dtype(args.dtype, runner.runtime.torch))
        if runner.device is not None and hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(runner.device)
        with runner.runtime.torch.inference_mode():
            response_text, _history = runner.model.chat(
                runner.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
    except Exception as exc:
        if media_mode != "video":
            raise
        used_storyboard_fallback = True
        fallback_reason = str(exc)
        storyboard = build_storyboard_frames(
            runtime=runner.runtime,
            sample=sample,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
            max_video_frames=args.max_video_frames,
        )
        pixel_values, num_patches_list = build_pixel_values(runner.runtime, [storyboard], max_num=12)
        pixel_values = pixel_values.to(dtype=local_grounding.resolve_dtype(args.dtype, runner.runtime.torch))
        if runner.device is not None and hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(runner.device)
        question = "<image>\n" + build_storyboard_prompt(prompt_text)
        with runner.runtime.torch.inference_mode():
            response_text, _history = runner.model.chat(
                runner.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
    response_text = str(response_text).strip()

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
        "prompt_style": args.prompt_style,
        "prompt_text": prompt_text,
        "response_text": response_text,
        "parsed_response": parsed_response,
        "resolved_object_row": resolved_object_row,
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
    print(f"Saved local InternVL3 3D response to: {output_path}")


if __name__ == "__main__":
    main()
