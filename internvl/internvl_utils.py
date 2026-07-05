#!/usr/bin/env python3
"""Small InternVL adapter shared by VR-TriRef baseline scripts."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVLError(Exception):
    """Raised when InternVL setup or inference fails."""


def parse_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    return value_float if math.isfinite(value_float) else None


def resolve_binary_path(configured_path: Optional[str], binary_name: str) -> str:
    if configured_path:
        path = Path(configured_path).expanduser()
        if path.exists():
            return str(path)
    return binary_name


def probe_video_duration(video_path: Path, ffprobe_path: Optional[str]) -> Optional[float]:
    ffprobe_binary = resolve_binary_path(ffprobe_path, "ffprobe")
    command = [
        ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    try:
        duration = float(completed.stdout.strip())
    except ValueError:
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def extract_video_frames(
    video_path: Path,
    max_video_frames: int,
    ffmpeg_path: Optional[str] = None,
    ffprobe_path: Optional[str] = None,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> List[Image.Image]:
    if not video_path.exists():
        raise InternVLError(f"Missing video file: {video_path}")
    if max_video_frames <= 0:
        raise InternVLError(f"max_video_frames must be positive, got {max_video_frames}")
    duration = probe_video_duration(video_path, ffprobe_path)
    clip_start = t_start if t_start is not None else 0.0
    clip_end = t_end
    if clip_end is not None and clip_end <= clip_start:
        raise InternVLError(f"t_end must be greater than t_start for video input: {video_path}")
    if clip_end is None and duration is not None:
        clip_end = duration
    clip_duration = None if clip_end is None else max(clip_end - clip_start, 1e-3)
    fps = min(8.0, max(1.0, max_video_frames / clip_duration)) if clip_duration is not None else 1.0
    ffmpeg_binary = resolve_binary_path(ffmpeg_path, "ffmpeg")
    with tempfile.TemporaryDirectory(prefix="internvl_video_") as temp_dir:
        frame_pattern = str(Path(temp_dir) / "frame_%04d.jpg")
        command = [ffmpeg_binary, "-y"]
        if clip_start > 0.0:
            command.extend(["-ss", f"{clip_start:.3f}"])
        command.extend(["-i", str(video_path)])
        if clip_duration is not None:
            command.extend(["-t", f"{clip_duration:.3f}"])
        command.extend(["-vf", f"fps={fps:.6f}", "-frames:v", str(max_video_frames), frame_pattern])
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            raise InternVLError(
                f"ffmpeg failed while extracting frames from {video_path}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        frame_paths = sorted(Path(temp_dir).glob("frame_*.jpg"))
        if not frame_paths:
            raise InternVLError(f"No video frames were extracted from {video_path}")
        frames: List[Image.Image] = []
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            image.load()
            frames.append(image.copy())
        return frames


def pil_image_to_pixel_values(image: Image.Image, image_size: int, torch: Any, transforms: Any) -> Any:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform(image).unsqueeze(0).to(torch.bfloat16)


def build_image_question(system_prompt: str, prompt_text: str, image_count: int) -> str:
    prefixes = "".join(f"Image-{idx + 1}: <image>\n" for idx in range(image_count))
    return f"{system_prompt}\n\n{prefixes}{prompt_text}"


def build_video_question(system_prompt: str, prompt_text: str, frame_count: int) -> str:
    prefixes = "".join(f"Frame{idx + 1}: <image>\n" for idx in range(frame_count))
    return f"{system_prompt}\n\n{prefixes}{prompt_text}"


@dataclass
class InternVLRuntime:
    torch: Any
    transforms: Any
    auto_model_cls: Any
    auto_tokenizer_cls: Any
    bitsandbytes_config_cls: Any


def ensure_runtime() -> InternVLRuntime:
    try:
        import torch
        from torchvision import transforms
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover
        raise InternVLError(f"Missing InternVL runtime dependency: {exc}") from exc
    return InternVLRuntime(
        torch=torch,
        transforms=transforms,
        auto_model_cls=AutoModel,
        auto_tokenizer_cls=AutoTokenizer,
        bitsandbytes_config_cls=BitsAndBytesConfig,
    )


def _safe_transformers_module_name(path: Path) -> str:
    return path.name.replace("-", "_hyphen_").replace(".", "_dot_")


def patch_internvl_chat_remote_code(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text()
    marker = "class InternVLChatModel(PreTrainedModel):\n"
    compat_line = "    all_tied_weights_keys = {}\n"
    if marker not in text or compat_line in text:
        return
    path.write_text(text.replace(marker, marker + compat_line, 1))


def copy_local_remote_code_to_hf_cache(model_name: str) -> None:
    """Populate HF dynamic module cache with local InternVL custom code files.

    Transformers may create the module cache entry before all relative imports
    are copied for local checkpoints. Copying the local *.py files into existing
    cache revisions keeps trust_remote_code loading reproducible offline.
    """
    model_dir = Path(model_name).expanduser()
    if not model_dir.exists():
        return
    patch_internvl_chat_remote_code(model_dir / "modeling_internvl_chat.py")
    py_files = list(model_dir.glob("*.py"))
    if not py_files:
        return

    modules_cache = Path(
        os.environ.get("HF_MODULES_CACHE", str(Path.home() / ".cache" / "huggingface" / "modules"))
    )
    transformers_modules = modules_cache / "transformers_modules"
    if not transformers_modules.exists():
        return

    safe_name = _safe_transformers_module_name(model_dir)
    model_prefix = model_dir.name.lower().split("-")[0]
    target_dirs: List[Path] = []
    for repo_cache in transformers_modules.iterdir():
        if not repo_cache.is_dir():
            continue
        if repo_cache.name not in {model_dir.name, safe_name} and model_prefix not in repo_cache.name.lower():
            continue
        target_dirs.append(repo_cache)
        for revision_cache in repo_cache.iterdir():
            if revision_cache.is_dir():
                target_dirs.append(revision_cache)

    for target_dir in target_dirs:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "__init__.py").touch(exist_ok=True)
        for source_file in py_files:
            shutil.copy2(source_file, target_dir / source_file.name)
        patch_internvl_chat_remote_code(target_dir / "modeling_internvl_chat.py")


class InternVLChatRunner:
    def __init__(
        self,
        model_name: str,
        dtype_name: str = "bfloat16",
        local_files_only: bool = True,
        load_in_8bit: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 1024,
        image_size: int = 448,
        offload_folder: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.dtype_name = dtype_name
        self.local_files_only = local_files_only
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.offload_folder = offload_folder
        self.runtime: Optional[InternVLRuntime] = None
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        runtime = ensure_runtime()
        self.runtime = runtime
        dtype = getattr(runtime.torch, self.dtype_name, runtime.torch.bfloat16)
        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
            "device_map": self.device_map,
        }
        if self.load_in_8bit:
            model_kwargs["quantization_config"] = runtime.bitsandbytes_config_cls(load_in_8bit=True)
        if self.offload_folder:
            offload_path = Path(self.offload_folder).expanduser().resolve()
            offload_path.mkdir(parents=True, exist_ok=True)
            model_kwargs["offload_folder"] = str(offload_path)
        self.tokenizer = runtime.auto_tokenizer_cls.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
            use_fast=False,
        )
        copy_local_remote_code_to_hf_cache(self.model_name)
        try:
            self.model = runtime.auto_model_cls.from_pretrained(self.model_name, **model_kwargs).eval()
        except FileNotFoundError:
            copy_local_remote_code_to_hf_cache(self.model_name)
            self.model = runtime.auto_model_cls.from_pretrained(self.model_name, **model_kwargs).eval()

    def _pixel_values_from_images(self, images: Sequence[Image.Image]) -> Tuple[Any, List[int]]:
        if self.runtime is None:
            raise InternVLError("InternVL runner is not loaded")
        tensors = [
            pil_image_to_pixel_values(image, self.image_size, self.runtime.torch, self.runtime.transforms)
            for image in images
        ]
        if not tensors:
            raise InternVLError("No images were provided to InternVL")
        pixel_values = self.runtime.torch.cat(tensors, dim=0).cuda()
        return pixel_values, [1 for _ in tensors]

    def chat_images(self, images: Sequence[Image.Image], system_prompt: str, prompt_text: str) -> str:
        if self.runtime is None or self.model is None or self.tokenizer is None:
            raise InternVLError("InternVL runner is not loaded")
        pixel_values, num_patches_list = self._pixel_values_from_images(images)
        question = build_image_question(system_prompt, prompt_text, len(images))
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        with self.runtime.torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        return str(response).strip()

    def chat_video_frames(self, frames: Sequence[Image.Image], system_prompt: str, prompt_text: str) -> str:
        if self.runtime is None or self.model is None or self.tokenizer is None:
            raise InternVLError("InternVL runner is not loaded")
        pixel_values, num_patches_list = self._pixel_values_from_images(frames)
        question = build_video_question(system_prompt, prompt_text, len(frames))
        generation_config = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        with self.runtime.torch.inference_mode():
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )
        return str(response).strip()

    def empty_cache(self) -> None:
        if self.runtime is not None:
            try:
                self.runtime.torch.cuda.empty_cache()
            except Exception:
                pass
