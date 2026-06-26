#!/usr/bin/env python3
"""Download a Hugging Face model repository via hf-mirror."""

from __future__ import annotations

import argparse
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
from pathlib import Path
from typing import List, Optional


DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.py",
    "*.safetensors",
    "tokenizer*",
    "processor*",
    "preprocessor_config.json",
    "generation_config.json",
    "chat_template.json",
]


class DownloadError(Exception):
    """Raised when model download cannot proceed."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository via hf-mirror."
    )
    parser.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Model repository name on Hugging Face.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Local directory where the model repository will be downloaded.",
    )
    parser.add_argument(
        "--hf_endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help="Mirror endpoint used for Hugging Face downloads.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision, branch, or commit to download.",
    )
    parser.add_argument(
        "--token",
        help="Optional Hugging Face access token for gated or private repos.",
    )
    parser.add_argument(
        "--cache_dir",
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--allow_patterns",
        default=",".join(DEFAULT_ALLOW_PATTERNS),
        help="Comma-separated file patterns to include. Use '*' to download everything.",
    )
    parser.add_argument(
        "--ignore_patterns",
        default="",
        help="Comma-separated file patterns to exclude.",
    )
    parser.add_argument(
        "--resume_download",
        action="store_true",
        help="Resume partial downloads when supported.",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if files already exist in cache.",
    )
    parser.add_argument(
        "--local_dir_use_symlinks",
        action="store_true",
        help="Allow symlinks in the local output directory when supported by huggingface_hub.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print resolved settings without starting the download.",
    )
    return parser.parse_args()


def parse_pattern_list(raw_value: str) -> Optional[List[str]]:
    patterns = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not patterns:
        return None
    if len(patterns) == 1 and patterns[0] == "*":
        return None
    return patterns


def ensure_runtime_dependencies():
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise DownloadError(
            "Missing dependency: huggingface_hub. Please install it before downloading."
        ) from exc
    return snapshot_download


def normalize_endpoint(raw_endpoint: str) -> str:
    endpoint = raw_endpoint.strip().rstrip("/")
    if not endpoint:
        raise DownloadError("hf_endpoint cannot be empty.")
    if not (endpoint.startswith("http://") or endpoint.startswith("https://")):
        raise DownloadError("hf_endpoint must start with http:// or https://")
    return endpoint


def main() -> int:
    args = parse_args()

    try:
        snapshot_download = ensure_runtime_dependencies()

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

        hf_endpoint = normalize_endpoint(args.hf_endpoint)
        allow_patterns = parse_pattern_list(args.allow_patterns)
        ignore_patterns = parse_pattern_list(args.ignore_patterns)

        os.environ["HF_ENDPOINT"] = hf_endpoint

        download_kwargs = {
            "repo_id": args.model_name,
            "revision": args.revision,
            "local_dir": str(output_dir),
            "local_dir_use_symlinks": args.local_dir_use_symlinks,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
            "token": args.token,
            "resume_download": args.resume_download,
            "force_download": args.force_download,
        }
        if cache_dir is not None:
            download_kwargs["cache_dir"] = str(cache_dir)

        print(f"Model name: {args.model_name}")
        print(f"Output dir: {output_dir}")
        print(f"HF endpoint: {hf_endpoint}")
        print(f"Revision: {args.revision}")
        print(f"Allow patterns: {allow_patterns if allow_patterns is not None else ['*']}")
        print(f"Ignore patterns: {ignore_patterns if ignore_patterns is not None else []}")

        if args.dry_run:
            print("Dry run only. No download started.")
            return 0

        downloaded_path = snapshot_download(**download_kwargs)
        print(f"Download completed: {downloaded_path}")
        return 0
    except DownloadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
