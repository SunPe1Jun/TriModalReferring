#!/usr/bin/env python3
"""Build API-friendly grounding input CSV directly from a scene sample directory and instruction CSV."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

OUTPUT_COLUMNS = (
    "event_id",
    "scene_id",
    "sample_id",
    "video_id",
    "video_path",
    "json_path",
    "t_start",
    "t_peak",
    "t_end",
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


class SceneApiInputError(Exception):
    """Raised when scene API input CSV construction fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 3D-API-friendly grounding input CSV from scene sample folders and an instruction CSV."
    )
    parser.add_argument("--scene-root", required=True, help="Root directory containing sample subdirectories, e.g. /ai/data/V3dMD/scene1")
    parser.add_argument("--instruction-csv", required=True, help="Instruction CSV extracted from the Scene1 instruction set.")
    parser.add_argument("--output-csv", required=True, help="Output API input CSV path.")
    parser.add_argument("--event-json-dir", help="Optional directory to save compact event JSON files.")
    parser.add_argument("--scene-id", default="1", help="Scene id to write into the output CSV. Default: 1")
    parser.add_argument("--instruction-scene-id", help="Instruction scene_id to filter in instruction CSV. Defaults to --scene-id.")
    parser.add_argument("--instruction-start-order", type=int, default=1, help="Instruction order to start from. Default: 1")
    parser.add_argument("--sample-start-index", type=int, default=0, help="Zero-based sample directory start index after sorting. Default: 0")
    parser.add_argument("--limit", type=int, help="Optional maximum number of rows to build.")
    parser.add_argument("--sample-glob", default="*", help="Glob for sample subdirectories under scene-root. Default: *")
    parser.add_argument("--video-patterns", default="*gaze_enhanced*.mp4,ScreenRecord_*.mp4,*.mp4", help="Comma-separated video patterns searched inside each sample directory.")
    parser.add_argument("--json-patterns", default="multimodal_data.json", help="Comma-separated multimodal JSON patterns searched inside each sample directory.")
    parser.add_argument("--peak-window-seconds", type=float, default=0.25, help="Half window size used to build compact event JSON. Default: 0.25")
    parser.add_argument("--max-gaze-points", type=int, default=3, help="Maximum representative gaze points in summary. Default: 3")
    parser.add_argument("--output-profile", choices=("gaze_only_api", "legacy"), default="gaze_only_api", help="Spatial context output style. Default: gaze_only_api")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it already exists.")
    return parser.parse_args()


def load_prep_module() -> Any:
    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "scripts" / "data_prep" / "build_keyframe_grounding_input.py"
    spec = importlib.util.spec_from_file_location("build_keyframe_grounding_input_module", module_path)
    if spec is None or spec.loader is None:
        raise SceneApiInputError(f"Failed to load helper module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_patterns(raw: str, label: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise SceneApiInputError(f"{label} must contain at least one pattern.")
    return values


def natural_key(path: Path) -> Tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    normalized: List[Any] = []
    for part in parts:
        if part.isdigit():
            normalized.append(int(part))
        elif part:
            normalized.append(part.lower())
    return tuple(normalized)


def resolve_single_match(sample_dir: Path, patterns: Sequence[str], label: str) -> Path:
    matches: List[Path] = []
    for pattern in patterns:
        found = sorted(sample_dir.glob(pattern))
        if found:
            matches = [path.resolve() for path in found]
            break
    if not matches:
        raise SceneApiInputError(f"Missing {label} under sample directory: {sample_dir}")
    if len(matches) > 1:
        preview = ", ".join(str(path) for path in matches[:5])
        raise SceneApiInputError(f"{label} matched multiple files under {sample_dir}: {preview}")
    return matches[0]


def discover_sample_dirs(scene_root: Path, sample_glob: str, json_patterns: Sequence[str], video_patterns: Sequence[str]) -> List[Path]:
    if not scene_root.exists() or not scene_root.is_dir():
        raise SceneApiInputError(f"scene-root does not exist or is not a directory: {scene_root}")
    sample_dirs: List[Path] = []
    for path in sorted(scene_root.glob(sample_glob), key=natural_key):
        if not path.is_dir():
            continue
        try:
            resolve_single_match(path, json_patterns, "multimodal JSON")
            resolve_single_match(path, video_patterns, "video")
        except SceneApiInputError:
            continue
        sample_dirs.append(path.resolve())
    if not sample_dirs:
        raise SceneApiInputError(f"No valid sample directories were discovered under: {scene_root}")
    return sample_dirs


def read_instruction_rows(path: Path, instruction_scene_id: str, start_order: int) -> List[Dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise SceneApiInputError(f"instruction-csv does not exist or is not a file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SceneApiInputError(f"instruction-csv has no header row: {path}")
        required = {"scene_id", "instruction_order", "instruction_text"}
        missing = [column for column in required if column not in reader.fieldnames]
        if missing:
            raise SceneApiInputError(f"instruction-csv is missing required columns: {', '.join(missing)}")
        rows: List[Dict[str, str]] = []
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items()}
            if normalized.get("scene_id") != str(instruction_scene_id):
                continue
            try:
                order = int(normalized.get("instruction_order", "0"))
            except ValueError:
                continue
            if order < start_order:
                continue
            if not normalized.get("instruction_text"):
                continue
            rows.append(normalized)
    rows.sort(key=lambda item: int(item["instruction_order"]))
    if not rows:
        raise SceneApiInputError(f"No instruction rows found for scene_id={instruction_scene_id} starting at order {start_order}")
    return rows


def build_instruction_fields(instruction_row: Mapping[str, str]) -> Dict[str, str]:
    instruction_text = str(instruction_row.get("instruction_text", "")).strip()
    utterance_text = str(instruction_row.get("utterance_text", "")).strip()
    target_description = str(instruction_row.get("target_description", "")).strip() or instruction_text
    return {
        "instruction_text": instruction_text,
        "utterance_text": utterance_text,
        "target_description": target_description,
    }


def write_rows(path: Path, rows: Sequence[Mapping[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SceneApiInputError(f"Output CSV already exists. Use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def main() -> int:
    args = parse_args()
    try:
        prep = load_prep_module()
        scene_root = Path(args.scene_root).expanduser().resolve()
        output_csv = Path(args.output_csv).expanduser().resolve()
        event_json_dir = (
            Path(args.event_json_dir).expanduser().resolve()
            if args.event_json_dir
            else output_csv.parent / f"{output_csv.stem}_event_json"
        )
        instruction_scene_id = args.instruction_scene_id or args.scene_id
        video_patterns = parse_patterns(args.video_patterns, "video-patterns")
        json_patterns = parse_patterns(args.json_patterns, "json-patterns")

        sample_dirs = discover_sample_dirs(scene_root, args.sample_glob, json_patterns, video_patterns)
        instruction_rows = read_instruction_rows(Path(args.instruction_csv).expanduser().resolve(), str(instruction_scene_id), args.instruction_start_order)

        if args.sample_start_index < 0:
            raise SceneApiInputError("sample-start-index must be non-negative.")
        if args.sample_start_index:
            if args.sample_start_index >= len(sample_dirs):
                raise SceneApiInputError(
                    f"sample-start-index out of range: start={args.sample_start_index}, sample_count={len(sample_dirs)}"
                )
            sample_dirs = sample_dirs[args.sample_start_index :]

        if args.limit is not None:
            if args.limit <= 0:
                raise SceneApiInputError("limit must be positive.")
            sample_dirs = sample_dirs[: args.limit]

        if len(instruction_rows) < len(sample_dirs):
            raise SceneApiInputError(
                f"Instruction rows are fewer than discovered sample dirs: instructions={len(instruction_rows)}, samples={len(sample_dirs)}"
            )

        output_rows: List[Dict[str, str]] = []
        for index, sample_dir in enumerate(sample_dirs, start=1):
            sample_id = sample_dir.name
            video_path = resolve_single_match(sample_dir, video_patterns, "video")
            json_path = resolve_single_match(sample_dir, json_patterns, "multimodal JSON")
            samples = prep.load_multimodal_samples(json_path)
            timed_samples = prep.collect_timed_samples(samples)
            if not timed_samples:
                raise SceneApiInputError(f"No timed samples found in: {json_path}")

            t_start = 0.0
            t_end = timed_samples[-1][0]
            t_peak = (t_start + t_end) / 2.0
            peak_sample = prep.select_peak_sample(timed_samples, t_peak)
            if peak_sample is None:
                raise SceneApiInputError(f"Failed to select peak sample for: {json_path}")
            peak_window_samples = prep.select_peak_window_samples(
                timed_samples,
                t_peak=t_peak,
                peak_window_seconds=args.peak_window_seconds,
                fallback_sample=peak_sample,
            )
            window_samples = [sample for _, sample in timed_samples]
            gaze_summary = prep.build_gaze_summary(window_samples, args.max_gaze_points)
            hand_summary = prep.build_hand_summary(window_samples)
            spatial_prior = {"source": "none", "u_norm": None, "v_norm": None, "world_point": None, "sample_count": 0}
            event_id = f"scene{args.scene_id}_{sample_id}"
            payload = prep.build_event_json_payload(
                event_id=event_id,
                json_path=json_path,
                t_start=t_start,
                t_peak=t_peak,
                t_end=t_end,
                peak_window_seconds=args.peak_window_seconds,
                window_samples=window_samples,
                peak_window_samples=peak_window_samples,
                peak_sample=peak_sample,
                image_width=0,
                image_height=0,
                spatial_prior=spatial_prior,
                include_prior_metadata=args.output_profile == "legacy",
            )
            event_json_path = prep.write_event_json(event_json_dir, event_id, payload)
            spatial_context_text = prep.build_spatial_context_text(payload, output_profile=args.output_profile)

            instruction_fields = build_instruction_fields(instruction_rows[index - 1])
            output_rows.append(
                {
                    "event_id": event_id,
                    "scene_id": str(args.scene_id),
                    "sample_id": sample_id,
                    "video_id": video_path.stem,
                    "video_path": str(video_path),
                    "json_path": str(json_path),
                    "t_start": prep.format_time_seconds(t_start),
                    "t_peak": prep.format_time_seconds(t_peak),
                    "t_end": prep.format_time_seconds(t_end),
                    "instruction_text": instruction_fields["instruction_text"],
                    "utterance_text": instruction_fields["utterance_text"],
                    "target_description": instruction_fields["target_description"],
                    "gaze_summary": gaze_summary,
                    "hand_summary": hand_summary,
                    "event_json_path": str(event_json_path),
                    "spatial_context_text": spatial_context_text,
                    "spatial_context_json": json.dumps(payload, ensure_ascii=False),
                    "spatial_prior_u_norm": "",
                    "spatial_prior_v_norm": "",
                    "spatial_prior_source": "none",
                }
            )
            print(f"Prepared API input row {index}/{len(sample_dirs)} | sample_id={sample_id}", flush=True)

        write_rows(output_csv, output_rows, args.overwrite)
        print(f"Saved scene API input CSV to: {output_csv}")
        print(f"Saved compact event JSON snippets to: {event_json_dir}")
        print(f"Rows written: {len(output_rows)}")
        return 0
    except SceneApiInputError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
