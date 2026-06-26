#!/usr/bin/env python3
"""Build an explicit mapping from event_manifest rows to instruction_set rows."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

MANIFEST_REQUIRED_COLUMNS = ("event_id", "scene_id")
INSTRUCTION_REQUIRED_COLUMNS = (
    "scene_id",
    "instruction_order",
    "instruction_id",
    "instruction_text",
    "utterance_text",
    "target_description",
)
OUTPUT_COLUMNS = (
    "event_id",
    "manifest_scene_id",
    "instruction_scene_id",
    "instruction_order",
    "instruction_id",
    "instruction_text",
    "utterance_text",
    "target_description",
    "source_file",
    "sheet_name",
    "source_row",
)


class AssignmentError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an explicit CSV assignment from event_manifest rows to instruction_set rows."
    )
    parser.add_argument("--manifest-csv", required=True, help="Path to event_manifest.csv.")
    parser.add_argument("--instruction-csv", required=True, help="Path to instruction_set_merged.csv.")
    parser.add_argument("--output-csv", required=True, help="Path to the assignment CSV.")
    parser.add_argument(
        "--assignment-spec",
        required=True,
        help=(
            "Comma-separated rules in the form manifestStart-manifestEnd:instructionScene:instructionStart. "
            "Example: 1-10:1:1,11-20:2:11"
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output CSV if it already exists.")
    return parser.parse_args()


def read_csv_rows(path: Path, required_columns: Sequence[str], label: str) -> List[Dict[str, str]]:
    if not path.exists() or not path.is_file():
        raise AssignmentError(f"{label} does not exist or is not a file: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AssignmentError(f"{label} has no header row: {path}")
        missing = [column for column in required_columns if column not in reader.fieldnames]
        if missing:
            raise AssignmentError(f"{label} is missing required columns: {', '.join(missing)}")
        rows = []
        for row in reader:
            normalized = {key: (value or "").strip() for key, value in row.items()}
            if any(normalized.values()):
                rows.append(normalized)
    if not rows:
        raise AssignmentError(f"{label} contains no valid rows: {path}")
    return rows


def parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise AssignmentError(f"Invalid integer for {label}: {value}") from exc


def parse_assignment_spec(spec: str) -> List[Tuple[int, int, int, int]]:
    rules: List[Tuple[int, int, int, int]] = []
    for item in spec.split(','):
        token = item.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d+)-(\d+):(\d+):(\d+)", token)
        if match is None:
            raise AssignmentError(
                f"Invalid assignment rule: {token}. Expected manifestStart-manifestEnd:instructionScene:instructionStart"
            )
        manifest_start, manifest_end, instruction_scene, instruction_start = map(int, match.groups())
        if manifest_end < manifest_start:
            raise AssignmentError(f"Invalid manifest range in rule: {token}")
        rules.append((manifest_start, manifest_end, instruction_scene, instruction_start))
    if not rules:
        raise AssignmentError("assignment-spec must contain at least one rule.")
    return rules


def build_instruction_index(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[int, int], Dict[str, str]]:
    index: Dict[Tuple[int, int], Dict[str, str]] = {}
    for row in rows:
        scene_id = parse_int(str(row.get("scene_id", "")), "instruction scene_id")
        order = parse_int(str(row.get("instruction_order", "")), "instruction_order")
        index[(scene_id, order)] = dict(row)
    return index


def build_assignment_rows(
    manifest_rows: Sequence[Mapping[str, str]],
    instruction_index: Mapping[Tuple[int, int], Mapping[str, str]],
    rules: Sequence[Tuple[int, int, int, int]],
) -> List[Dict[str, str]]:
    manifest_by_scene: Dict[int, Dict[str, str]] = {}
    for row in manifest_rows:
        scene_id = parse_int(str(row.get("scene_id", "")), "manifest scene_id")
        manifest_by_scene[scene_id] = dict(row)
    output_rows: List[Dict[str, str]] = []
    assigned_event_ids = set()
    for manifest_start, manifest_end, instruction_scene, instruction_start in rules:
        for offset, manifest_scene_id in enumerate(range(manifest_start, manifest_end + 1)):
            manifest_row = manifest_by_scene.get(manifest_scene_id)
            if manifest_row is None:
                raise AssignmentError(f"No manifest row found for scene_id={manifest_scene_id}")
            event_id = str(manifest_row.get("event_id", "")).strip()
            if not event_id:
                raise AssignmentError(f"Manifest row scene_id={manifest_scene_id} is missing event_id")
            instruction_order = instruction_start + offset
            instruction_row = instruction_index.get((instruction_scene, instruction_order))
            if instruction_row is None:
                raise AssignmentError(
                    f"No instruction row found for instruction_scene={instruction_scene}, instruction_order={instruction_order}"
                )
            if event_id in assigned_event_ids:
                raise AssignmentError(f"Duplicate assignment generated for event_id={event_id}")
            assigned_event_ids.add(event_id)
            instruction_text = str(instruction_row.get("instruction_text", "")).strip()
            utterance_text = str(instruction_row.get("utterance_text", "")).strip()
            target_description = str(instruction_row.get("target_description", "")).strip()
            if not target_description:
                target_description = instruction_text
            output_rows.append(
                {
                    "event_id": event_id,
                    "manifest_scene_id": str(manifest_scene_id),
                    "instruction_scene_id": str(instruction_scene),
                    "instruction_order": str(instruction_order),
                    "instruction_id": str(instruction_row.get("instruction_id", "")),
                    "instruction_text": instruction_text,
                    "utterance_text": utterance_text,
                    "target_description": target_description,
                    "source_file": str(instruction_row.get("source_file", "")),
                    "sheet_name": str(instruction_row.get("sheet_name", "")),
                    "source_row": str(instruction_row.get("source_row", "")),
                }
            )
    return output_rows


def write_rows(path: Path, rows: Sequence[Mapping[str, str]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise AssignmentError(f"Output CSV already exists. Use --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def main() -> int:
    args = parse_args()
    try:
        manifest_rows = read_csv_rows(Path(args.manifest_csv).resolve(), MANIFEST_REQUIRED_COLUMNS, "event_manifest.csv")
        instruction_rows = read_csv_rows(Path(args.instruction_csv).resolve(), INSTRUCTION_REQUIRED_COLUMNS, "instruction_set_merged.csv")
        rules = parse_assignment_spec(args.assignment_spec)
        instruction_index = build_instruction_index(instruction_rows)
        output_rows = build_assignment_rows(manifest_rows, instruction_index, rules)
        write_rows(Path(args.output_csv).resolve(), output_rows, args.overwrite)
        print(f"Saved instruction assignment CSV to: {Path(args.output_csv).resolve()}")
        print(f"Assigned rows: {len(output_rows)}")
        return 0
    except AssignmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
