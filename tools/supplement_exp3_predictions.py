#!/usr/bin/env python3
"""Prepare, merge, and audit experiment-3 completion predictions."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCENES = (
    "scene1",
    "scene2",
    "scene3",
    "scene4_room1",
    "scene4_room2",
    "scene4_room3",
    "scene4_room4",
    "scene5",
)
SCENE_ORDER = {scene: index for index, scene in enumerate(SCENES)}
COMPLETION_IDS = {
    ("scene2", 260), ("scene2", 265), ("scene2", 335), ("scene2", 361),
    ("scene2", 374), ("scene2", 498), ("scene2", 569), ("scene2", 751),
    ("scene2", 792), ("scene4_room1", 15), ("scene4_room1", 20),
    ("scene4_room1", 33), ("scene4_room1", 34), ("scene4_room1", 37),
    ("scene4_room1", 44), ("scene4_room1", 45), ("scene4_room1", 64),
    ("scene4_room1", 70), ("scene4_room1", 76), ("scene4_room1", 94),
    ("scene4_room1", 99), ("scene4_room1", 130), ("scene4_room1", 138),
    ("scene4_room3", 19), ("scene4_room4", 61), ("scene5", 9),
    ("scene5", 35), ("scene5", 58), ("scene5", 131),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("scene", "")).strip(), int(str(row.get("row_index", "")).strip())


def sorted_keys(values: Iterable[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return sorted(values, key=lambda item: (SCENE_ORDER[item[0]], item[1]))


def expected_ids() -> set[Tuple[str, int]]:
    return {
        (scene, row_index)
        for scene in SCENES
        for row_index in range(200 if scene.startswith("scene4_") else 800)
    }


def prediction_path(root: Path, model: str) -> Path:
    if model == "qwen30":
        return root / "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/predictions.csv"
    if model == "qwen8":
        return root / "qwen8/outputs/exam3_qwen3vl8b_point_grounding_merged_20260713/predictions.csv"
    if model == "internvl":
        return root / "internvl/outputs/exam3_internvl38b_point_grounding_merged_20260714/predictions.csv"
    raise ValueError(model)


def prepare(root: Path, manifest_path: Path, gt_path: Path, output_dir: Path) -> None:
    manifest = read_csv(manifest_path)
    gt_rows = read_csv(gt_path)
    manifest_by_key = {key(row): row for row in manifest}
    gt_by_key = {key(row): row for row in gt_rows}
    expected = expected_ids()
    if len(manifest_by_key) != len(manifest) or set(manifest_by_key) != expected:
        raise RuntimeError("experiment-3 model-facing manifest is not a unique 4,000-row interaction set")
    if len(gt_by_key) != len(gt_rows) or set(gt_by_key) != expected:
        raise RuntimeError("experiment-3 GT manifest is not a unique 4,000-row interaction set")
    if any(row.get("status") != "ok" for row in manifest):
        bad = [f"{scene}::{index}" for (scene, index), row in manifest_by_key.items() if row.get("status") != "ok"]
        raise RuntimeError(f"non-ok model-facing manifest rows: {bad[:20]}")
    if any(row.get("gt_mapping_status") != "valid" for row in gt_rows):
        bad = [f"{scene}::{index}" for (scene, index), row in gt_by_key.items() if row.get("gt_mapping_status") != "valid"]
        raise RuntimeError(f"non-valid GT manifest rows: {bad[:20]}")
    scene2_498 = gt_by_key[("scene2", 498)]
    ids_498 = {item.strip() for item in scene2_498["gt_anchor_ids"].split(",") if item.strip()}
    if ids_498 != {"desk4", "file4", "desk1"} or int(scene2_498["gt_count"]) != 3:
        raise RuntimeError(f"scene2::498 GT mismatch: {scene2_498}")
    scene2_569 = gt_by_key[("scene2", 569)]
    if scene2_569["gt_anchor_ids"] != "drawer1" or int(scene2_569["gt_count"]) != 1:
        raise RuntimeError(f"scene2::569 GT mismatch: {scene2_569}")

    target_rows: List[Dict[str, Any]] = []
    for model in ("qwen30", "qwen8", "internvl"):
        predictions = read_csv(prediction_path(root, model))
        by_key = {key(row): row for row in predictions}
        if len(by_key) != len(predictions):
            raise RuntimeError(f"duplicate existing prediction IDs: {model}")
        missing = expected - set(by_key)
        if missing != COMPLETION_IDS:
            raise RuntimeError(f"unexpected missing prediction IDs for {model}: {sorted_keys(missing ^ COMPLETION_IDS)}")
        existing_invalid = [item for item, row in by_key.items() if row.get("parse_ok", "").lower() not in {"true", "1"}]
        if existing_invalid:
            raise RuntimeError(f"unexpected invalid existing predictions for {model}: {sorted_keys(existing_invalid)}")
        for scene, row_index in sorted_keys(missing):
            target_rows.append({
                "model": model,
                "unified_id": f"{scene}::{row_index}",
                "scene": scene,
                "row_index": row_index,
                "reason": "missing_prediction",
            })
    write_csv(
        output_dir / "exp3_supplement_targets.csv",
        target_rows,
        ("model", "unified_id", "scene", "row_index", "reason"),
    )
    key_file = output_dir / "completion_sample_keys.txt"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_text("".join(f"{scene}:{row_index}\n" for scene, row_index in sorted_keys(COMPLETION_IDS)), encoding="utf-8")
    summary = {
        "manifest_rows": len(manifest),
        "gt_rows": len(gt_rows),
        "completion_id_count": len(COMPLETION_IDS),
        "model_target_count": {model: sum(row["model"] == model for row in target_rows) for model in ("qwen30", "qwen8", "internvl")},
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def merge(root: Path, output_dir: Path, model: str, supplement_paths: Sequence[Path]) -> None:
    final_path = prediction_path(root, model)
    original = read_csv(final_path)
    supplement: List[Dict[str, str]] = []
    for supplement_path in supplement_paths:
        supplement.extend(read_csv(supplement_path))
    original_by_key = {key(row): row for row in original}
    supplement_by_key = {key(row): row for row in supplement}
    if len(supplement_by_key) != len(supplement) or set(supplement_by_key) != COMPLETION_IDS:
        raise RuntimeError(f"invalid supplement key set for {model}")
    backup = output_dir / "original_prediction_csv" / f"{model}_predictions_before_completion.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(final_path, backup)
    original_by_key.update(supplement_by_key)
    if set(original_by_key) != expected_ids():
        raise RuntimeError(f"merged prediction key mismatch for {model}")
    merged = [original_by_key[item] for item in sorted_keys(original_by_key)]
    write_csv(final_path, merged, original[0].keys())
    audit = {
        "model": model,
        "original_rows": len(original),
        "supplement_rows": len(supplement),
        "merged_rows": len(merged),
        "parse_failures": sum(row.get("parse_ok", "").lower() not in {"true", "1"} for row in merged),
    }
    (output_dir / f"{model}_merge_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "merge"))
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--manifest", default="exam3_point_grounding/outputs_full_v9_20260709/manifest.csv")
    parser.add_argument("--gt_manifest", default="exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv")
    parser.add_argument("--output_dir", default="analysis_outputs/experiment_completion/exp3")
    parser.add_argument("--model", choices=("qwen30", "qwen8", "internvl"))
    parser.add_argument("--supplement_csv", action="append")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if args.command == "prepare":
        prepare(root, (root / args.manifest).resolve(), (root / args.gt_manifest).resolve(), output_dir)
    else:
        if not args.model or not args.supplement_csv:
            parser.error("merge requires --model and --supplement_csv")
        merge(
            root,
            output_dir,
            args.model,
            [(root / path).resolve() for path in args.supplement_csv],
        )


if __name__ == "__main__":
    main()
