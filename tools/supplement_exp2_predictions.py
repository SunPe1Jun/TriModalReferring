#!/usr/bin/env python3
"""Prepare, merge, and audit experiment-2 completion predictions."""

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
    ("scene2", 260),
    ("scene2", 265),
    ("scene2", 335),
    ("scene2", 361),
    ("scene2", 374),
    ("scene2", 498),
    ("scene2", 569),
    ("scene2", 751),
    ("scene2", 792),
    ("scene4_room1", 15),
    ("scene4_room1", 20),
    ("scene4_room1", 33),
    ("scene4_room1", 34),
    ("scene4_room1", 37),
    ("scene4_room1", 44),
    ("scene4_room1", 45),
    ("scene4_room1", 64),
    ("scene4_room1", 70),
    ("scene4_room1", 76),
    ("scene4_room1", 94),
    ("scene4_room1", 99),
    ("scene4_room1", 130),
    ("scene4_room1", 138),
    ("scene4_room3", 19),
    ("scene4_room4", 61),
    ("scene5", 9),
    ("scene5", 35),
    ("scene5", 58),
    ("scene5", 131),
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


def parse_ok(row: Mapping[str, Any]) -> bool:
    return str(row.get("parse_ok", "")).strip().lower() in {"true", "1", "yes"}


def sorted_keys(values: Iterable[Tuple[str, int]]) -> List[Tuple[str, int]]:
    return sorted(values, key=lambda item: (SCENE_ORDER[item[0]], item[1]))


def prediction_config(root: Path, model: str) -> Tuple[Path, Path]:
    if model == "qwen30":
        base = root / "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/predictions"
        return base / "qwen3vl_2d_predictions.csv", base / "json"
    if model == "qwen8":
        base = root / "qwen8/outputs/exam2_qwen3vl8b_baseline_2d_point_hybrid_v10/predictions"
        return base / "qwen3vl_2d_predictions.csv", base / "json"
    if model == "internvl":
        base = root / "internvl/outputs/exam2_internvl3_38b_baseline/predictions"
        return base / "internvl_2d_predictions.csv", base / "json"
    raise ValueError(model)


def validate_manifest(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int], List[Mapping[str, str]]]:
    grouped: Dict[Tuple[str, int], List[Mapping[str, str]]] = {}
    for row in rows:
        grouped.setdefault(key(row), []).append(row)
    expected = {(scene, index) for scene in SCENES for index in range(200 if scene.startswith("scene4_") else 800)}
    if set(grouped) != expected:
        missing = sorted_keys(expected - set(grouped))
        extra = sorted_keys(set(grouped) - expected)
        raise RuntimeError(f"manifest key mismatch: missing={missing[:10]}, extra={extra[:10]}")
    for target in COMPLETION_IDS:
        target_rows = grouped[target]
        if not any(row.get("referent_name", "").strip() for row in target_rows):
            raise RuntimeError(f"completion target still has no GT referent: {target}")
        if any(row.get("status", "") in {"missing_gt_referents", "missing_anchor"} for row in target_rows):
            raise RuntimeError(f"completion target still has placeholder status: {target}")
    scene2_498 = {row.get("referent_name", "").strip() for row in grouped[("scene2", 498)]}
    if scene2_498 != {"desk4", "file4", "desk1"}:
        raise RuntimeError(f"scene2::498 GT mismatch: {sorted(scene2_498)}")
    return grouped


def prepare(root: Path, manifest_path: Path, output_dir: Path) -> None:
    manifest_rows = read_csv(manifest_path)
    grouped = validate_manifest(manifest_rows)
    target_records: List[Dict[str, Any]] = []
    for model in ("qwen30", "qwen8", "internvl"):
        prediction_csv, json_dir = prediction_config(root, model)
        predictions = read_csv(prediction_csv)
        prediction_by_key = {key(row): row for row in predictions}
        if len(prediction_by_key) != len(predictions):
            raise RuntimeError(f"duplicate prediction keys before completion: {model}")
        missing = set(grouped) - set(prediction_by_key)
        if missing != COMPLETION_IDS:
            raise RuntimeError(
                f"unexpected missing IDs for {model}: expected={len(COMPLETION_IDS)}, actual={len(missing)}, "
                f"difference={sorted_keys(missing ^ COMPLETION_IDS)[:20]}"
            )
        failures = {item for item, row in prediction_by_key.items() if not parse_ok(row)}
        targets = set(missing)
        if model == "qwen30":
            targets.update(failures)
            backup_dir = output_dir / "original_qwen30_parse_failures"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for scene, row_index in sorted_keys(failures):
                source = json_dir / f"{scene}_row_{row_index}.json"
                if not source.exists():
                    raise RuntimeError(f"missing original parse-failure JSON: {source}")
                destination = backup_dir / source.name
                if not destination.exists():
                    shutil.copy2(source, destination)
        elif failures:
            raise RuntimeError(f"unexpected existing parse failures for {model}: {sorted_keys(failures)}")

        sparse_rows = [row for target in sorted_keys(targets) for row in grouped[target]]
        sparse_path = output_dir / f"{model}_sparse_manifest.csv"
        write_csv(sparse_path, sparse_rows, manifest_rows[0].keys())
        for scene, row_index in sorted_keys(targets):
            target_records.append(
                {
                    "model": model,
                    "unified_id": f"{scene}::{row_index}",
                    "scene": scene,
                    "row_index": row_index,
                    "reason": "missing_prediction" if (scene, row_index) in missing else "parse_failure_retry",
                    "sparse_manifest": str(sparse_path.relative_to(root)),
                }
            )

    write_csv(
        output_dir / "exp2_supplement_targets.csv",
        target_records,
        ("model", "unified_id", "scene", "row_index", "reason", "sparse_manifest"),
    )
    summary = {
        "manifest_unique_interactions": len(grouped),
        "completion_id_count": len(COMPLETION_IDS),
        "qwen30_target_count": sum(row["model"] == "qwen30" for row in target_records),
        "qwen8_target_count": sum(row["model"] == "qwen8" for row in target_records),
        "internvl_target_count": sum(row["model"] == "internvl" for row in target_records),
    }
    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


def retry_manifest(manifest_path: Path, prediction_csv: Path, output_path: Path) -> None:
    manifest_rows = read_csv(manifest_path)
    grouped = validate_manifest(manifest_rows)
    predictions = read_csv(prediction_csv)
    failures = {key(row) for row in predictions if not parse_ok(row)}
    if not failures:
        raise RuntimeError(f"no parse failures in {prediction_csv}")
    unknown = failures - set(grouped)
    if unknown:
        raise RuntimeError(f"retry keys not present in manifest: {sorted_keys(unknown)}")
    retry_rows = [row for target in sorted_keys(failures) for row in grouped[target]]
    write_csv(output_path, retry_rows, manifest_rows[0].keys())
    print(json.dumps({"retry_count": len(failures), "retry_ids": [f"{s}::{i}" for s, i in sorted_keys(failures)]}))


def merge(root: Path, output_dir: Path, model: str, supplement_csvs: Sequence[Path]) -> None:
    prediction_csv, _ = prediction_config(root, model)
    original = read_csv(prediction_csv)
    original_by_key = {key(row): row for row in original}
    supplement_by_key: Dict[Tuple[str, int], Dict[str, str]] = {}
    supplement_row_count = 0
    for supplement_csv in supplement_csvs:
        supplement = read_csv(supplement_csv)
        if not supplement:
            raise RuntimeError(f"empty supplement CSV: {supplement_csv}")
        current = {key(row): row for row in supplement}
        if len(current) != len(supplement):
            raise RuntimeError(f"duplicate keys within supplement: {supplement_csv}")
        supplement_by_key.update(current)
        supplement_row_count += len(supplement)
    expected_targets = {
        (row["scene"], int(row["row_index"]))
        for row in read_csv(output_dir / "exp2_supplement_targets.csv")
        if row["model"] == model
    }
    if set(supplement_by_key) != expected_targets:
        raise RuntimeError(
            f"supplement target mismatch for {model}: "
            f"missing={sorted_keys(expected_targets - set(supplement_by_key))}, "
            f"extra={sorted_keys(set(supplement_by_key) - expected_targets)}"
        )
    backup = output_dir / "original_prediction_csv" / f"{model}_predictions_before_completion.csv"
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        shutil.copy2(prediction_csv, backup)
    original_by_key.update(supplement_by_key)
    expected = {(scene, index) for scene in SCENES for index in range(200 if scene.startswith("scene4_") else 800)}
    if set(original_by_key) != expected:
        raise RuntimeError(f"merged prediction key mismatch for {model}")
    merged = [original_by_key[item] for item in sorted_keys(original_by_key)]
    write_csv(prediction_csv, merged, original[0].keys())
    audit = {
        "model": model,
        "original_rows": len(original),
        "supplement_attempt_count": len(supplement_csvs),
        "supplement_rows_across_attempts": supplement_row_count,
        "supplement_unique_rows": len(supplement_by_key),
        "merged_rows": len(merged),
        "merged_unique_interactions": len(original_by_key),
        "parse_failures_after_merge": sum(not parse_ok(row) for row in merged),
    }
    (output_dir / f"{model}_merge_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "retry-manifest", "merge"))
    parser.add_argument("--repo_root", default=".")
    parser.add_argument(
        "--manifest",
        default="exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv",
    )
    parser.add_argument("--output_dir", default="analysis_outputs/experiment_completion/exp2")
    parser.add_argument("--model", choices=("qwen30", "qwen8", "internvl"))
    parser.add_argument("--supplement_csv", action="append")
    parser.add_argument("--retry_manifest")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    if args.command == "prepare":
        prepare(root, (root / args.manifest).resolve(), output_dir)
    elif args.command == "retry-manifest":
        if not args.supplement_csv or len(args.supplement_csv) != 1 or not args.retry_manifest:
            parser.error("retry-manifest requires one --supplement_csv and --retry_manifest")
        retry_manifest(
            (root / args.manifest).resolve(),
            (root / args.supplement_csv[0]).resolve(),
            (root / args.retry_manifest).resolve(),
        )
    else:
        if not args.model or not args.supplement_csv:
            parser.error("merge requires --model and at least one --supplement_csv")
        merge(root, output_dir, args.model, [(root / path).resolve() for path in args.supplement_csv])


if __name__ == "__main__":
    main()
