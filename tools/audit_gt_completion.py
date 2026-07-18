#!/usr/bin/env python3
"""Audit the confirmed GT completion against all three experiment loaders."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import openpyxl

from repair_unmapped_gt_annotations import ALIASES, CONFIRMED_CANONICAL


SCENE_FILES = {
    "scene2": "scene2_cleaned_v2.xlsx",
    "scene4_room1": "scene4_room1.xlsx",
    "scene4_room3": "scene4_room3.xlsx",
    "scene4_room4": "scene4_room4.xlsx",
    "scene5": "scene5.xlsx",
}


def key(value: Any) -> str:
    return re.sub(r"[\s_-]+", "", "" if value is None else str(value).strip().lower())


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: List[Mapping[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_anchor_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output = root / "analysis_outputs/gt_completion"

    eval_module = load_module("gt_audit_eval", root / "scripts/eval/evaluate_local_3d_object_match.py")
    exam2_module = load_module("gt_audit_exam2", root / "exam2/build_2d_eval_manifest.py")
    sys.path.insert(0, str(root / "exam3_point_grounding"))
    import point_grounding_common as exam3_common

    mapping_rows: List[Dict[str, Any]] = []
    anchor_rows_out: List[Dict[str, Any]] = []
    loader_rows: List[Dict[str, Any]] = []
    current_canonicals: Dict[str, set[str]] = {}

    for scene, workbook_name in SCENE_FILES.items():
        anchor_path = root / "data" / f"{scene}_anchor_table.tsv"
        workbook_path = root / "data" / workbook_name
        alias_map, canonical_labels = eval_module.load_anchor_alias_map(anchor_path)
        gt_rows = eval_module.load_gt_rows(workbook_path, alias_map)
        gt_by_index = {
            int(row["excel_row_index"]) - 2: row for row in gt_rows
        }

        seen_canonical: Dict[str, str] = {}
        seen_alias: Dict[str, str] = {}
        rows = read_anchor_rows(anchor_path)
        for row in rows:
            canonical = str(row.get("object_name", "")).strip()
            canonical_key = key(canonical)
            if not canonical_key or canonical_key in seen_canonical:
                raise RuntimeError(f"duplicate/empty canonical in {anchor_path}: {canonical!r}")
            seen_canonical[canonical_key] = canonical
            point = [float(row[field]) for field in ("x_world", "y_world", "z_world")]
            if not all(math.isfinite(value) for value in point):
                raise RuntimeError(f"nonfinite point for {scene}:{canonical}")
            for alias in [canonical] + eval_module.split_alias_text(row.get("aliases")):
                alias_key = key(alias)
                previous = seen_alias.get(alias_key)
                if previous and previous != canonical:
                    raise RuntimeError(f"alias collision in {scene}: {alias!r} -> {previous!r}/{canonical!r}")
                seen_alias[alias_key] = canonical
            anchor_rows_out.append({
                "scene": scene, "canonical_anchor": canonical,
                "x_world": point[0], "y_world": point[1], "z_world": point[2],
                "aliases": row.get("aliases", ""), "status": "ok",
            })
        current_canonicals[scene] = set(seen_canonical)

        exp2_loaded = exam2_module.load_anchor_table(anchor_path)
        exp3_loaded = exam3_common.load_anchor_table(root, scene)
        exp3_names = {anchor.anchor_id for anchor in exp3_loaded}
        for loader_name, names in (
            ("experiment1", set(canonical_labels)),
            ("experiment2", set(exp2_loaded)),
            ("experiment3", exp3_names),
        ):
            loader_rows.append({
                "scene": scene, "loader": loader_name, "anchor_count": len(names),
                "drawer1_loaded": "drawer1" in names,
                "drawer2_loaded": "drawer2" in names,
                "status": "ok",
            })

        for row_index, expected in CONFIRMED_CANONICAL[scene].items():
            actual = gt_by_index[row_index]["mapped_referents"]
            unmapped = gt_by_index[row_index]["unmapped_referents"]
            status = "ok" if actual == expected and not unmapped else "mismatch"
            mapping_rows.append({
                "unified_id": f"{scene}::{row_index}", "scene": scene,
                "row_index": row_index, "raw_referents": gt_by_index[row_index]["referents_raw"],
                "expected_canonical": ", ".join(expected), "actual_canonical": ", ".join(actual),
                "unmapped_referents": ", ".join(unmapped), "status": status,
            })
            if status != "ok":
                raise RuntimeError(f"GT mismatch for {scene}::{row_index}: expected={expected}, actual={actual}, unmapped={unmapped}")

        if scene == "scene2":
            row = gt_by_index[569]
            status = "ok" if row["mapped_referents"] == ["drawer1"] and not row["unmapped_referents"] else "mismatch"
            mapping_rows.append({
                "unified_id": "scene2::569", "scene": "scene2", "row_index": 569,
                "raw_referents": row["referents_raw"], "expected_canonical": "drawer1",
                "actual_canonical": ", ".join(row["mapped_referents"]),
                "unmapped_referents": ", ".join(row["unmapped_referents"]), "status": status,
            })
            if status != "ok":
                raise RuntimeError(f"scene2::569 audit failed: {row}")

    # The evidence anchor tables are rebuilt from the repaired source files, so
    # comparing them to the current evidence copy is intentionally idempotent.
    # The repair manifest records the approved canonical-anchor addition.
    new_anchors = ["scene2:drawer2"] if "drawer2" in current_canonicals["scene2"] else []
    if new_anchors != ["scene2:drawer2"]:
        raise RuntimeError(f"approved new canonical anchor is absent: {new_anchors}")

    forbidden = {key(value) for value in ("laptop3", "Printer1", "heater", "ceiling", "room1_front_area", "centerpiece")}
    bad = sorted(f"{scene}:{name}" for scene, names in current_canonicals.items() for name in names if name in forbidden)
    if bad:
        raise RuntimeError(f"forbidden canonical anchors present: {bad}")

    mapping_rows.sort(key=lambda row: (row["scene"], int(row["row_index"])))
    write_csv(output / "final_gt_mapping_audit.csv", mapping_rows, list(mapping_rows[0]))
    write_csv(output / "anchor_table_audit.csv", anchor_rows_out, list(anchor_rows_out[0]))
    write_csv(output / "loader_compatibility.csv", loader_rows, list(loader_rows[0]))
    summary = {
        "confirmed_mapping_count": 28,
        "additional_scene2_569_ok": True,
        "mapping_audit_row_count": len(mapping_rows),
        "all_confirmed_mappings_ok": all(row["status"] == "ok" for row in mapping_rows),
        "anchor_rows_audited": len(anchor_rows_out),
        "loader_checks": len(loader_rows),
        "new_canonical_anchors": new_anchors,
        "alias_addition_count": sum(len(items) for mapping in ALIASES.values() for items in mapping.values()),
    }
    (output / "gt_completion_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
