#!/usr/bin/env python3
"""Recompute location/region anchor and interaction statistics from explicit taxonomy."""

from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCENES = (
    "scene1", "scene2", "scene3", "scene4_room1", "scene4_room2",
    "scene4_room3", "scene4_room4", "scene5",
)


def read_csv(path: Path, delimiter: str = ",") -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def split_ids(value: Any) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def load_taxonomy(path: Path) -> Tuple[Dict[str, set[str]], List[Dict[str, str]]]:
    rows = read_csv(path)
    by_scene: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        scene = row["scene"].strip()
        anchor = row["canonical_anchor"].strip()
        if scene not in SCENES or not anchor or anchor in by_scene[scene]:
            raise RuntimeError(f"invalid or duplicate taxonomy row: {row}")
        by_scene[scene].add(anchor)
    return by_scene, rows


def load_anchor_ids(root: Path, scene: str) -> set[str]:
    return {
        row["object_name"].strip()
        for row in read_csv(root / "data" / f"{scene}_anchor_table.tsv", delimiter="\t")
        if row.get("object_name", "").strip()
    }


def current_interactions(root: Path, taxonomy: Mapping[str, set[str]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for scene in SCENES:
        rows = read_csv(root / "data/match_eval_qwen3vl30b_mention_first_v3" / f"{scene}_match_eval.csv")
        for row in rows:
            mapped = split_ids(row.get("gt_referents_mapped"))
            matched = sorted(set(mapped) & taxonomy.get(scene, set()))
            output.append({
                "unified_id": f"{scene}::{int(row['row_index'])}",
                "scene": scene,
                "row_index": int(row["row_index"]),
                "raw_referents": row.get("gt_referents_raw", ""),
                "mapped_referents": ",".join(mapped),
                "matched_location_region_anchors": ",".join(matched),
                "is_location_region_interaction": bool(matched),
            })
    return output


def previous_interaction_ids(root: Path, taxonomy: Mapping[str, set[str]]) -> set[str]:
    try:
        text = subprocess.check_output(
            ["git", "show", "c21c5ea:paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv"],
            cwd=root, text=True,
        )
    except subprocess.CalledProcessError:
        path = root / "paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv"
        if not path.exists():
            return set()
        text = path.read_text(encoding="utf-8-sig")
    result = set()
    for row in csv.DictReader(io.StringIO(text)):
        try:
            mapped = json.loads(row.get("gt_anchor_ids", "[]"))
        except json.JSONDecodeError:
            mapped = []
        if set(mapped) & taxonomy.get(row["scene"], set()):
            result.add(row["unified_id"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--taxonomy", default="analysis_configs/location_region_anchor_classification.csv")
    parser.add_argument("--output_dir", default="analysis_outputs/location_region_audit")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    taxonomy, taxonomy_rows = load_taxonomy((root / args.taxonomy).resolve())
    output_dir = (root / args.output_dir).resolve()

    missing = []
    for scene, names in taxonomy.items():
        for name in sorted(names - load_anchor_ids(root, scene)):
            missing.append(f"{scene}:{name}")
    if missing:
        raise RuntimeError(f"taxonomy anchors absent from anchor tables: {missing}")

    interactions = current_interactions(root, taxonomy)
    current_ids = {row["unified_id"] for row in interactions if row["is_location_region_interaction"]}
    previous_ids = previous_interaction_ids(root, taxonomy)
    by_scene = []
    for scene in SCENES:
        scene_rows = [row for row in interactions if row["scene"] == scene]
        by_scene.append({
            "scene": scene,
            "location_region_anchor_count": len(taxonomy.get(scene, set())),
            "location_region_interaction_count": sum(bool(row["is_location_region_interaction"]) for row in scene_rows),
            "total_interaction_count": len(scene_rows),
        })
    changed_rows = []
    interaction_by_id = {row["unified_id"]: row for row in interactions}
    for unified_id in sorted(current_ids ^ previous_ids):
        row = interaction_by_id[unified_id]
        changed_rows.append({
            "unified_id": unified_id,
            "scene": row["scene"],
            "row_index": row["row_index"],
            "change": "added" if unified_id in current_ids else "removed",
            "mapped_referents": row["mapped_referents"],
            "matched_location_region_anchors": row["matched_location_region_anchors"],
        })

    write_csv(output_dir / "location_region_anchor_classification.csv", taxonomy_rows, taxonomy_rows[0].keys())
    write_csv(output_dir / "location_region_interaction_audit.csv", interactions, interactions[0].keys())
    write_csv(output_dir / "location_region_by_scene.csv", by_scene, by_scene[0].keys())
    write_csv(
        output_dir / "location_region_changed_ids.csv",
        changed_rows,
        ("unified_id", "scene", "row_index", "change", "mapped_referents", "matched_location_region_anchors"),
    )
    summary = {
        "taxonomy_source": args.taxonomy,
        "location_region_anchor_count": sum(len(names) for names in taxonomy.values()),
        "location_region_interaction_count": len(current_ids),
        "previous_evidence_interaction_count_under_same_taxonomy": len(previous_ids),
        "added_interaction_count": len(current_ids - previous_ids),
        "removed_interaction_count": len(previous_ids - current_ids),
        "added_ids": sorted(current_ids - previous_ids),
        "removed_ids": sorted(previous_ids - current_ids),
        "legacy_1461_reproducible": len(previous_ids) == 1461,
        "legacy_1461_note": "The prior 1,461 value has no committed taxonomy and is not reproduced by this explicit classification.",
    }
    (output_dir / "location_region_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
