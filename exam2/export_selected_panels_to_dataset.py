#!/usr/bin/env python3
"""Export Qwen-selected exam2 panels back into each V3dMD sample folder.

This creates a derived, post-hoc selected-panel dataset. The source selection
comes from an already completed multi-panel model run; the copied images must
not be described as the original input to that run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


OUTPUT_COLUMNS = (
    "scene",
    "row_index",
    "event_id",
    "instruction",
    "policy",
    "panel_id",
    "referent_mentions",
    "source_frame_path",
    "dataset_sample_dir",
    "output_image_path",
    "provenance_path",
    "status",
    "status_detail",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy post-hoc selected panels into V3dMD sample folders.")
    parser.add_argument(
        "--manifest",
        default="exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv",
        help="Exam2 v10 manifest_all.csv.",
    )
    parser.add_argument(
        "--pred_csv",
        default="exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/predictions/qwen3vl_2d_predictions.csv",
        help="Exam2 v10 prediction CSV.",
    )
    parser.add_argument(
        "--output_manifest",
        default="exam2/selected_panel_export_manifest.csv",
        help="Global export manifest CSV.",
    )
    parser.add_argument("--dataset_root", default="/workspace/usr3/V3dMD", help="Dataset root used for safety checks.")
    parser.add_argument(
        "--selection_policy",
        choices=("all_unique", "majority", "first"),
        default="all_unique",
        help="How to collapse referent-level panel choices. Default: all_unique.",
    )
    parser.add_argument(
        "--folder_name",
        default="qwen_selected_panels",
        help="Subfolder created inside each V3dMD sample directory. Default: qwen_selected_panels.",
    )
    parser.add_argument("--start_index", type=int, default=0, help="Minimum row_index per scene.")
    parser.add_argument("--limit", type=int, help="Maximum row_index count per scene.")
    parser.add_argument("--scenes", nargs="*", help="Optional scene/partition names to export.")
    parser.add_argument("--dry_run", action="store_true", help="Validate and write manifests without copying images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing copied images/provenance.")
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_int(value: Any) -> Optional[int]:
    text = normalize_text(value)
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in OUTPUT_COLUMNS})


def selected_key(scene: str, row_index: int, args: argparse.Namespace) -> bool:
    if args.scenes and scene not in set(args.scenes):
        return False
    if row_index < args.start_index:
        return False
    if args.limit is not None and row_index >= args.start_index + args.limit:
        return False
    return True


def panel_sort_key(panel_id: str) -> int:
    value = parse_int(panel_id.upper().replace("P", ""))
    return value if value is not None else 999


def normalize_panel_id(value: Any) -> str:
    panel_id = normalize_text(value).upper()
    if panel_id and not panel_id.startswith("P"):
        panel_id = "P" + panel_id
    return panel_id


def parse_prediction_items(row: Mapping[str, str]) -> List[Dict[str, str]]:
    try:
        payload = json.loads(normalize_text(row.get("parsed_json")) or "{}")
    except json.JSONDecodeError:
        return []
    raw = payload.get("referents") if isinstance(payload, Mapping) else None
    if not isinstance(raw, list):
        return []
    items: List[Dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        panel_id = normalize_panel_id(item.get("panel_id"))
        if not panel_id:
            continue
        items.append(
            {
                "panel_id": panel_id,
                "mention": normalize_text(item.get("mention")),
            }
        )
    return items


def select_panels(items: Sequence[Mapping[str, str]], policy: str) -> List[Dict[str, Any]]:
    if not items:
        return []
    if policy == "first":
        first = items[0]
        return [{"panel_id": first["panel_id"], "mentions": [first.get("mention", "")]}]
    if policy == "majority":
        counts = Counter(item["panel_id"] for item in items)
        first_order: Dict[str, int] = {}
        for index, item in enumerate(items):
            first_order.setdefault(item["panel_id"], index)
        best_panel = min(counts, key=lambda panel: (-counts[panel], first_order.get(panel, 999)))
        mentions = [item.get("mention", "") for item in items if item["panel_id"] == best_panel]
        return [{"panel_id": best_panel, "mentions": mentions}]
    selected: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        panel_id = item["panel_id"]
        if panel_id in seen:
            for selected_item in selected:
                if selected_item["panel_id"] == panel_id:
                    selected_item["mentions"].append(item.get("mention", ""))
                    break
            continue
        seen.add(panel_id)
        selected.append({"panel_id": panel_id, "mentions": [item.get("mention", "")]})
    return selected


def manifest_panel_map(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, int, str], Mapping[str, str]]:
    result: Dict[Tuple[str, int, str], Mapping[str, str]] = {}
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        panel_id = normalize_panel_id(row.get("panel_id"))
        if not scene or row_index is None or not panel_id:
            continue
        key = (scene, row_index, panel_id)
        if key in result:
            continue
        if normalize_text(row.get("frame_extracted")) == "False":
            continue
        result[key] = row
    return result


def resolve_dataset_sample_dir(manifest_row: Mapping[str, str], dataset_root: Path) -> Tuple[Optional[Path], str]:
    json_path = Path(normalize_text(manifest_row.get("json_path")))
    if not json_path:
        return None, "empty_json_path"
    sample_dir = json_path.parent
    try:
        sample_dir.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        return None, f"json_path_not_under_dataset_root:{sample_dir}"
    if not sample_dir.exists():
        return None, f"sample_dir_missing:{sample_dir}"
    return sample_dir, ""


def output_image_name(panel_id: str, selected_count: int) -> str:
    if selected_count == 1:
        return "qwen_selected_panel.jpg"
    return f"qwen_selected_panel_{panel_id}.jpg"


def write_provenance(path: Path, payload: Mapping[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_panel(source: Path, destination: Path, overwrite: bool, dry_run: bool) -> Tuple[str, str]:
    if not source.exists():
        return "missing_source_frame", str(source)
    if dry_run:
        return "dry_run", ""
    if destination.exists() and not overwrite:
        return "exists", ""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return "copied", ""


def prediction_rows_by_event(rows: Sequence[Mapping[str, str]]) -> Iterable[Tuple[Tuple[str, int], Mapping[str, str]]]:
    for row in rows:
        scene = normalize_text(row.get("scene"))
        row_index = parse_int(row.get("row_index"))
        if scene and row_index is not None:
            yield (scene, row_index), row


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    pred_path = Path(args.pred_csv).resolve()
    output_manifest = Path(args.output_manifest).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    panel_map = manifest_panel_map(read_csv_rows(manifest_path))
    pred_rows = list(prediction_rows_by_event(read_csv_rows(pred_path)))

    output_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "manifest": str(manifest_path),
        "pred_csv": str(pred_path),
        "dataset_root": str(dataset_root),
        "selection_policy": args.selection_policy,
        "folder_name": args.folder_name,
        "dry_run": bool(args.dry_run),
        "events_seen": 0,
        "events_selected": 0,
        "export_rows": 0,
        "multi_panel_events": 0,
        "status_counts": {},
    }

    for (scene, row_index), pred_row in pred_rows:
        if not selected_key(scene, row_index, args):
            continue
        summary["events_seen"] += 1
        prediction_items = parse_prediction_items(pred_row)
        selected = select_panels(prediction_items, args.selection_policy)
        if not selected:
            output_rows.append(
                {
                    "scene": scene,
                    "row_index": row_index,
                    "event_id": normalize_text(pred_row.get("event_id")),
                    "instruction": normalize_text(pred_row.get("instruction")),
                    "policy": args.selection_policy,
                    "status": "no_predicted_panel",
                    "status_detail": "No parseable panel_id in parsed_json.",
                }
            )
            continue
        summary["events_selected"] += 1
        if len(selected) > 1:
            summary["multi_panel_events"] += 1

        provenance_items: List[Dict[str, Any]] = []
        event_rows: List[Dict[str, Any]] = []
        for selected_item in selected:
            panel_id = selected_item["panel_id"]
            manifest_row = panel_map.get((scene, row_index, panel_id))
            base = {
                "scene": scene,
                "row_index": row_index,
                "event_id": normalize_text(pred_row.get("event_id")),
                "instruction": normalize_text(pred_row.get("instruction")),
                "policy": args.selection_policy,
                "panel_id": panel_id,
                "referent_mentions": ";".join(item for item in selected_item.get("mentions", []) if item),
            }
            if manifest_row is None:
                event_rows.append({**base, "status": "missing_manifest_panel", "status_detail": f"{scene} row_{row_index} {panel_id}"})
                continue
            sample_dir, sample_error = resolve_dataset_sample_dir(manifest_row, dataset_root)
            if sample_dir is None:
                event_rows.append({**base, "status": "unresolved_dataset_dir", "status_detail": sample_error})
                continue
            out_dir = sample_dir / args.folder_name
            source = Path(normalize_text(manifest_row.get("frame_path")))
            destination = out_dir / output_image_name(panel_id, len(selected))
            provenance_path = out_dir / "qwen_selected_panels.json"
            status, detail = copy_panel(source, destination, args.overwrite, args.dry_run)
            event_row = {
                **base,
                "source_frame_path": str(source),
                "dataset_sample_dir": str(sample_dir),
                "output_image_path": str(destination),
                "provenance_path": str(provenance_path),
                "status": status,
                "status_detail": detail,
            }
            event_rows.append(event_row)
            provenance_items.append(
                {
                    "panel_id": panel_id,
                    "referent_mentions": selected_item.get("mentions", []),
                    "source_frame_path": str(source),
                    "output_image_path": str(destination),
                    "video_frame_time": normalize_text(manifest_row.get("video_frame_time")),
                    "json_sample_time": normalize_text(manifest_row.get("json_sample_time")),
                    "panel_selection_score": normalize_text(manifest_row.get("panel_selection_score")),
                    "panel_selection_reason": normalize_text(manifest_row.get("panel_selection_reason")),
                }
            )

        valid_dirs = {row.get("provenance_path") for row in event_rows if row.get("provenance_path")}
        if valid_dirs:
            provenance_path = Path(sorted(valid_dirs)[0])
            provenance_payload = {
                "derived_dataset": "post_hoc_qwen_selected_panels",
                "selection_source": "exam2 v10 multi-panel Qwen3-VL predictions",
                "method_note": (
                    "These images were selected after a multi-panel model run. "
                    "They are a derived selected-panel dataset, not the original input protocol."
                ),
                "scene": scene,
                "row_index": row_index,
                "event_id": normalize_text(pred_row.get("event_id")),
                "instruction": normalize_text(pred_row.get("instruction")),
                "selection_policy": args.selection_policy,
                "items": provenance_items,
            }
            if not args.dry_run:
                write_provenance(provenance_path, provenance_payload, args.overwrite)
        output_rows.extend(event_rows)

    status_counts = Counter(row.get("status", "") for row in output_rows)
    summary["export_rows"] = len(output_rows)
    summary["status_counts"] = dict(status_counts)

    write_csv(output_manifest, output_rows)
    summary_path = output_manifest.with_suffix(output_manifest.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote export manifest: {output_manifest}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
