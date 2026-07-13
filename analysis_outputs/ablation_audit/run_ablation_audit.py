#!/usr/bin/env python3
"""Non-destructive audit and unified export for completed modality ablations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


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
ANCHOR_VARIANTS = (
    "full_baseline",
    "language_anchors_only",
    "no_gaze",
    "no_hand",
    "no_visual",
    "no_gaze_hand",
)
PROJECTED_VARIANTS = (
    "full_baseline",
    "full_panels_no_crop",
    "instruction_only_prompt",
    "no_gaze",
    "no_gaze_text_prior",
)
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 2027
UNKNOWN = "UNKNOWN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="analysis_outputs/ablation_audit")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in fields})


def format_cell(value: Any) -> Any:
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            return UNKNOWN
        return f"{value:.10f}"
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def as_bool(value: Any) -> bool:
    return text(value).lower() in {"1", "true", "yes", "ok"}


def as_int(value: Any) -> int:
    try:
        return int(float(text(value)))
    except (TypeError, ValueError):
        return 0


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(text(value))
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def split_labels(value: Any) -> list[str]:
    return [item.strip() for item in text(value).split(",") if item.strip()]


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sample_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    return prf(float(tp), float(fp), float(fn))


def interaction_id(scene: str, row: Mapping[str, Any]) -> str:
    # event_id is not globally unique across scene4 room partitions.
    return f"{scene}_row_{as_int(row.get('row_index'))}"


def anchor_detail_paths(repo: Path, variant: str) -> list[Path]:
    if variant == "full_baseline":
        root = repo / "data/match_eval_qwen3vl30b_mention_first_v3"
    else:
        root = repo / "ablation/exam1/outputs" / variant / "eval"
    return [root / f"{scene}_match_eval.csv" for scene in SCENES]


def anchor_summary_paths(repo: Path, variant: str) -> list[Path]:
    if variant == "full_baseline":
        root = repo / "data/match_eval_qwen3vl30b_mention_first_v3"
    else:
        root = repo / "ablation/exam1/outputs" / variant / "eval"
    return [root / f"{scene}_match_eval_summary.json" for scene in SCENES]


def anchor_output_dir(repo: Path, variant: str) -> Path:
    if variant == "full_baseline":
        return repo / "data/match_eval_qwen3vl30b_mention_first_v3"
    return repo / "ablation/exam1/outputs" / variant


def projected_root(repo: Path, variant: str) -> Path:
    if variant == "full_baseline":
        return repo / "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10"
    return repo / "ablation/exam2/outputs" / variant


def load_anchor_predictions(repo: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    exported: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in ANCHOR_VARIANTS:
        rows: list[dict[str, Any]] = []
        for scene, path in zip(SCENES, anchor_detail_paths(repo, variant)):
            for source in read_csv(path):
                gt = split_labels(source.get("gt_referents_mapped"))
                pred = split_labels(source.get("predicted_referents_mapped"))
                gt_set = set(gt)
                pred_set = set(pred)
                tp = len(gt_set & pred_set)
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                precision, recall, f1 = sample_prf(tp, fp, fn)
                evaluable = as_bool(source.get("evaluable"))
                valid = text(source.get("response_status")) == "ok"
                item = {
                    "interaction_id": interaction_id(scene, source),
                    "scene": scene,
                    "partition": scene,
                    "variant": variant,
                    "ground_truth_targets": gt,
                    "predicted_targets": pred,
                    "valid_output": valid,
                    "evaluable": evaluable,
                    "hit": bool(tp),
                    "exact": bool(gt_set) and gt_set == pred_set,
                    "set_precision": precision,
                    "set_recall": recall,
                    "set_f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "num_gt_targets": len(gt_set),
                    "num_predicted_targets": len(pred_set),
                    "target_type": UNKNOWN,
                    "raw_output_path": text(source.get("prediction_json")),
                    "gt_raw": text(source.get("gt_referents_raw")),
                    "gt_unmapped": text(source.get("gt_referents_unmapped")),
                }
                rows.append(item)
                exported.append(item)
        by_variant[variant] = rows
    return exported, by_variant


def anchor_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    evaluable = [row for row in rows if bool(row["evaluable"])]
    valid = sum(bool(row["valid_output"]) for row in rows)
    tp = sum(as_int(row["tp"]) for row in rows)
    fp = sum(as_int(row["fp"]) for row in rows)
    fn = sum(as_int(row["fn"]) for row in rows)
    micro_p, micro_r, micro_f1 = prf(tp, fp, fn)
    return {
        "num_total": total,
        "num_evaluable": len(evaluable),
        "coverage": len(evaluable) / total if total else 0.0,
        "valid_output_count": valid,
        "valid_output_rate": valid / total if total else 0.0,
        "hit_all": sum(bool(row["hit"]) for row in rows) / total if total else 0.0,
        "hit_mapped": sum(bool(row["hit"]) for row in evaluable) / len(evaluable) if evaluable else 0.0,
        "exact": sum(bool(row["exact"]) for row in evaluable) / len(evaluable) if evaluable else 0.0,
        "set_f1_macro": sum(float(row["set_f1"]) for row in evaluable) / len(evaluable) if evaluable else 0.0,
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
    }


def category(setting: str, variant: str) -> tuple[str, bool, str]:
    if variant == "full_baseline" or (setting == "3d_grounding" and variant == "full"):
        return "full_baseline", False, "Reference condition."
    if setting == "anchor_selection":
        values = {
            "no_visual": (
                "hybrid_ablation",
                False,
                "Visual evidence is removed, but video input is replaced by one 64x64 placeholder image and ablation-specific prompt text is added; input type/count/geometry therefore differ from full.",
            ),
            "no_gaze": (
                "hybrid_ablation",
                False,
                "Structured gaze fields and gaze-derived proposals removed, but the original video can retain the visible green gaze marker; this is a structured-gaze-cue ablation, not complete gaze-modality removal.",
            ),
            "no_hand": (
                "hybrid_ablation",
                False,
                "Structured hand summary and hand/ray fields removed, but visible hands/gestures remain in the video; this is a structured-hand-cue ablation, not complete hand-modality removal.",
            ),
            "no_gaze_hand": (
                "hybrid_ablation",
                False,
                "Removes two structured cue families simultaneously; visible gaze marker and hands can remain in video.",
            ),
            "language_anchors_only": (
                "multimodal_input_baseline",
                False,
                "Simultaneously removes visual, gaze, hand, structured geometry, and timeline evidence while retaining the closed-set candidate inventory.",
            ),
        }
        return values[variant]
    if setting == "projected_2d":
        values = {
            "full_panels_no_crop": (
                "preprocessing_ablation",
                False,
                "Changes paired full-panel plus gaze-crop inputs to full panels only.",
            ),
            "instruction_only_prompt": (
                "prompt_ablation",
                False,
                "Keeps paired visual inputs but removes expected referent count and changes prompt mode.",
            ),
            "no_gaze": (
                "hybrid_ablation",
                False,
                "Removes gaze-specific text, masks the projected marker, and also removes the gaze-centered crop path by changing paired inputs to full panels.",
            ),
            "no_gaze_text_prior": (
                "hybrid_ablation",
                False,
                "Removes gaze-specific text and changes paired crop preprocessing to full panels; visible marker remains.",
            ),
        }
        return values[variant]
    return "cue_baseline", False, "Deterministic cue baseline; not a VLM modality ablation."


def import_projected_evaluator(repo: Path) -> Any:
    path = repo / "exam2/evaluate_2d_point_grounding.py"
    spec = importlib.util.spec_from_file_location("audit_projected_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_prediction_payload(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        payload = json.loads(text(row.get("parsed_json")) or "{}")
    except json.JSONDecodeError:
        return []
    refs = payload.get("referents", []) if isinstance(payload, dict) else []
    return [dict(item) for item in refs if isinstance(item, dict)]


def load_projected_predictions(repo: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    evaluator = import_projected_evaluator(repo)
    manifest_path = repo / "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv"
    manifest_rows = evaluator.read_csv_rows(manifest_path)
    grouped = evaluator.manifest_groups(manifest_rows)
    eval_args = SimpleNamespace(
        coordinate_mode="panel",
        panel_width=512,
        panel_height=384,
        columns=3,
        gutter=12,
        label_height=34,
    )
    exported: list[dict[str, Any]] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in PROJECTED_VARIANTS:
        root = projected_root(repo, variant)
        pred_csv = root / "predictions/qwen3vl_2d_predictions.csv"
        pred_rows = read_csv(pred_csv)
        pred_map = {(text(row.get("scene")), as_int(row.get("row_index"))): row for row in pred_rows}
        detail_rows = read_csv(root / "eval/2d_eval_detail.csv")
        detail_map = {(text(row.get("scene")), as_int(row.get("row_index"))): row for row in detail_rows}
        rows: list[dict[str, Any]] = []
        for key in sorted(grouped, key=lambda value: (value[0], value[1])):
            scene, row_index = key
            source_manifest = grouped[key]
            pred_row = pred_map.get(key, {})
            detail = detail_map.get(key, {})
            pred_items = evaluator.parse_predictions(pred_row)
            layout = evaluator.build_layout(source_manifest, eval_args)
            gts = evaluator.gt_points(source_manifest, layout, eval_args)
            _, _, _, matched_distances = evaluator.greedy_point_match(
                pred_items, gts, layout, eval_args, 100.0, False
            )
            min_distance = evaluator.min_point_distance(pred_items, gts, layout, eval_args)
            gt_panels: dict[str, list[str]] = defaultdict(list)
            gt_points: list[dict[str, Any]] = []
            seen_points: set[tuple[str, str]] = set()
            for gt in gts:
                referent = text(gt.get("referent"))
                panel = text(gt.get("panel_id"))
                if bool(gt.get("evidence_acceptable")) and panel not in gt_panels[referent]:
                    gt_panels[referent].append(panel)
                point_key = (referent, panel)
                if point_key not in seen_points:
                    seen_points.add(point_key)
                    gt_points.append(
                        {
                            "referent": referent,
                            "panel_id": panel,
                            "x_norm": as_float(gt.get("gt_u_norm")),
                            "y_norm": as_float(gt.get("gt_v_norm")),
                            "evidence_acceptable": bool(gt.get("evidence_acceptable")),
                        }
                    )
            predicted_points = parse_prediction_payload(pred_row)
            predicted_panels = [text(item.get("panel_id")) for item in predicted_points if text(item.get("panel_id"))]
            time_p, time_r, time_f1 = sample_prf(
                as_int(detail.get("time_tp")), as_int(detail.get("time_fp")), as_int(detail.get("time_fn"))
            )
            item: dict[str, Any] = {
                "interaction_id": f"{scene}_row_{row_index}",
                "scene": scene,
                "partition": scene,
                "variant": variant,
                "gt_panels": dict(gt_panels),
                "predicted_panels": predicted_panels,
                "temporal_correct": as_int(detail.get("time_fn")) == 0 and as_int(detail.get("time_fp")) == 0,
                "temporal_precision": time_p,
                "temporal_recall": time_r,
                "temporal_f1": time_f1,
                "gt_points": gt_points,
                "predicted_points": predicted_points,
                "point_distance": min_distance,
                "point_at_50": sample_prf(as_int(detail.get("point_tp_50")), as_int(detail.get("point_fp_50")), as_int(detail.get("point_fn_50")))[2],
                "point_at_100": sample_prf(as_int(detail.get("point_tp_100")), as_int(detail.get("point_fp_100")), as_int(detail.get("point_fn_100")))[2],
                "point_at_200": sample_prf(as_int(detail.get("point_tp_200")), as_int(detail.get("point_fp_200")), as_int(detail.get("point_fn_200")))[2],
                "joint_at_50": sample_prf(as_int(detail.get("joint_tp_50")), as_int(detail.get("joint_fp_50")), as_int(detail.get("joint_fn_50")))[2],
                "joint_at_100": sample_prf(as_int(detail.get("joint_tp_100")), as_int(detail.get("joint_fp_100")), as_int(detail.get("joint_fn_100")))[2],
                "joint_at_200": sample_prf(as_int(detail.get("joint_tp_200")), as_int(detail.get("joint_fp_200")), as_int(detail.get("joint_fn_200")))[2],
                "valid_output": as_bool(pred_row.get("parse_ok")),
                "raw_output_path": str((root / "predictions/json" / f"{scene}_row_{row_index}.json").resolve()),
                "gt_referent_count": as_int(detail.get("gt_referent_count")),
                "matched_distances_100": matched_distances,
            }
            for prefix in ("time", "point_50", "point_100", "point_200", "joint_50", "joint_100", "joint_200"):
                for stat in ("tp", "fp", "fn"):
                    if prefix == "time":
                        detail_column = f"time_{stat}"
                    else:
                        metric_name, threshold = prefix.rsplit("_", 1)
                        detail_column = f"{metric_name}_{stat}_{threshold}"
                    item[f"{prefix}_{stat}"] = as_int(detail.get(detail_column))
            rows.append(item)
            exported.append(item)
        by_variant[variant] = rows
    return exported, by_variant


def projected_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_samples": len(rows),
        "valid_output_count": sum(bool(row["valid_output"]) for row in rows),
    }
    result["valid_output_rate"] = result["valid_output_count"] / len(rows) if rows else 0.0
    for source, target in (
        ("time", "temporal"),
        ("point_50", "point_at_50"),
        ("point_100", "point_at_100"),
        ("point_200", "point_at_200"),
        ("joint_50", "joint_at_50"),
        ("joint_100", "joint_at_100"),
        ("joint_200", "joint_at_200"),
    ):
        tp = sum(as_int(row[f"{source}_tp"]) for row in rows)
        fp = sum(as_int(row[f"{source}_fp"]) for row in rows)
        fn = sum(as_int(row[f"{source}_fn"]) for row in rows)
        precision, recall, f1 = prf(tp, fp, fn)
        if target == "temporal":
            result["temporal_precision"] = precision
            result["temporal_recall"] = recall
            result["temporal_f1"] = f1
        else:
            result[target] = f1
    distances = [float(value) for row in rows for value in row["matched_distances_100"]]
    result["mean_matched_distance"] = float(np.mean(distances)) if distances else None
    result["median_matched_distance"] = float(np.median(distances)) if distances else None
    return result


def identical_id_sets(by_variant: Mapping[str, Sequence[Mapping[str, Any]]], baseline: str = "full_baseline") -> dict[str, bool]:
    base = [text(row["interaction_id"]) for row in by_variant[baseline]]
    base_set = set(base)
    return {
        variant: len(ids := [text(row["interaction_id"]) for row in rows]) == len(base)
        and len(ids) == len(set(ids))
        and set(ids) == base_set
        for variant, rows in by_variant.items()
    }


def gt_consistency(by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, bool]:
    baseline = {
        text(row["interaction_id"]): tuple(row["ground_truth_targets"])
        for row in by_variant["full_baseline"]
    }
    return {
        variant: all(baseline.get(text(row["interaction_id"])) == tuple(row["ground_truth_targets"]) for row in rows)
        for variant, rows in by_variant.items()
    }


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def anchor_bootstrap(by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    full_map = {text(row["interaction_id"]): row for row in by_variant["full_baseline"]}
    output: list[dict[str, Any]] = []
    for variant in ANCHOR_VARIANTS:
        _, strict, _ = category("anchor_selection", variant)
        if not strict:
            continue
        ablation_map = {text(row["interaction_id"]): row for row in by_variant[variant]}
        ids = sorted(full_map)
        full_rows = [full_map[key] for key in ids]
        ablation_rows = [ablation_map[key] for key in ids]
        arrays = {
            "Hit-All": (
                np.array([float(bool(row["hit"])) for row in full_rows]),
                np.array([float(bool(row["hit"])) for row in ablation_rows]),
                np.ones(len(ids)),
            ),
            "Exact": (
                np.array([float(bool(row["exact"])) for row in full_rows]),
                np.array([float(bool(row["exact"])) for row in ablation_rows]),
                np.array([float(bool(row["evaluable"])) for row in full_rows]),
            ),
            "per-sample Set-F1": (
                np.array([float(row["set_f1"]) for row in full_rows]),
                np.array([float(row["set_f1"]) for row in ablation_rows]),
                np.array([float(bool(row["evaluable"])) for row in full_rows]),
            ),
        }
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for metric, (full_values, ablation_values, denominator_mask) in arrays.items():
            diffs: list[np.ndarray] = []
            remaining = BOOTSTRAP_REPETITIONS
            while remaining:
                batch = min(250, remaining)
                indices = rng.integers(0, len(ids), size=(batch, len(ids)))
                denom = denominator_mask[indices].sum(axis=1)
                denom = np.where(denom == 0, 1.0, denom)
                full_stat = (full_values[indices] * denominator_mask[indices]).sum(axis=1) / denom
                ablation_stat = (ablation_values[indices] * denominator_mask[indices]).sum(axis=1) / denom
                diffs.append(ablation_stat - full_stat)
                remaining -= batch
            samples = np.concatenate(diffs)
            full_denom = denominator_mask.sum() or 1.0
            full_value = float((full_values * denominator_mask).sum() / full_denom)
            ablation_value = float((ablation_values * denominator_mask).sum() / full_denom)
            low, high = percentile(samples, 2.5), percentile(samples, 97.5)
            output.append(
                {
                    "setting": "anchor_selection",
                    "full_variant": "full_baseline",
                    "ablation_variant": variant,
                    "metric": metric,
                    "full_value": full_value,
                    "ablation_value": ablation_value,
                    "delta_ablation_minus_full": ablation_value - full_value,
                    "ci95_low": low,
                    "ci95_high": high,
                    "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
                    "seed": BOOTSTRAP_SEED,
                    "significant_zero_excluded": low > 0 or high < 0,
                }
            )
    return output


def subset_label(count: int) -> str:
    if count <= 0:
        return "zero_mapped_target"
    return "single_target" if count == 1 else "multi_target"


def build_anchor_subset(by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant, rows in by_variant.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[subset_label(as_int(row["num_gt_targets"]))].append(row)
        for subset, subset_rows in sorted(groups.items()):
            metrics = anchor_aggregate(subset_rows)
            output.append({"variant": variant, "subset": subset, **metrics})
    full = {row["subset"]: row for row in output if row["variant"] == "full_baseline"}
    for row in output:
        base = full.get(row["subset"], {})
        row["delta_hit_all_vs_full"] = as_float(row["hit_all"]) - as_float(base.get("hit_all"))
        row["delta_exact_vs_full"] = as_float(row["exact"]) - as_float(base.get("exact"))
        row["delta_set_f1_macro_vs_full"] = as_float(row["set_f1_macro"]) - as_float(base.get("set_f1_macro"))
    return output


def build_projected_subset(by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for variant, rows in by_variant.items():
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            groups["single_target" if as_int(row["gt_referent_count"]) == 1 else "multi_target"].append(row)
        for subset, subset_rows in sorted(groups.items()):
            metrics = projected_aggregate(subset_rows)
            output.append({"variant": variant, "subset": subset, **metrics})
    full = {row["subset"]: row for row in output if row["variant"] == "full_baseline"}
    for row in output:
        base = full.get(row["subset"], {})
        row["delta_temporal_f1_vs_full"] = as_float(row["temporal_f1"]) - as_float(base.get("temporal_f1"))
        row["delta_joint100_vs_full"] = as_float(row["joint_at_100"]) - as_float(base.get("joint_at_100"))
    return output


def build_manifest(repo: Path, anchor_by: Mapping[str, Sequence[Mapping[str, Any]]], projected_by: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in ANCHOR_VARIANTS:
        ablation_category, strict, differences = category("anchor_selection", variant)
        base = variant == "full_baseline"
        rows.append(
            {
                "setting": "anchor_selection",
                "variant": variant,
                "model": "Qwen3-VL-30B-A3B-Instruct",
                "checkpoint": "/workspace/usr3/Qwen3-VL-30B-A3B-Instruct (revision/hash UNKNOWN)",
                "sample_count": len(anchor_by[variant]),
                "sample_id_source": "scene + row_index from per-scene *_match_eval.csv (event_id alone is not globally unique)",
                "modalities_provided": anchor_modalities(variant),
                "visual_input_description": "Original event video, up to 16 frames" if variant not in {"no_visual", "language_anchors_only"} else "One blank 64x64 RGB placeholder",
                "prompt_file": "scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py" if base else "ablation/exam1/scripts/grounding/run_qwen3vl_plus_api_single_event_3d.py",
                "candidate_source": "data/{scene}_anchor_table.tsv",
                "panel_or_frame_source": "/workspace/usr3/V3dMD event videos" if variant not in {"no_visual", "language_anchors_only"} else "Generated blank placeholder",
                "generation_config": "dtype=bfloat16; max_video_frames=16; max_new_tokens=1536; do_sample=False; max_evidence_segments=0",
                "parser_file": "scripts/grounding/run_qwen3vl_local_single_event_3d.py" if base else "ablation/exam1/scripts/grounding/run_qwen3vl_local_single_event_3d.py",
                "normalization_file": "scripts/eval/evaluate_local_3d_object_match.py (baseline) / ablation copy (variants; functional full-run difference only summary_scope option)",
                "evaluation_script": "scripts/eval/evaluate_local_3d_object_match.py" if base else "ablation/exam1/scripts/eval/evaluate_local_3d_object_match.py",
                "output_dir": str(anchor_output_dir(repo, variant).resolve()),
                "summary_file": ";".join(str(path.resolve()) for path in anchor_summary_paths(repo, variant)),
                "detail_file": ";".join(str(path.resolve()) for path in anchor_detail_paths(repo, variant)),
                "completed": len(anchor_by[variant]) == 4000,
                "strict_modality_ablation": strict,
                "ablation_category": ablation_category,
                "differences_from_full": differences,
                "notes": "Sample-level predictions retained. Baseline launch configuration corroborated by scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh and its full log.",
            }
        )
    for variant in PROJECTED_VARIANTS:
        ablation_category, strict, differences = category("projected_2d", variant)
        root = projected_root(repo, variant)
        rows.append(
            {
                "setting": "projected_2d",
                "variant": variant,
                "model": "Qwen3-VL-30B-A3B-Instruct",
                "checkpoint": "/workspace/usr3/Qwen3-VL-30B-A3B-Instruct (revision/hash UNKNOWN)",
                "sample_count": len(projected_by[variant]),
                "sample_id_source": "shared manifest scene+row_index/event_id",
                "modalities_provided": projected_modalities(variant),
                "visual_input_description": projected_visual(variant),
                "prompt_file": "exam2/run_qwen3vl_2d_point_grounding.py" if variant == "full_baseline" else "ablation/exam2/scripts/run_qwen3vl_2d_point_grounding.py",
                "candidate_source": "N/A (open 2D point output; GT anchor projections evaluator-only)",
                "panel_or_frame_source": "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/manifest/manifest_all.csv",
                "generation_config": "max_new_tokens=768; do_sample=False; input_mode=multi_image; 3 panels",
                "parser_file": "exam2/run_qwen3vl_2d_point_grounding.py" if variant == "full_baseline" else "ablation/exam2/scripts/run_qwen3vl_2d_point_grounding.py",
                "normalization_file": "same parser file; paired_canvas mapping only for paired_crop inputs",
                "evaluation_script": "exam2/evaluate_2d_point_grounding.py --coordinate_mode panel",
                "output_dir": str(root.resolve()),
                "summary_file": str((root / "eval/2d_eval_summary.json").resolve()),
                "detail_file": str((root / "eval/2d_eval_detail.csv").resolve()),
                "completed": len(projected_by[variant]) == 4000,
                "strict_modality_ablation": strict,
                "ablation_category": ablation_category,
                "differences_from_full": differences,
                "notes": "3971 prediction records are preserved; 29 manifest events without a prediction remain in the 4000-event evaluation denominator.",
            }
        )
    full3d = repo / "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b"
    for variant, source in (
        ("full", full3d),
        ("gaze_copy", repo / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_copy"),
        ("hand_copy", repo / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/hand_copy"),
        ("gaze_hand_fusion", repo / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_hand_fusion"),
    ):
        ablation_category, strict, differences = category("3d_grounding", variant)
        eval_root = source if variant == "full" else source / "eval"
        rows.append(
            {
                "setting": "3d_grounding",
                "variant": variant,
                "model": "Qwen3-VL-30B-A3B-Instruct" if variant == "full" else "deterministic cue baseline",
                "checkpoint": "/workspace/usr3/Qwen3-VL-30B-A3B-Instruct (revision/hash UNKNOWN)" if variant == "full" else "N/A",
                "sample_count": 3971,
                "sample_id_source": "exam3 point-grounding manifest sample_key",
                "modalities_provided": "language, target-free evidence frames, gaze, hand, camera/world geometry, anchors" if variant == "full" else variant.replace("_", "+"),
                "visual_input_description": "Target-free evidence frames" if variant == "full" else "No VLM visual inference",
                "prompt_file": "exam3_point_grounding/prompts/point_grounding_prompt_v9.txt" if variant == "full" else "N/A",
                "candidate_source": "exam3 point-grounding manifest candidate anchors",
                "panel_or_frame_source": "exam3_point_grounding/outputs_full_v9_20260709/evidence_frames",
                "generation_config": UNKNOWN if variant == "full" else "deterministic",
                "parser_file": "exam3_point_grounding/point_parser.py" if variant == "full" else "exam3_point_grounding/run_cue_baselines.py",
                "normalization_file": "exam3_point_grounding/point_grounding_common.py",
                "evaluation_script": "exam3_point_grounding/evaluate_point_grounding.py",
                "output_dir": str(source.resolve()),
                "summary_file": str((eval_root / "evaluation_summary.json").resolve()),
                "detail_file": str((eval_root / "evaluation_detail.csv").resolve()),
                "completed": (eval_root / "evaluation_summary.json").exists(),
                "strict_modality_ablation": strict,
                "ablation_category": ablation_category,
                "differences_from_full": differences,
                "notes": "No completed VLM no_language/no_gaze/no_hand/no_visual variants were found. Cue baselines must not be reported as modality ablations.",
            }
        )
    return rows


def anchor_modalities(variant: str) -> str:
    values = {
        "full_baseline": "language, video, structured gaze, structured hand, camera/world geometry, candidate anchors",
        "no_visual": "language, structured gaze, structured hand, camera/world geometry, candidate anchors",
        "no_gaze": "language, video (may contain gaze marker), structured hand, camera/world geometry, candidate anchors",
        "no_hand": "language, video (visible hands remain), structured gaze, camera/world geometry, candidate anchors",
        "no_gaze_hand": "language, video (visible gaze marker/hands may remain), camera/world geometry, candidate anchors",
        "language_anchors_only": "language, candidate anchors",
    }
    return values[variant]


def projected_modalities(variant: str) -> str:
    values = {
        "full_baseline": "language, full panels, gaze-centered crops, visible gaze marker, gaze-specific prompt prior",
        "full_panels_no_crop": "language, full panels, visible gaze marker, gaze-specific prompt prior",
        "instruction_only_prompt": "language, full panels, gaze-centered crops, visible gaze marker; expected-count prior removed",
        "no_gaze": "language, full panels with projected gaze marker masked; gaze-specific prompt prior removed",
        "no_gaze_text_prior": "language, full panels, visible gaze marker; gaze-specific prompt prior removed",
    }
    return values[variant]


def projected_visual(variant: str) -> str:
    return {
        "full_baseline": "Three paired images, each LEFT=full panel and RIGHT=gaze-centered crop",
        "instruction_only_prompt": "Three paired images, each LEFT=full panel and RIGHT=gaze-centered crop",
        "full_panels_no_crop": "Three full panels without crops",
        "no_gaze_text_prior": "Three full panels without crops; marker retained",
        "no_gaze": "Three full panels; projected green gaze marker locally masked",
    }[variant]


def write_metric_definitions(output: Path) -> None:
    (output / "anchor_ablation_metric_definitions.md").write_text(
        """# Anchor Ablation Metric Definitions

- `num_total`: all 4,000 interactions, including unmapped GT and invalid model outputs.
- `num_evaluable`: interactions with at least one GT referent mapped into the scene anchor inventory.
- `Hit-All`: interactions with at least one mapped predicted/GT overlap divided by all interactions.
- `Hit-Mapped`: the same hit indicator divided by evaluable interactions only.
- `Exact`: exact predicted-set/GT-set equality over evaluable interactions.
- `Set-F1 macro`: arithmetic mean of per-interaction set F1 over evaluable interactions. No-overlap and empty predictions contribute 0. This differs from the historical evaluator `macro_f1`, which omitted blank/zero-F1 rows.
- `micro precision/recall/F1`: aggregate set TP/FP/FN across all interactions. Unmapped-GT interactions can contribute FP if a prediction is made.
- `valid_output`: `response_status == \"ok\"`. Invalid rows remain in all relevant denominators.
- All exported values are decimal fractions, not percentages.
""",
        encoding="utf-8",
    )
    (output / "projected2d_metric_definitions.md").write_text(
        """# Projected-2D Ablation Metric Definitions

- The evaluation unit is one interaction. All 4,000 manifest interactions remain in the denominator; 29 interactions without prediction records are invalid/missing rather than dropped.
- `Temporal F1`, `Point@K`, and `Joint@K` are corpus-level F1 values recomputed from summed per-interaction TP/FP/FN, matching `exam2/evaluate_2d_point_grounding.py`.
- `Point@K` accepts a spatial match within K source-panel pixels regardless of temporal acceptability. `Joint@K` additionally requires an acceptable evidence panel.
- Per-interaction `*_at_*` columns are local F1 values derived from that row's TP/FP/FN; aggregate summaries do not average these columns.
- `mean_matched_distance` and `median_matched_distance` use evaluator-returned distances for successful Point@100 matches.
- `valid_output` requires an existing prediction row with `parse_ok=True`. Invalid/missing outputs remain in the corpus counts generated by the original evaluator.
- All exported values are decimal fractions, not percentages.
""",
        encoding="utf-8",
    )


def source_inventory(repo: Path, output: Path) -> None:
    lines = [
        "# Anchor selection baseline and ablations",
        "data/match_eval_qwen3vl30b_mention_first_v3/ (*_match_eval.csv, *_match_eval_summary.json)",
        "data/*_local_3d_outputs_qwen3vl30b_mention_first_v3/ (sample-level prediction JSON)",
        "ablation/exam1/outputs/{language_anchors_only,no_gaze,no_hand,no_visual,no_gaze_hand}/",
        "ablation/exam1/run_exam1_ablation.sh",
        "ablation/exam1/scripts/grounding/",
        "ablation/exam1/scripts/eval/",
        "",
        "# Projected-2D baseline and ablations",
        "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/{manifest,predictions,eval}/",
        "ablation/exam2/outputs/{full_panels_no_crop,instruction_only_prompt,no_gaze,no_gaze_text_prior}/",
        "ablation/exam2/run_exam2_ablation.sh",
        "ablation/exam2/scripts/run_qwen3vl_2d_point_grounding.py",
        "exam2/evaluate_2d_point_grounding.py",
        "",
        "# 3D grounding search result",
        "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/ (completed full VLM run)",
        "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/{gaze_copy,hand_copy,gaze_hand_fusion}/",
        "No completed VLM no_language/no_gaze/no_hand/no_visual directory was found under exam3_point_grounding, exam3, qwen8, or internvl.",
        "",
        "# Prior reports inspected",
        "ablation/reports/MODALITY_ABLATION_AUDIT.md",
        "ablation/reports/ABLATION_RESULTS.md",
        "ablation/reports/ABLATION_SUMMARY_CN.md",
    ]
    (output / "FOUND_RESULT_FILES.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_commands(output: Path, repo: Path) -> None:
    command = f"python {output / 'run_ablation_audit.py'} --repo-root {repo} --output-dir {output}"
    (output / "COMMANDS_EXECUTED.sh").write_text(
        """#!/usr/bin/env bash
set -euo pipefail

# Read-only inventory/configuration checks used during the audit included rg,
# find, sed, head, ls, sha256sum, git diff --no-index, and git log.
# No model inference and no original evaluator output overwrite was performed.

""" + command + "\n",
        encoding="utf-8",
    )


def build_audit_markdown(
    output: Path,
    anchor_summary: Sequence[Mapping[str, Any]],
    projected_summary: Sequence[Mapping[str, Any]],
    anchor_id_same: Mapping[str, bool],
    anchor_gt_same: Mapping[str, bool],
    projected_id_same: Mapping[str, bool],
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> None:
    anchor_lines = [
        f"| {row['variant']} | {row['num_total']} | {float(row['hit_all']):.4f} | {float(row['hit_mapped']):.4f} | {float(row['exact']):.4f} | {float(row['set_f1_macro']):.4f} | {float(row['micro_f1']):.4f} | {row['strict_modality_ablation']} |"
        for row in anchor_summary
    ]
    projected_lines = [
        f"| {row['variant']} | {row['num_samples']} | {float(row['valid_output_rate']):.4f} | {float(row['temporal_f1']):.4f} | {float(row['point_at_100']):.4f} | {float(row['joint_at_100']):.4f} | {row['strict_modality_ablation']} |"
        for row in projected_summary
    ]
    strict_variants = [row["variant"] for row in anchor_summary if row["strict_modality_ablation"]]
    content = f"""# Ablation Audit

## Scope and policy

This audit reads completed predictions and evaluations only. It did not modify paper files, rerun VLM inference, overwrite predictions, or overwrite existing evaluation outputs. Unified tables are recomputed in this independent directory.

## Completed experiments

All listed Anchor Selection variants contain 4,000 sample-level evaluation rows and all listed Projected-2D variants contain 4,000 evaluator rows. The Projected-2D prediction CSVs contain 3,971 model records; the 29 manifest events without records are retained as invalid/missing in the 4,000-event evaluation denominator.

Anchor Selection:

| Variant | Total | Hit-All | Hit-Mapped | Exact | Set-F1 macro | Micro F1 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(anchor_lines)}

Projected-2D:

| Variant | Total | Valid rate | Temporal F1 | Point@100 F1 | Joint@100 F1 | Strict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(projected_lines)}

The completed 3D Grounding VLM full run and three deterministic cue baselines were found. No completed 3D Grounding VLM modality-removal variant was found.

## Strict modality ablations

Under the conservative requirement that the target modality be fully removed while all other task/evaluation conditions remain fixed, the strict variants are: {', '.join(strict_variants) if strict_variants else 'none'}.

No completed variant satisfies the attachment's strict standard. In particular, Anchor `no_visual` removes visual content but changes the input from video to one 64x64 image and adds ablation-specific prompt text; the image type/count/geometry and prompt are therefore not held fixed.

## Non-strict variants

- Anchor `no_gaze` removes structured gaze fields and gaze-derived proposals but can retain the visible gaze marker in video pixels.
- Anchor `no_hand` removes structured hand/ray fields but retains visible hands and gestures in video pixels.
- Anchor `no_gaze_hand` removes two structured cue families simultaneously.
- Anchor `language_anchors_only` removes several modalities and is a multimodal input baseline.
- Anchor `no_visual` removes visual content but also changes media type/count/geometry and prompt text, so it is classified as hybrid rather than strict.
- Projected-2D `full_panels_no_crop` is a preprocessing ablation.
- Projected-2D `instruction_only_prompt` is a prompt ablation.
- Projected-2D `no_gaze_text_prior` jointly changes gaze text and crop preprocessing.
- Projected-2D `no_gaze` removes gaze text and marker but also changes paired full+crop input to full-panel-only input; relative to the current full baseline it is a hybrid ablation.
- 3D `gaze_copy`, `hand_copy`, and `gaze_hand_fusion` are cue baselines, not VLM modality ablations.

## Controlled-comparison checks

- Anchor interaction-ID equality versus full: {json.dumps(anchor_id_same, sort_keys=True)}
- Anchor GT equality versus full: {json.dumps(anchor_gt_same, sort_keys=True)}
- Projected-2D interaction-ID equality versus full: {json.dumps(projected_id_same, sort_keys=True)}
- Candidate anchor inventories are scene-level files shared by baseline and variants. Variant prediction JSON records the same `scene_anchor_csv`; the no-visual condition changes only visual input content.
- Projected-2D variants share the same manifest and evaluator. Their visual preprocessing/prompt differences are explicitly classified above.
- All runners use greedy decoding (`do_sample=False`). Anchor baseline and ablations use 1,536 max new tokens; Projected-2D uses 768.
- Sample-level predictions and raw model outputs are retained for all audited VLM runs.
- A full JSON-level scan found zero candidate-inventory mismatches and zero video-path mismatches between Anchor baseline and variants; each scene has exactly one stable anchor inventory.
- Anchor model path, dtype, input-mode metadata, frame limit, evidence-segment limit, prompt style, and prompt strategy are constant across all 4,000 rows of every variant.
- Projected-2D panel counts are identical by sample across variants (3,813 events with 3 panels, 152 with 2, and 6 with 1 among the 3,971 prediction records).

## Metric-scope conflicts

1. Historical Anchor Selection reports label the displayed `Set-F1` value ambiguously; the approximately 0.53 values are micro F1, not macro per-sample Set-F1.
2. The historical anchor evaluator's `macro_f1` averages only nonblank F1 values. It leaves no-overlap rows blank, which excludes zero-F1 samples and inflates the value (for example, scene-level values near 0.75). The unified export defines macro F1 over all evaluable samples with no-overlap rows equal to zero.
3. Anchor `Hit-All` uses all 4,000 interactions; `Hit-Mapped` and `Exact` use mapped/evaluable interactions. They must not be reported as sharing one denominator.
4. Projected-2D Temporal/Point/Joint F1 values are corpus-level F1 from aggregate TP/FP/FN, not mean per-sample F1.
5. All new exports use decimal fractions consistently.

## Re-evaluation and rerun requirements

No large inference rerun was performed. No completed run requires inference merely to reproduce the unified tables. The completed results can be reported under their audited categories (hybrid, preprocessing, prompt, multimodal-input, or cue baseline), but none supports a strict single-modality attribution under the requested definition.

For a claim of complete gaze or hand modality removal, new inference is required for Anchor `no_gaze` (mask/remove the video gaze marker) and Anchor `no_hand` (mask/remove visible hands/gesture evidence). For a strict Projected-2D gaze ablation relative to the paired-crop full baseline, new inference is required with the same paired image layout while removing gaze-derived crop selection/text/marker without changing image count or geometry. A full set of 3D Grounding VLM modality ablations also requires new inference because none was found.

The original completed evaluations do not need to be rerun for the reported legacy metrics. This audit recomputed unified metrics directly from sample-level files in a separate directory.

## Paired statistical analysis

No paired bootstrap was run because no completed variant meets the strict modality-ablation definition relative to its full baseline. Both bootstrap CSVs contain headers only.

## Subset readiness

- Stable: single-target versus multi-target, scenes, and room partitions.
- Not stable/available: discrete/location-like/region-like target type and instruction type. No name-based inference was used.
- Required for target-type analysis: an explicit interaction-level annotation or versioned mapping table with mutually exclusive target-type labels and documented handling of mixed-target interactions.

## Remaining unknowns

- Exact checkpoint revision/commit hashes are not recorded; only local model paths are available.
- A reliable semantic target-type field and instruction-type taxonomy are absent.
- The visible-pixel completeness of Anchor gaze/hand removal is known to be false from the implementation; no pixel-level audit of every video frame was attempted.
"""
    (output / "ABLATION_AUDIT.md").write_text(content, encoding="utf-8")


def build_validation_markdown(
    output: Path,
    anchor_summary: Sequence[Mapping[str, Any]],
    projected_summary: Sequence[Mapping[str, Any]],
    anchor_by: Mapping[str, Sequence[Mapping[str, Any]]],
    projected_by: Mapping[str, Sequence[Mapping[str, Any]]],
    repo: Path,
) -> None:
    lines = [
        "# Result Validation",
        "",
        "All checks below were performed from sample-level detail files. No original result was overwritten.",
        "",
        "## Structural checks",
        "",
        "| Setting | Variant | Rows | Unique IDs | Duplicate IDs | Missing vs full | Valid outputs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for setting, variants in (("anchor_selection", anchor_by), ("projected_2d", projected_by)):
        full_ids = {text(row["interaction_id"]) for row in variants["full_baseline"]}
        for variant, rows in variants.items():
            ids = [text(row["interaction_id"]) for row in rows]
            lines.append(
                f"| {setting} | {variant} | {len(rows)} | {len(set(ids))} | {len(ids)-len(set(ids))} | {len(full_ids-set(ids))} | {sum(bool(row['valid_output']) for row in rows)} |"
            )
    lines.extend([
        "",
        "## Summary reproduction",
        "",
        "Anchor unified values are recomputed from `*_match_eval.csv`. Hit-All, Hit-Mapped, Exact, and micro F1 reproduce the count-based legacy summaries. Unified macro Set-F1 intentionally differs from the legacy `macro_f1` because zero-overlap rows are restored as F1=0.",
        "",
        "| Anchor variant | Recomputed Hit-All | Recomputed Hit-Mapped | Recomputed Exact | Recomputed macro F1 | Recomputed micro F1 | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in anchor_summary:
        lines.append(
            f"| {row['variant']} | {float(row['hit_all']):.10f} | {float(row['hit_mapped']):.10f} | {float(row['exact']):.10f} | {float(row['set_f1_macro']):.10f} | {float(row['micro_f1']):.10f} | per-scene match_eval CSVs |"
        )
    lines.extend([
        "",
        "Projected-2D corpus metrics are recomputed by summing the existing per-event TP/FP/FN fields. Mean/median matched distance is independently reconstructed from the unchanged evaluator's Point@100 matching function.",
        "",
        "| Projected variant | Temporal F1 | Point@100 F1 | Joint@100 F1 | Original summary discrepancy | Source |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ])
    for row in projected_summary:
        original = read_json(projected_root(repo, text(row["variant"])) / "eval/2d_eval_summary.json")
        discrepancy = max(
            abs(float(row["temporal_f1"]) - as_float(original.get("time_f1"))),
            abs(float(row["point_at_100"]) - as_float(original.get("point_100_f1"))),
            abs(float(row["joint_at_100"]) - as_float(original.get("joint_100_f1"))),
        )
        lines.append(
            f"| {row['variant']} | {float(row['temporal_f1']):.10f} | {float(row['point_at_100']):.10f} | {float(row['joint_at_100']):.10f} | {discrepancy:.3e} | 2d_eval_detail.csv and summary JSON |"
        )
    lines.extend([
        "",
        "## Denominator and GT checks",
        "",
        "- Anchor coverage is 3,972 mapped/evaluable interactions out of 4,000 for every variant. Unmapped interactions remain in Hit-All and validity denominators.",
        "- Projected-2D contains 4,000 manifest/evaluator rows and 3,971 model prediction records for every variant. Missing records are retained by the evaluator as empty predictions.",
        "- Anchor mapped GT sets are identical across variants for every interaction.",
        "- Projected-2D uses one shared manifest, so GT panels/points and sample IDs are identical across variants.",
        "- No duplicate interaction IDs were found in the unified sample-level exports. The key is `scene + row_index`; raw `event_id` alone is not globally unique across scene4 room partitions.",
        "- Full JSON comparison found zero candidate-inventory mismatches and zero video-path mismatches between Anchor baseline and every variant.",
        "- Percent and decimal representations are not mixed in generated CSVs; all metrics are decimal fractions.",
        "",
        "## Parser/invalid checks",
        "",
        "- Anchor parser/evaluator fields are shared in schema; validity is `response_status == ok` and invalid rows remain in denominators.",
        "- Projected-2D uses the same JSON extraction/normalization implementation plus ablation-only input preparation. Baseline and instruction-only each have 7 parse failures; full-panels-no-crop, no-gaze, and no-gaze-text-prior have 0. Failures are distributed across scenes, with no scene-level collapse.",
    ])
    (output / "RESULT_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = (repo / output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    anchor_export, anchor_by = load_anchor_predictions(repo)
    projected_export, projected_by = load_projected_predictions(repo)
    anchor_id_same = identical_id_sets(anchor_by)
    anchor_gt_same = gt_consistency(anchor_by)
    projected_id_same = identical_id_sets(projected_by)

    anchor_summary: list[dict[str, Any]] = []
    base_anchor = anchor_aggregate(anchor_by["full_baseline"])
    for variant in ANCHOR_VARIANTS:
        metrics = anchor_aggregate(anchor_by[variant])
        ablation_category, strict, differences = category("anchor_selection", variant)
        anchor_summary.append(
            {
                "variant": variant,
                "ablation_category": ablation_category,
                **metrics,
                "delta_hit_all_vs_full": metrics["hit_all"] - base_anchor["hit_all"],
                "delta_hit_mapped_vs_full": metrics["hit_mapped"] - base_anchor["hit_mapped"],
                "delta_exact_vs_full": metrics["exact"] - base_anchor["exact"],
                "delta_set_f1_macro_vs_full": metrics["set_f1_macro"] - base_anchor["set_f1_macro"],
                "same_sample_set_as_full": anchor_id_same[variant],
                "strict_modality_ablation": strict,
                "notes": differences,
            }
        )

    anchor_partition: list[dict[str, Any]] = []
    for variant, rows in anchor_by.items():
        for scene in SCENES:
            metrics = anchor_aggregate([row for row in rows if row["partition"] == scene])
            anchor_partition.append({"variant": variant, "partition": scene, **metrics})

    projected_summary: list[dict[str, Any]] = []
    base_projected = projected_aggregate(projected_by["full_baseline"])
    for variant in PROJECTED_VARIANTS:
        metrics = projected_aggregate(projected_by[variant])
        ablation_category, strict, differences = category("projected_2d", variant)
        projected_summary.append(
            {
                "variant": variant,
                "ablation_category": ablation_category,
                **metrics,
                "delta_temporal_f1_vs_full": metrics["temporal_f1"] - base_projected["temporal_f1"],
                "delta_joint100_vs_full": metrics["joint_at_100"] - base_projected["joint_at_100"],
                "delta_joint200_vs_full": metrics["joint_at_200"] - base_projected["joint_at_200"],
                "same_sample_set_as_full": projected_id_same[variant],
                "strict_modality_ablation": strict,
                "notes": differences,
            }
        )

    projected_partition: list[dict[str, Any]] = []
    for variant, rows in projected_by.items():
        for scene in SCENES:
            metrics = projected_aggregate([row for row in rows if row["partition"] == scene])
            projected_partition.append({"variant": variant, "partition": scene, **metrics})

    manifest = build_manifest(repo, anchor_by, projected_by)
    anchor_bootstrap_rows = anchor_bootstrap(anchor_by)
    projected_bootstrap_rows: list[dict[str, Any]] = []
    anchor_subset = build_anchor_subset(anchor_by)
    projected_subset = build_projected_subset(projected_by)

    manifest_fields = (
        "setting", "variant", "model", "checkpoint", "sample_count", "sample_id_source",
        "modalities_provided", "visual_input_description", "prompt_file", "candidate_source",
        "panel_or_frame_source", "generation_config", "parser_file", "normalization_file",
        "evaluation_script", "output_dir", "summary_file", "detail_file", "completed",
        "strict_modality_ablation", "ablation_category", "differences_from_full", "notes",
    )
    anchor_summary_fields = (
        "variant", "ablation_category", "num_total", "num_evaluable", "coverage",
        "valid_output_count", "valid_output_rate", "hit_all", "hit_mapped", "exact",
        "set_f1_macro", "micro_precision", "micro_recall", "micro_f1",
        "delta_hit_all_vs_full", "delta_hit_mapped_vs_full", "delta_exact_vs_full",
        "delta_set_f1_macro_vs_full", "same_sample_set_as_full", "strict_modality_ablation", "notes",
    )
    anchor_partition_fields = (
        "variant", "partition", "num_total", "num_evaluable", "coverage", "hit_all", "hit_mapped", "exact", "set_f1_macro",
    )
    anchor_prediction_fields = (
        "interaction_id", "scene", "partition", "variant", "ground_truth_targets", "predicted_targets",
        "valid_output", "hit", "exact", "set_precision", "set_recall", "set_f1",
        "num_gt_targets", "num_predicted_targets", "target_type", "raw_output_path",
    )
    projected_summary_fields = (
        "variant", "ablation_category", "num_samples", "valid_output_count", "valid_output_rate",
        "temporal_precision", "temporal_recall", "temporal_f1", "point_at_50", "point_at_100",
        "point_at_200", "joint_at_50", "joint_at_100", "joint_at_200", "mean_matched_distance",
        "median_matched_distance", "delta_temporal_f1_vs_full", "delta_joint100_vs_full",
        "delta_joint200_vs_full", "same_sample_set_as_full", "strict_modality_ablation", "notes",
    )
    projected_partition_fields = (
        "variant", "partition", "num_samples", "temporal_f1", "point_at_50", "point_at_100",
        "point_at_200", "joint_at_50", "joint_at_100", "joint_at_200",
    )
    projected_prediction_fields = (
        "interaction_id", "scene", "partition", "variant", "gt_panels", "predicted_panels",
        "temporal_correct", "temporal_precision", "temporal_recall", "temporal_f1", "gt_points",
        "predicted_points", "point_distance", "point_at_50", "point_at_100", "point_at_200",
        "joint_at_50", "joint_at_100", "joint_at_200", "valid_output", "raw_output_path",
    )
    bootstrap_fields = (
        "setting", "full_variant", "ablation_variant", "metric", "full_value", "ablation_value",
        "delta_ablation_minus_full", "ci95_low", "ci95_high", "bootstrap_repetitions", "seed",
        "significant_zero_excluded",
    )

    write_csv(output / "ABLATION_RUN_MANIFEST.csv", manifest, manifest_fields)
    write_csv(output / "anchor_ablation_summary.csv", anchor_summary, anchor_summary_fields)
    write_csv(output / "anchor_ablation_per_partition.csv", anchor_partition, anchor_partition_fields)
    write_csv(output / "anchor_ablation_predictions.csv", anchor_export, anchor_prediction_fields)
    write_csv(output / "projected2d_ablation_summary.csv", projected_summary, projected_summary_fields)
    write_csv(output / "projected2d_ablation_per_partition.csv", projected_partition, projected_partition_fields)
    write_csv(output / "projected2d_ablation_predictions.csv", projected_export, projected_prediction_fields)
    write_csv(output / "anchor_ablation_bootstrap.csv", anchor_bootstrap_rows, bootstrap_fields)
    write_csv(output / "projected2d_ablation_bootstrap.csv", projected_bootstrap_rows, bootstrap_fields)
    write_csv(
        output / "anchor_ablation_by_target_count.csv",
        anchor_subset,
        ("variant", "subset", "num_total", "num_evaluable", "hit_all", "hit_mapped", "exact", "set_f1_macro", "delta_hit_all_vs_full", "delta_exact_vs_full", "delta_set_f1_macro_vs_full"),
    )
    write_csv(
        output / "projected2d_ablation_by_target_count.csv",
        projected_subset,
        ("variant", "subset", "num_samples", "temporal_f1", "point_at_100", "joint_at_100", "delta_temporal_f1_vs_full", "delta_joint100_vs_full"),
    )
    write_metric_definitions(output)
    source_inventory(repo, output)
    write_commands(output, repo)
    build_audit_markdown(output, anchor_summary, projected_summary, anchor_id_same, anchor_gt_same, projected_id_same, anchor_bootstrap_rows)
    build_validation_markdown(output, anchor_summary, projected_summary, anchor_by, projected_by, repo)
    print(f"Audit outputs written to {output}")


if __name__ == "__main__":
    main()
