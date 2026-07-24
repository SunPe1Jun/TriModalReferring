#!/usr/bin/env python3
"""Export compact evidence for the completed strict hand-input ablations.

The exporter reads final inference/evaluator artifacts only. It never rewrites
raw model outputs and never runs model inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


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
THRESHOLDS = (50, 100, 150, 200)
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def truth(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"true", "1", "yes"}


def split_labels(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def uid(scene: str, row_index: Any) -> str:
    return f"{scene}::{int(row_index)}"


def digest(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def as_int(value: Any) -> int:
    return int(float(value or 0))


def ordered_ids(rows: Mapping[str, Any]) -> List[str]:
    return sorted(rows, key=lambda value: (value.rsplit("::", 1)[0], int(value.rsplit("::", 1)[1])))


def load_exp1_eval(eval_dir: Path) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for scene in SCENES:
        for row in read_csv(eval_dir / f"{scene}_match_eval.csv"):
            key = uid(scene, row["row_index"])
            if key in result:
                raise RuntimeError(f"duplicate Exp.1 id: {key}")
            result[key] = {"scene": scene, **row}
    return result


def summarize_exp1(rows: Sequence[Mapping[str, Any]], variant: str, partition: str = "overall") -> Dict[str, Any]:
    evaluable = [row for row in rows if truth(row["evaluable"])]
    tp = sum(len(split_labels(str(row["true_positive_referents"]))) for row in rows)
    fp = sum(len(split_labels(str(row["false_positive_referents"]))) for row in rows)
    fn = sum(len(split_labels(str(row["false_negative_referents"]))) for row in rows)
    precision, recall, f1 = prf(tp, fp, fn)
    row_f1: List[float] = []
    for row in rows:
        value = row.get("set_f1")
        row_f1.append(float(value) if value not in (None, "") else 0.0)
    return {
        "experiment": "experiment1",
        "model_name": MODEL_NAME,
        "variant": variant,
        "partition": partition,
        "total_samples": len(rows),
        "valid_output_count": sum(str(row.get("response_status", "")).lower() == "ok" for row in rows),
        "invalid_output_count": sum(str(row.get("response_status", "")).lower() != "ok" for row in rows),
        "mapped_count": len(evaluable),
        "hit_all": sum(truth(row["match_success"]) for row in rows) / len(rows),
        "hit_mapped": sum(truth(row["match_success"]) for row in evaluable) / len(evaluable),
        "exact": sum(truth(row["exact_match"]) for row in rows) / len(rows),
        "macro_set_f1": statistics.fmean(row_f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def export_exp1(root: Path, evidence_dir: Path, commit: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    strict_root = root / "ablation/exam1/outputs_strict_hand_v1_full/no_hand_strict"
    baseline_eval = load_exp1_eval(root / "data/match_eval_qwen3vl30b_mention_first_v3")
    strict_eval = load_exp1_eval(strict_root / "eval")
    gt_rows = read_csv(root / "paper_experiment_evidence/predictions/exp1/qwen3_vl_30b.csv")
    gt_by_id = {row["unified_id"]: row for row in gt_rows}
    expected = set(gt_by_id)
    if len(expected) != 4000 or set(baseline_eval) != expected or set(strict_eval) != expected:
        raise RuntimeError("Exp.1 sample sets are not the same 4,000 unified ids")

    evaluator = root / "paper_experiment_evidence/evaluators/evaluate_local_3d_object_match.py"
    evaluator_version = f"git:{commit};sha256:{sha256(evaluator)[:16]}"
    fields = [
        "model_name", "experiment", "ablation_variant", "scene", "row_index", "unified_id", "event_id",
        "gt_anchor_ids", "predicted_anchor_ids", "gt_mapped", "gt_unmapped_labels", "valid_output",
        "parse_status", "invalid_reason", "single_or_multi_target", "tp", "fp", "fn", "set_f1",
        "exact", "hit", "evaluator_version_or_commit",
    ]
    compact: List[Dict[str, Any]] = []
    frame_audit: List[Dict[str, Any]] = []
    prompt_leaks: List[str] = []
    for key in ordered_ids(strict_eval):
        gt = gt_by_id[key]
        row = strict_eval[key]
        scene = row["scene"]
        prediction_path = strict_root / "predictions" / scene / f"row_{int(row['row_index'])}.json"
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        predicted = split_labels(row["predicted_referents_mapped"])
        gt_ids = json.loads(gt["gt_anchor_ids"])
        if sorted(split_labels(row["gt_referents_mapped"])) != sorted(gt_ids):
            raise RuntimeError(f"Exp.1 strict GT differs from final paper GT for {key}")
        tp = len(split_labels(row["true_positive_referents"]))
        fp = len(split_labels(row["false_positive_referents"]))
        fn = len(split_labels(row["false_negative_referents"]))
        _, _, set_f1 = prf(tp, fp, fn)
        valid = str(row["response_status"]).lower() == "ok"
        compact.append({
            "model_name": MODEL_NAME,
            "experiment": "experiment1",
            "ablation_variant": "no_hand_strict",
            "scene": scene,
            "row_index": int(row["row_index"]),
            "unified_id": key,
            "event_id": row["event_id"],
            "gt_anchor_ids": canonical(gt_ids),
            "predicted_anchor_ids": canonical(predicted),
            "gt_mapped": truth(row["evaluable"]),
            "gt_unmapped_labels": canonical(split_labels(row["gt_referents_unmapped"])),
            "valid_output": valid,
            "parse_status": "ok" if valid else (row["response_status"] or "invalid_output"),
            "invalid_reason": "" if valid else (row["response_status"] or "invalid_output"),
            "single_or_multi_target": "single" if len(gt_ids) == 1 else "multi",
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "set_f1": set_f1,
            "exact": truth(row["exact_match"]),
            "hit": truth(row["match_success"]),
            "evaluator_version_or_commit": evaluator_version,
        })
        if payload.get("strict_hand_visual") is not True or "hand" not in payload.get("ablate_modalities", []):
            raise RuntimeError(f"Exp.1 strict hand flags missing for {key}")
        prompt = str(payload.get("prompt_text") or "").lower()
        structured_patterns = ("hand telemetry", "left hand ray", "right hand ray", "hand endpoint", "hand joints")
        if any(pattern in prompt for pattern in structured_patterns):
            prompt_leaks.append(key)
        audits = payload.get("hand_mask_audit") or []
        if not audits:
            raise RuntimeError(f"Exp.1 mask audit missing for {key}")
        for item in audits:
            boxes = item.get("boxes") or []
            frame_audit.append({
                "scene": scene,
                "row_index": int(row["row_index"]),
                "unified_id": key,
                "frame_index": int(item.get("frame_index") or 0),
                "status": item.get("status", ""),
                "tracked_sides": canonical(item.get("tracked_sides") or []),
                "masked_side_count": sum(box.get("status") == "masked" for box in boxes),
                "projected_joint_count": sum(as_int(box.get("joint_count")) for box in boxes),
                "mask_pixels": as_int(item.get("mask_pixels")),
                "mask_fraction": float(item.get("mask_fraction") or 0.0),
                "image_width": as_int((item.get("image_size") or [0, 0])[0]),
                "image_height": as_int((item.get("image_size") or [0, 0])[1]),
                "boxes_json": canonical(boxes),
                "video_time": item.get("video_time", ""),
                "target_sample_time": item.get("target_sample_time", ""),
                "nearest_sample_time": item.get("nearest_sample_time", ""),
                "sample_time_delta": item.get("sample_time_delta", ""),
                "mask_version": item.get("mask_version", ""),
                "mask_mode": item.get("mask_mode", ""),
                "mask_color_rgb": canonical(item.get("mask_color_rgb") or []),
            })

    out = evidence_dir / "experiment1_qwen30b_strict_hand"
    write_csv(out / "predictions/qwen3_vl_30b_no_hand_strict.csv", compact, fields)
    write_csv(out / "hand_mask_audit.csv", frame_audit)
    invalid = [row for row in compact if not truth(row["valid_output"])]
    write_csv(out / "invalid_outputs.csv", invalid, fields)

    summary_rows: List[Dict[str, Any]] = []
    for variant, source in (("full", baseline_eval), ("no_hand_strict", strict_eval)):
        values = list(source.values())
        summary_rows.append(summarize_exp1(values, variant))
        for scene in SCENES:
            summary_rows.append(summarize_exp1([row for row in values if row["scene"] == scene], variant, f"scene:{scene}"))
    write_csv(out / "evaluation_summary.csv", summary_rows)

    statuses = Counter(row["status"] for row in frame_audit)
    fractions = [float(row["mask_fraction"]) for row in frame_audit]
    mask_summary = {
        "mask_version": "hand_mask_v1",
        "sample_count": len(compact),
        "frame_count": len(frame_audit),
        "status_counts": dict(statuses),
        "masked_frame_count": statuses["masked"],
        "tracked_offscreen_frame_count": statuses["tracked_offscreen"],
        "no_tracked_hand_frame_count": statuses["no_tracked_hand"],
        "mean_mask_fraction": statistics.fmean(fractions),
        "median_mask_fraction": statistics.median(fractions),
        "max_mask_fraction": max(fractions),
        "frames_over_25_percent": sum(value > 0.25 for value in fractions),
        "frames_over_50_percent": sum(value > 0.50 for value in fractions),
        "prompt_structured_hand_leak_count": len(prompt_leaks),
        "prompt_structured_hand_leak_ids": prompt_leaks,
    }
    write_json(out / "hand_mask_summary.json", mask_summary)
    strict_summary = next(row for row in summary_rows if row["variant"] == "no_hand_strict" and row["partition"] == "overall")
    baseline_summary = next(row for row in summary_rows if row["variant"] == "full" and row["partition"] == "overall")
    checks = {
        "prediction_rows_4000": len(compact) == 4000,
        "unique_ids_4000": len({row["unified_id"] for row in compact}) == 4000,
        "sample_set_matches_final_gt": {row["unified_id"] for row in compact} == expected,
        "all_predictions_have_mask_audit": len({row["unified_id"] for row in frame_audit}) == 4000,
        "mask_statuses_valid": set(statuses) <= {"masked", "tracked_offscreen", "no_tracked_hand"},
        "structured_hand_prompt_leaks_zero": not prompt_leaks,
    }
    validation = {
        "validation_pass": all(checks.values()),
        "checks": checks,
        "sample_set_sha256": digest(expected),
        "prediction_count": len(compact),
        "valid_count": len(compact) - len(invalid),
        "invalid_count": len(invalid),
        "mask_summary": mask_summary,
    }
    write_json(out / "validation.json", validation)
    return [baseline_summary, strict_summary], validation


def summarize_exp2_rows(rows: Sequence[Mapping[str, Any]], variant: str, partition: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "experiment": "experiment2",
        "model_name": MODEL_NAME,
        "variant": variant,
        "partition": partition,
        "total_samples": len(rows),
        "valid_output_count": sum(truth(row["parse_ok"]) for row in rows),
        "invalid_output_count": sum(not truth(row["parse_ok"]) for row in rows),
        "gt_referents": sum(as_int(row["gt_referent_count"]) for row in rows),
        "predictions": sum(as_int(row["prediction_count"]) for row in rows),
    }
    metric_sources = [("temporal", "time_tp", "time_fp", "time_fn")]
    metric_sources.extend(
        (f"{family}_{threshold}", f"{family}_tp_{threshold}", f"{family}_fp_{threshold}", f"{family}_fn_{threshold}")
        for family in ("point", "joint") for threshold in THRESHOLDS
    )
    for family, tp_field, fp_field, fn_field in metric_sources:
        tp = sum(as_int(row[tp_field]) for row in rows)
        fp = sum(as_int(row[fp_field]) for row in rows)
        fn = sum(as_int(row[fn_field]) for row in rows)
        precision, recall, f1 = prf(tp, fp, fn)
        result.update({f"{family}_tp": tp, f"{family}_fp": fp, f"{family}_fn": fn,
                       f"{family}_precision": precision, f"{family}_recall": recall, f"{family}_f1": f1})
    return result


def export_exp2(root: Path, evidence_dir: Path, commit: str) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    strict_root = root / "ablation/exam2/outputs_strict_hand_v1_full/no_hand_strict"
    pred_rows = read_csv(strict_root / "predictions/qwen3vl_2d_predictions.csv")
    detail_rows = read_csv(strict_root / "eval/2d_eval_detail.csv")
    baseline_rows = read_csv(root / "paper_experiment_evidence/predictions/exp2/qwen3_vl_30b.csv")
    baseline_by_id = {row["unified_id"]: row for row in baseline_rows}
    pred_by_id = {uid(row["scene"], row["row_index"]): row for row in pred_rows}
    detail_by_id = {uid(row["scene"], row["row_index"]): row for row in detail_rows}
    expected = set(baseline_by_id)
    if len(pred_rows) != 4000 or len(pred_by_id) != 4000 or set(pred_by_id) != expected:
        raise RuntimeError("Exp.2 strict predictions do not contain the final 4,000 sample ids")
    if len(detail_rows) != 4000 or len(detail_by_id) != 4000 or set(detail_by_id) != expected:
        raise RuntimeError("Exp.2 strict evaluator detail does not contain the final 4,000 sample ids")

    evaluator = root / "paper_experiment_evidence/evaluators/evaluate_2d_point_grounding.py"
    evaluator_version = f"git:{commit};sha256:{sha256(evaluator)[:16]}"
    base_fields = list(baseline_rows[0])
    fields = base_fields[:2] + ["ablation_variant"] + base_fields[2:]
    compact: List[Dict[str, Any]] = []
    panel_audit: List[Dict[str, Any]] = []
    json_root = strict_root / "predictions/json"
    for key in ordered_ids(pred_by_id):
        gt = baseline_by_id[key]
        pred = pred_by_id[key]
        detail = detail_by_id[key]
        parsed = json.loads(pred["parsed_json"] or "{}") if truth(pred["parse_ok"]) else {}
        referents = parsed.get("referents") or []
        predicted_panels = [str(item.get("panel_id") or "") for item in referents if item.get("panel_id")]
        predicted_points = [
            {name: item.get(name) for name in ("panel_id", "coordinate_space", "x_norm", "y_norm")}
            for item in referents
        ]
        valid = truth(pred["parse_ok"])
        item: Dict[str, Any] = {
            "model_name": MODEL_NAME,
            "experiment": "experiment2",
            "ablation_variant": "no_hand_strict",
            "scene": pred["scene"],
            "row_index": int(pred["row_index"]),
            "unified_id": key,
            "event_id": pred["event_id"],
            "gt_targets": gt["gt_targets"],
            "gt_panel_ids": gt["gt_panel_ids"],
            "predicted_panel_ids": canonical(predicted_panels),
            "gt_points": gt["gt_points"],
            "predicted_points": canonical(predicted_points),
            "valid_output": valid,
            "parse_status": "ok" if valid else "parse_failure",
            "invalid_reason": pred.get("error_message") or detail.get("error_message") or ("parse_failure" if not valid else ""),
            "missing_prediction": False,
            "parse_failure": not valid,
            "single_or_multi_target": gt["single_or_multi_target"],
            "evaluator_version_or_commit": evaluator_version,
        }
        for threshold in THRESHOLDS:
            for family in ("point", "joint"):
                for metric in ("tp", "fp", "fn"):
                    item[f"{family}_{metric}_{threshold}"] = as_int(detail[f"{family}_{metric}_{threshold}"])
        for metric in ("tp", "fp", "fn"):
            item[f"time_{metric}"] = as_int(detail[f"time_{metric}"])
        compact.append(item)

        raw_path = json_root / f"{pred['scene']}_row_{int(pred['row_index'])}.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        panel_meta = raw.get("panel_meta") or []
        model_paths = [Path(path) for path in raw.get("model_input_images") or []]
        if not model_paths or any("hand_masked" not in path.parts or not path.is_file() for path in model_paths):
            raise RuntimeError(f"Exp.2 model input did not resolve to saved hand-masked panels for {key}")
        for panel in panel_meta:
            frame_path = Path(panel["frame_path"])
            panel_audit.append({
                "scene": pred["scene"],
                "row_index": int(pred["row_index"]),
                "unified_id": key,
                "panel_id": panel["panel_id"],
                "status": panel.get("hand_mask_status", ""),
                "mask_fraction": float(panel.get("hand_mask_fraction") or 0.0),
                "nearest_sample_time": panel.get("hand_mask_nearest_sample_time", ""),
                "target_sample_time": panel.get("json_sample_time", ""),
                "masked_file_exists": frame_path.is_file(),
                "model_input_uses_masked_path": "hand_masked" in frame_path.parts,
                "mask_version": "hand_mask_v1",
            })

    out = evidence_dir / "experiment2_qwen30b_strict_hand"
    write_csv(out / "predictions/qwen3_vl_30b_no_hand_strict.csv", compact, fields)
    write_csv(out / "hand_mask_audit.csv", panel_audit)
    invalid = [row for row in compact if not truth(row["valid_output"])]
    write_csv(out / "invalid_outputs.csv", invalid, fields)

    strict_summary_rows = [summarize_exp2_rows(detail_rows, "no_hand_strict", "overall")]
    for scene in SCENES:
        strict_summary_rows.append(summarize_exp2_rows(
            [row for row in detail_rows if row["scene"] == scene], "no_hand_strict", f"scene:{scene}"
        ))
    baseline_compact_detail: List[Dict[str, Any]] = []
    for row in baseline_rows:
        converted: Dict[str, Any] = {
            "scene": row["scene"], "parse_ok": row["valid_output"],
            "gt_referent_count": len(json.loads(row["gt_targets"])),
            "prediction_count": len(json.loads(row["predicted_panel_ids"])),
        }
        for metric in ("tp", "fp", "fn"):
            converted[f"time_{metric}"] = row[f"time_{metric}"]
        for threshold in THRESHOLDS:
            for family in ("point", "joint"):
                for metric in ("tp", "fp", "fn"):
                    converted[f"{family}_{metric}_{threshold}"] = row[f"{family}_{metric}_{threshold}"]
        baseline_compact_detail.append(converted)
    baseline_summary_rows = [summarize_exp2_rows(baseline_compact_detail, "full", "overall")]
    for scene in SCENES:
        baseline_summary_rows.append(summarize_exp2_rows(
            [row for row in baseline_compact_detail if row["scene"] == scene], "full", f"scene:{scene}"
        ))
    write_csv(out / "evaluation_summary.csv", baseline_summary_rows + strict_summary_rows)

    statuses = Counter(row["status"] for row in panel_audit)
    fractions = [float(row["mask_fraction"]) for row in panel_audit]
    mask_summary = {
        "mask_version": "hand_mask_v1",
        "sample_count": len(compact),
        "panel_count": len(panel_audit),
        "status_counts": dict(statuses),
        "masked_panel_count": statuses["masked"],
        "tracked_offscreen_panel_count": statuses["tracked_offscreen"],
        "no_tracked_hand_panel_count": statuses["no_tracked_hand"],
        "mean_mask_fraction": statistics.fmean(fractions),
        "median_mask_fraction": statistics.median(fractions),
        "max_mask_fraction": max(fractions),
        "panels_over_25_percent": sum(value > 0.25 for value in fractions),
        "panels_over_50_percent": sum(value > 0.50 for value in fractions),
        "all_saved_model_input_files_exist": all(truth(row["masked_file_exists"]) for row in panel_audit),
        "all_model_input_paths_are_masked": all(truth(row["model_input_uses_masked_path"]) for row in panel_audit),
    }
    write_json(out / "hand_mask_summary.json", mask_summary)
    checks = {
        "prediction_rows_4000": len(compact) == 4000,
        "unique_ids_4000": len({row["unified_id"] for row in compact}) == 4000,
        "sample_set_matches_final_gt": {row["unified_id"] for row in compact} == expected,
        "all_predictions_have_panel_audit": len({row["unified_id"] for row in panel_audit}) == 4000,
        "all_masked_input_files_exist": mask_summary["all_saved_model_input_files_exist"],
        "all_model_input_paths_are_masked": mask_summary["all_model_input_paths_are_masked"],
        "mask_statuses_valid": set(statuses) <= {"masked", "tracked_offscreen", "no_tracked_hand"},
    }
    validation = {
        "validation_pass": all(checks.values()),
        "checks": checks,
        "sample_set_sha256": digest(expected),
        "prediction_count": len(compact),
        "valid_count": len(compact) - len(invalid),
        "invalid_count": len(invalid),
        "mask_summary": mask_summary,
    }
    write_json(out / "validation.json", validation)
    return [baseline_summary_rows[0], strict_summary_rows[0]], validation, panel_audit


def wide_summary(exp1: Sequence[Mapping[str, Any]], exp2: Sequence[Mapping[str, Any]], root: Path) -> List[Dict[str, Any]]:
    fields = (
        "experiment", "model_name", "variant", "total_samples", "valid_output_count", "invalid_output_count",
        "hit_all", "hit_mapped", "exact", "macro_set_f1", "micro_precision", "micro_recall", "micro_f1",
        "temporal_f1", "point_100_f1", "joint_100_f1", "anchor_set_precision", "anchor_set_recall",
        "anchor_set_f1", "margin_f1_1_0", "margin_f1_2_0", "mean_scene_normalized_error",
    )
    rows = [{name: row.get(name, "") for name in fields} for row in (*exp1, *exp2)]
    exp3_rows = read_csv(root / "paper_experiment_evidence/ablation/experiment3_qwen30b_strict_hand/ablation_summary.csv")
    for variant in ("full", "no_hand_strict"):
        source = next(row for row in exp3_rows if row["variant"] == variant and row["partition"] == "overall")
        item = {name: "" for name in fields}
        item.update({
            "experiment": "experiment3", "model_name": MODEL_NAME, "variant": variant,
            "total_samples": source["total_samples"], "valid_output_count": source["valid_output_count"],
            "invalid_output_count": source["invalid_output_count"], "exact": source["anchor_set_exact_rate"],
            "anchor_set_precision": source["anchor_set_precision_micro"],
            "anchor_set_recall": source["anchor_set_recall_micro"], "anchor_set_f1": source["anchor_set_f1_micro"],
            "margin_f1_1_0": source["margin_f1_at_1_0"], "margin_f1_2_0": source["margin_f1_at_2_0"],
            "mean_scene_normalized_error": source["mean_scene_normalized_error"],
        })
        rows.append(item)
    return rows


def build_montage(root: Path, evidence_dir: Path, panel_audit: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    from PIL import Image, ImageDraw

    masked = sorted((row for row in panel_audit if row["status"] == "masked"), key=lambda row: float(row["mask_fraction"]))
    indexes = [int((len(masked) - 1) * quantile) for quantile in (0.10, 0.50, 0.90, 0.99, 1.00)]
    selected = [masked[index] for index in indexes]
    selected_rows: List[Dict[str, Any]] = []
    thumb_w, thumb_h, label_h = 360, 360, 28
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_number, row in enumerate(selected):
        scene, row_index, panel_id = row["scene"], int(row["row_index"]), row["panel_id"]
        raw_path = root / "ablation/exam2/outputs_strict_hand_v1_full/no_hand_strict/predictions/json" / f"{scene}_row_{row_index}.json"
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        panel = next(item for item in payload["panel_meta"] if item["panel_id"] == panel_id)
        original = Path(panel["source_frame_path"])
        masked_path = Path(panel["frame_path"])
        selected_rows.append({
            "scene": scene, "row_index": row_index, "unified_id": row["unified_id"], "panel_id": panel_id,
            "mask_fraction": row["mask_fraction"], "status": row["status"],
        })
        for column, (label, path) in enumerate((("original", original), ("model input", masked_path))):
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = column * thumb_w + (thumb_w - image.width) // 2
            y = row_number * (thumb_h + label_h) + label_h + (thumb_h - image.height) // 2
            canvas.paste(image, (x, y))
            draw.text((column * thumb_w + 6, row_number * (thumb_h + label_h) + 6),
                      f"{scene} row {row_index} {panel_id} {label} mask={float(row['mask_fraction']):.3f}", fill="black")
    output = evidence_dir / "qualitative_cases/strict_hand_mask_examples.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=70, optimize=True)
    write_csv(evidence_dir / "qualitative_cases/strict_hand_mask_examples.csv", selected_rows)
    return selected_rows


def write_provenance(root: Path, evidence_dir: Path, commit: str, exp1_summary: Sequence[Mapping[str, Any]],
                     exp2_summary: Sequence[Mapping[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    evaluator1 = root / "paper_experiment_evidence/evaluators/evaluate_local_3d_object_match.py"
    evaluator2 = root / "paper_experiment_evidence/evaluators/evaluate_2d_point_grounding.py"
    evaluator3 = root / "exam3_point_grounding/evaluate_point_grounding.py"
    common = {
        "model_name": MODEL_NAME,
        "status": "completed_input_occlusion_diagnostic",
        "mask_version": "hand_mask_v1",
        "decoding": "greedy; do_sample=false",
        "evaluator_version_or_commit": f"git:{commit}",
        "sample_set_sha256": digest(f"{scene}::{index}" for scene, count in (("scene1", 800), ("scene2", 800), ("scene3", 800), ("scene4_room1", 200), ("scene4_room2", 200), ("scene4_room3", 200), ("scene4_room4", 200), ("scene5", 800)) for index in range(count)),
        "strict_model_input_feasibility_verified": "True",
        "paper_ready_for_strict_visual_hand_claim": "False",
    }
    configs = [
        ("experiment1", "full", "video; max_video_frames=16", "hand fields retained; no image mask", "1536", "bfloat16", "data/match_eval_qwen3vl30b_mention_first_v3", evaluator1),
        ("experiment1", "no_hand_strict", "video; max_video_frames=16", "hand fields removed; projected hand mask applied in memory", "1536", "bfloat16", "ablation/exam1/outputs_strict_hand_v1_full/no_hand_strict", evaluator1),
        ("experiment2", "full", "multi_image; frozen panels", "no image mask", "768", "auto", "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10", evaluator2),
        ("experiment2", "no_hand_strict", "multi_image; frozen panels", "projected hand mask saved before processor", "768", "auto", "ablation/exam2/outputs_strict_hand_v1_full/no_hand_strict", evaluator2),
        ("experiment3", "full", "candidate-free measured point hypotheses", "no image mask", "512", "auto", "exam3_point_grounding/outputs_full_v9_20260709", evaluator3),
        ("experiment3", "no_hand_strict", "candidate-free measured point hypotheses", "hand fields removed; projected hand mask applied to frozen panels", "512", "auto", "ablation/exam3/outputs_qwen3vl30b_v9_strict_hand_v1_full/no_hand_strict", evaluator3),
    ]
    summary_lookup = {(row["experiment"], row["variant"]): row for row in (*exp1_summary, *exp2_summary)}
    exp3_summary = {row["variant"]: row for row in read_csv(evidence_dir / "experiment3_qwen30b_strict_hand/ablation_summary.csv") if row["partition"] == "overall"}
    for experiment, variant, protocol, mask, max_tokens, dtype, source, evaluator in configs:
        metrics = summary_lookup.get((experiment, variant), exp3_summary.get(variant, {}))
        row = dict(common)
        row.update({
            "experiment": experiment,
            "variant": variant,
            "input_protocol": protocol,
            "visual_mask": mask,
            "max_new_tokens": max_tokens,
            "dtype": dtype,
            "source_output_directory": source,
            "evaluator": str(evaluator.relative_to(root)),
            "total_samples": metrics.get("total_samples", ""),
            "valid_output_count": metrics.get("valid_output_count", ""),
            "invalid_output_count": metrics.get("invalid_output_count", ""),
        })
        rows.append(row)
    write_csv(evidence_dir / "strict_hand_run_provenance.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    evidence = root / "paper_experiment_evidence/ablation"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()

    exp1_summary, exp1_validation = export_exp1(root, evidence, commit)
    exp2_summary, exp2_validation, exp2_panels = export_exp2(root, evidence, commit)
    combined = wide_summary(exp1_summary, exp2_summary, root)
    write_csv(evidence / "strict_hand_ablation_summary.csv", combined)
    examples = build_montage(root, root / "paper_experiment_evidence", exp2_panels)
    write_provenance(root, evidence, commit, exp1_summary, exp2_summary)

    exp3_validation = json.loads((evidence / "experiment3_qwen30b_strict_hand/compact_evidence_validation.json").read_text())
    validation = {
        "validation_pass": bool(exp1_validation["validation_pass"] and exp2_validation["validation_pass"]
                                and exp3_validation["validation_pass"]),
        "strict_model_input_feasibility_verified": True,
        "paper_ready_for_strict_visual_hand_claim": False,
        "semantic_visual_audit_status": "failed",
        "semantic_visual_audit_reference": "paper_experiment_evidence/ablation/STRICT_HAND_ABLATION_AUDIT.md",
        "source_commit_before_export": commit,
        "raw_model_outputs_modified": False,
        "experiment1": exp1_validation,
        "experiment2": exp2_validation,
        "experiment3": exp3_validation,
        "qualitative_examples": examples,
    }
    write_json(evidence / "strict_hand_validation.json", validation)
    if not validation["validation_pass"]:
        raise RuntimeError("strict hand compact evidence validation failed")
    print(json.dumps({
        "validation_pass": True,
        "experiment1_rows": exp1_validation["prediction_count"],
        "experiment2_rows": exp2_validation["prediction_count"],
        "experiment3_rows": 4000,
        "summary": "paper_experiment_evidence/ablation/strict_hand_ablation_summary.csv",
    }, indent=2))


if __name__ == "__main__":
    main()
