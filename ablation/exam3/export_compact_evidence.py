#!/usr/bin/env python3
"""Export compact, auditable Experiment 3 ablation evidence without raw prompts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


VARIANTS = ("no_visual", "no_gaze", "no_hand", "no_gaze_hand", "no_instruction")
MODEL_NAME = "Qwen3-VL-30B-A3B-Instruct"
INPUT_MASKS = {
    "no_visual": "remove image tensors and image paths",
    "no_gaze": "remove gaze telemetry, hypotheses, distances, and gaze-derived selection metadata",
    "no_hand": "remove hand telemetry, hypotheses, distances, and hand-derived selection metadata",
    "no_gaze_hand": "remove both gaze and hand cue families and their derived metadata",
    "no_instruction": "remove event instruction and utterance values while retaining task instructions",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Mapping[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(fields or (rows[0].keys() if rows else ()))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1"}


def unified_id(row: Mapping[str, Any]) -> str:
    return f"{row['scene']}::{int(row['row_index'])}"


def digest(values: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def split_ids(value: str) -> List[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_config(path: Path) -> Dict[str, str]:
    config: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            config[key] = value
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_root", default="ablation/exam3/outputs_qwen3vl30b_v9_input_mask_v3_full")
    parser.add_argument("--gt_manifest", default="exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv")
    parser.add_argument("--summary_dir", default="ablation/exam3/reports/full_v3")
    parser.add_argument("--evidence_dir", default="paper_experiment_evidence/ablation/experiment3_qwen30b")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    output_root = (root / args.output_root).resolve()
    summary_dir = (root / args.summary_dir).resolve()
    evidence_dir = (root / args.evidence_dir).resolve()
    gt_path = (root / args.gt_manifest).resolve()
    evaluator_path = root / "exam3_point_grounding/evaluate_point_grounding.py"
    runner_path = root / "exam3_point_grounding/run_qwen3vl_point_grounding.py"
    prompt_path = root / "exam3_point_grounding/prompts/qwen3vl_point_grounding.md"
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    evaluator_sha = hashlib.sha256(evaluator_path.read_bytes()).hexdigest()
    runner_sha = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    evaluator_version = f"git:{source_commit};sha256:{evaluator_sha[:16]}"

    gt_rows = read_csv(gt_path)
    gt_by_id = {unified_id(row): row for row in gt_rows}
    if len(gt_rows) != 4000 or len(gt_by_id) != 4000:
        raise RuntimeError(f"expected 4000 unique GT rows, found {len(gt_rows)}/{len(gt_by_id)}")

    input_audit = load_json(summary_dir / "input_audit.json")
    validation_rows = {row["variant"]: row for row in read_csv(summary_dir / "validation_summary.csv")}
    provenance_rows: List[Dict[str, Any]] = []
    invalid_rows: List[Dict[str, Any]] = []
    validation: Dict[str, Any] = {}

    fields = [
        "model_name", "experiment", "ablation_variant", "scene", "row_index", "unified_id", "event_id",
        "gt_anchor_ids", "gt_points", "predicted_point_hypotheses", "nearest_anchor_predictions",
        "valid_output", "parse_status", "invalid_reason", "single_or_multi_target",
        "tp", "fp", "fn", "anchor_set_precision", "anchor_set_recall", "anchor_set_f1", "exact",
        "margin_tp_0.5", "margin_fp_0.5", "margin_fn_0.5", "margin_f1_0.5",
        "margin_tp_1.0", "margin_fp_1.0", "margin_fn_1.0", "margin_f1_1.0",
        "margin_tp_2.0", "margin_fp_2.0", "margin_fn_2.0", "margin_f1_2.0",
        "scene_normalized_errors", "mean_scene_normalized_error", "evaluator_version_or_commit",
    ]

    expected_ids = set(gt_by_id)
    expected_hash = digest(expected_ids)
    gt_hash = digest(
        f"{key}:{gt_by_id[key]['gt_anchor_ids']}:{gt_by_id[key]['gt_points_json']}" for key in expected_ids
    )
    for variant in VARIANTS:
        variant_root = output_root / variant
        pred_rows = read_csv(variant_root / "predictions.csv")
        detail_rows = read_csv(variant_root / "eval/evaluation_detail.csv")
        pred_by_id = {unified_id(row): row for row in pred_rows}
        detail_by_id = {unified_id(row): row for row in detail_rows}
        actual_ids = set(pred_by_id)
        if len(pred_rows) != 4000 or len(pred_by_id) != 4000 or actual_ids != expected_ids:
            raise RuntimeError(f"{variant}: prediction sample set is incomplete or duplicated")
        if len(detail_rows) != 4000 or len(detail_by_id) != 4000 or set(detail_by_id) != expected_ids:
            raise RuntimeError(f"{variant}: evaluation detail sample set is incomplete or duplicated")

        compact_rows: List[Dict[str, Any]] = []
        for key in sorted(expected_ids, key=lambda value: (value.split("::")[0], int(value.split("::")[1]))):
            gt = gt_by_id[key]
            pred = pred_by_id[key]
            detail = detail_by_id[key]
            valid = truth(pred["parse_ok"])
            reason = pred.get("invalid_reason") or detail.get("invalid_reason") or ("parse_failure" if not valid else "")
            item: Dict[str, Any] = {
                "model_name": MODEL_NAME,
                "experiment": "experiment3",
                "ablation_variant": variant,
                "scene": gt["scene"],
                "row_index": int(gt["row_index"]),
                "unified_id": key,
                "event_id": gt["event_id"],
                "gt_anchor_ids": canonical(split_ids(gt["gt_anchor_ids"])),
                "gt_points": gt["gt_points_json"],
                "predicted_point_hypotheses": pred["pred_points_json"] if valid else "[]",
                "nearest_anchor_predictions": canonical(split_ids(detail["nearest_pred_anchor_ids"])),
                "valid_output": valid,
                "parse_status": "ok" if valid else "parse_failure",
                "invalid_reason": reason,
                "single_or_multi_target": "single" if int(gt["gt_count"]) == 1 else "multi",
                "tp": int(detail["set_tp"]),
                "fp": int(detail["set_fp"]),
                "fn": int(detail["set_fn"]),
                "anchor_set_precision": detail["anchor_set_precision"],
                "anchor_set_recall": detail["anchor_set_recall"],
                "anchor_set_f1": detail["anchor_set_f1"],
                "exact": truth(detail["anchor_set_exact"]),
                "scene_normalized_errors": detail["scene_normalized_errors_json"],
                "mean_scene_normalized_error": detail["mean_scene_normalized_error"],
                "evaluator_version_or_commit": evaluator_version,
            }
            for threshold, suffix in (("0.5", "0_5"), ("1.0", "1_0"), ("2.0", "2_0")):
                for metric in ("tp", "fp", "fn"):
                    item[f"margin_{metric}_{threshold}"] = int(detail[f"margin_{metric}_at_{suffix}"])
                item[f"margin_f1_{threshold}"] = detail[f"margin_f1_at_{suffix}"]
            compact_rows.append(item)
            if not valid:
                invalid_rows.append({
                    "model_name": MODEL_NAME,
                    "experiment": "experiment3",
                    "ablation_variant": variant,
                    "scene": gt["scene"],
                    "row_index": int(gt["row_index"]),
                    "unified_id": key,
                    "event_id": gt["event_id"],
                    "parse_status": "parse_failure",
                    "invalid_reason": reason,
                    "denominator_policy": "retained as empty prediction",
                })

        evidence_path = evidence_dir / "predictions" / f"qwen3_vl_30b_{variant}.csv"
        write_csv(evidence_path, compact_rows, fields)
        config = parse_config(variant_root / "run_config.txt")
        summary = load_json(variant_root / "eval/evaluation_summary.json")["overall"]
        audit = input_audit[variant]
        recorded_validation = validation_rows[variant]
        sample_hash = digest(row["unified_id"] for row in compact_rows)
        invalid_count = sum(not truth(row["valid_output"]) for row in compact_rows)
        set_tp, set_fp, set_fn = (sum(int(row[name]) for row in compact_rows) for name in ("tp", "fp", "fn"))
        set_precision, set_recall, set_f1 = prf(set_tp, set_fp, set_fn)
        exact_rate = sum(truth(row["exact"]) for row in compact_rows) / len(compact_rows)
        margin_f1 = {}
        for threshold in ("0.5", "1.0", "2.0"):
            tp, fp, fn = (
                sum(int(row[f"margin_{metric}_{threshold}"]) for row in compact_rows)
                for metric in ("tp", "fp", "fn")
            )
            margin_f1[threshold] = prf(tp, fp, fn)[2]
        checks = {
            "row_count_4000": len(compact_rows) == 4000,
            "unique_ids_4000": len({row["unified_id"] for row in compact_rows}) == 4000,
            "sample_set_matches_gt": sample_hash == expected_hash,
            "raw_prompt_mask_audit_pass": audit["failure_count"] == 0 and audit["checked_raw_outputs"] == 4000,
            "invalid_count_matches_evaluator": invalid_count == int(summary["invalid_output_count"]),
            "anchor_metrics_match_evaluator": all(abs(actual - float(summary[name])) < 1e-12 for actual, name in (
                (set_precision, "anchor_set_precision_micro"),
                (set_recall, "anchor_set_recall_micro"),
                (set_f1, "anchor_set_f1_micro"),
            )),
            "exact_rate_matches_evaluator": abs(exact_rate - float(summary["anchor_set_exact_rate"])) < 1e-12,
            "margin_metrics_match_evaluator": all(
                abs(margin_f1[threshold] - float(summary[f"margin_f1_at_{threshold.replace('.', '_')}"])) < 1e-12
                for threshold in ("0.5", "1.0", "2.0")
            ),
            "recorded_validation_pass": all(
                int(recorded_validation[name]) == 0 for name in (
                    "duplicate_id_count", "missing_id_count", "extra_id_count",
                    "variant_column_mismatch_count", "raw_prompt_failure_count",
                )
            ),
        }
        validation[variant] = {
            "checks": checks,
            "validation_pass": all(checks.values()),
            "sample_set_sha256": sample_hash,
            "gt_sha256": gt_hash,
            "row_count": len(compact_rows),
            "valid_count": len(compact_rows) - invalid_count,
            "invalid_count": invalid_count,
            "evidence_file": str(evidence_path.relative_to(root)),
        }
        provenance_rows.append({
            "experiment": "experiment3",
            "model_name": MODEL_NAME,
            "ablation_variant": variant,
            "status": "final",
            "checkpoint": config["model_name"],
            "input_protocol": "up to three frozen target-free evidence frames plus Unity telemetry",
            "input_mask": INPUT_MASKS[variant],
            "prompt_template": config["prompt_template"],
            "decoding": "greedy; do_sample=false",
            "max_new_tokens": config["max_new_tokens"],
            "dtype": config["dtype"],
            "manifest": config["manifest"],
            "gt_manifest": config["gt_manifest"],
            "evaluator": str(evaluator_path.relative_to(root)),
            "evaluator_version_or_commit": evaluator_version,
            "runner_sha256": runner_sha,
            "prompt_sha256": prompt_sha,
            "sample_set_sha256": sample_hash,
            "gt_sha256": gt_hash,
            "total_samples": summary["total_samples"],
            "valid_output_count": summary["valid_output_count"],
            "invalid_output_count": summary["invalid_output_count"],
            "source_output_directory": str(variant_root.relative_to(root)),
            "compact_evidence_file": str(evidence_path.relative_to(root)),
            "raw_outputs_exported": False,
        })

    if not all(item["validation_pass"] for item in validation.values()):
        raise RuntimeError(f"compact evidence validation failed: {validation}")
    write_csv(evidence_dir / "run_provenance.csv", provenance_rows)
    write_csv(
        evidence_dir / "invalid_outputs.csv",
        invalid_rows,
        ("model_name", "experiment", "ablation_variant", "scene", "row_index", "unified_id", "event_id",
         "parse_status", "invalid_reason", "denominator_policy"),
    )
    (evidence_dir / "compact_evidence_validation.json").write_text(
        json.dumps({"validation_pass": True, "variants": validation}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"variants": len(VARIANTS), "rows_per_variant": 4000, "invalid_outputs": len(invalid_rows),
                      "evidence_dir": str(evidence_dir)}, indent=2))


if __name__ == "__main__":
    main()
