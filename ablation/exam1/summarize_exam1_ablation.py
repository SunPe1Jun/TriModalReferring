#!/usr/bin/env python3
"""Summarize experiment-1 ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


FIELDS = (
    "variant",
    "scene",
    "total_gt_rows",
    "total_prediction_files",
    "response_status_ok_count",
    "match_count_overall",
    "overall_accuracy",
    "exact_match_accuracy_overall",
    "evaluable_row_count",
    "mapped_only_accuracy",
    "exact_match_accuracy_evaluable_only",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "delta_overall_accuracy_vs_baseline",
    "delta_micro_f1_vs_baseline",
    "source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exam1 ablation eval summaries.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_root", default="ablation/exam1/outputs")
    parser.add_argument("--report_dir", default="ablation/exam1/reports")
    parser.add_argument("--baseline_eval_dir", default="data/match_eval_qwen3vl30b_mention_first_v3")
    return parser.parse_args()


def safe_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def safe_div(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def infer_scene(path: Path) -> str:
    name = path.name
    return name.replace("_match_eval_summary.json", "")


def read_summary(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_files(variant: str, files: Sequence[Path], source: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    totals = {
        "total_gt_rows": 0,
        "total_prediction_files": 0,
        "response_status_ok_count": 0,
        "match_count_overall": 0,
        "exact_match_count_overall": 0,
        "evaluable_row_count": 0,
        "match_count_evaluable_only": 0,
        "exact_match_count_evaluable_only": 0,
        "set_true_positive_count": 0,
        "set_false_positive_count": 0,
        "set_false_negative_count": 0,
    }
    for path in sorted(files):
        payload = read_summary(path)
        scene = infer_scene(path)
        row = build_row(variant, scene, payload, source)
        rows.append(row)
        for key in totals:
            totals[key] += safe_int(payload.get(key))
    if files:
        rows.append(build_row(variant, "ALL", totals, source))
    return rows


def build_row(variant: str, scene: str, payload: Mapping[str, Any], source: str) -> Dict[str, Any]:
    total_gt = safe_int(payload.get("total_gt_rows"))
    evaluable = safe_int(payload.get("evaluable_row_count"))
    tp = safe_int(payload.get("set_true_positive_count"))
    fp = safe_int(payload.get("set_false_positive_count"))
    fn = safe_int(payload.get("set_false_negative_count"))
    micro_precision = safe_float(payload.get("micro_precision"))
    if micro_precision is None:
        micro_precision = safe_div(tp, tp + fp)
    micro_recall = safe_float(payload.get("micro_recall"))
    if micro_recall is None:
        micro_recall = safe_div(tp, tp + fn)
    micro_f1 = safe_float(payload.get("micro_f1"))
    if micro_f1 is None and micro_precision is not None and micro_recall is not None and micro_precision + micro_recall > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    return {
        "variant": variant,
        "scene": scene,
        "total_gt_rows": total_gt,
        "total_prediction_files": safe_int(payload.get("total_prediction_files")),
        "response_status_ok_count": safe_int(payload.get("response_status_ok_count")),
        "match_count_overall": safe_int(payload.get("match_count_overall")),
        "overall_accuracy": safe_float(payload.get("overall_accuracy")) if payload.get("overall_accuracy") is not None else safe_div(safe_int(payload.get("match_count_overall")), total_gt),
        "exact_match_accuracy_overall": safe_float(payload.get("exact_match_accuracy_overall")) if payload.get("exact_match_accuracy_overall") is not None else safe_div(safe_int(payload.get("exact_match_count_overall")), total_gt),
        "evaluable_row_count": evaluable,
        "mapped_only_accuracy": safe_float(payload.get("mapped_only_accuracy")) if payload.get("mapped_only_accuracy") is not None else safe_div(safe_int(payload.get("match_count_evaluable_only")), evaluable),
        "exact_match_accuracy_evaluable_only": safe_float(payload.get("exact_match_accuracy_evaluable_only")) if payload.get("exact_match_accuracy_evaluable_only") is not None else safe_div(safe_int(payload.get("exact_match_count_evaluable_only")), evaluable),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "delta_overall_accuracy_vs_baseline": "",
        "delta_micro_f1_vs_baseline": "",
        "source": source,
    }


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in FIELDS})


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    all_rows = [row for row in rows if row.get("scene") == "ALL"]
    lines = [
        "# Experiment 1 Modality Ablation Summary",
        "",
        "Baseline is reused from `data/match_eval_qwen3vl30b_mention_first_v3/`. Candidate anchors and evaluator are unchanged.",
        "",
        "| Variant | Predictions | Overall Acc | Delta Acc | Mapped Acc | Exact Set Acc | Micro F1 | Delta F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in all_rows:
        lines.append(
            "| {variant} | {pred} | {acc} | {dacc} | {mapped} | {exact} | {f1} | {df1} |".format(
                variant=row.get("variant", ""),
                pred=row.get("total_prediction_files", ""),
                acc=fmt(row.get("overall_accuracy")),
                dacc=fmt(row.get("delta_overall_accuracy_vs_baseline")),
                mapped=fmt(row.get("mapped_only_accuracy")),
                exact=fmt(row.get("exact_match_accuracy_evaluable_only")),
                f1=fmt(row.get("micro_f1")),
                df1=fmt(row.get("delta_micro_f1_vs_baseline")),
            )
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `no_gaze` hides structured gaze fields and gaze-derived sparse timeline proposals. If the source video contains a green gaze marker, the prompt tells the model to ignore it but the pixels are not edited.",
        "- `no_visual` is the clean visual-removal control because the model receives only a blank placeholder image.",
        "- `language_anchors_only` keeps the candidate anchor interface because removing anchors would change the closed-set task definition.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    report_dir = (repo_root / args.report_dir).resolve() if not Path(args.report_dir).is_absolute() else Path(args.report_dir)
    baseline_dir = (repo_root / args.baseline_eval_dir).resolve() if not Path(args.baseline_eval_dir).is_absolute() else Path(args.baseline_eval_dir)

    rows: List[Dict[str, Any]] = []
    baseline_files = sorted(baseline_dir.glob("*_match_eval_summary.json")) if baseline_dir.exists() else []
    rows.extend(summarize_files("full_baseline", baseline_files, str(baseline_dir)))

    for variant_dir in sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.exists() else []:
        files = sorted((variant_dir / "eval").glob("*_match_eval_summary.json"))
        if files:
            rows.extend(summarize_files(variant_dir.name, files, str(variant_dir)))

    baseline_all = next((row for row in rows if row.get("variant") == "full_baseline" and row.get("scene") == "ALL"), None)
    if baseline_all:
        base_acc = baseline_all.get("overall_accuracy")
        base_f1 = baseline_all.get("micro_f1")
        for row in rows:
            if row.get("scene") != "ALL" or row.get("variant") == "full_baseline":
                continue
            if isinstance(row.get("overall_accuracy"), float) and isinstance(base_acc, float):
                row["delta_overall_accuracy_vs_baseline"] = row["overall_accuracy"] - base_acc
            if isinstance(row.get("micro_f1"), float) and isinstance(base_f1, float):
                row["delta_micro_f1_vs_baseline"] = row["micro_f1"] - base_f1

    write_csv(report_dir / "exam1_ablation_summary.csv", rows)
    write_markdown(report_dir / "EXAM1_ABLATION_RESULTS.md", rows)
    print(f"Wrote {report_dir / 'exam1_ablation_summary.csv'}")
    print(f"Wrote {report_dir / 'EXAM1_ABLATION_RESULTS.md'}")


if __name__ == "__main__":
    main()
