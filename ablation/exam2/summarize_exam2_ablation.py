#!/usr/bin/env python3
"""Summarize experiment-2 ablation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


FIELDS = (
    "variant",
    "events",
    "gt_referents",
    "time_evaluable_referents",
    "predictions",
    "time_f1",
    "point_50_f1",
    "point_100_f1",
    "point_150_f1",
    "joint_50_f1",
    "joint_100_f1",
    "joint_150_f1",
    "mean_point_distance_px_100",
    "mean_joint_distance_px_100",
    "delta_joint_100_f1_vs_baseline",
    "delta_point_100_f1_vs_baseline",
    "source",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize exam2 ablation summaries.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_root", default="ablation/exam2/outputs")
    parser.add_argument("--report_dir", default="ablation/exam2/reports")
    parser.add_argument("--baseline_summary", default="exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/eval/2d_eval_summary.json")
    return parser.parse_args()


def safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def safe_int(value: Any) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_row(variant: str, payload: Mapping[str, Any], source: str) -> Dict[str, Any]:
    return {
        "variant": variant,
        "events": safe_int(payload.get("events")),
        "gt_referents": safe_int(payload.get("gt_referents")),
        "time_evaluable_referents": safe_int(payload.get("time_evaluable_referents")),
        "predictions": safe_int(payload.get("predictions")),
        "time_f1": safe_float(payload.get("time_f1")),
        "point_50_f1": safe_float(payload.get("point_50_f1")),
        "point_100_f1": safe_float(payload.get("point_100_f1")),
        "point_150_f1": safe_float(payload.get("point_150_f1")),
        "joint_50_f1": safe_float(payload.get("joint_50_f1")),
        "joint_100_f1": safe_float(payload.get("joint_100_f1")),
        "joint_150_f1": safe_float(payload.get("joint_150_f1")),
        "mean_point_distance_px_100": safe_float(payload.get("mean_point_distance_px_100")),
        "mean_joint_distance_px_100": safe_float(payload.get("mean_joint_distance_px_100")),
        "delta_joint_100_f1_vs_baseline": "",
        "delta_point_100_f1_vs_baseline": "",
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
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field, "")) for field in FIELDS})


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Experiment 2 Modality Ablation Summary",
        "",
        "Baseline is reused from `exam2/outputs_qwen3vl30b_2d_point_hybrid_v10/`. Manifest construction and evaluator are unchanged unless a variant explicitly changes the number of panels.",
        "",
        "| Variant | Events | Predictions | Time F1 | Point@100 F1 | Joint@100 F1 | Delta Joint@100 | Mean Point Dist@100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {events} | {pred} | {time} | {point} | {joint} | {delta} | {dist} |".format(
                variant=row.get("variant", ""),
                events=row.get("events", ""),
                pred=row.get("predictions", ""),
                time=fmt(row.get("time_f1")),
                point=fmt(row.get("point_100_f1")),
                joint=fmt(row.get("joint_100_f1")),
                delta=fmt(row.get("delta_joint_100_f1_vs_baseline")),
                dist=fmt(row.get("mean_point_distance_px_100")),
            )
        )
    lines.extend([
        "",
        "## Notes",
        "",
        "- `full_panels_no_crop` removes the gaze-centered crop path but keeps full visual panels.",
        "- `no_gaze_text_prior` removes gaze-specific prompt wording and uses full panels, but it does not edit any visible gaze marker.",
        "- `no_gaze` additionally masks the projected green gaze marker in copied panel images before inference.",
        "- The current experiment-2 manifest has no explicit hand summary field, so hand contribution is not claimed from this workflow.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    report_dir = (repo_root / args.report_dir).resolve() if not Path(args.report_dir).is_absolute() else Path(args.report_dir)
    baseline_path = (repo_root / args.baseline_summary).resolve() if not Path(args.baseline_summary).is_absolute() else Path(args.baseline_summary)

    rows: List[Dict[str, Any]] = []
    if baseline_path.exists():
        rows.append(build_row("full_baseline", read_json(baseline_path), str(baseline_path)))

    if output_root.exists():
        for variant_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
            summary_path = variant_dir / "eval" / "2d_eval_summary.json"
            if summary_path.exists():
                rows.append(build_row(variant_dir.name, read_json(summary_path), str(summary_path)))

    baseline = next((row for row in rows if row.get("variant") == "full_baseline"), None)
    if baseline:
        base_joint = baseline.get("joint_100_f1")
        base_point = baseline.get("point_100_f1")
        for row in rows:
            if row.get("variant") == "full_baseline":
                continue
            if isinstance(row.get("joint_100_f1"), float) and isinstance(base_joint, float):
                row["delta_joint_100_f1_vs_baseline"] = row["joint_100_f1"] - base_joint
            if isinstance(row.get("point_100_f1"), float) and isinstance(base_point, float):
                row["delta_point_100_f1_vs_baseline"] = row["point_100_f1"] - base_point

    write_csv(report_dir / "exam2_ablation_summary.csv", rows)
    write_markdown(report_dir / "EXAM2_ABLATION_RESULTS.md", rows)
    print(f"Wrote {report_dir / 'exam2_ablation_summary.csv'}")
    print(f"Wrote {report_dir / 'EXAM2_ABLATION_RESULTS.md'}")


if __name__ == "__main__":
    main()
