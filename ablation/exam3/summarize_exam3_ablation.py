#!/usr/bin/env python3
"""Validate and summarize Qwen3-VL-30B Experiment 3 input ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple


VARIANTS = ("no_visual", "no_gaze", "no_hand", "no_gaze_hand", "no_instruction")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def key(row: Mapping[str, Any]) -> str:
    return f"{row.get('scene')}::{row.get('row_index')}"


def stable_hash(items: Iterable[str]) -> str:
    text = "\n".join(sorted(items)) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_points(row: Mapping[str, Any]) -> str:
    raw = row.get("parsed_json") or row.get("pred_points_json") or ""
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        value = {"points_3d": []}
    if isinstance(value, list):
        value = {"points_3d": value}
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: List[str] = []
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def expected_audit(variant: str) -> Dict[str, bool]:
    return {
        "uses_images": variant != "no_visual",
        "has_visual_path": variant != "no_visual",
        "has_gaze_payload": variant not in {"no_gaze", "no_gaze_hand"},
        "has_hand_payload": variant not in {"no_hand", "no_gaze_hand"},
        "has_instruction_value": variant != "no_instruction",
        "contains_nan_or_inf": False,
    }


def validate_raw_prompts(pred_rows: List[Mapping[str, str]], variant: str, repo_root: Path) -> Tuple[int, List[str]]:
    failures: List[str] = []
    checked = 0
    expected = expected_audit(variant)
    for row in pred_rows:
        raw_path = Path(str(row.get("raw_json_path", "")))
        if not raw_path.is_absolute():
            raw_path = repo_root / raw_path
        if not raw_path.exists():
            failures.append(f"{key(row)}:missing_raw_json")
            continue
        payload = read_json(raw_path)
        audit = payload.get("prompt_audit")
        if not isinstance(audit, Mapping):
            failures.append(f"{key(row)}:missing_prompt_audit")
            continue
        checked += 1
        if payload.get("ablation_variant") != variant:
            failures.append(f"{key(row)}:raw_variant_mismatch")
        if variant == "no_visual" and payload.get("frame_paths"):
            failures.append(f"{key(row)}:visual_frames_present")
        for field, expected_value in expected.items():
            if bool(audit.get(field)) != expected_value:
                failures.append(f"{key(row)}:{field}={audit.get(field)!r}")
    return checked, failures


def summary_rows(name: str, payload: Mapping[str, Any], baseline: Mapping[str, Any]) -> List[Dict[str, Any]]:
    partitions = {
        "overall": payload["overall"],
        "single_target": payload["single_target"],
        "multi_target": payload["multi_target"],
    }
    base_partitions = {
        "overall": baseline["overall"],
        "single_target": baseline["single_target"],
        "multi_target": baseline["multi_target"],
    }
    for scene, item in payload.get("per_scene", {}).items():
        partitions[f"scene:{scene}"] = item
        base_partitions[f"scene:{scene}"] = baseline["per_scene"][scene]
    rows = []
    for partition, item in partitions.items():
        base = base_partitions[partition]
        rows.append({
            "variant": name,
            "partition": partition,
            "total_samples": item["total_samples"],
            "valid_output_count": item["valid_output_count"],
            "invalid_output_count": item["invalid_output_count"],
            "valid_output_rate": item["valid_output_rate"],
            "anchor_set_precision_micro": item["anchor_set_precision_micro"],
            "anchor_set_recall_micro": item["anchor_set_recall_micro"],
            "anchor_set_f1_micro": item["anchor_set_f1_micro"],
            "anchor_set_exact_rate": item["anchor_set_exact_rate"],
            "margin_f1_at_0_5": item["margin_f1_at_0_5"],
            "margin_f1_at_1_0": item["margin_f1_at_1_0"],
            "margin_f1_at_2_0": item["margin_f1_at_2_0"],
            "mean_scene_normalized_error": item["mean_scene_normalized_error"],
            "mean_matched_euclidean_error": item["mean_matched_euclidean_error"],
            "delta_anchor_f1_vs_full": item["anchor_set_f1_micro"] - base["anchor_set_f1_micro"],
            "delta_margin_f1_at_1_vs_full": item["margin_f1_at_1_0"] - base["margin_f1_at_1_0"],
        })
    return rows


def fmt(value: Any) -> str:
    return "n/a" if value in (None, "") else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--expected_gt", required=True)
    parser.add_argument("--report_dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    report_dir = (repo_root / args.report_dir).resolve()
    gt_rows = read_csv((repo_root / args.expected_gt).resolve())
    expected_ids = {key(row) for row in gt_rows}
    baseline_pred_path = repo_root / "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/predictions.csv"
    baseline_summary_path = repo_root / "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b/evaluation_summary.json"
    baseline_predictions = {key(row): canonical_points(row) for row in read_csv(baseline_pred_path)}
    baseline_summary = read_json(baseline_summary_path)
    gaze_copy_summary = read_json(
        repo_root / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_copy/eval/evaluation_summary.json"
    )

    metric_rows: List[Dict[str, Any]] = summary_rows("full", baseline_summary, baseline_summary)
    metric_rows.extend(summary_rows("gaze_copy_reference", gaze_copy_summary, baseline_summary))
    validation_rows: List[Dict[str, Any]] = []
    prompt_audit: Dict[str, Any] = {}
    summaries: Dict[str, Any] = {"full": baseline_summary, "gaze_copy_reference": gaze_copy_summary}
    for variant in VARIANTS:
        pred_path = output_root / variant / "predictions.csv"
        summary_path = output_root / variant / "eval/evaluation_summary.json"
        if not pred_path.exists() or not summary_path.exists():
            raise RuntimeError(f"incomplete variant {variant}: missing predictions or summary")
        predictions = read_csv(pred_path)
        summary = read_json(summary_path)
        summaries[variant] = summary
        actual_ids = [key(row) for row in predictions]
        predictions_by_id = {key(row): canonical_points(row) for row in predictions}
        actual_set = set(actual_ids)
        duplicates = len(actual_ids) - len(actual_set)
        missing = expected_ids - actual_set
        extra = actual_set - expected_ids
        variant_mismatch = sum(1 for row in predictions if row.get("ablation_variant") != variant)
        same_denominator_predictions = [item for item in actual_set & expected_ids if item in baseline_predictions]
        same_as_full = sum(
            1 for item in same_denominator_predictions
            if predictions_by_id[item] == baseline_predictions[item]
        )
        checked, prompt_failures = validate_raw_prompts(predictions, variant, repo_root)
        prompt_audit[variant] = {
            "checked_raw_outputs": checked,
            "failure_count": len(prompt_failures),
            "failure_examples": prompt_failures[:20],
            "expected": expected_audit(variant),
        }
        validation_rows.append({
            "variant": variant,
            "expected_count": len(expected_ids),
            "prediction_count": len(predictions),
            "unique_id_count": len(actual_set),
            "duplicate_id_count": duplicates,
            "missing_id_count": len(missing),
            "extra_id_count": len(extra),
            "sample_id_sha256": stable_hash(actual_set),
            "expected_id_sha256": stable_hash(expected_ids),
            "variant_column_mismatch_count": variant_mismatch,
            "raw_prompt_audit_count": checked,
            "raw_prompt_failure_count": len(prompt_failures),
            "valid_output_count": summary["overall"]["valid_output_count"],
            "invalid_output_count": summary["overall"]["invalid_output_count"],
            "same_prediction_as_full_count": same_as_full,
            "same_prediction_as_full_rate": same_as_full / len(same_denominator_predictions) if same_denominator_predictions else 0.0,
        })
        metric_rows.extend(summary_rows(variant, summary, baseline_summary))

    failed = [
        row for row in validation_rows
        if any(int(row[field]) for field in (
            "duplicate_id_count", "missing_id_count", "extra_id_count",
            "variant_column_mismatch_count", "raw_prompt_failure_count",
        ))
    ]
    if failed:
        raise RuntimeError(f"validation failed: {failed}")

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "ablation_summary.csv", metric_rows)
    write_csv(report_dir / "validation_summary.csv", validation_rows)
    (report_dir / "input_audit.json").write_text(json.dumps(prompt_audit, indent=2) + "\n", encoding="utf-8")
    (report_dir / "evaluation_summaries.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    overall = {row["variant"]: row for row in metric_rows if row["partition"] == "overall"}
    table_lines = []
    for variant in ("full", "gaze_copy_reference") + VARIANTS:
        item = overall[variant]
        validation = next((row for row in validation_rows if row["variant"] == variant), None)
        table_lines.append(
            f"| {variant} | {item['total_samples']} | {item['valid_output_count']} | "
            f"{fmt(item['anchor_set_precision_micro'])} | {fmt(item['anchor_set_recall_micro'])} | "
            f"{fmt(item['anchor_set_f1_micro'])} | {fmt(item['anchor_set_exact_rate'])} | "
            f"{fmt(item['margin_f1_at_0_5'])} | {fmt(item['margin_f1_at_1_0'])} | "
            f"{fmt(item['margin_f1_at_2_0'])} | {fmt(item['mean_scene_normalized_error'])} | "
            f"{fmt(validation['same_prediction_as_full_rate']) if validation else 'n/a'} |"
        )
    partition_lines = []
    partition_lookup = {(row["variant"], row["partition"]): row for row in metric_rows}
    for variant in ("full", "gaze_copy_reference") + VARIANTS:
        for partition in ("single_target", "multi_target"):
            item = partition_lookup[(variant, partition)]
            partition_lines.append(
                f"| {variant} | {partition} | {item['total_samples']} | "
                f"{fmt(item['anchor_set_f1_micro'])} | {fmt(item['margin_f1_at_1_0'])} | "
                f"{fmt(item['margin_f1_at_2_0'])} | {fmt(item['mean_scene_normalized_error'])} |"
            )
    base = baseline_summary["overall"]
    report = f"""# Experiment 3 Qwen3-VL-30B Input Ablation

## Scope

This is a descriptive model-input ablation of the frozen v9 candidate-free measured point-hypothesis diagnostic. All variants reuse the same evidence-frame selection, GT manifest, model checkpoint, parser, greedy decoding (`do_sample=false`, `max_new_tokens=512`), and evaluator. The current evaluation denominator is {len(expected_ids)} samples. Only model-visible input fields are masked.

Because the frozen target-free frame selector itself used gaze/hand availability and stability, `no_gaze`, `no_hand`, and `no_gaze_hand` do not constitute strict causal single-modality ablations. They remove those fields after panel selection and must be reported as controlled input ablations.

## Frozen Baseline

- samples: {base['total_samples']}
- nearest-anchor set F1: {fmt(base['anchor_set_f1_micro'])}
- exact set: {fmt(base['anchor_set_exact_rate'])}
- Margin-F1@0.5/1.0/2.0: {fmt(base['margin_f1_at_0_5'])} / {fmt(base['margin_f1_at_1_0'])} / {fmt(base['margin_f1_at_2_0'])}
- mean scene-normalized error: {fmt(base['mean_scene_normalized_error'])}

## Overall Results

| variant | N | valid | anchor P | anchor R | anchor F1 | exact | M-F1@0.5 | M-F1@1.0 | M-F1@2.0 | scene norm err | same outputs as full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

`gaze_copy_reference` is the frozen deterministic gaze-copy baseline and is included because the v9 prompt explicitly exposes copyable measured gaze hypotheses.

## Target-Count Partitions

| variant | partition | N | anchor F1 | M-F1@1.0 | M-F1@2.0 | scene norm err |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(partition_lines)}

## Variant Definitions

- `no_visual`: sends no image tensors and removes image paths from the prompt; language, camera, gaze, and hand telemetry remain.
- `no_gaze`: removes gaze coordinates, validity, directions, copyable hypotheses, gaze-derived distances, and selection metadata; images, language, camera, and hand remain.
- `no_hand`: removes hand state, coordinates, directions, copyable hypotheses, hand-derived distances, and selection metadata; images, language, camera, and gaze remain.
- `no_gaze_hand`: removes both behavioral cue families and their derived metadata; images, language, and camera remain.
- `no_instruction`: removes instruction and utterance values while retaining the task instruction, images, camera, gaze, and hand.

## Interpretation Boundary

The v9 task prompt defaults to copying distinct measured gaze hypotheses and uses hand only as fallback. Therefore limited changes under `no_visual`, `no_hand`, or `no_instruction` are evidence that the frozen protocol is dominated by exposed gaze point hypotheses, not evidence that those modalities are generally unnecessary for referential grounding. Conversely, degradation under `no_gaze` measures dependence on model-visible gaze hypotheses under this protocol.

No bootstrap significance test is included. These results must not be described as unconstrained 3D reconstruction, 3D box grounding, or strict single-modality causal attribution.

## Validation

Every variant passed exact sample-set, unique-key, variant-label, and raw prompt-mask validation. Invalid model outputs remain empty predictions in the {len(expected_ids)}-sample denominator. Machine-readable files in this directory contain overall/single/multi metrics and the full input audit.
"""
    (report_dir / "EXPERIMENT3_QWEN30B_ABLATION.md").write_text(report, encoding="utf-8")
    print(json.dumps({"variants": len(VARIANTS), "samples": len(expected_ids), "report_dir": str(report_dir)}))


if __name__ == "__main__":
    main()
