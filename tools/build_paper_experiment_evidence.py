#!/usr/bin/env python3
"""Build compact paper evidence from existing immutable evaluator outputs."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_experiment_evidence"
COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
).strip()
SCENES = ["scene1", "scene2", "scene3", "scene4_room1", "scene4_room2", "scene4_room3", "scene4_room4", "scene5"]
MODELS = ["Qwen3-VL-30B-A3B", "Qwen3-VL-8B", "InternVL3-38B"]
EXP1 = {
    MODELS[0]: ROOT / "data/match_eval_qwen3vl30b_mention_first_v3",
    MODELS[1]: ROOT / "qwen8/outputs/exam1_qwen3vl8b_baseline/eval",
    MODELS[2]: ROOT / "internvl/outputs/exam1_internvl3_38b_baseline/eval",
}
EXP2 = {
    MODELS[0]: ROOT / "exam2/outputs_qwen3vl30b_2d_point_hybrid_v10",
    MODELS[1]: ROOT / "qwen8/outputs/exam2_qwen3vl8b_baseline_2d_point_hybrid_v10",
    MODELS[2]: ROOT / "internvl/outputs/exam2_internvl3_38b_baseline",
}
EXP3 = {
    MODELS[0]: ROOT / "exam3_point_grounding/outputs_full_v9_20260709/qwen3vl30b",
    MODELS[1]: ROOT / "qwen8/outputs/exam3_qwen3vl8b_point_grounding_merged_20260713/eval",
    MODELS[2]: ROOT / "internvl/outputs/exam3_internvl38b_point_grounding_merged_20260714/eval",
}
EXP3_GT = ROOT / "exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv"
EVALUATORS = {
    "experiment1": ROOT / "scripts/eval/evaluate_local_3d_object_match.py",
    "experiment2": ROOT / "exam2/evaluate_2d_point_grounding.py",
    "experiment3": ROOT / "exam3_point_grounding/evaluate_point_grounding.py",
}


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"fields are required when writing an empty CSV: {path}")
        fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def split_ids(value):
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def truth(value):
    return value is True or str(value).lower() == "true"


def uid(scene, row_index):
    return f"{scene}::{int(row_index)}"


def digest(values):
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def slug(model):
    return model.lower().replace("-", "_").replace("_a3b", "")


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def evaluator_version(experiment):
    path = EVALUATORS[experiment]
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return f"git:{COMMIT};sha256:{checksum}"


def exp2_prediction_csv(model, source):
    name = "internvl_2d_predictions.csv" if model == MODELS[2] else "qwen3vl_2d_predictions.csv"
    return source / "predictions" / name


def common_stats(rows, path):
    ids = [r["unified_id"] for r in rows]
    return {"row_count": len(rows), "unique_ids": len(set(ids)), "duplicate_ids": len(ids) - len(set(ids)),
            "valid_count": sum(truth(r["valid_output"]) for r in rows),
            "invalid_count": sum(not truth(r["valid_output"]) for r in rows),
            "sample_set_hash": digest(ids), "file": str(path.relative_to(ROOT))}


def export_exp1():
    stats, expected_ids = {}, set()
    for model, source in EXP1.items():
        output = []
        for scene in SCENES:
            for row in read_csv(source / f"{scene}_match_eval.csv"):
                key = uid(scene, row["row_index"])
                expected_ids.add(key)
                gt = split_ids(row["gt_referents_mapped"])
                mapped = truth(row["evaluable"])
                status = row["response_status"] or "unknown"
                output.append({
                    "model_name": model, "experiment": "experiment1", "scene": scene,
                    "row_index": int(row["row_index"]), "unified_id": key, "event_id": row["event_id"],
                    "gt_anchor_ids": canonical(gt), "predicted_anchor_ids": canonical(split_ids(row["predicted_referents_mapped"])),
                    "gt_mapped": mapped, "gt_unmapped_labels": canonical(split_ids(row["gt_referents_unmapped"])),
                    "valid_output": status == "ok", "parse_status": status,
                    "invalid_reason": "" if status == "ok" else status,
                    "single_or_multi_target": "single" if len(gt) == 1 else ("multi" if len(gt) > 1 else "unmapped"),
                    "tp": len(split_ids(row["true_positive_referents"])),
                    "fp": len(split_ids(row["false_positive_referents"])),
                    "fn": len(split_ids(row["false_negative_referents"])),
                    "set_f1": float(row["set_f1"] or 0.0),
                    "exact": truth(row["exact_match"]) if mapped else False,
                    "hit": truth(row["match_success"]), "evaluator_version_or_commit": evaluator_version("experiment1"),
                })
        path = OUT / "predictions/exp1" / f"{slug(model)}.csv"
        write_csv(path, output)
        s = common_stats(output, path)
        mapped_rows = [r for r in output if truth(r["gt_mapped"])]
        # The audited micro set metrics use all 4,000 rows; unmapped GT rows can
        # still contribute false positives. Hit-Mapped/exact retain mapped-only scope.
        tp, fp, fn = (sum(int(r[x]) for r in output) for x in ("tp", "fp", "fn"))
        p, rec, f1 = prf(tp, fp, fn)
        s.update({"mapped_count": len(mapped_rows), "unmapped_count": len(output)-len(mapped_rows), "tp": tp, "fp": fp, "fn": fn,
                  "hit_all": sum(truth(r["hit"]) for r in output)/len(output),
                  "hit_mapped": sum(truth(r["hit"]) for r in mapped_rows)/len(mapped_rows),
                  "exact": sum(truth(r["exact"]) for r in mapped_rows)/len(mapped_rows),
                  "macro_set_f1": sum(float(r["set_f1"]) for r in mapped_rows)/len(mapped_rows),
                  "micro_precision": p, "micro_recall": rec, "micro_f1": f1,
                  "gt_hash": digest([r["unified_id"]+":"+r["gt_anchor_ids"] for r in output])})
        stats[("experiment1", model)] = s
    return stats, expected_ids


def group_exp2_gt(manifest):
    grouped = defaultdict(lambda: {"panels": [], "points": [], "targets": []})
    for row in read_csv(manifest):
        key = uid(row["scene"], row["row_index"])
        if truth(row["projection_valid"]):
            grouped[key]["panels"].append(row["panel_id"])
            grouped[key]["points"].append({"panel_id": row["panel_id"], "x": float(row["gt_x"]), "y": float(row["gt_y"])})
            grouped[key]["targets"].append(row["referent_name"])
    return grouped


def export_exp2():
    stats = {}
    for model, source in EXP2.items():
        gt = group_exp2_gt(source / "manifest/manifest_all.csv")
        prediction_rows = read_csv(exp2_prediction_csv(model, source))
        prediction_by_id = {uid(row["scene"], row["row_index"]): row for row in prediction_rows}
        if len(prediction_by_id) != len(prediction_rows):
            raise RuntimeError(f"duplicate experiment-2 prediction IDs: {model}")
        output = []
        for row in read_csv(source / "eval/2d_eval_detail.csv"):
            key = uid(row["scene"], row["row_index"])
            prediction = prediction_by_id.get(key)
            missing = prediction is None
            parsed, parse_ok, reason = [], False, "missing_prediction" if missing else ""
            if not missing:
                parse_ok = truth(prediction.get("parse_ok", False))
                try:
                    parsed_payload = json.loads(prediction.get("parsed_json") or "{}")
                except json.JSONDecodeError:
                    parsed_payload = {}
                parsed = parsed_payload.get("referents", []) if parse_ok and isinstance(parsed_payload, dict) else []
                reason = "" if parse_ok else (prediction.get("error_message") or row["error_message"] or "parse_failure")
            gt_item = gt[key]
            item = {
                "model_name": model, "experiment": "experiment2", "scene": row["scene"],
                "row_index": int(row["row_index"]), "unified_id": key, "event_id": row["event_id"],
                "gt_targets": canonical(gt_item["targets"]), "gt_panel_ids": canonical(gt_item["panels"]),
                "predicted_panel_ids": canonical([x.get("panel_id") for x in parsed if x.get("panel_id")]),
                "gt_points": canonical(gt_item["points"]),
                "predicted_points": canonical([{"panel_id": x.get("panel_id"), "x_norm": x.get("x_norm"), "y_norm": x.get("y_norm")} for x in parsed]),
                "valid_output": parse_ok, "parse_status": "missing" if missing else ("ok" if parse_ok else "parse_failure"),
                "invalid_reason": reason, "missing_prediction": missing, "parse_failure": not missing and not parse_ok,
                "single_or_multi_target": "single" if int(row["gt_referent_count"]) == 1 else "multi",
                "evaluator_version_or_commit": evaluator_version("experiment2"),
            }
            for name in ("time_tp", "time_fp", "time_fn"):
                item[name] = int(row[name])
            for threshold in (50, 100, 150, 200):
                for kind in ("point", "joint"):
                    for metric in ("tp", "fp", "fn"):
                        name = f"{kind}_{metric}_{threshold}"
                        item[name] = int(row[name])
            output.append(item)
        path = OUT / "predictions/exp2" / f"{slug(model)}.csv"
        write_csv(path, output)
        s = common_stats(output, path)
        s.update({"missing_count": sum(truth(r["missing_prediction"]) for r in output),
                  "parse_failure_count": sum(truth(r["parse_failure"]) for r in output)})
        for prefix in ("time", "point_50", "point_100", "point_150", "point_200", "joint_50", "joint_100", "joint_150", "joint_200"):
            names = ("time_tp", "time_fp", "time_fn") if prefix == "time" else tuple(f'{prefix.split("_")[0]}_{x}_{prefix.split("_")[1]}' for x in ("tp", "fp", "fn"))
            tp, fp, fn = (sum(int(r[x]) for r in output) for x in names)
            p, rec, f1 = prf(tp, fp, fn)
            s.update({f"{prefix}_tp": tp, f"{prefix}_fp": fp, f"{prefix}_fn": fn,
                      f"{prefix}_precision": p, f"{prefix}_recall": rec, f"{prefix}_f1": f1})
        s["gt_hash"] = digest([r["unified_id"]+":"+r["gt_panel_ids"]+":"+r["gt_points"] for r in output])
        stats[("experiment2", model)] = s
    return stats


def export_exp3():
    gt_rows = read_csv(EXP3_GT)
    gt = {uid(x["scene"], x["row_index"]): x for x in gt_rows}
    stats = {}
    for model, eval_root in EXP3.items():
        output = []
        for row in read_csv(eval_root / "evaluation_detail.csv"):
            key = uid(row["scene"], row["row_index"])
            raw_path = Path(row["raw_json_path"])
            raw = load_json(raw_path) if raw_path.exists() else {}
            parse_ok = truth(row["parse_ok"])
            pred_points = (raw.get("parsed_json") or {}).get("points_3d", []) if parse_ok else []
            gt_item = gt[key]
            item = {
                "model_name": model, "experiment": "experiment3", "scene": row["scene"],
                "row_index": int(row["row_index"]), "unified_id": key, "event_id": row["event_id"],
                "gt_anchor_ids": canonical(split_ids(gt_item["gt_anchor_ids"])),
                "gt_points": gt_item["gt_points_json"], "predicted_point_hypotheses": canonical(pred_points),
                "nearest_anchor_predictions": canonical(split_ids(row["nearest_pred_anchor_ids"])),
                "point_to_anchor_margins": row["matched_margin_errors_json"],
                "point_to_anchor_distances": row["matched_euclidean_errors_json"],
                "scene_normalized_errors": row["scene_normalized_errors_json"],
                "valid_output": parse_ok, "parse_status": "ok" if parse_ok else "parse_failure",
                "invalid_reason": row["invalid_reason"],
                "single_or_multi_target": "single" if int(row["gt_count"]) == 1 else "multi",
                "tp": int(row["set_tp"]), "fp": int(row["set_fp"]), "fn": int(row["set_fn"]),
                "exact": truth(row["anchor_set_exact"]),
                "mean_scene_normalized_error": row["mean_scene_normalized_error"],
                "evaluator_version_or_commit": evaluator_version("experiment3"),
            }
            for threshold, suffix in [(0.5, "0_5"), (1.0, "1_0"), (2.0, "2_0")]:
                for metric in ("tp", "fp", "fn"):
                    item[f"margin_{metric}_{threshold}"] = int(row[f"margin_{metric}_at_{suffix}"])
            output.append(item)
        path = OUT / "predictions/exp3" / f"{slug(model)}.csv"
        write_csv(path, output)
        s = common_stats(output, path)
        tp, fp, fn = (sum(int(r[x]) for r in output) for x in ("tp", "fp", "fn"))
        p, rec, f1 = prf(tp, fp, fn)
        s.update({"tp": tp, "fp": fp, "fn": fn, "anchor_precision": p, "anchor_recall": rec, "anchor_f1": f1,
                  "exact": sum(truth(r["exact"]) for r in output)/len(output),
                  "single_count": sum(r["single_or_multi_target"] == "single" for r in output),
                  "multi_count": sum(r["single_or_multi_target"] == "multi" for r in output)})
        for threshold in (0.5, 1.0, 2.0):
            mtp, mfp, mfn = (sum(int(r[f"margin_{x}_{threshold}"]) for r in output) for x in ("tp", "fp", "fn"))
            s[f"margin_f1_{threshold}"] = prf(mtp, mfp, mfn)[2]
        official = load_json(eval_root / "evaluation_summary.json")["overall"]
        s["mean_scene_normalized_error"] = official["mean_scene_normalized_error"]
        s["gt_hash"] = digest([r["unified_id"]+":"+r["gt_anchor_ids"]+":"+r["gt_points"] for r in output])
        stats[("experiment3", model)] = s
    return stats, set(gt)


def export_audits(exp1_ids, exp3_ids):
    reference = read_csv(OUT / "predictions/exp1" / f"{slug(MODELS[0])}.csv")
    unmapped = [{"unified_id": r["unified_id"], "scene": r["scene"], "row_index": r["row_index"],
                 "experiment": "experiment1", "reason": "GT label has no mapped scene anchor",
                 "source_path": "data/match_eval_qwen3vl30b_mention_first_v3/*_match_eval.csv"}
                for r in reference if not truth(r["gt_mapped"])]
    write_csv(
        OUT / "denominator_audit/unmapped_gt_ids.csv",
        unmapped,
        ("unified_id", "scene", "row_index", "experiment", "reason", "source_path"),
    )
    missing = []
    for model in MODELS:
        for row in read_csv(OUT / "predictions/exp2" / f"{slug(model)}.csv"):
            if truth(row["missing_prediction"]):
                missing.append({"unified_id": row["unified_id"], "scene": row["scene"], "row_index": row["row_index"],
                                "experiment": "experiment2", "model": model, "reason": "no prediction record",
                                "source_path": str(EXP2[model].relative_to(ROOT)) + "/predictions/json"})
    write_csv(
        OUT / "denominator_audit/missing_prediction_ids.csv",
        missing,
        ("unified_id", "scene", "row_index", "experiment", "model", "reason", "source_path"),
    )
    by_id = {r["unified_id"]: r for r in reference}
    excluded = []
    for key in sorted(exp1_ids - exp3_ids):
        row = by_id[key]
        reason = "unmapped GT anchor" if not truth(row["gt_mapped"]) else "mapped GT but excluded by Exp.3 manifest builder"
        excluded.append({"unified_id": key, "scene": row["scene"], "row_index": row["row_index"],
                         "experiment": "experiment3", "reason": reason,
                         "source_path": "exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv"})
    write_csv(
        OUT / "denominator_audit/exp3_excluded_ids.csv",
        excluded,
        ("unified_id", "scene", "row_index", "experiment", "reason", "source_path"),
    )
    invalid = []
    for experiment in ("exp1", "exp2", "exp3"):
        for model in MODELS:
            for row in read_csv(OUT / "predictions" / experiment / f"{slug(model)}.csv"):
                if not truth(row["valid_output"]):
                    invalid.append({
                        "experiment": experiment.replace("exp", "experiment"),
                        "model": model,
                        "unified_id": row["unified_id"],
                        "scene": row["scene"],
                        "row_index": row["row_index"],
                        "parse_status": row["parse_status"],
                        "invalid_reason": row["invalid_reason"],
                    })
    write_csv(
        OUT / "denominator_audit/invalid_output_ids.csv",
        invalid,
        ("experiment", "model", "unified_id", "scene", "row_index", "parse_status", "invalid_reason"),
    )


def copy_repro_files():
    copies = {
        ROOT / "scripts/eval/evaluate_local_3d_object_match.py": OUT / "evaluators/evaluate_local_3d_object_match.py",
        ROOT / "exam2/evaluate_2d_point_grounding.py": OUT / "evaluators/evaluate_2d_point_grounding.py",
        ROOT / "exam3_point_grounding/evaluate_point_grounding.py": OUT / "evaluators/evaluate_point_grounding.py",
        ROOT / "exam2/build_2d_eval_manifest.py": OUT / "evaluators/build_2d_eval_manifest.py",
        ROOT / "exam3_point_grounding/build_point_grounding_manifest.py": OUT / "evaluators/build_point_grounding_manifest.py",
        ROOT / "exam3_point_grounding/prompts/qwen3vl_point_grounding.md": OUT / "prompts_and_configs/exp3_qwen3vl_point_grounding.md",
        ROOT / "scripts/grounding/run_qwen3vl_30b_mention_first_all_scenes_full.sh": OUT / "prompts_and_configs/exp1_qwen30_final.sh",
        ROOT / "qwen8/run_exam1_qwen3vl8b_baseline.sh": OUT / "prompts_and_configs/exp1_qwen8_final.sh",
        ROOT / "internvl/run_exam1_internvl38b_baseline.sh": OUT / "prompts_and_configs/exp1_internvl_final.sh",
        ROOT / "exam2/run_qwen3vl_30b_2d_full.sh": OUT / "prompts_and_configs/exp2_qwen30_final.sh",
        ROOT / "qwen8/run_exam2_qwen3vl8b_baseline.sh": OUT / "prompts_and_configs/exp2_qwen8_final.sh",
        ROOT / "internvl/run_exam2_internvl38b_baseline.sh": OUT / "prompts_and_configs/exp2_internvl_final.sh",
        ROOT / "exam3_point_grounding/run_qwen3vl_30b_point_grounding.sh": OUT / "prompts_and_configs/exp3_qwen30_final.sh",
        ROOT / "qwen8/run_exam3_qwen3vl8b_point_grounding.sh": OUT / "prompts_and_configs/exp3_qwen8_final.sh",
        ROOT / "internvl/run_exam3_internvl38b_point_grounding.sh": OUT / "prompts_and_configs/exp3_internvl_final.sh",
        ROOT / "analysis_configs/location_region_anchor_classification.csv": OUT / "prompts_and_configs/location_region_anchor_classification.csv",
    }
    for source, dest in copies.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write_csv(OUT / "manifests/experiment3_evaluable_manifest.csv", read_csv(EXP3_GT))
    rows = read_csv(EXP2[MODELS[0]] / "manifest/manifest_all.csv")
    fields = ("scene", "row_index", "event_id", "referent_name", "panel_id", "gt_x", "gt_y", "projection_valid", "status")
    write_csv(OUT / "manifests/experiment2_projected_gt_manifest.csv", [{k: row[k] for k in fields} for row in rows])
    (OUT / "anchor_tables").mkdir(parents=True, exist_ok=True)
    for source in sorted(ROOT.glob("data/*_anchor_table.tsv")):
        shutil.copy2(source, OUT / "anchor_tables" / source.name)
    location_dest = OUT / "location_region_audit"
    location_dest.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "analysis_outputs/location_region_audit").glob("*")):
        if source.is_file():
            shutil.copy2(source, location_dest / source.name)
    gt_dest = OUT / "gt_completion"
    gt_dest.mkdir(parents=True, exist_ok=True)
    for source in sorted((ROOT / "analysis_outputs/gt_completion").glob("*")):
        if source.is_file():
            shutil.copy2(source, gt_dest / source.name)


def validate(stats):
    validation = []
    cross_checks = {}
    for experiment in ("experiment1", "experiment2", "experiment3"):
        values = [stats[(experiment, model)] for model in MODELS]
        cross_checks[experiment] = {
            "same_sample_set_hash": len({item["sample_set_hash"] for item in values}) == 1,
            "same_gt_hash": len({item["gt_hash"] for item in values}) == 1,
            "all_4000_rows": all(item["row_count"] == 4000 for item in values),
        }
    for (experiment, model), values in stats.items():
        expected = 4000
        checks = {"row_count": values["row_count"] == expected,
                  "unique_ids": values["unique_ids"] == expected,
                  "duplicate_ids": values["duplicate_ids"] == 0,
                  "finite_metrics": all(not isinstance(v, float) or math.isfinite(v) for v in values.values()),
                  **cross_checks[experiment]}
        if experiment == "experiment1":
            checks.update({"mapped_4000": values["mapped_count"] == 4000,
                           "unmapped_zero": values["unmapped_count"] == 0})
        elif experiment == "experiment2":
            checks["missing_zero"] = values["missing_count"] == 0
        validation.append({"experiment": experiment, "model": model, **values,
                           "validation_pass": all(checks.values()), "checks_json": canonical(checks)})
    write_csv(OUT / "validation/validation_results.csv", validation)
    with (OUT / "validation/validation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=True)
    denominator_counts = {
        name: len(read_csv(OUT / "denominator_audit" / name))
        for name in ("unmapped_gt_ids.csv", "missing_prediction_ids.csv", "exp3_excluded_ids.csv")
    }
    gt_audit = load_json(ROOT / "analysis_outputs/gt_completion/gt_completion_audit.json")
    location = load_json(ROOT / "analysis_outputs/location_region_audit/location_region_summary.json")
    final_audit = {
        "validation_pass": all(row["validation_pass"] for row in validation) and all(value == 0 for value in denominator_counts.values()),
        "cross_model_checks": cross_checks,
        "denominator_file_row_counts": denominator_counts,
        "confirmed_mapping_count": gt_audit["confirmed_mapping_count"],
        "all_confirmed_mappings_ok": gt_audit["all_confirmed_mappings_ok"],
        "scene2_569_ok": gt_audit["additional_scene2_569_ok"],
        "new_canonical_anchors": gt_audit["new_canonical_anchors"],
        "location_region_anchor_count": location["location_region_anchor_count"],
        "location_region_interaction_count": location["location_region_interaction_count"],
    }
    (OUT / "validation/final_audit.json").write_text(json.dumps(final_audit, indent=2) + "\n", encoding="utf-8")
    if not final_audit["validation_pass"]:
        raise RuntimeError("Evidence validation failed")


def aggregate_evidence(experiment, rows):
    result = {
        "n": len(rows),
        "valid": sum(truth(row["valid_output"]) for row in rows),
        "invalid": sum(not truth(row["valid_output"]) for row in rows),
    }
    if not rows:
        return result
    if experiment == "experiment1":
        mapped = [row for row in rows if truth(row["gt_mapped"])]
        tp, fp, fn = (sum(int(row[name]) for row in rows) for name in ("tp", "fp", "fn"))
        p, r, f1 = prf(tp, fp, fn)
        result.update({
            "mapped": len(mapped), "unmapped": len(rows) - len(mapped),
            "hit_all": sum(truth(row["hit"]) for row in rows) / len(rows),
            "hit_mapped": sum(truth(row["hit"]) for row in mapped) / len(mapped) if mapped else 0.0,
            "exact": sum(truth(row["exact"]) for row in mapped) / len(mapped) if mapped else 0.0,
            "macro_set_f1": sum(float(row["set_f1"]) for row in mapped) / len(mapped) if mapped else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": p, "micro_recall": r, "micro_f1": f1,
        })
    elif experiment == "experiment2":
        result.update({
            "missing": sum(truth(row["missing_prediction"]) for row in rows),
            "parse_failures": sum(truth(row["parse_failure"]) for row in rows),
        })
        specs = [("temporal", "time")]
        specs += [(f"{kind}{threshold}", f"{kind}_{threshold}") for kind in ("point", "joint") for threshold in (50, 100, 150, 200)]
        for label, prefix in specs:
            names = ("time_tp", "time_fp", "time_fn") if prefix == "time" else tuple(
                f"{prefix.split('_')[0]}_{metric}_{prefix.split('_')[1]}" for metric in ("tp", "fp", "fn")
            )
            tp, fp, fn = (sum(int(row[name]) for row in rows) for name in names)
            p, r, f1 = prf(tp, fp, fn)
            result.update({f"{label}_tp": tp, f"{label}_fp": fp, f"{label}_fn": fn,
                           f"{label}_precision": p, f"{label}_recall": r, f"{label}_f1": f1})
    else:
        tp, fp, fn = (sum(int(row[name]) for row in rows) for name in ("tp", "fp", "fn"))
        p, r, f1 = prf(tp, fp, fn)
        errors = []
        for row in rows:
            try:
                errors.extend(float(value) for value in json.loads(row["scene_normalized_errors"]) if math.isfinite(float(value)))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        result.update({"anchor_tp": tp, "anchor_fp": fp, "anchor_fn": fn,
                       "anchor_precision": p, "anchor_recall": r, "anchor_f1": f1,
                       "exact": sum(truth(row["exact"]) for row in rows) / len(rows),
                       "mean_scene_normalized_error": sum(errors) / len(errors) if errors else ""})
        for threshold in (0.5, 1.0, 2.0):
            mtp, mfp, mfn = (sum(int(row[f"margin_{metric}_{threshold}"]) for row in rows) for metric in ("tp", "fp", "fn"))
            mp, mr, mf1 = prf(mtp, mfp, mfn)
            result.update({f"margin_tp_{threshold}": mtp, f"margin_fp_{threshold}": mfp,
                           f"margin_fn_{threshold}": mfn, f"margin_precision_{threshold}": mp,
                           f"margin_recall_{threshold}": mr, f"margin_f1_{threshold}": mf1})
    return result


def export_grouped_results():
    by_scene, by_target = [], []
    for experiment, folder in (("experiment1", "exp1"), ("experiment2", "exp2"), ("experiment3", "exp3")):
        for model in MODELS:
            rows = read_csv(OUT / "predictions" / folder / f"{slug(model)}.csv")
            for scene in SCENES:
                subset = [row for row in rows if row["scene"] == scene]
                by_scene.append({"experiment": experiment, "model": model, "scene": scene, **aggregate_evidence(experiment, subset)})
            for partition in ("single", "multi"):
                subset = [row for row in rows if row["single_or_multi_target"] == partition]
                by_target.append({"experiment": experiment, "model": model, "partition": partition, **aggregate_evidence(experiment, subset)})
    scene_fields = ["experiment", "model", "scene"] + sorted({key for row in by_scene for key in row if key not in {"experiment", "model", "scene"}})
    target_fields = ["experiment", "model", "partition"] + sorted({key for row in by_target for key in row if key not in {"experiment", "model", "partition"}})
    write_csv(OUT / "by_scene/model_results_by_scene.csv", by_scene, scene_fields)
    write_csv(OUT / "model_results_by_scene.csv", by_scene, scene_fields)
    write_csv(OUT / "by_target_count/model_results_by_target_count.csv", by_target, target_fields)
    write_csv(OUT / "model_results_by_target_count.csv", by_target, target_fields)


def baseline_result_rows():
    result = []
    baseline_root = ROOT / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines"
    for name in ("gaze_copy", "hand_copy", "gaze_hand_fusion"):
        rows = read_csv(baseline_root / name / "eval/evaluation_summary.csv")
        row = next(item for item in rows if item["partition"] == "overall")
        result.append({
            "experiment": "experiment3_baseline", "model": name, "n": int(row["total_samples"]),
            "valid": int(row["valid_output_count"]), "invalid": int(row["invalid_output_count"]),
            "anchor_precision": float(row["anchor_set_precision_micro"]),
            "anchor_recall": float(row["anchor_set_recall_micro"]),
            "anchor_f1": float(row["anchor_set_f1_micro"]), "exact": float(row["anchor_set_exact_rate"]),
            "margin_f1_0.5": float(row["margin_f1_at_0_5"]),
            "margin_f1_1.0": float(row["margin_f1_at_1_0"]),
            "margin_f1_2.0": float(row["margin_f1_at_2_0"]),
            "mean_scene_normalized_error": float(row["mean_scene_normalized_error"]),
        })
    write_csv(OUT / "experiment3_baselines.csv", result)
    return result


def export_ablation_results():
    exp1_meta = {
        "full_baseline": ("full_baseline", "Reference condition."),
        "language_anchors_only": ("multimodal_input_baseline", "Removes visual, gaze, hand, structured geometry, and timeline evidence while retaining the closed-set anchor inventory."),
        "no_gaze": ("hybrid_ablation", "Removes structured gaze fields and gaze-derived proposals; visible video markers may remain."),
        "no_hand": ("hybrid_ablation", "Removes structured hand summaries and hand/ray fields; visible hands remain in video."),
        "no_visual": ("hybrid_ablation", "Replaces visual evidence with a 64x64 placeholder and changes prompt text."),
        "no_gaze_hand": ("hybrid_ablation", "Removes both structured gaze and hand cue families; visible cues may remain."),
    }
    exp2_meta = {
        "full_baseline": ("full_baseline", "Reference condition."),
        "full_panels_no_crop": ("preprocessing_ablation", "Uses full panels without the gaze-centered crop path."),
        "instruction_only_prompt": ("prompt_ablation", "Removes expected referent count and changes prompt mode."),
        "no_gaze": ("hybrid_ablation", "Removes gaze text, masks the marker, and changes paired inputs to full panels."),
        "no_gaze_text_prior": ("hybrid_ablation", "Removes gaze text and paired crop preprocessing; visible marker remains."),
    }
    output = []
    baseline_exp1 = OUT / "predictions/exp1" / f"{slug(MODELS[0])}.csv"
    for variant in exp1_meta:
        if variant == "full_baseline":
            rows = read_csv(baseline_exp1)
        else:
            rows = []
            for scene in SCENES:
                rows.extend(read_csv(ROOT / f"ablation/exam1/outputs/{variant}/eval/{scene}_match_eval.csv"))
        if variant == "full_baseline":
            mapped = [row for row in rows if truth(row["gt_mapped"])]
            valid = sum(truth(row["valid_output"]) for row in rows)
            hit = sum(truth(row["hit"]) for row in rows)
            exact = sum(truth(row["exact"]) for row in mapped)
            macro = sum(float(row["set_f1"]) for row in mapped) / len(mapped)
            tp, fp, fn = (sum(int(row[name]) for row in rows) for name in ("tp", "fp", "fn"))
        else:
            mapped = [row for row in rows if truth(row["evaluable"])]
            valid = sum((row["response_status"] or "unknown") == "ok" for row in rows)
            hit = sum(truth(row["match_success"]) for row in rows)
            exact = sum(truth(row["exact_match"]) for row in mapped)
            macro = sum(float(row["set_f1"] or 0.0) for row in mapped) / len(mapped)
            tp = sum(len(split_ids(row["true_positive_referents"])) for row in rows)
            fp = sum(len(split_ids(row["false_positive_referents"])) for row in rows)
            fn = sum(len(split_ids(row["false_negative_referents"])) for row in rows)
        p, r, f1 = prf(tp, fp, fn)
        output.append({"experiment": "experiment1", "variant": variant,
                       "ablation_category": exp1_meta[variant][0], "n": len(rows), "mapped": len(mapped),
                       "valid": valid, "invalid": len(rows) - valid, "hit_all": hit / len(rows),
                       "hit_mapped": hit / len(mapped), "exact": exact / len(mapped), "macro_set_f1": macro,
                       "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1,
                       "strict_modality_ablation": False, "notes": exp1_meta[variant][1]})

    for variant in exp2_meta:
        detail = (EXP2[MODELS[0]] / "eval/2d_eval_detail.csv" if variant == "full_baseline" else
                  ROOT / f"ablation/exam2/outputs/{variant}/eval/2d_eval_detail.csv")
        rows = read_csv(detail)
        item = {"experiment": "experiment2", "variant": variant,
                "ablation_category": exp2_meta[variant][0], "n": len(rows),
                "valid": sum(truth(row["parse_ok"]) for row in rows),
                "invalid": sum(not truth(row["parse_ok"]) for row in rows),
                "strict_modality_ablation": False, "notes": exp2_meta[variant][1]}
        for label, names in [("temporal", ("time_tp", "time_fp", "time_fn"))] + [
            (f"{kind}{threshold}", tuple(f"{kind}_{metric}_{threshold}" for metric in ("tp", "fp", "fn")))
            for kind in ("point", "joint") for threshold in (50, 100, 150, 200)
        ]:
            tp, fp, fn = (sum(int(row[name]) for row in rows) for name in names)
            p, r, f1 = prf(tp, fp, fn)
            item.update({f"{label}_tp": tp, f"{label}_fp": fp, f"{label}_fn": fn,
                         f"{label}_precision": p, f"{label}_recall": r, f"{label}_f1": f1})
        output.append(item)
    fields = ["experiment", "variant", "ablation_category", "n", "mapped", "valid", "invalid",
              "strict_modality_ablation", "notes"]
    fields += sorted({key for row in output for key in row if key not in fields})
    write_csv(OUT / "ablation_results.csv", output, fields)


def write_provenance(stats):
    configs = {
        ("experiment1", MODELS[0]): ("/workspace/usr3/Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL video adapter", "bfloat16", "none", "video; 16 frames", "mention_first/full", 1536),
        ("experiment1", MODELS[1]): ("Qwen3-VL-8B-Instruct", "Qwen3-VL video adapter", "bfloat16", "none", "video; 8 frames", "mention_first/full", 1536),
        ("experiment1", MODELS[2]): ("/workspace/usr3/InternVL3-38B-Instruct", "InternVL video adapter", "bfloat16", "none", "video; 16 frames", "mention_first/full", 1536),
        ("experiment2", MODELS[0]): ("/workspace/usr3/Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL multi-image adapter", "auto", "none", "3 panels; paired crop; paired_canvas_map", "expected_count", 768),
        ("experiment2", MODELS[1]): ("Qwen3-VL-8B-Instruct", "Qwen3-VL multi-image adapter", "bfloat16", "none", "3 panels; paired crop; paired_canvas_map", "expected_count", 768),
        ("experiment2", MODELS[2]): ("/workspace/usr3/InternVL3-38B-Instruct", "InternVL multi-image adapter", "bfloat16", "none", "3 panels; paired crop; paired_canvas_map", "expected_count", 768),
        ("experiment3", MODELS[0]): ("/workspace/usr3/Qwen3-VL-30B-A3B-Instruct", "Qwen3-VL multi-image adapter", "auto", "none", "up to 3 target-free evidence frames", "measured point hypotheses v9", 512),
        ("experiment3", MODELS[1]): ("Qwen3-VL-8B-Instruct", "Qwen3-VL multi-image adapter", "auto", "none", "up to 3 target-free evidence frames", "measured point hypotheses v9", 512),
        ("experiment3", MODELS[2]): ("/workspace/usr3/InternVL3-38B-Instruct", "InternVL multi-image adapter", "bfloat16", "bitsandbytes 8-bit", "up to 2 target-free evidence frames", "measured point hypotheses v9", 256),
    }
    mappings = {"experiment1": EXP1, "experiment2": EXP2, "experiment3": EXP3}
    rows = []
    for key, s in stats.items():
        experiment, model = key
        checkpoint, adapter, dtype, quantization, protocol, prompt, max_tokens = configs[key]
        source = mappings[experiment][model]
        manifest = "scene-specific GT workbooks and anchor tables"
        if experiment == "experiment2": manifest = str((source / "manifest/manifest_all.csv").relative_to(ROOT))
        if experiment == "experiment3": manifest = str(EXP3_GT.relative_to(ROOT))
        rows.append({
            "experiment": experiment, "model": model, "status": "final", "checkpoint": checkpoint,
            "model_adapter": adapter, "dtype": dtype, "quantization": quantization,
            "input_protocol": protocol, "prompt_mode_or_template": prompt,
            "decoding": "greedy; do_sample=false", "max_new_tokens": max_tokens,
            "output_directory": str(source.relative_to(ROOT)), "manifest": manifest,
            "evaluator": str(EVALUATORS[experiment].relative_to(ROOT)),
            "evaluator_version_or_commit": evaluator_version(experiment),
            "sample_set_hash": s["sample_set_hash"], "gt_hash": s["gt_hash"],
            "completion_policy": "existing valid predictions preserved; only absent/invalid IDs inferred with the original run configuration",
        })
    write_csv(OUT / "run_provenance/run_provenance.csv", rows)
    comparison = [{"experiment": e, "model": m, "sample_set_hash": s["sample_set_hash"],
                   "gt_hash": s["gt_hash"], "row_count": s["row_count"]} for (e, m), s in stats.items()]
    write_csv(OUT / "run_provenance/sample_set_comparison.csv", comparison)
    schemas = {
        "experiment1": "JSON anchor ID set",
        "experiment2": "JSON panel IDs and normalized 2D points",
        "experiment3": "JSON points_3d coordinate triples",
    }
    protocol_rows = [{
        "experiment": row["experiment"], "model": row["model"], "checkpoint": row["checkpoint"],
        "input_protocol": row["input_protocol"], "prompt_strategy": row["prompt_mode_or_template"],
        "output_schema": schemas[row["experiment"]], "decoding": row["decoding"],
        "max_new_tokens": row["max_new_tokens"], "manifest": row["manifest"],
        "evaluator": row["evaluator"], "run_status": row["status"],
        "source_path": row["output_directory"],
    } for row in rows]
    protocol_rows.append({
        "experiment": "experiment1", "model": "Qwen3-VL-8B", "checkpoint": "Qwen3-VL-8B-Instruct",
        "input_protocol": "egocentric video plus structured cues and candidate anchors",
        "prompt_strategy": "standard non-mention-first", "output_schema": "single-candidate JSON protocol",
        "decoding": "legacy", "max_new_tokens": 512, "manifest": "4,000 scene rows",
        "evaluator": "scripts/eval/evaluate_local_3d_object_match.py", "run_status": "legacy",
        "source_path": "data/match_eval_qwen3vl8b",
    })
    write_csv(OUT / "model_protocol_comparison.csv", protocol_rows)
    qwen30_summary = EXP3[MODELS[0]] / "evaluation_summary.json"
    gaze_root = ROOT / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_copy/eval"
    shutil.copy2(qwen30_summary, OUT / "run_provenance/exp3_qwen30_official_summary.json")
    shutil.copy2(gaze_root / "evaluation_summary.json", OUT / "run_provenance/exp3_gaze_copy_evaluation_summary.json")
    shutil.copy2(gaze_root / "evaluation_summary.csv", OUT / "run_provenance/exp3_gaze_copy_evaluation_summary.csv")


def export_qualitative_cases():
    rows = read_csv(OUT / "predictions/exp3" / f"{slug(MODELS[0])}.csv")
    scored = []
    for row in rows:
        values = json.loads(row["scene_normalized_errors"] or "[]")
        score = sum(float(value) for value in values) / len(values) if values else float("inf")
        scored.append((score, row))
    finite = [(score, row) for score, row in scored if math.isfinite(score)]
    selected = [("lowest_error", score, row) for score, row in sorted(finite)[:6]]
    selected += [("highest_error", score, row) for score, row in sorted(finite, reverse=True)[:6]]
    output = [{"case_type": kind, "unified_id": row["unified_id"], "scene": row["scene"],
               "row_index": row["row_index"], "gt_anchor_ids": row["gt_anchor_ids"],
               "predicted_point_hypotheses": row["predicted_point_hypotheses"],
               "nearest_anchor_predictions": row["nearest_anchor_predictions"],
               "mean_scene_normalized_error": score} for kind, score, row in selected]
    write_csv(OUT / "qualitative_cases/experiment3_qwen30_error_extremes.csv", output)


def previous_result_rows():
    try:
        text = subprocess.check_output(
            ["git", "show", "c21c5ea:paper_experiment_evidence/model_results.csv"],
            cwd=ROOT, text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return list(csv.DictReader(io.StringIO(text)))


def sync_exp3_repository_reports():
    sources = {
        "qwen3vl30b_v9": EXP3[MODELS[0]],
        "qwen3vl8b": EXP3[MODELS[1]],
        "internvl3_38b": EXP3[MODELS[2]],
        "gaze_copy": ROOT / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_copy/eval",
        "hand_copy": ROOT / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/hand_copy/eval",
        "gaze_hand_fusion": ROOT / "exam3_point_grounding/outputs_full_v9_20260709/cue_baselines/gaze_hand_fusion/eval",
    }
    all_rows = []
    summary = {"sample_count": 4000, "models_and_baselines": {}}
    for name, source in sources.items():
        rows = read_csv(source / "evaluation_summary.csv")
        summary["models_and_baselines"][name] = rows
        all_rows.extend({"model_or_baseline": name, **row} for row in rows)
    report_dir = ROOT / "exam3_point_grounding/reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0])
    write_csv(report_dir / "experiment3_full_results_v9_overall.csv",
              [row for row in all_rows if row["partition"] == "overall"], fields)
    write_csv(report_dir / "experiment3_full_results_v9_partitions.csv",
              [row for row in all_rows if row["partition"] in {"single_target", "multi_target"}], fields)
    write_csv(report_dir / "experiment3_full_results_v9_per_scene.csv",
              [row for row in all_rows if row["partition"] in SCENES], fields)
    (report_dir / "experiment3_full_results_v9_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(OUT / "EXPERIMENT3_FULL_RESULTS_V9.md", report_dir / "EXPERIMENT3_FULL_RESULTS_V9.md")


def write_documents(stats):
    result_rows = []
    for (experiment, model), s in stats.items():
        row = {"experiment": experiment, "model": model, "n": s["row_count"],
               "valid": s["valid_count"], "invalid": s["invalid_count"]}
        if experiment == "experiment1":
            for name in ("mapped_count", "unmapped_count", "tp", "fp", "fn", "hit_all", "hit_mapped", "exact", "macro_set_f1", "micro_precision", "micro_recall", "micro_f1"):
                row[name] = s[name]
        elif experiment == "experiment2":
            row.update({"missing": s["missing_count"], "parse_failures": s["parse_failure_count"]})
            for prefix in ("time", "point_50", "point_100", "point_150", "point_200", "joint_50", "joint_100", "joint_150", "joint_200"):
                label = "temporal" if prefix == "time" else prefix.replace("_", "")
                for metric in ("tp", "fp", "fn", "precision", "recall", "f1"):
                    row[f"{label}_{metric}"] = s[f"{prefix}_{metric}"]
        else:
            for name in ("tp", "fp", "fn", "anchor_precision", "anchor_recall", "anchor_f1", "exact", "margin_f1_0.5", "margin_f1_1.0", "margin_f1_2.0", "mean_scene_normalized_error", "single_count", "multi_count"):
                row[name] = s[name]
        result_rows.append(row)
    result_rows.extend(baseline_result_rows())
    result_fields = ["experiment", "model", "n", "valid", "invalid"]
    result_fields += sorted({key for row in result_rows for key in row if key not in result_fields})
    write_csv(OUT / "model_results.csv", result_rows, result_fields)
    export_ablation_results()

    previous = {(row["experiment"], row["model"]): row for row in previous_result_rows()}
    comparisons = []
    metric_names = {
        "experiment1": ("hit_all", "exact", "micro_f1"),
        "experiment2": ("temporal_f1", "point100_f1", "joint100_f1"),
        "experiment3": ("anchor_f1", "exact", "margin_f1_1.0", "mean_scene_normalized_error"),
    }
    for row in result_rows:
        if row["experiment"] not in metric_names:
            continue
        old = previous.get((row["experiment"], row["model"]), {})
        for metric in metric_names[row["experiment"]]:
            new_name = metric
            if metric == "point100_f1": new_name = "point100_f1"
            if metric == "joint100_f1": new_name = "joint100_f1"
            before = old.get(metric, "")
            after = row.get(new_name, "")
            comparisons.append({"experiment": row["experiment"], "model": row["model"], "metric": metric,
                                "before_commit": "c21c5ea", "before": before, "after": after,
                                "delta": float(after) - float(before) if before not in ("", None) and after not in ("", None) else ""})
    write_csv(OUT / "run_provenance/core_metrics_before_after.csv", comparisons)
    write_provenance(stats)
    export_grouped_results()
    export_qualitative_cases()

    q30_exp2 = stats[("experiment2", MODELS[0])]
    location = load_json(ROOT / "analysis_outputs/location_region_audit/location_region_summary.json")
    readme = f"""# VR-TriRef Paper Experiment Evidence

This compact evidence bundle is rebuilt from the final VR-TriRef evaluator outputs at source commit `{COMMIT}`. Existing model outputs were not edited: the newly evaluable interactions were inferred with each run's recorded configuration, then all metrics were recomputed with one evaluator per experiment.

## Experiments and denominators

- **Experiment 1: closed-set 3D anchor selection.** All 4,000 interactions now have fully mapped GT. Hit-All, Hit-Mapped, exact set, macro set F1, and micro P/R/F1 therefore share the 4,000-row denominator.
- **Experiment 2: projected-2D point diagnostic.** Temporal, Point@50/100/150/200, and Joint@50/100/150/200 are corpus-level TP/FP/FN metrics over 4,000 interactions. Missing predictions remain empty predictions; the final runs have zero missing records. Qwen3-VL-30B retains {q30_exp2['parse_failure_count']} deterministic parse failures after three same-configuration attempts, listed in `denominator_audit/invalid_output_ids.csv`.
- **Experiment 3: candidate-free measured point-hypothesis diagnostic.** All 4,000 interactions are evaluated. Models emit measured world-coordinate point hypotheses without candidate IDs or hidden GT in the model-facing manifest. Nearest-anchor association occurs only in evaluation. This is not unconstrained 3D reconstruction, 3D box grounding, or 3D IoU.

## Fairness and provenance

Within each experiment, all three models have identical `scene::row_index` sample hashes and GT hashes. The semantic input protocol, prompt objective, evaluator, and greedy decoding policy are shared. Model-specific chat/vision adapters remain necessary; InternVL Exp.3 uses two evidence images and 8-bit loading with 256 output tokens, while Qwen uses up to three images and 512 tokens. These differences are explicit in `run_provenance/run_provenance.csv`.

Qwen3-VL-8B Exp.1's paper run is `qwen8/outputs/exam1_qwen3vl8b_baseline`, reevaluated on the repaired 4,000-row GT. Its earlier `0.6625` value used the same final raw run before GT completion; `data/match_eval_qwen3vl8b` with Hit-All `0.60075` is a separate legacy prompt/input run. Neither value should replace the rebuilt result in `model_results.csv`.

## Ablations and limits

The Exp.1/Exp.2 ablations are descriptive hybrid/input/preprocessing/prompt ablations, not strict single-modality causal ablations. No bootstrap samples exist beyond the header-only file, so no p-value, significance, or confidence interval is reported.

The explicit location/region taxonomy contains {location['location_region_anchor_count']} canonical anchors and identifies {location['location_region_interaction_count']} interactions. The old 1,461 count is not reproducible because no committed taxonomy supported it; it must not be cited.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    exp3_lines = ["# Experiment 3 Full Results (4,000 interactions)", "",
                  "Experiment 3 is a candidate-free measured point-hypothesis diagnostic. The model receives language, target-free evidence frames, camera/gaze/hand telemetry, and scene-scale context, then emits measured Unity-world 3D points. Hidden anchors are used only by the evaluator.", "",
                  "| Model or baseline | Valid | Anchor F1 | Exact | Margin-F1@1.0 | Scene-normalized error |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in [item for item in result_rows if item["experiment"] in {"experiment3", "experiment3_baseline"}]:
        exp3_lines.append(f"| {row['model']} | {row['valid']}/{row['n']} | {float(row['anchor_f1']):.4f} | {float(row['exact']):.4f} | {float(row['margin_f1_1.0']):.4f} | {float(row['mean_scene_normalized_error']):.4f} |")
    exp3_lines += ["", "The gaze-copy baseline is reported because gaze hypotheses are exposed by the task input and provide a strong copy-based control. The diagnostic does not measure object extents, boxes, 3D IoU, or unconstrained reconstruction."]
    (OUT / "EXPERIMENT3_FULL_RESULTS_V9.md").write_text("\n".join(exp3_lines) + "\n", encoding="utf-8")
    sync_exp3_repository_reports()

    validation = f"""# Result Validation

Nine sample-level model files contain 4,000 unique `scene::row_index` keys each. Within every experiment, all three sample-set hashes and GT hashes match. Exp.1 has 4,000 mapped GT and zero unmapped rows; Exp.2 has zero missing prediction records; Exp.3 has 4,000 manifest rows and zero excluded IDs.

Qwen3-VL-30B Exp.2 has {q30_exp2['valid_count']} valid outputs and {q30_exp2['parse_failure_count']} invalid records. The two invalid records were produced on all three same-configuration retries and remain empty predictions in the 4,000-row corpus metrics. No output was manually repaired. Qwen3-VL-8B and InternVL3-38B have 4,000 valid Exp.2 outputs. All three Exp.3 runs have 4,000 valid outputs.

Machine-readable checks are in `validation/validation_results.csv` and `.json`; explicit invalid IDs are in `denominator_audit/invalid_output_ids.csv`. The 28 repaired mappings, `scene2::569`, anchor loader checks, aliases, and the sole new canonical anchor `drawer2 = (0.638, -1.227, 5.241)` are audited under `gt_completion/` and `location_region_audit/`.
"""
    (OUT / "RESULT_VALIDATION.md").write_text(validation, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp1_stats, exp1_ids = export_exp1()
    exp2_stats = export_exp2()
    exp3_stats, exp3_ids = export_exp3()
    stats = {**exp1_stats, **exp2_stats, **exp3_stats}
    export_audits(exp1_ids, exp3_ids)
    copy_repro_files()
    write_documents(stats)
    validate(stats)
    print(json.dumps({f"{key[0]}::{key[1]}": value for key, value in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
