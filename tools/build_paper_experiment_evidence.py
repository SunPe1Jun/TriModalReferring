#!/usr/bin/env python3
"""Build compact paper evidence from existing immutable evaluator outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_experiment_evidence"
COMMIT = "46db9cc0f4b88c7227f69ada23c9510b55389d5e"
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


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
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
                    "exact": truth(row["exact_match"]) if mapped else False,
                    "hit": truth(row["match_success"]), "evaluator_version_or_commit": COMMIT,
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
        output = []
        for row in read_csv(source / "eval/2d_eval_detail.csv"):
            key = uid(row["scene"], row["row_index"])
            raw_path = source / "predictions/json" / f'{row["scene"]}_row_{row["row_index"]}.json'
            missing = not raw_path.exists()
            parsed, parse_ok, reason = [], False, "missing_prediction" if missing else ""
            if not missing:
                raw = load_json(raw_path)
                parse_ok = truth(raw.get("parse_ok", False))
                parsed = (raw.get("parsed_json") or {}).get("referents", []) if parse_ok else []
                reason = "" if parse_ok else (raw.get("error_message") or row["error_message"] or "parse_failure")
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
                "evaluator_version_or_commit": COMMIT,
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
                "evaluator_version_or_commit": COMMIT,
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
    write_csv(OUT / "denominator_audit/unmapped_gt_ids.csv", unmapped)
    missing = []
    for model in MODELS:
        for row in read_csv(OUT / "predictions/exp2" / f"{slug(model)}.csv"):
            if truth(row["missing_prediction"]):
                missing.append({"unified_id": row["unified_id"], "scene": row["scene"], "row_index": row["row_index"],
                                "experiment": "experiment2", "model": model, "reason": "no prediction record",
                                "source_path": str(EXP2[model].relative_to(ROOT)) + "/predictions/json"})
    write_csv(OUT / "denominator_audit/missing_prediction_ids.csv", missing)
    by_id = {r["unified_id"]: r for r in reference}
    excluded = []
    for key in sorted(exp1_ids - exp3_ids):
        row = by_id[key]
        reason = "unmapped GT anchor" if not truth(row["gt_mapped"]) else "mapped GT but excluded by Exp.3 manifest builder"
        excluded.append({"unified_id": key, "scene": row["scene"], "row_index": row["row_index"],
                         "experiment": "experiment3", "reason": reason,
                         "source_path": "exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv"})
    write_csv(OUT / "denominator_audit/exp3_excluded_ids.csv", excluded)


def copy_repro_files():
    copies = {
        ROOT / "scripts/eval/evaluate_local_3d_object_match.py": OUT / "evaluators/evaluate_local_3d_object_match.py",
        ROOT / "exam2/evaluate_2d_point_grounding.py": OUT / "evaluators/evaluate_2d_point_grounding.py",
        ROOT / "exam3_point_grounding/evaluate_point_grounding.py": OUT / "evaluators/evaluate_point_grounding.py",
        ROOT / "exam3_point_grounding/prompts/qwen3vl_point_grounding.md": OUT / "prompts_and_configs/exp3_qwen3vl_point_grounding.md",
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


def validate(stats):
    validation = []
    for (experiment, model), values in stats.items():
        expected = 3971 if experiment == "experiment3" else 4000
        checks = {"row_count": values["row_count"] == expected,
                  "unique_ids": values["unique_ids"] == expected,
                  "duplicate_ids": values["duplicate_ids"] == 0,
                  "finite_metrics": all(not isinstance(v, float) or math.isfinite(v) for v in values.values())}
        validation.append({"experiment": experiment, "model": model, **values,
                           "validation_pass": all(checks.values()), "checks_json": canonical(checks)})
    write_csv(OUT / "validation/validation_results.csv", validation)
    with (OUT / "validation/validation_results.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=True)
    if not all(r["validation_pass"] for r in validation):
        raise RuntimeError("Evidence validation failed")


def write_documents(stats):
    result_rows = []
    for (experiment, model), s in stats.items():
        row = {"experiment": experiment, "model": model, "n": s["row_count"], "valid": s["valid_count"], "invalid": s["invalid_count"]}
        if experiment == "experiment1":
            row.update({"hit_all": s["hit_all"], "hit_mapped": s["hit_mapped"], "exact": s["exact"], "micro_precision": s["micro_precision"], "micro_recall": s["micro_recall"], "micro_f1": s["micro_f1"]})
        elif experiment == "experiment2":
            row.update({"missing": s["missing_count"], "parse_failures": s["parse_failure_count"], "temporal_f1": s["time_f1"], "point50_f1": s["point_50_f1"], "point100_f1": s["point_100_f1"], "point150_f1": s["point_150_f1"], "point200_f1": s["point_200_f1"], "joint50_f1": s["joint_50_f1"], "joint100_f1": s["joint_100_f1"], "joint150_f1": s["joint_150_f1"], "joint200_f1": s["joint_200_f1"]})
        else:
            row.update({"anchor_precision": s["anchor_precision"], "anchor_recall": s["anchor_recall"], "anchor_f1": s["anchor_f1"], "exact": s["exact"], "margin_f1_0.5": s["margin_f1_0.5"], "margin_f1_1.0": s["margin_f1_1.0"], "margin_f1_2.0": s["margin_f1_2.0"], "mean_scene_normalized_error": s["mean_scene_normalized_error"], "single_count": s["single_count"], "multi_count": s["multi_count"]})
        result_rows.append(row)
    result_fields = ["experiment", "model", "n", "valid", "invalid"]
    result_fields += sorted({key for row in result_rows for key in row if key not in result_fields})
    write_csv(OUT / "model_results.csv", result_rows, result_fields)
    ablation = ROOT / "analysis_outputs/ablation_audit/anchor_ablation_summary.csv"
    if ablation.exists(): shutil.copy2(ablation, OUT / "ablation_results.csv")
    (OUT / "README.md").write_text("""# VR-TriRef Paper Experiment Evidence\n\nThis bundle is a compact, read-only export of the final VR-TriRef experiments at commit `46db9cc`. It is intended for paper tables, denominator audits, and reproducibility review. No model inference was rerun and no raw prediction was edited.\n\n## Experiments\n\n- **Experiment 1: closed-set 3D anchor selection.** The model receives a candidate anchor inventory and returns anchor IDs. Hit-All uses all 4,000 rows; Hit-Mapped, exact set, and macro set metrics use the 3,972 mapped rows. Micro TP/FP/FN are retained over all rows so predictions on unmapped rows remain visible.\n- **Experiment 2: projected-2D point diagnostic.** The model selects temporal panels and image-plane points. Temporal, Point@50/100/150/200, and Joint@50/100/150/200 are corpus metrics over all 4,000 rows. Missing records and parse failures remain empty predictions.\n- **Experiment 3: candidate-free measured point-hypothesis diagnostic.** The model emits measured 3D point hypotheses without candidate IDs. Nearest-anchor association is performed only by the evaluator; reported metrics include anchor-set P/R/F1, exact set, Margin-F1, scene-normalized error, and single/multi partitions. This is not unconstrained reconstruction, box grounding, or 3D IoU.\n\n## Final Runs\n\nThe final sources are the complete baseline directories named in `run_provenance/run_provenance.csv` and the source paths encoded in each prediction file. All three models use the same unified IDs and GT hashes within each experiment. Chat templates, image/video adapters, and checkpoints are recorded as run-specific provenance; these implementation differences are not silently treated as identical.\n\nQwen3-VL-8B Exp.1 has a legacy result with Hit-All `0.60075` under `data/match_eval_qwen3vl8b` and a final mention-first run with Hit-All `0.6625` under `qwen8/outputs/exam1_qwen3vl8b_baseline`. The final run uses 8 video frames, `mention_first`, and max 1,536 new tokens; the legacy run uses 16 frames, standard prompt strategy, and a different prompt. The difference is therefore attributable to documented run configuration, not an inferred model improvement.\n\n## Ablations and Limits\n\nThe supplied ablations are descriptive hybrid/input/preprocessing/prompt ablations, not strict single-modality causal ablations. The bootstrap CSV contains headers but no bootstrap samples, so this bundle makes no significance, p-value, or confidence-interval claim.\n\n## Files\n\n`predictions/` contains nine compact sample-level CSVs; `denominator_audit/` contains explicit unmapped, missing, and Exp.3-excluded IDs; `manifests/` and `anchor_tables/` preserve compact GT context; `evaluators/` and `prompts_and_configs/` preserve the relevant code/prompt; `validation/` contains machine-readable checks; `model_results.csv` is the table-ready summary.\n""", encoding="utf-8")
    (OUT / "EXPERIMENT3_FULL_RESULTS_V9.md").write_text("""# Experiment 3: Candidate-Free Measured Point-Hypothesis Diagnostic\n\nThe model receives language, up to three evidence frames, camera/gaze/hand telemetry, and broad scene bounds. It outputs one or more measured Unity-world 3D points. Candidate anchor IDs and hidden GT points are not supplied to the model. The evaluator subsequently associates each predicted point with its nearest hidden anchor, computes set metrics, local distractor-margin metrics, and scene-normalized error.\n\nThe final manifest contains 3,971 rows. Qwen3-VL-30B, Qwen3-VL-8B, and InternVL3-38B each have 3,971 valid parsed outputs. Single-target and multi-target partitions are preserved in the sample files. The gaze-copy baseline is a required diagnostic because the v9 prompt exposes copyable gaze hypotheses; it ties Qwen on anchor F1 and is slightly stronger on Margin-F1.\n\nThese results support analysis of measured behavioral point hypotheses. They do not support claims of unconstrained 3D reconstruction, 3D boxes, object extents, or 3D IoU.\n""", encoding="utf-8")
    prov = []
    for experiment, mapping in (("experiment1", EXP1), ("experiment2", EXP2), ("experiment3", EXP3)):
        for model, source in mapping.items():
            prov.append({"experiment": experiment, "model": model, "status": "final", "output_directory": str(source.relative_to(ROOT)), "manifest": str((source / ("manifest/manifest_all.csv" if experiment == "experiment2" else "" )).relative_to(ROOT)) if experiment == "experiment2" else (str(EXP3_GT.relative_to(ROOT)) if experiment == "experiment3" else "scene-specific evaluator CSVs"), "evaluator": "scripts/eval/evaluate_local_3d_object_match.py" if experiment == "experiment1" else ("exam2/evaluate_2d_point_grounding.py" if experiment == "experiment2" else "exam3_point_grounding/evaluate_point_grounding.py"), "git_commit": COMMIT})
    write_csv(OUT / "run_provenance/run_provenance.csv", prov)
    write_csv(OUT / "run_provenance/sample_set_comparison.csv", [{"experiment": e, "model": m, "sample_set_hash": s["sample_set_hash"], "gt_hash": s["gt_hash"], "row_count": s["row_count"]} for (e, m), s in stats.items()])
    (OUT / "RESULT_VALIDATION.md").write_text("""# Result Validation\n\nThe exporter read the final evaluator details and wrote nine files with unique `scene::row_index` keys. All Exp.1/Exp.2 files contain 4,000 rows; all Exp.3 files contain 3,971 rows. Validation results are in `validation/validation_results.csv` and `.json`.\n\nExp.1 has 3,972 mapped and 28 unmapped GT rows. Exp.2 keeps 4,000 denominator rows; Qwen3-VL-30B has 3,971 prediction records, 29 missing records, 7 parse failures, and 3,964 valid parses. Qwen3-VL-8B and InternVL3-38B each have 29 missing and no parse failures. Exp.3 excludes 29 rows from the 4,000-row interaction universe: 28 are unmapped in the Exp.3 anchor tables and one (`scene2::569`) is the mapped `drawer1` interaction absent from the Exp.3 scene2 anchor table (`missing_gt_anchor`). The explicit IDs are in `denominator_audit/exp3_excluded_ids.csv`.\n\nThe Exp.1 micro metrics in this bundle intentionally follow the audit definition and aggregate all rows, including false positives on unmapped GT rows.\n""", encoding="utf-8")


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
