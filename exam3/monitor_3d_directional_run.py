#!/usr/bin/env python3
"""Monitor a camera-centered 3D directional full run.

This script is intentionally read-mostly: it counts raw model outputs, checks
whether the inference process is still active, runs evaluation after inference
finishes if needed, and appends progress/results to an iteration log.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor experiment 3 full run and append iteration notes.")
    parser.add_argument("--repo_root", default="/workspace/usr3/TriModal-Referring")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--run_log", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--iteration_log", required=True)
    parser.add_argument("--interval_seconds", type=int, default=600)
    parser.add_argument("--min_valid_rate", type=float, default=0.98)
    parser.add_argument("--max_median_deg", type=float, default=15.0)
    parser.add_argument("--min_acc30_all", type=float, default=0.70)
    parser.add_argument("--once", action="store_true", help="Write one progress entry and exit.")
    return parser.parse_args()


def now_text() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def read_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text.rstrip() + "\n")


def count_manifest_samples(path: Path) -> int:
    """Count experiment-3 inference samples.

    The exam2 manifest has one row per referent/panel combination, while
    run_qwen3vl_3d_directional.py groups rows by (scene, row_index). Progress
    should therefore use unique event groups, not raw CSV rows.
    """
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            keys = {
                (str(row.get("scene", "")).strip(), str(row.get("row_index", "")).strip())
                for row in reader
                if str(row.get("scene", "")).strip() and str(row.get("row_index", "")).strip()
            }
    except OSError:
        return 0
    return len(keys)


def count_raw_outputs(output_root: Path) -> int:
    raw_dir = output_root / "predictions" / "raw"
    if not raw_dir.exists():
        return 0
    return sum(1 for _path in raw_dir.glob("*.json"))


def latest_run_log_line(path: Path) -> str:
    if not path.exists():
        return "run log missing"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"could not read run log: {exc}"
    for line in reversed(lines):
        line = line.strip()
        if line:
            return line
    return "run log empty"


def process_running(output_root: Path) -> bool:
    try:
        result = subprocess.run(["ps", "-ef"], check=False, capture_output=True, text=True)
    except OSError:
        return False
    marker = str(output_root)
    for line in result.stdout.splitlines():
        if "run_qwen3vl_3d_directional.py" in line and marker in line:
            return True
    return False


def run_evaluation(repo_root: Path, output_root: Path) -> Optional[str]:
    pred_csv = output_root / "predictions" / "qwen3vl_3d_directional_predictions.csv"
    if not pred_csv.exists():
        return "prediction CSV not found"
    eval_script = repo_root / "exam3" / "evaluate_3d_directional.py"
    cmd = [
        sys.executable,
        str(eval_script),
        "--pred_csv",
        str(pred_csv),
        "--output_dir",
        str(output_root / "eval"),
        "--report_path",
        str(output_root / "report.md"),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-5:]
        return "evaluation failed: " + " | ".join(tail)
    return None


def load_summary(output_root: Path) -> Optional[Dict[str, Any]]:
    path = output_root / "eval" / "3d_directional_eval_summary.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def metric(summary: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = summary.get("overall", {}).get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def final_decision(summary: Dict[str, Any], args: argparse.Namespace) -> str:
    valid_rate = metric(summary, "valid_rate")
    median_deg = metric(summary, "median_angular_error_deg_valid_only", default=9999.0)
    acc30 = metric(summary, "angular_accuracy_at_30_deg_all_samples")
    checks = [
        ("valid_rate", valid_rate >= args.min_valid_rate, format_percent(valid_rate), f">= {format_percent(args.min_valid_rate)}"),
        ("median_deg", median_deg <= args.max_median_deg, f"{median_deg:.2f}", f"<= {args.max_median_deg:.2f}"),
        ("acc30_all", acc30 >= args.min_acc30_all, format_percent(acc30), f">= {format_percent(args.min_acc30_all)}"),
    ]
    failed = [f"{name} {observed} (target {target})" for name, ok, observed, target in checks if not ok]
    if not failed:
        return "ACCEPT baseline: all predefined engineering thresholds passed."
    return "NEEDS ITERATION: " + "; ".join(failed)


def append_progress(args: argparse.Namespace, state: Dict[str, Any]) -> bool:
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve()
    run_log = Path(args.run_log).resolve()
    manifest = Path(args.manifest).resolve()
    iteration_log = Path(args.iteration_log).resolve()

    expected = count_manifest_samples(manifest)
    raw_count = count_raw_outputs(output_root)
    running = process_running(output_root)
    latest_line = latest_run_log_line(run_log)
    progress = (raw_count / expected) if expected else 0.0

    progress_text = (
        f"\n### Monitor Update - {now_text()}\n"
        f"- output_root: `{output_root}`\n"
        f"- raw outputs: {raw_count} / {expected} ({progress:.2%})\n"
        f"- inference_process_running: {running}\n"
        f"- latest run log line: `{latest_line}`\n"
    )
    append_log(iteration_log, progress_text)

    summary = load_summary(output_root)
    if summary is None and not running and raw_count > 0:
        eval_error = run_evaluation(repo_root, output_root)
        if eval_error:
            append_log(iteration_log, f"\n### Evaluation Attempt - {now_text()}\n- status: {eval_error}\n")
        summary = load_summary(output_root)

    if summary is not None and not state.get("final_logged"):
        overall = summary.get("overall", {})
        thresholds = {
            "min_valid_rate": args.min_valid_rate,
            "max_median_deg": args.max_median_deg,
            "min_acc30_all": args.min_acc30_all,
        }
        final_text = (
            f"\n## Baseline Full-Run Result - {now_text()}\n"
            f"- total_samples: {overall.get('total_samples')}\n"
            f"- valid_prediction_count: {overall.get('valid_prediction_count')}\n"
            f"- invalid_count: {overall.get('invalid_count')}\n"
            f"- valid_rate: {format_percent(metric(summary, 'valid_rate'))}\n"
            f"- mean_angular_error_deg_valid_only: {metric(summary, 'mean_angular_error_deg_valid_only', 0.0):.2f}\n"
            f"- median_angular_error_deg_valid_only: {metric(summary, 'median_angular_error_deg_valid_only', 0.0):.2f}\n"
            f"- acc@5_all: {format_percent(metric(summary, 'angular_accuracy_at_5_deg_all_samples'))}\n"
            f"- acc@10_all: {format_percent(metric(summary, 'angular_accuracy_at_10_deg_all_samples'))}\n"
            f"- acc@15_all: {format_percent(metric(summary, 'angular_accuracy_at_15_deg_all_samples'))}\n"
            f"- acc@30_all: {format_percent(metric(summary, 'angular_accuracy_at_30_deg_all_samples'))}\n"
            f"- invalid_reason_counts: `{json.dumps(overall.get('invalid_reason_counts', {}), ensure_ascii=False)}`\n"
            f"- thresholds: `{json.dumps(thresholds, sort_keys=True)}`\n"
            f"- decision: {final_decision(summary, args)}\n"
            f"- summary_json: `{output_root / 'eval' / '3d_directional_eval_summary.json'}`\n"
            f"- report_md: `{output_root / 'report.md'}`\n"
        )
        append_log(iteration_log, final_text)
        state["final_logged"] = True
        return True
    return bool(summary is not None and state.get("final_logged"))


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    state_path = output_root / "monitor_state_3d_directional.json"
    state = read_state(state_path)
    while True:
        done = append_progress(args, state)
        state["last_monitor_time"] = now_text()
        write_state(state_path, state)
        if args.once or done:
            break
        time.sleep(max(10, args.interval_seconds))


if __name__ == "__main__":
    main()
