#!/usr/bin/env python3
"""Lightweight progress monitor for detached InternVL baseline runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

SCENE_ROWS = {
    "scene1": 800,
    "scene2": 800,
    "scene3": 800,
    "scene4_room1": 200,
    "scene4_room2": 200,
    "scene4_room3": 200,
    "scene4_room4": 200,
    "scene5": 800,
}
OK_RE = re.compile(r"\[ok\]\s+row\s+(\d+)")
ROW_RE = re.compile(r"\[row\s+(\d+)\]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def tail_text(path: Path, max_bytes: int = 65536) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        return handle.read().decode("utf-8", errors="replace")


def latest_row(text: str):
    matches = OK_RE.findall(text) or ROW_RE.findall(text)
    return int(matches[-1]) if matches else None


def snapshot(repo_root: Path, pid: int) -> dict:
    pred_root = repo_root / "internvl" / "outputs" / "exam1_internvl3_38b_baseline" / "predictions"
    counts = {}
    for scene in SCENE_ROWS:
        scene_dir = pred_root / scene
        counts[scene] = len(list(scene_dir.glob("row_*.json"))) if scene_dir.exists() else 0
    exam1_log = repo_root / "internvl" / "logs" / "exam1_internvl3_38b_baseline_full.log"
    exam2_summary = repo_root / "internvl" / "outputs" / "exam2_internvl3_38b_baseline" / "eval" / "2d_eval_summary.json"
    full_log = repo_root / "internvl" / "logs" / "internvl3_38b_full_sequence.log"
    exam1_tail = tail_text(exam1_log)
    full_tail = tail_text(full_log, 8192)
    return {
        "timestamp_utc": utc_now(),
        "pid": pid,
        "pid_alive": pid_alive(pid),
        "exam1_predictions_done": sum(counts.values()),
        "exam1_predictions_total": sum(SCENE_ROWS.values()),
        "exam1_scene_counts": counts,
        "exam1_latest_row_in_log": latest_row(exam1_tail),
        "exam2_summary_exists": exam2_summary.exists(),
        "tracebacks_in_recent_logs": exam1_tail.count("Traceback") + full_tail.count("Traceback"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/usr3/TriModal-Referring")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=96.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output) if args.output else repo_root / "internvl" / "logs" / "internvl3_38b_full_monitor.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    while time.time() < deadline:
        item = snapshot(repo_root, args.pid)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if not item["pid_alive"]:
            return 0
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
