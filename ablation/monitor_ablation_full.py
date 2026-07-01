#!/usr/bin/env python3
"""Lightweight detached monitor for the ablation full runs."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from pathlib import Path


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    return Path(f"/proc/{pid}").exists()


def tail_lines(path: Path, n: int = 120) -> list[str]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return [f"<read error: {exc}>"]
    return [line.rstrip("\n") for line in lines[-n:]]


def latest_matching(lines: list[str], patterns: list[str]) -> str:
    regexes = [re.compile(p) for p in patterns]
    for line in reversed(lines):
        if any(r.search(line) for r in regexes):
            return line
    return "<none>"


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(f"*{suffix}") if p.is_file())


def write_snapshot(args: argparse.Namespace) -> None:
    root = Path(args.repo_root)
    log_dir = root / "ablation" / "logs"
    monitor_log = Path(args.monitor_log)
    monitor_log.parent.mkdir(parents=True, exist_ok=True)

    exam1_log = root / "ablation" / "logs" / "exam1_full.log"
    exam2_log = root / "ablation" / "logs" / "exam2_full.log"
    exam1_lines = tail_lines(exam1_log)
    exam2_lines = tail_lines(exam2_log)

    exam1_last = latest_matching(
        exam1_lines,
        [
            r"^========== exam1 variant=",
            r"^\[infer\]",
            r"^\[row ",
            r"^\[ok\] row ",
            r"^\[eval\]",
            r"^\[summary\]",
            r"Traceback",
            r"Error|ERROR|error",
        ],
    )
    exam2_last = latest_matching(
        exam2_lines,
        [
            r"^========== exam2 variant=",
            r"^\[ok\] ",
            r"^\[eval\]",
            r"^\[summary\]",
            r"Traceback",
            r"Error|ERROR|error",
        ],
    )

    exam1_counts = []
    exam1_root = root / "ablation" / "exam1" / "outputs"
    if exam1_root.exists():
        for variant_dir in sorted(p for p in exam1_root.iterdir() if p.is_dir()):
            predictions = variant_dir / "predictions"
            exam1_counts.append(f"{variant_dir.name}:{count_files(predictions, '.json')}")

    exam2_counts = []
    exam2_root = root / "ablation" / "exam2" / "outputs"
    if exam2_root.exists():
        for variant_dir in sorted(p for p in exam2_root.iterdir() if p.is_dir()):
            csv_path = variant_dir / "predictions" / "qwen3vl_2d_predictions.csv"
            rows = 0
            if csv_path.exists():
                try:
                    rows = max(0, sum(1 for _ in csv_path.open("r", encoding="utf-8", errors="replace")) - 1)
                except OSError:
                    rows = -1
            exam2_counts.append(f"{variant_dir.name}:{rows}")

    now = dt.datetime.now().isoformat(timespec="seconds")
    block = [
        f"[{now}]",
        f"exam1_pid={args.exam1_pid} alive={pid_alive(args.exam1_pid)} last={exam1_last}",
        f"exam1_json_counts={', '.join(exam1_counts) if exam1_counts else '<none>'}",
        f"exam2_pid={args.exam2_pid} alive={pid_alive(args.exam2_pid)} last={exam2_last}",
        f"exam2_csv_rows={', '.join(exam2_counts) if exam2_counts else '<none>'}",
        "",
    ]
    with monitor_log.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/usr3/TriModal-Referring")
    parser.add_argument("--exam1-pid", type=int, default=None)
    parser.add_argument("--exam2-pid", type=int, default=None)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=72.0)
    parser.add_argument(
        "--monitor-log",
        default="/workspace/usr3/TriModal-Referring/ablation/logs/ablation_full_monitor.log",
    )
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    while True:
        write_snapshot(args)
        if not pid_alive(args.exam1_pid) and not pid_alive(args.exam2_pid):
            break
        if time.time() >= deadline:
            break
        time.sleep(max(10, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
