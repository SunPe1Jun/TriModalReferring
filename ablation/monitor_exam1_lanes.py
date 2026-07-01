#!/usr/bin/env python3
"""Monitor the recovered Experiment 1 ablation lanes."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import subprocess
import time
from pathlib import Path


PATTERNS = [
    re.compile(pattern)
    for pattern in [
        r"^========== exam1 variant=",
        r"^\[infer\]",
        r"^\[row ",
        r"^\[ok\] row ",
        r"^\[eval\]",
        r"^Finished exam1",
        r"Traceback",
        r"Error|ERROR|error",
    ]
]


def pid_state(pid: int | None) -> str:
    if not pid:
        return "none"
    stat_path = Path(f"/proc/{pid}/stat")
    if not stat_path.exists():
        return "missing"
    try:
        fields = stat_path.read_text(encoding="utf-8", errors="replace").split()
    except OSError as exc:
        return f"read_error:{exc}"
    if len(fields) >= 3 and fields[2] == "Z":
        return "zombie"
    return "alive"


def tail_lines(path: Path, n: int = 160) -> list[str]:
    if not path.exists():
        return []
    try:
        return [line.rstrip("\n") for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]]
    except OSError as exc:
        return [f"<read error: {exc}>"]


def latest_relevant(path: Path) -> str:
    for line in reversed(tail_lines(path)):
        if any(pattern.search(line) for pattern in PATTERNS):
            return line
    return "<none>"


def count_json_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.json") if item.is_file())


def exam1_counts(repo_root: Path) -> str:
    output_root = repo_root / "ablation" / "exam1" / "outputs"
    if not output_root.exists():
        return "<missing outputs>"
    parts: list[str] = []
    for variant_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        pred_root = variant_dir / "predictions"
        scene_counts = []
        total = 0
        if pred_root.exists():
            for scene_dir in sorted(path for path in pred_root.iterdir() if path.is_dir()):
                count = count_json_files(scene_dir)
                total += count
                if count:
                    scene_counts.append(f"{scene_dir.name}={count}")
        detail = ",".join(scene_counts) if scene_counts else "none"
        parts.append(f"{variant_dir.name}:{total}({detail})")
    return "; ".join(parts) if parts else "<none>"


def gpu_snapshot() -> str:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return f"<nvidia-smi error: {type(exc).__name__}: {exc}>"
    if result.returncode != 0:
        return f"<nvidia-smi rc={result.returncode}: {result.stderr.strip()}>"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return " | ".join(lines) if lines else "<empty>"


def write_snapshot(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root)
    monitor_log = Path(args.monitor_log)
    monitor_log.parent.mkdir(parents=True, exist_ok=True)
    gpu0_log = Path(args.gpu0_log)
    gpu1_log = Path(args.gpu1_log)

    block = [
        f"[{dt.datetime.now().isoformat(timespec='seconds')}]",
        f"gpu0_pid={args.gpu0_pid} state={pid_state(args.gpu0_pid)} last={latest_relevant(gpu0_log)}",
        f"gpu1_pid={args.gpu1_pid} state={pid_state(args.gpu1_pid)} last={latest_relevant(gpu1_log)}",
        f"gpu_snapshot={gpu_snapshot()}",
        f"exam1_counts={exam1_counts(repo_root)}",
        "",
    ]
    with monitor_log.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/workspace/usr3/TriModal-Referring")
    parser.add_argument("--gpu0-pid", type=int, required=True)
    parser.add_argument("--gpu1-pid", type=int, required=True)
    parser.add_argument("--gpu0-log", default="/workspace/usr3/TriModal-Referring/ablation/logs/exam1_gpu0_full.log")
    parser.add_argument("--gpu1-log", default="/workspace/usr3/TriModal-Referring/ablation/logs/exam1_gpu1_recovery.log")
    parser.add_argument("--monitor-log", default="/workspace/usr3/TriModal-Referring/ablation/logs/exam1_lanes_recovery_monitor.log")
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=120.0)
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    while True:
        write_snapshot(args)
        states = {pid_state(args.gpu0_pid), pid_state(args.gpu1_pid)}
        if states <= {"missing", "zombie", "none"}:
            break
        if time.time() >= deadline:
            break
        time.sleep(max(10, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
