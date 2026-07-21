#!/usr/bin/env python3
"""Report progress for the detached Qwen30B Experiment 3 ablation lanes."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


VARIANTS = ("no_hand", "no_instruction", "no_visual", "no_gaze", "no_gaze_hand")


def raw_count(path: Path) -> int:
    raw = path / "raw"
    return sum(1 for item in raw.glob("*.json") if item.is_file()) if raw.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output_root", default="ablation/exam3/outputs_qwen3vl30b_v9_input_mask_v3_full")
    parser.add_argument("--expected", type=int, default=4000)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    output_root = root / args.output_root
    process_text = subprocess.run(
        ["ps", "-eo", "pid,etime,cmd"], check=True, capture_output=True, text=True
    ).stdout
    rows = []
    for variant in VARIANTS:
        count = raw_count(output_root / variant)
        active = f"--ablation_variant {variant}" in process_text
        complete = (output_root / variant / "eval/evaluation_summary.json").exists()
        rows.append({
            "variant": variant,
            "raw_count": count,
            "expected": args.expected,
            "progress": count / args.expected,
            "active": active,
            "evaluated": complete,
        })
    print(json.dumps({"output_root": str(output_root), "variants": rows}, indent=2))


if __name__ == "__main__":
    main()
