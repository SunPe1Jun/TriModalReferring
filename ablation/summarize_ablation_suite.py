#!/usr/bin/env python3
"""Build a compact handoff report for experiment-1 and experiment-2 ablations."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine ablation reports into one markdown handoff.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--output", default="ablation/reports/ABLATION_RESULTS.md")
    return parser.parse_args()


def read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else f"Missing report: `{path}`\n"


def strip_h1(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        return "\n".join(lines[1:]).strip()
    return text.strip()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    exam1 = repo_root / "ablation/exam1/reports/EXAM1_ABLATION_RESULTS.md"
    exam2 = repo_root / "ablation/exam2/reports/EXAM2_ABLATION_RESULTS.md"

    lines = [
        "# VR-TriRef Modality Ablation Results",
        "",
        "This report covers ablations for the two established experiments only.",
        "Experiment 3 is excluded from this round.",
        "",
        "## Experiment 1",
        "",
        strip_h1(read_optional(exam1)),
        "",
        "## Experiment 2",
        "",
        strip_h1(read_optional(exam2)),
        "",
        "## Reproducibility",
        "",
        "- Experiment 1 runner: `ablation/exam1/run_exam1_ablation.sh`",
        "- Experiment 2 runner: `ablation/exam2/run_exam2_ablation.sh`",
        "- Parallel smoke: `ablation/run_parallel_smoke.sh`",
        "- Parallel full run: `ablation/run_parallel_full.sh`",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
