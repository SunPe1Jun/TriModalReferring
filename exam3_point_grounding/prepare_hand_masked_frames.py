#!/usr/bin/env python3
"""Precompute image-level hand masks for a frozen Experiment 3 manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from hand_masking import prepare_manifest_hand_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--audit_path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = prepare_manifest_hand_masks(
        Path(args.manifest), Path(args.output_dir), Path(args.audit_path), overwrite=args.overwrite
    )
    print({key: value for key, value in summary.items() if key != "panels"})


if __name__ == "__main__":
    main()
