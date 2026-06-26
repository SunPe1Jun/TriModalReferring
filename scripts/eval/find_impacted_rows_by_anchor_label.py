#!/usr/bin/env python3
"""Find rows impacted by a missing anchor label in cleaned scene annotations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List


class ImpactedRowsError(Exception):
    """Raised when impacted-row detection fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find row indices whose cleaned annotations mention a target anchor label.")
    parser.add_argument("--input_csv", required=True, help="Cleaned CSV path, e.g. scene1_cleaned_v2.csv")
    parser.add_argument("--target_label", required=True, help="Anchor label to search for, e.g. Platform1 or truck2")
    parser.add_argument("--output_txt", required=True, help="Path to save one zero-based row_index per line.")
    parser.add_argument("--output_csv", required=True, help="Path to save detailed impacted rows.")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return (value or "").strip()


def split_labels(text: str) -> List[str]:
    return [part.strip() for part in normalize_text(text).split(",") if part and part.strip()]


def main() -> int:
    args = parse_args()
    try:
        input_csv = Path(args.input_csv).expanduser().resolve()
        output_txt = Path(args.output_txt).expanduser().resolve()
        output_csv = Path(args.output_csv).expanduser().resolve()
        target_label = normalize_text(args.target_label)
        if not target_label:
            raise ImpactedRowsError("target_label must not be empty.")
        if not input_csv.exists() or not input_csv.is_file():
            raise ImpactedRowsError(f"Input CSV does not exist: {input_csv}")

        impacted_rows: List[Dict[str, str]] = []
        with input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ImpactedRowsError(f"CSV has no header: {input_csv}")
            for zero_based_index, row in enumerate(reader):
                cleaned = normalize_text(row.get("Referents_Cleaned", ""))
                anchor_mapped = normalize_text(row.get("Referents_AnchorMapped", ""))
                if target_label in split_labels(cleaned) or target_label in split_labels(anchor_mapped):
                    impacted_rows.append(
                        {
                            "row_index": str(zero_based_index),
                            "number": normalize_text(row.get("Number", "")),
                            "instruction": normalize_text(row.get("Instruction", "")),
                            "referents_cleaned": cleaned,
                            "referents_anchor_mapped": anchor_mapped,
                        }
                    )

        output_txt.parent.mkdir(parents=True, exist_ok=True)
        output_txt.write_text("\n".join(row["row_index"] for row in impacted_rows) + ("\n" if impacted_rows else ""), encoding="utf-8")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["row_index", "number", "instruction", "referents_cleaned", "referents_anchor_mapped"],
            )
            writer.writeheader()
            writer.writerows(impacted_rows)

        print(f"Found {len(impacted_rows)} impacted rows for label: {target_label}")
        print(f"Saved row indices to: {output_txt}")
        print(f"Saved detailed CSV to: {output_csv}")
        return 0
    except ImpactedRowsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
