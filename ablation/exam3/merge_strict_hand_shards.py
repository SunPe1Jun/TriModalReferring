#!/usr/bin/env python3
"""Merge non-overlapping strict-hand prediction CSV shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def key(row: Dict[str, str]) -> str:
    return f"{row.get('scene')}::{int(row.get('row_index', '0'))}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction_csvs", nargs="+", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    rows: List[Dict[str, str]] = []
    fields: List[str] = []
    for csv_path in args.prediction_csvs:
        with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames:
                for field in reader.fieldnames:
                    if field not in fields:
                        fields.append(field)
            rows.extend(reader)
    by_key = {key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise RuntimeError(f"duplicate strict-hand shard keys: {len(rows)} rows, {len(by_key)} unique")
    rows = sorted(by_key.values(), key=lambda row: (row.get("scene", ""), int(row.get("row_index", "0"))))
    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print({"rows": len(rows), "unique_ids": len(by_key), "output": str(output)})


if __name__ == "__main__":
    main()
