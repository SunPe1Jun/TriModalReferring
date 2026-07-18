#!/usr/bin/env python3
"""Write a GT-manifest subset matching scene:row_index keys."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--keys", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keys = {
        line.strip()
        for line in Path(args.keys).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    with Path(args.input).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise RuntimeError("input GT manifest has no header")
        rows = [row for row in reader if f"{row.get('scene')}:{row.get('row_index')}" in keys]
    missing = keys - {f"{row.get('scene')}:{row.get('row_index')}" for row in rows}
    if missing:
        raise RuntimeError(f"keys missing from GT manifest: {sorted(missing)[:10]}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
