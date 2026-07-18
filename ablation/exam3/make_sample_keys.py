#!/usr/bin/env python3
"""Create deterministic scene:row_index keys from a multiline CSV manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    with Path(args.manifest).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [f"{row['scene']}:{row['row_index']}" for row in rows[: args.limit]]
    if len(keys) != args.limit:
        raise RuntimeError(f"requested {args.limit} keys but manifest has {len(rows)} rows")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(keys) + "\n", encoding="utf-8")
    print(f"wrote {len(keys)} keys to {output}")


if __name__ == "__main__":
    main()
