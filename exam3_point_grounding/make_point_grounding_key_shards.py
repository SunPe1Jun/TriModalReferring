#!/usr/bin/env python3
"""Create non-overlapping scene:row_index key shards for Experiment 3 runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from point_grounding_common import normalize_text, read_csv_rows, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Experiment 3 eval keys into non-overlapping shards.")
    parser.add_argument("--repo_root", default=".")
    parser.add_argument("--gt_manifest", default="exam3_point_grounding/outputs_full_v9_20260709/gt_manifest_eval.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefix", default="shard")
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--existing_raw_dirs", nargs="*", default=[])
    parser.add_argument("--sample_keys_file", help="Optional allowed scene:row_index key list.")
    parser.add_argument("--round_robin", action="store_true", help="Use round-robin assignment instead of contiguous chunks.")
    return parser.parse_args()


def key_from_row(row: dict[str, str]) -> str:
    return f"{normalize_text(row.get('scene'))}:{normalize_text(row.get('row_index'))}"


def raw_name(key: str) -> str:
    scene, row_index = key.rsplit(":", 1)
    return f"{scene}_row_{row_index}.json"


def has_existing_raw(key: str, raw_dirs: Sequence[Path]) -> bool:
    name = raw_name(key)
    return any((raw_dir / name).exists() for raw_dir in raw_dirs)


def split_keys(keys: Sequence[str], num_shards: int, round_robin: bool) -> List[List[str]]:
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    if round_robin:
        for idx, key in enumerate(keys):
            shards[idx % num_shards].append(key)
        return shards
    chunk = (len(keys) + num_shards - 1) // num_shards
    for shard_idx in range(num_shards):
        start = shard_idx * chunk
        end = min(len(keys), start + chunk)
        shards[shard_idx].extend(keys[start:end])
    return shards


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num_shards must be positive")
    repo_root = Path(args.repo_root).resolve()
    gt_rows = read_csv_rows((repo_root / args.gt_manifest).resolve())
    raw_dirs = [Path(item).resolve() for item in args.existing_raw_dirs]
    all_keys = [key_from_row(row) for row in gt_rows]
    if args.sample_keys_file:
        allowed = {
            normalize_text(line)
            for line in Path(args.sample_keys_file).read_text(encoding="utf-8").splitlines()
            if normalize_text(line) and not normalize_text(line).startswith("#")
        }
        unknown = allowed - set(all_keys)
        if unknown:
            raise ValueError(f"sample keys not found in GT manifest: {sorted(unknown)}")
        all_keys = [key for key in all_keys if key in allowed]
    remaining_keys = [key for key in all_keys if not has_existing_raw(key, raw_dirs)]
    shards = split_keys(remaining_keys, args.num_shards, args.round_robin)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_files = []
    for idx, shard in enumerate(shards):
        path = output_dir / f"{args.prefix}_{idx}.txt"
        path.write_text("\n".join(shard) + ("\n" if shard else ""), encoding="utf-8")
        shard_files.append(str(path))
    summary = {
        "gt_total": len(all_keys),
        "existing_raw_dirs": [str(path) for path in raw_dirs],
        "existing_count": len(all_keys) - len(remaining_keys),
        "remaining_count": len(remaining_keys),
        "num_shards": args.num_shards,
        "round_robin": bool(args.round_robin),
        "shard_files": shard_files,
        "shard_counts": [len(shard) for shard in shards],
    }
    write_json(output_dir / f"{args.prefix}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
