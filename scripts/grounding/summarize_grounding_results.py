#!/usr/bin/env python3
"""Summarize grounding output results into a readable CSV and Markdown report."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Dict, List


class SummaryError(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize grounding output results.")
    parser.add_argument("--pred-csv", required=True, help="Path to grounding output CSV.")
    parser.add_argument("--output-csv", required=True, help="Path to simplified summary CSV.")
    parser.add_argument("--output-md", required=True, help="Path to markdown summary report.")
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists() or not csv_path.is_file():
        raise SummaryError(f"CSV file does not exist or is not a file: {csv_path}")
    with csv_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SummaryError(f"CSV file has no header row: {csv_path}")
        return [{key: (value or '').strip() for key, value in row.items()} for row in reader if any(row.values())]


def parse_float(value: str):
    if value == '':
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_summary_csv(output_csv: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        'event_id', 'referent_text', 'confidence', 'u_norm', 'v_norm',
        'x_world', 'y_world', 'z_world', 'parse_ok', 'error_message'
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_markdown(output_md: Path, rows: List[Dict[str, str]]) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    parse_ok_rows = [row for row in rows if row.get('parse_ok', '').lower() == 'true']
    parse_fail_rows = [row for row in rows if row.get('parse_ok', '').lower() != 'true']
    confidences = [parse_float(row.get('confidence', '')) for row in parse_ok_rows]
    confidences = [value for value in confidences if value is not None]

    lines = [
        '# Grounding Result Summary',
        '',
        f'- Total rows: {total}',
        f'- Parsed successfully: {len(parse_ok_rows)}',
        f'- Parse failed: {len(parse_fail_rows)}',
        f'- Average confidence: {statistics.mean(confidences):.4f}' if confidences else '- Average confidence: n/a',
        '',
        '## Simplified Table',
        '',
        '| event_id | referent_text | confidence | u_norm | v_norm | parse_ok | error_message |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for row in rows:
        lines.append(
            f"| {row.get('event_id', '')} | {row.get('referent_text', '')} | {row.get('confidence', '')} | {row.get('u_norm', '')} | {row.get('v_norm', '')} | {row.get('parse_ok', '')} | {row.get('error_message', '')} |"
        )
    output_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    args = parse_args()
    try:
        pred_rows = load_rows(Path(args.pred_csv).expanduser().resolve())
        write_summary_csv(Path(args.output_csv).expanduser().resolve(), pred_rows)
        write_markdown(Path(args.output_md).expanduser().resolve(), pred_rows)
        print(f"Saved summary CSV to: {Path(args.output_csv).expanduser().resolve()}")
        print(f"Saved summary Markdown to: {Path(args.output_md).expanduser().resolve()}")
        return 0
    except SummaryError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
