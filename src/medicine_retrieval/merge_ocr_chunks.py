"""Merge chunked OCR CSV outputs into one OCR result file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OCR chunk CSV files.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--image-pattern", default="easyocr_gpu_chunk_*_*.csv")
    parser.add_argument("--line-pattern", default="easyocr_gpu_chunk_*_*_lines.csv")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--line-output-csv", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def dedupe_by_image_id(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_image_id: dict[str, dict[str, str]] = {}
    for row in rows:
        by_image_id[row["image_id"]] = row
    return [by_image_id[key] for key in sorted(by_image_id)]


def main() -> None:
    args = parse_args()
    image_paths = sorted(
        path
        for path in args.input_dir.glob(args.image_pattern)
        if "_lines" not in path.stem and "test" not in path.stem
    )
    if not image_paths:
        raise SystemExit(f"No OCR image chunks found in {args.input_dir}")

    image_rows: list[dict[str, str]] = []
    for path in image_paths:
        image_rows.extend(read_csv(path))
    image_rows = dedupe_by_image_id(image_rows)
    write_csv(args.output_csv, image_rows, list(image_rows[0]))

    line_count = 0
    if args.line_output_csv:
        line_paths = sorted(
            path for path in args.input_dir.glob(args.line_pattern) if "test" not in path.stem
        )
        line_rows: list[dict[str, str]] = []
        for path in line_paths:
            line_rows.extend(read_csv(path))
        line_count = len(line_rows)
        if line_rows:
            write_csv(args.line_output_csv, line_rows, list(line_rows[0]))

    print(f"Merged image rows: {len(image_rows)} from {len(image_paths)} chunks")
    print(f"Saved: {args.output_csv}")
    if args.line_output_csv:
        print(f"Merged line rows: {line_count}")
        print(f"Saved: {args.line_output_csv}")


if __name__ == "__main__":
    main()
