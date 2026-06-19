"""Run PaddleOCR on manifest query images and save OCR text/confidence CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OCR text from query package images.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--line-output-csv", type=Path)
    parser.add_argument("--role", default="query")
    parser.add_argument("--lang", default="korean")
    parser.add_argument("--cache-dir", type=Path, default=Path(r"D:\medicine_data\paddle_home"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--use-angle-cls", action="store_true")
    parser.add_argument("--use-gpu", action="store_true")
    return parser.parse_args()


def configure_paddle_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(cache_dir)
    os.environ["HOME"] = str(cache_dir)
    os.environ["XDG_CACHE_HOME"] = str(cache_dir / ".cache")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_paddleocr_result(result: Any) -> list[dict[str, Any]]:
    """Return line-level OCR records from PaddleOCR 2.x style results."""

    if not result:
        return []
    pages = result if isinstance(result, list) else [result]
    lines: list[dict[str, Any]] = []
    for page in pages:
        if not page:
            continue
        for entry in page:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            box = entry[0]
            text_score = entry[1]
            if not isinstance(text_score, (list, tuple)) or len(text_score) < 2:
                continue
            text = str(text_score[0]).strip()
            try:
                score = float(text_score[1])
            except (TypeError, ValueError):
                score = 0.0
            if text:
                lines.append({"text": text, "confidence": score, "box_json": json.dumps(box)})
    return lines


def main() -> None:
    args = parse_args()
    configure_paddle_cache(args.cache_dir)

    from paddleocr import PaddleOCR

    rows = [row for row in read_csv(args.manifest_csv) if row.get("role") == args.role]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No manifest rows found for role={args.role}")

    ocr = PaddleOCR(
        use_angle_cls=args.use_angle_cls,
        use_gpu=args.use_gpu,
        lang=args.lang,
        show_log=False,
    )

    image_rows: list[dict[str, str]] = []
    line_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        started_at = time.perf_counter()
        print(f"[{index}/{len(rows)}] OCR {row['image_id']} {image_path}", flush=True)
        error = ""
        lines: list[dict[str, Any]] = []
        try:
            result = ocr.ocr(image_path, cls=args.use_angle_cls)
            lines = flatten_paddleocr_result(result)
        except Exception as exc:
            error = repr(exc)

        elapsed = time.perf_counter() - started_at
        text_values = [line["text"] for line in lines]
        confidences = [line["confidence"] for line in lines]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        ocr_text = " ".join(text_values)
        image_rows.append(
            {
                "image_id": row["image_id"],
                "product_id": row.get("product_id", ""),
                "item_seq": row.get("item_seq", ""),
                "product_name": row.get("product_name", ""),
                "role": row.get("role", ""),
                "image_path": image_path,
                "ocr_text": ocr_text,
                "ocr_confidence": f"{mean_confidence:.6f}",
                "ocr_line_count": str(len(lines)),
                "ocr_model": f"paddleocr:{args.lang}",
                "elapsed_sec": f"{elapsed:.3f}",
                "error": error,
                "ocr_ground_truth": row.get("ocr_ground_truth", ""),
            }
        )
        for line_index, line in enumerate(lines, start=1):
            line_rows.append(
                {
                    "image_id": row["image_id"],
                    "item_seq": row.get("item_seq", ""),
                    "line_index": str(line_index),
                    "text": line["text"],
                    "confidence": f"{line['confidence']:.6f}",
                    "box_json": line["box_json"],
                }
            )

    write_csv(
        args.output_csv,
        image_rows,
        [
            "image_id",
            "product_id",
            "item_seq",
            "product_name",
            "role",
            "image_path",
            "ocr_text",
            "ocr_confidence",
            "ocr_line_count",
            "ocr_model",
            "elapsed_sec",
            "error",
            "ocr_ground_truth",
        ],
    )
    if args.line_output_csv:
        write_csv(
            args.line_output_csv,
            line_rows,
            ["image_id", "item_seq", "line_index", "text", "confidence", "box_json"],
        )

    successful = sum(1 for row in image_rows if not row["error"])
    print(f"Saved OCR image rows: {args.output_csv}")
    if args.line_output_csv:
        print(f"Saved OCR line rows:  {args.line_output_csv}")
    print(f"Successful images: {successful}/{len(image_rows)}")


if __name__ == "__main__":
    main()
