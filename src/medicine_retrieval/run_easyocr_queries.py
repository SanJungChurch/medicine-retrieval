"""Run EasyOCR on manifest query images and save OCR text/confidence CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract OCR text from query images with EasyOCR.")
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--line-output-csv", type=Path)
    parser.add_argument("--role", default="query")
    parser.add_argument("--languages", nargs="+", default=["ko", "en"])
    parser.add_argument("--model-storage-dir", type=Path, default=Path(r"D:\medicine_data\easyocr_models"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--canvas-size", type=int, default=1280)
    parser.add_argument("--mag-ratio", type=float, default=1.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--gpu", action="store_true")
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


def flatten_easyocr_result(result: list[Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for entry in result:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            continue
        box, text, confidence = entry[0], str(entry[1]).strip(), entry[2]
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if text:
            lines.append(
                {
                    "text": text,
                    "confidence": score,
                    "box_json": json.dumps(
                        box,
                        ensure_ascii=False,
                        default=lambda value: value.item() if hasattr(value, "item") else str(value),
                    ),
                }
            )
    return lines


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    args.model_storage_dir.mkdir(parents=True, exist_ok=True)

    import easyocr

    rows = [row for row in read_csv(args.manifest_csv) if row.get("role") == args.role]
    if args.start_index:
        rows = rows[args.start_index :]
    if args.max_images:
        rows = rows[: args.max_images]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"No manifest rows found for role={args.role}")

    reader = easyocr.Reader(
        args.languages,
        gpu=args.gpu,
        model_storage_directory=str(args.model_storage_dir),
        verbose=False,
    )

    image_rows: list[dict[str, str]] = []
    line_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image_path = row["image_path"]
        started_at = time.perf_counter()
        print(f"[{index}/{len(rows)}] EasyOCR {row['image_id']} {image_path}", flush=True)
        error = ""
        lines: list[dict[str, Any]] = []
        try:
            result = reader.readtext(
                image_path,
                canvas_size=args.canvas_size,
                mag_ratio=args.mag_ratio,
            )
            lines = flatten_easyocr_result(result)
        except Exception as exc:
            error = repr(exc)

        elapsed = time.perf_counter() - started_at
        text_values = [line["text"] for line in lines]
        confidences = [line["confidence"] for line in lines]
        mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        image_rows.append(
            {
                "image_id": row["image_id"],
                "product_id": row.get("product_id", ""),
                "item_seq": row.get("item_seq", ""),
                "product_name": row.get("product_name", ""),
                "role": row.get("role", ""),
                "image_path": image_path,
                "ocr_text": " ".join(text_values),
                "ocr_confidence": f"{mean_confidence:.6f}",
                "ocr_line_count": str(len(lines)),
                "ocr_model": f"easyocr:{','.join(args.languages)}:{'gpu' if args.gpu else 'cpu'}",
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
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

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
