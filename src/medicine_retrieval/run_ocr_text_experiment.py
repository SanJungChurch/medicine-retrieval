"""Run OCR-text retrieval over a real medicine subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import evaluate_rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR-RAG text retrieval.")
    parser.add_argument("--products-csv", type=Path, required=True)
    parser.add_argument("--ocr-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-rankings", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def product_document(row: dict[str, str]) -> str:
    parts = [
        row.get("aihub_product_name", ""),
        row.get("matched_product_name", ""),
        row.get("manufacturer", ""),
        row.get("classification", ""),
        row.get("standard_code", ""),
        row.get("ingredient_names", "").replace("|", " "),
    ]
    return " ".join(part for part in parts if part)


def main() -> None:
    args = parse_args()
    products = read_csv(args.products_csv)
    ocr_rows = read_csv(args.ocr_csv)

    item_codes = [row["item_seq"] for row in products]
    documents = [product_document(row) for row in products]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
    )
    document_matrix = vectorizer.fit_transform(documents)

    rankings: dict[str, list[str]] = {}
    result_rows: list[dict[str, str]] = []
    labels = {row["image_id"]: row["item_seq"] for row in ocr_rows}
    for row in ocr_rows:
        query_matrix = vectorizer.transform([row["ocr_text"]])
        scores = cosine_similarity(query_matrix, document_matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][: args.top_k]
        ranked_items = [item_codes[index] for index in ranked_indexes]
        rankings[row["image_id"]] = ranked_items
        for rank, index in enumerate(ranked_indexes, start=1):
            result_rows.append(
                {
                    "query_image_id": row["image_id"],
                    "gold_item_code": row["item_seq"],
                    "rank": str(rank),
                    "matched_item_code": item_codes[index],
                    "matched_product_name": products[index]["aihub_product_name"],
                    "score": f"{scores[index]:.6f}",
                    "ocr_confidence": row["ocr_confidence"],
                    "correct": "Y" if item_codes[index] == row["item_seq"] else "N",
                }
            )

    metrics = evaluate_rankings(rankings, labels, k_values=(1, 3, 5))
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_rankings, result_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved rankings: {args.output_rankings}")
    print(f"Saved metrics:  {args.output_metrics}")


if __name__ == "__main__":
    main()
