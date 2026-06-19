"""Run the OCR-RAG text retrieval baseline on the mock package image data."""

from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import evaluate_rankings
from .ocr_rag import build_retriever, retrieve_safety_context
from .schemas import load_processed_tables, project_root


def write_results_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "method", "rank", "item_code", "score"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = project_root()
    tables = load_processed_tables()
    retriever = build_retriever(tables)

    labels = {
        row["image_id"]: row["gold_item_code"]
        for row in tables["evaluation_labels"]
    }

    rankings: dict[str, list[str]] = {}
    result_rows: list[dict[str, str]] = []
    for image_row in tables["product_images"]:
        results = retriever.search(
            image_id=image_row["image_id"],
            ocr_text=image_row["ocr_text"],
            top_k=5,
        )
        rankings[image_row["image_id"]] = [result.item_code for result in results]
        for result in results:
            result_rows.append(
                {
                    "image_id": result.image_id,
                    "method": result.method,
                    "rank": str(result.rank),
                    "item_code": result.item_code,
                    "score": f"{result.score:.6f}",
                }
            )

    output_path = root / "outputs" / "reports" / "ocr_rag_results.csv"
    write_results_csv(output_path, result_rows)

    metrics = evaluate_rankings(rankings, labels, k_values=(3, 5))
    print("OCR-RAG baseline metrics")
    print(metrics)
    print()
    print(f"Saved rankings: {output_path}")

    first_image = tables["product_images"][0]
    first_item = rankings[first_image["image_id"]][0]
    safety_context = retrieve_safety_context(first_item, tables)
    print()
    print(f"Top-1 safety context for {first_image['image_id']} -> {first_item}")
    print(
        {
            key: len(value)
            for key, value in safety_context.items()
        }
    )


if __name__ == "__main__":
    main()
