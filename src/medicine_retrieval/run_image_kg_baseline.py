"""Run the Package Image-KG retrieval baseline."""

from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import evaluate_rankings
from .image_kg import build_image_kg_retriever, retrieve_image_kg_safety_context
from .schemas import load_processed_tables, project_root


def write_results_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_image_id",
                "method",
                "rank",
                "matched_image_id",
                "item_code",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = project_root()
    tables = load_processed_tables()
    retriever = build_image_kg_retriever(tables)

    labels = {
        row["image_id"]: row["gold_item_code"]
        for row in tables["evaluation_labels"]
    }

    rankings: dict[str, list[str]] = {}
    result_rows: list[dict[str, str]] = []
    for image_row in tables["product_images"]:
        results = retriever.search(image_row["image_id"], top_k=5, exclude_self=False)
        rankings[image_row["image_id"]] = [result.item_code for result in results]
        for result in results:
            result_rows.append(
                {
                    "query_image_id": result.query_image_id,
                    "method": result.method,
                    "rank": str(result.rank),
                    "matched_image_id": result.matched_image_id,
                    "item_code": result.item_code,
                    "score": f"{result.score:.6f}",
                }
            )

    output_path = root / "outputs" / "reports" / "image_kg_results.csv"
    write_results_csv(output_path, result_rows)

    metrics = evaluate_rankings(rankings, labels, k_values=(3, 5))
    print("Image-KG baseline metrics")
    print(metrics)
    print()
    print(f"Saved rankings: {output_path}")

    first_image = tables["product_images"][0]
    first_item = rankings[first_image["image_id"]][0]
    safety_context = retrieve_image_kg_safety_context(first_item, tables)
    print()
    print(f"Top-1 safety context for {first_image['image_id']} -> {first_item}")
    print({key: len(value) for key, value in safety_context.items()})


if __name__ == "__main__":
    main()
