"""Run Hybrid GraphRAG candidate ranking and compare all baselines."""

from __future__ import annotations

import csv
from pathlib import Path

from .evaluation import evaluate_rankings
from .hybrid import build_hybrid_retriever
from .image_kg import build_image_kg_retriever
from .ocr_rag import build_retriever
from .schemas import load_processed_tables, project_root


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = project_root()
    tables = load_processed_tables()
    alpha = 0.5

    labels = {
        row["image_id"]: row["gold_item_code"]
        for row in tables["evaluation_labels"]
    }

    ocr_retriever = build_retriever(tables)
    image_retriever = build_image_kg_retriever(tables)
    hybrid_retriever = build_hybrid_retriever(tables, alpha=alpha)

    ocr_rankings = ocr_retriever.search_dataset(top_k=5)
    image_rankings = image_retriever.search_dataset(top_k=5, exclude_self=False)
    hybrid_rankings: dict[str, list[str]] = {}
    hybrid_rows: list[dict[str, str]] = []

    for image_row in tables["product_images"]:
        results = hybrid_retriever.search(
            image_id=image_row["image_id"],
            ocr_text=image_row["ocr_text"],
            top_k=5,
        )
        hybrid_rankings[image_row["image_id"]] = [result.item_code for result in results]
        for result in results:
            hybrid_rows.append(
                {
                    "image_id": result.image_id,
                    "method": result.method,
                    "rank": str(result.rank),
                    "item_code": result.item_code,
                    "ocr_score": f"{result.ocr_score:.6f}",
                    "image_score": f"{result.image_score:.6f}",
                    "final_score": f"{result.final_score:.6f}",
                    "alpha": f"{result.alpha:.2f}",
                }
            )

    hybrid_path = root / "outputs" / "reports" / "hybrid_results.csv"
    write_csv(
        hybrid_path,
        [
            "image_id",
            "method",
            "rank",
            "item_code",
            "ocr_score",
            "image_score",
            "final_score",
            "alpha",
        ],
        hybrid_rows,
    )

    comparison_rows: list[dict[str, str]] = []
    method_rankings = {
        "ocr_rag_tfidf": ocr_rankings,
        "image_kg_mock_embedding": image_rankings,
        "hybrid_graphrag_fixed_alpha": hybrid_rankings,
    }
    for method, rankings in method_rankings.items():
        metrics = evaluate_rankings(rankings, labels, k_values=(3, 5))
        comparison_rows.append(
            {
                "method": method,
                "top1_accuracy": f"{metrics['top1_accuracy']:.6f}",
                "mrr": f"{metrics['mrr']:.6f}",
                "recall_at_3": f"{metrics['recall_at_3']:.6f}",
                "recall_at_5": f"{metrics['recall_at_5']:.6f}",
            }
        )

    comparison_path = root / "outputs" / "reports" / "baseline_comparison.csv"
    write_csv(
        comparison_path,
        ["method", "top1_accuracy", "mrr", "recall_at_3", "recall_at_5"],
        comparison_rows,
    )

    print("Hybrid GraphRAG baseline metrics")
    print(evaluate_rankings(hybrid_rankings, labels, k_values=(3, 5)))
    print()
    print(f"Saved hybrid rankings: {hybrid_path}")
    print(f"Saved baseline comparison: {comparison_path}")


if __name__ == "__main__":
    main()
