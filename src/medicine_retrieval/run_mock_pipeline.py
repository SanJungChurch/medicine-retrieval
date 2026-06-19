"""Run a smoke test over the mock dataset and KG skeleton."""

from __future__ import annotations

from .evaluation import evaluate_rankings
from .kg_builder import build_kg, graph_summary
from .schemas import load_processed_tables


def main() -> None:
    tables = load_processed_tables()
    graph = build_kg(tables)

    labels = {
        row["image_id"]: row["gold_item_code"]
        for row in tables["evaluation_labels"]
    }

    # Temporary deterministic rankings for smoke testing the evaluation layer.
    mock_rankings = {
        "IMG001": ["ITEM001", "ITEM002", "ITEM003"],
        "IMG002": ["ITEM002", "ITEM001", "ITEM003"],
        "IMG003": ["ITEM001", "ITEM003", "ITEM004"],
        "IMG004": ["ITEM004", "ITEM005", "ITEM003"],
        "IMG005": ["ITEM002", "ITEM005", "ITEM004"],
    }

    print("KG summary")
    print(graph_summary(graph))
    print()
    print("Mock ranking metrics")
    print(evaluate_rankings(mock_rankings, labels, k_values=(3, 5)))


if __name__ == "__main__":
    main()
