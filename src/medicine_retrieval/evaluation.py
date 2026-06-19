"""Ranking metrics for the OCR-RAG, Image-KG, and Hybrid retrieval experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RankedResult:
    query_id: str
    item_code: str
    score: float


def top1_accuracy(results: dict[str, list[str]], labels: dict[str, str]) -> float:
    hits = 0
    total = 0
    for query_id, gold in labels.items():
        ranked = results.get(query_id, [])
        if ranked:
            hits += int(ranked[0] == gold)
        total += 1
    return hits / total if total else 0.0


def recall_at_k(results: dict[str, list[str]], labels: dict[str, str], k: int) -> float:
    hits = 0
    total = 0
    for query_id, gold in labels.items():
        ranked = results.get(query_id, [])[:k]
        hits += int(gold in ranked)
        total += 1
    return hits / total if total else 0.0


def mean_reciprocal_rank(results: dict[str, list[str]], labels: dict[str, str]) -> float:
    total_score = 0.0
    total = 0
    for query_id, gold in labels.items():
        ranked = results.get(query_id, [])
        reciprocal_rank = 0.0
        for index, item_code in enumerate(ranked, start=1):
            if item_code == gold:
                reciprocal_rank = 1.0 / index
                break
        total_score += reciprocal_rank
        total += 1
    return total_score / total if total else 0.0


def normalize_ranked_results(rows: Iterable[RankedResult]) -> dict[str, list[str]]:
    grouped: dict[str, list[RankedResult]] = defaultdict(list)
    for row in rows:
        grouped[row.query_id].append(row)

    return {
        query_id: [row.item_code for row in sorted(items, key=lambda item: item.score, reverse=True)]
        for query_id, items in grouped.items()
    }


def evaluate_rankings(results: dict[str, list[str]], labels: dict[str, str], k_values: tuple[int, ...] = (3, 5)) -> dict[str, float]:
    metrics = {
        "top1_accuracy": top1_accuracy(results, labels),
        "mrr": mean_reciprocal_rank(results, labels),
    }
    for k in k_values:
        metrics[f"recall_at_{k}"] = recall_at_k(results, labels, k)
    return metrics
