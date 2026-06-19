"""Evaluate query image embeddings against a gallery FAISS image index."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate_rankings
from .faiss_store import FaissImageStore, read_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image retrieval artifacts.")
    parser.add_argument("--gallery-index", type=Path, required=True)
    parser.add_argument("--gallery-meta", type=Path, required=True)
    parser.add_argument("--gallery-embeddings", type=Path)
    parser.add_argument("--query-meta", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-images", type=int, default=50)
    parser.add_argument("--output-rankings", type=Path, required=True)
    parser.add_argument("--output-metrics", type=Path, required=True)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    gallery_store = FaissImageStore.load(
        index_path=args.gallery_index,
        meta_path=args.gallery_meta,
        embeddings_path=args.gallery_embeddings,
    )
    query_meta = read_meta(args.query_meta)
    query_embeddings = np.load(args.query_embeddings).astype(np.float32)
    if len(query_meta) != len(query_embeddings):
        raise SystemExit(
            f"Query metadata/vector length mismatch: {len(query_meta)} != {len(query_embeddings)}"
        )

    labels = {record.image_id: record.item_code for record in query_meta}
    rankings: dict[str, list[str]] = {}
    rows: list[dict[str, str]] = []
    for record, vector in zip(query_meta, query_embeddings):
        image_hits = gallery_store.search_by_vector(
            vector,
            top_k=max(args.candidate_images, args.top_k),
        )
        best_by_item = {}
        for hit in image_hits:
            current = best_by_item.get(hit.item_code)
            if current is None or hit.score > current.score:
                best_by_item[hit.item_code] = hit

        item_hits = sorted(
            best_by_item.values(),
            key=lambda hit: hit.score,
            reverse=True,
        )[: args.top_k]
        rankings[record.image_id] = [hit.item_code for hit in item_hits]
        for rank, hit in enumerate(item_hits, start=1):
            rows.append(
                {
                    "query_image_id": record.image_id,
                    "gold_item_code": record.item_code,
                    "rank": str(rank),
                    "matched_image_id": hit.image_id,
                    "matched_item_code": hit.item_code,
                    "score": f"{hit.score:.6f}",
                    "correct": "Y" if hit.item_code == record.item_code else "N",
                }
            )

    metrics = evaluate_rankings(rankings, labels, k_values=(1, 3, 5))
    args.output_metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(args.output_rankings, rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved rankings: {args.output_rankings}")
    print(f"Saved metrics:  {args.output_metrics}")


if __name__ == "__main__":
    main()
