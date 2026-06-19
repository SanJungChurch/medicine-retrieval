"""Run CLIP image retrieval experiments and write a comparison report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .evaluation import evaluate_rankings
from .faiss_store import FaissImageStore, read_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLIP image retrieval experiment sets.")
    parser.add_argument(
        "--single-dir",
        type=Path,
        default=Path(r"D:\medicine_data\validation_dur_positive_100"),
    )
    parser.add_argument(
        "--multi-dir",
        type=Path,
        default=Path(r"D:\medicine_data\validation_dur_positive_50_multigallery"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"D:\medicine_data\experiment_reports"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_dataset(dataset_dir: Path, experiment_name: str, top_k: int) -> dict[str, str]:
    embeddings_dir = dataset_dir / "embeddings"
    reports_dir = dataset_dir / "reports"
    gallery_meta_path = embeddings_dir / "clip_gallery_meta.csv"
    query_meta_path = embeddings_dir / "clip_query_meta.csv"

    gallery_store = FaissImageStore.load(
        index_path=embeddings_dir / "clip_gallery_index.faiss",
        meta_path=gallery_meta_path,
        embeddings_path=embeddings_dir / "clip_gallery_embeddings.npy",
    )
    gallery_meta = read_meta(gallery_meta_path)
    query_meta = read_meta(query_meta_path)
    query_embeddings = np.load(embeddings_dir / "clip_query_embeddings.npy").astype(np.float32)
    if len(query_meta) != len(query_embeddings):
        raise SystemExit(
            f"{experiment_name}: query metadata/vector length mismatch "
            f"{len(query_meta)} != {len(query_embeddings)}"
        )

    labels = {record.image_id: record.item_code for record in query_meta}
    rankings: dict[str, list[str]] = {}
    ranking_rows: list[dict[str, str]] = []
    for record, vector in zip(query_meta, query_embeddings):
        image_hits = gallery_store.search_by_vector(vector, top_k=len(gallery_meta))
        best_by_item = {}
        for hit in image_hits:
            current = best_by_item.get(hit.item_code)
            if current is None or hit.score > current.score:
                best_by_item[hit.item_code] = hit
        item_hits = sorted(
            best_by_item.values(),
            key=lambda hit: hit.score,
            reverse=True,
        )[:top_k]
        rankings[record.image_id] = [hit.item_code for hit in item_hits]
        for rank, hit in enumerate(item_hits, start=1):
            ranking_rows.append(
                {
                    "experiment": experiment_name,
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
    rankings_path = reports_dir / f"{experiment_name}_clip_rankings.csv"
    metrics_path = reports_dir / f"{experiment_name}_clip_metrics.json"
    write_csv(rankings_path, ranking_rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "experiment": experiment_name,
        "dataset_dir": str(dataset_dir),
        "gallery_images": str(len(gallery_meta)),
        "query_images": str(len(query_meta)),
        "candidate_items": str(len({row.item_code for row in gallery_meta})),
        "top1_accuracy": f"{metrics['top1_accuracy']:.6f}",
        "mrr": f"{metrics['mrr']:.6f}",
        "recall_at_1": f"{metrics['recall_at_1']:.6f}",
        "recall_at_3": f"{metrics['recall_at_3']:.6f}",
        "recall_at_5": f"{metrics['recall_at_5']:.6f}",
        "rankings_path": str(rankings_path),
        "metrics_path": str(metrics_path),
    }


def main() -> None:
    args = parse_args()
    rows = [
        evaluate_dataset(args.single_dir, "single_gallery_100", args.top_k),
        evaluate_dataset(args.multi_dir, "multi_gallery_50", args.top_k),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = args.output_dir / "clip_image_experiment_comparison.csv"
    comparison_json = args.output_dir / "clip_image_experiment_comparison.json"
    write_csv(comparison_csv, rows)
    comparison_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Saved comparison CSV:  {comparison_csv}")
    print(f"Saved comparison JSON: {comparison_json}")


if __name__ == "__main__":
    main()
