"""Run real Hybrid GraphRAG candidate ranking over OCR and image scores."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .evaluation import evaluate_rankings
from .faiss_store import FaissImageStore, read_meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OCR + Image hybrid ranking.")
    parser.add_argument("--products-csv", type=Path, required=True)
    parser.add_argument("--ocr-csv", type=Path, required=True)
    parser.add_argument("--gallery-index", type=Path, required=True)
    parser.add_argument("--gallery-meta", type=Path, required=True)
    parser.add_argument("--gallery-embeddings", type=Path, required=True)
    parser.add_argument("--query-meta", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-rankings", type=Path, required=True)
    parser.add_argument("--output-comparison", type=Path, required=True)
    parser.add_argument("--output-metrics-json", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
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


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max <= score_min:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - score_min) / (score_max - score_min)


def dynamic_alpha(ocr_confidence: float) -> float:
    """Map OCR confidence to a conservative OCR weight range."""

    return max(0.3, min(0.8, 0.2 + 0.6 * ocr_confidence))


def main() -> None:
    args = parse_args()
    products = read_csv(args.products_csv)
    ocr_rows = read_csv(args.ocr_csv)
    item_codes = [row["item_seq"] for row in products]
    item_names = {row["item_seq"]: row["aihub_product_name"] for row in products}

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
    )
    document_matrix = vectorizer.fit_transform([product_document(row) for row in products])

    gallery_store = FaissImageStore.load(
        index_path=args.gallery_index,
        meta_path=args.gallery_meta,
        embeddings_path=args.gallery_embeddings,
    )
    gallery_meta = read_meta(args.gallery_meta)
    query_meta = read_meta(args.query_meta)
    query_embeddings = np.load(args.query_embeddings).astype(np.float32)
    query_vector_by_image_id = {
        record.image_id: vector for record, vector in zip(query_meta, query_embeddings)
    }

    labels = {row["image_id"]: row["item_seq"] for row in ocr_rows}
    methods: dict[str, dict[str, list[str]]] = {
        "ocr_rag_paddle_cpu": {},
        "image_kg_clip_single_gallery": {},
        **{f"hybrid_alpha_{alpha:.1f}": {} for alpha in args.alphas},
        "hybrid_dynamic_confidence": {},
    }
    ranking_rows: list[dict[str, str]] = []

    for row in ocr_rows:
        image_id = row["image_id"]
        gold_item = row["item_seq"]
        ocr_conf = float(row.get("ocr_confidence") or 0.0)

        ocr_query = vectorizer.transform([row["ocr_text"]])
        ocr_scores = cosine_similarity(ocr_query, document_matrix).ravel().astype(np.float32)
        ocr_scores_norm = minmax(ocr_scores)

        image_scores_by_item = {item_code: 0.0 for item_code in item_codes}
        image_vector = query_vector_by_image_id[image_id]
        image_hits = gallery_store.search_by_vector(image_vector, top_k=len(gallery_meta))
        for hit in image_hits:
            image_scores_by_item[hit.item_code] = max(
                image_scores_by_item.get(hit.item_code, 0.0),
                hit.score,
            )
        image_scores = np.asarray([image_scores_by_item[item_code] for item_code in item_codes])
        image_scores_norm = minmax(image_scores)

        method_scores = {
            "ocr_rag_paddle_cpu": ocr_scores_norm,
            "image_kg_clip_single_gallery": image_scores_norm,
        }
        for alpha in args.alphas:
            method_scores[f"hybrid_alpha_{alpha:.1f}"] = (
                alpha * ocr_scores_norm + (1.0 - alpha) * image_scores_norm
            )
        conf_alpha = dynamic_alpha(ocr_conf)
        method_scores["hybrid_dynamic_confidence"] = (
            conf_alpha * ocr_scores_norm + (1.0 - conf_alpha) * image_scores_norm
        )

        for method, scores in method_scores.items():
            ranked_indexes = scores.argsort()[::-1][: args.top_k]
            ranked_items = [item_codes[index] for index in ranked_indexes]
            methods[method][image_id] = ranked_items
            for rank, index in enumerate(ranked_indexes, start=1):
                matched_item = item_codes[index]
                ranking_rows.append(
                    {
                        "method": method,
                        "query_image_id": image_id,
                        "gold_item_code": gold_item,
                        "rank": str(rank),
                        "matched_item_code": matched_item,
                        "matched_product_name": item_names.get(matched_item, ""),
                        "ocr_score_raw": f"{ocr_scores[index]:.6f}",
                        "image_score_raw": f"{image_scores[index]:.6f}",
                        "ocr_score_norm": f"{ocr_scores_norm[index]:.6f}",
                        "image_score_norm": f"{image_scores_norm[index]:.6f}",
                        "final_score": f"{scores[index]:.6f}",
                        "ocr_confidence": f"{ocr_conf:.6f}",
                        "alpha": (
                            f"{conf_alpha:.3f}"
                            if method == "hybrid_dynamic_confidence"
                            else method.replace("hybrid_alpha_", "")
                            if method.startswith("hybrid_alpha_")
                            else ""
                        ),
                        "correct": "Y" if matched_item == gold_item else "N",
                    }
                )

    comparison_rows: list[dict[str, str]] = []
    metrics_by_method: dict[str, dict[str, float]] = {}
    for method, rankings in methods.items():
        metrics = evaluate_rankings(rankings, labels, k_values=(1, 3, 5))
        metrics_by_method[method] = metrics
        comparison_rows.append(
            {
                "method": method,
                "top1_accuracy": f"{metrics['top1_accuracy']:.6f}",
                "mrr": f"{metrics['mrr']:.6f}",
                "recall_at_1": f"{metrics['recall_at_1']:.6f}",
                "recall_at_3": f"{metrics['recall_at_3']:.6f}",
                "recall_at_5": f"{metrics['recall_at_5']:.6f}",
            }
        )

    write_csv(args.output_rankings, ranking_rows)
    write_csv(args.output_comparison, comparison_rows)
    args.output_metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_metrics_json.write_text(
        json.dumps(metrics_by_method, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(comparison_rows, ensure_ascii=False, indent=2))
    print(f"Saved rankings:   {args.output_rankings}")
    print(f"Saved comparison: {args.output_comparison}")
    print(f"Saved metrics:    {args.output_metrics_json}")


if __name__ == "__main__":
    main()
