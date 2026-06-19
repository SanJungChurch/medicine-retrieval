"""Hybrid GraphRAG candidate ranking from OCR-RAG and Image-KG scores."""

from __future__ import annotations

from dataclasses import dataclass

from .image_kg import ImageKgRetriever, build_image_kg_retriever
from .ocr_rag import OcrRagRetriever, build_retriever
from .schemas import ProjectTables, load_processed_tables


@dataclass(frozen=True)
class HybridResult:
    image_id: str
    item_code: str
    ocr_score: float
    image_score: float
    final_score: float
    rank: int
    alpha: float
    method: str = "hybrid_graphrag_fixed_alpha"


def combine_scores(
    image_id: str,
    ocr_scores: dict[str, float],
    image_scores: dict[str, float],
    alpha: float = 0.5,
    top_k: int = 5,
) -> list[HybridResult]:
    """Combine OCR and image candidate scores with a fixed alpha."""

    candidate_items = sorted(set(ocr_scores) | set(image_scores))
    scored: list[tuple[str, float, float, float]] = []
    for item_code in candidate_items:
        ocr_score = ocr_scores.get(item_code, 0.0)
        image_score = image_scores.get(item_code, 0.0)
        final_score = alpha * ocr_score + (1.0 - alpha) * image_score
        scored.append((item_code, ocr_score, image_score, final_score))

    ranked = sorted(scored, key=lambda row: row[3], reverse=True)[:top_k]
    return [
        HybridResult(
            image_id=image_id,
            item_code=item_code,
            ocr_score=ocr_score,
            image_score=image_score,
            final_score=final_score,
            rank=index,
            alpha=alpha,
        )
        for index, (item_code, ocr_score, image_score, final_score) in enumerate(ranked, start=1)
    ]


class HybridGraphRagRetriever:
    """Fixed-alpha hybrid ranker over the two baseline retrievers."""

    def __init__(
        self,
        tables: ProjectTables,
        ocr_retriever: OcrRagRetriever,
        image_retriever: ImageKgRetriever,
        alpha: float = 0.5,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")

        self.tables = tables
        self.ocr_retriever = ocr_retriever
        self.image_retriever = image_retriever
        self.alpha = alpha

    def search(self, image_id: str, ocr_text: str, top_k: int = 5) -> list[HybridResult]:
        ocr_results = self.ocr_retriever.search(image_id=image_id, ocr_text=ocr_text, top_k=top_k)
        image_results = self.image_retriever.search(image_id=image_id, top_k=top_k, exclude_self=False)

        ocr_scores = {result.item_code: result.score for result in ocr_results}
        image_scores = {result.item_code: result.score for result in image_results}

        return combine_scores(
            image_id=image_id,
            ocr_scores=ocr_scores,
            image_scores=image_scores,
            alpha=self.alpha,
            top_k=top_k,
        )

    def search_dataset(self, top_k: int = 5) -> dict[str, list[str]]:
        rankings: dict[str, list[str]] = {}
        for row in self.tables["product_images"]:
            results = self.search(row["image_id"], row["ocr_text"], top_k=top_k)
            rankings[row["image_id"]] = [result.item_code for result in results]
        return rankings


def build_hybrid_retriever(
    tables: ProjectTables | None = None,
    alpha: float = 0.5,
) -> HybridGraphRagRetriever:
    loaded_tables = tables or load_processed_tables()
    return HybridGraphRagRetriever(
        tables=loaded_tables,
        ocr_retriever=build_retriever(loaded_tables),
        image_retriever=build_image_kg_retriever(loaded_tables),
        alpha=alpha,
    )
