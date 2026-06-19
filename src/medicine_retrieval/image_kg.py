"""Package Image-KG retrieval baseline."""

from __future__ import annotations

from dataclasses import dataclass

from .embedding_store import ImageEmbeddingStore
from .ocr_rag import retrieve_safety_context
from .schemas import ProjectTables, load_processed_tables


@dataclass(frozen=True)
class ImageKgResult:
    query_image_id: str
    matched_image_id: str
    item_code: str
    score: float
    rank: int
    method: str = "image_kg_mock_embedding"


class ImageKgRetriever:
    """Retrieve drug candidates via similar product_image nodes."""

    def __init__(self, tables: ProjectTables, embedding_store: ImageEmbeddingStore):
        self.tables = tables
        self.embedding_store = embedding_store
        self.item_code_by_image = {
            row["image_id"]: row["item_code"]
            for row in tables["product_images"]
        }

    def search(
        self,
        image_id: str,
        top_k: int = 5,
        exclude_self: bool = False,
    ) -> list[ImageKgResult]:
        image_hits = self.embedding_store.search_by_image_id(
            image_id=image_id,
            top_k=top_k,
            exclude_self=exclude_self,
        )

        results: list[ImageKgResult] = []
        seen_items: set[str] = set()
        for hit in image_hits:
            item_code = self.item_code_by_image[hit.image_id]
            if item_code in seen_items:
                continue
            seen_items.add(item_code)
            results.append(
                ImageKgResult(
                    query_image_id=image_id,
                    matched_image_id=hit.image_id,
                    item_code=item_code,
                    score=hit.score,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def search_dataset(self, top_k: int = 5, exclude_self: bool = False) -> dict[str, list[str]]:
        rankings: dict[str, list[str]] = {}
        for row in self.tables["product_images"]:
            results = self.search(row["image_id"], top_k=top_k, exclude_self=exclude_self)
            rankings[row["image_id"]] = [result.item_code for result in results]
        return rankings


def build_image_kg_retriever(tables: ProjectTables | None = None) -> ImageKgRetriever:
    loaded_tables = tables or load_processed_tables()
    return ImageKgRetriever(
        tables=loaded_tables,
        embedding_store=ImageEmbeddingStore.from_tables(loaded_tables),
    )


def retrieve_image_kg_safety_context(item_code: str, tables: ProjectTables):
    return retrieve_safety_context(item_code, tables)
