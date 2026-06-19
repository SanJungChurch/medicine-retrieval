"""Embedding storage and cosine retrieval utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schemas import ProjectTables, load_processed_tables


@dataclass(frozen=True)
class EmbeddingRecord:
    image_id: str
    embedding_model: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class EmbeddingSearchHit:
    image_id: str
    score: float
    rank: int


def parse_embedding(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in raw.split())


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Embedding dimensions do not match: {len(left)} != {len(right)}")

    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class ImageEmbeddingStore:
    """In-memory image embedding store.

    The current implementation loads vectors from CSV so the retrieval pipeline can
    be tested before real CLIP/SigLIP embeddings are generated.
    """

    def __init__(self, records: list[EmbeddingRecord]):
        self.records = records
        self.by_image_id = {record.image_id: record for record in records}

    @classmethod
    def from_tables(cls, tables: ProjectTables | None = None) -> "ImageEmbeddingStore":
        loaded_tables = tables or load_processed_tables()
        records = [
            EmbeddingRecord(
                image_id=row["image_id"],
                embedding_model=row["embedding_model"],
                vector=parse_embedding(row["embedding"]),
            )
            for row in loaded_tables["image_embeddings"]
        ]
        return cls(records)

    def get(self, image_id: str) -> EmbeddingRecord:
        try:
            return self.by_image_id[image_id]
        except KeyError as error:
            raise KeyError(f"No image embedding found for image_id={image_id}") from error

    def search_by_vector(
        self,
        query_vector: tuple[float, ...],
        top_k: int = 5,
        exclude_image_id: str | None = None,
    ) -> list[EmbeddingSearchHit]:
        hits: list[EmbeddingSearchHit] = []
        for record in self.records:
            if exclude_image_id and record.image_id == exclude_image_id:
                continue
            hits.append(
                EmbeddingSearchHit(
                    image_id=record.image_id,
                    score=cosine_similarity(query_vector, record.vector),
                    rank=0,
                )
            )

        ranked_hits = sorted(hits, key=lambda hit: hit.score, reverse=True)[:top_k]
        return [
            EmbeddingSearchHit(image_id=hit.image_id, score=hit.score, rank=index)
            for index, hit in enumerate(ranked_hits, start=1)
        ]

    def search_by_image_id(
        self,
        image_id: str,
        top_k: int = 5,
        exclude_self: bool = False,
    ) -> list[EmbeddingSearchHit]:
        query = self.get(image_id)
        return self.search_by_vector(
            query_vector=query.vector,
            top_k=top_k,
            exclude_image_id=image_id if exclude_self else None,
        )
