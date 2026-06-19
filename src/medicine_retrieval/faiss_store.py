"""FAISS-backed image embedding storage and retrieval."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FaissMetaRecord:
    row_id: int
    image_id: str
    item_code: str
    image_path: str
    embedding_model: str


@dataclass(frozen=True)
class FaissSearchHit:
    row_id: int
    image_id: str
    item_code: str
    image_path: str
    score: float
    rank: int


def require_faiss():
    try:
        import faiss

        return faiss
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "faiss is not installed. Install it in the medicine environment with: "
            "pip install faiss-cpu"
        ) from error


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def build_inner_product_index(embeddings: np.ndarray):
    faiss = require_faiss()
    vectors = l2_normalize(embeddings)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, path: Path) -> None:
    faiss = require_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path):
    faiss = require_faiss()
    return faiss.read_index(str(path))


def write_meta(path: Path, records: Iterable[FaissMetaRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "image_id", "item_code", "image_path", "embedding_model"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "row_id": record.row_id,
                    "image_id": record.image_id,
                    "item_code": record.item_code,
                    "image_path": record.image_path,
                    "embedding_model": record.embedding_model,
                }
            )


def read_meta(path: Path) -> list[FaissMetaRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return [
            FaissMetaRecord(
                row_id=int(row["row_id"]),
                image_id=row["image_id"],
                item_code=row["item_code"],
                image_path=row["image_path"],
                embedding_model=row["embedding_model"],
            )
            for row in rows
        ]


class FaissImageStore:
    """Local vector store for product package image embeddings."""

    def __init__(
        self,
        index,
        meta_records: list[FaissMetaRecord],
        embeddings: np.ndarray | None = None,
    ):
        self.index = index
        self.meta_records = meta_records
        self.embeddings = l2_normalize(embeddings) if embeddings is not None else None
        self.meta_by_row = {record.row_id: record for record in meta_records}
        self.row_by_image_id = {record.image_id: record.row_id for record in meta_records}

    @classmethod
    def load(
        cls,
        index_path: Path,
        meta_path: Path,
        embeddings_path: Path | None = None,
    ) -> "FaissImageStore":
        index = load_index(index_path)
        meta_records = read_meta(meta_path)
        embeddings = np.load(embeddings_path).astype(np.float32) if embeddings_path else None
        return cls(index=index, meta_records=meta_records, embeddings=embeddings)

    def get_vector(self, image_id: str) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("embeddings_path is required for search_by_image_id.")
        row_id = self.row_by_image_id[image_id]
        return self.embeddings[row_id]

    def search_by_vector(self, query_vector: np.ndarray, top_k: int = 5) -> list[FaissSearchHit]:
        query = l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))
        scores, indexes = self.index.search(query, top_k)

        hits: list[FaissSearchHit] = []
        for rank, (score, row_id) in enumerate(zip(scores[0], indexes[0]), start=1):
            if row_id < 0:
                continue
            meta = self.meta_by_row[int(row_id)]
            hits.append(
                FaissSearchHit(
                    row_id=meta.row_id,
                    image_id=meta.image_id,
                    item_code=meta.item_code,
                    image_path=meta.image_path,
                    score=float(score),
                    rank=rank,
                )
            )
        return hits

    def search_by_image_id(self, image_id: str, top_k: int = 5) -> list[FaissSearchHit]:
        return self.search_by_vector(self.get_vector(image_id), top_k=top_k)
