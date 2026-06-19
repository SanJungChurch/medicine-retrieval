"""OCR-text retrieval baseline for package image drug identification."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .kg_builder import build_kg
from .schemas import ProjectTables, load_processed_tables


@dataclass(frozen=True)
class SearchResult:
    image_id: str
    item_code: str
    score: float
    rank: int
    method: str = "ocr_rag_tfidf"


def _ingredient_names_by_item(tables: ProjectTables) -> dict[str, list[str]]:
    ingredients = {
        row["ingredient_id"]: row
        for row in tables["ingredients"]
    }
    names_by_item: dict[str, list[str]] = {}
    for row in tables["drug_item_ingredients"]:
        ingredient = ingredients[row["ingredient_id"]]
        names_by_item.setdefault(row["item_code"], []).extend(
            [ingredient["name_kr"], ingredient["name_en"]]
        )
    return names_by_item


def build_drug_search_corpus(tables: ProjectTables) -> tuple[list[str], list[str]]:
    """Create searchable text documents for each drug item."""

    ingredient_names = _ingredient_names_by_item(tables)
    item_codes: list[str] = []
    documents: list[str] = []

    for row in tables["drug_items"]:
        item_codes.append(row["item_code"])
        document_parts = [
            row["name_kr"],
            row["manufacturer"],
            row["dosage_form"],
            row["strength"],
            *ingredient_names.get(row["item_code"], []),
        ]
        documents.append(" ".join(part for part in document_parts if part))

    return item_codes, documents


class OcrRagRetriever:
    """Character n-gram TF-IDF baseline for Korean OCR text retrieval."""

    def __init__(self, tables: ProjectTables):
        self.tables = tables
        self.item_codes, self.documents = build_drug_search_corpus(tables)
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
        )
        self.document_matrix = self.vectorizer.fit_transform(self.documents)

    def search(self, image_id: str, ocr_text: str, top_k: int = 5) -> list[SearchResult]:
        query_matrix = self.vectorizer.transform([ocr_text])
        scores = cosine_similarity(query_matrix, self.document_matrix).ravel()
        ranked_indexes = scores.argsort()[::-1][:top_k]

        return [
            SearchResult(
                image_id=image_id,
                item_code=self.item_codes[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indexes, start=1)
        ]

    def search_dataset(self, top_k: int = 5) -> dict[str, list[str]]:
        rankings: dict[str, list[str]] = {}
        for row in self.tables["product_images"]:
            results = self.search(row["image_id"], row["ocr_text"], top_k=top_k)
            rankings[row["image_id"]] = [result.item_code for result in results]
        return rankings


def retrieve_safety_context(item_code: str, tables: ProjectTables) -> dict[str, list[dict[str, str]]]:
    """Return KG-backed safety rows for a retrieved drug item."""

    item_ingredients = [
        row for row in tables["drug_item_ingredients"]
        if row["item_code"] == item_code
    ]
    ingredient_ids = {row["ingredient_id"] for row in item_ingredients}

    dur_rules = [
        row for row in tables["dur_rules"]
        if (row["ref_type"] == "item" and row["ref_id"] == item_code)
        or (row["ref_type"] == "ingredient" and row["ref_id"] in ingredient_ids)
    ]
    food_rules = [
        row for row in tables["food_rules"]
        if row["item_code"] == item_code or row["ingredient_id"] in ingredient_ids
    ]
    ingredients = [
        row for row in tables["ingredients"]
        if row["ingredient_id"] in ingredient_ids
    ]
    sources = {
        row["source_id"]: row
        for row in tables["evidence_sources"]
    }
    used_source_ids = {
        row["source_id"]
        for row in [*dur_rules, *food_rules]
        if row.get("source_id")
    }

    return {
        "ingredients": ingredients,
        "dur_rules": dur_rules,
        "food_rules": food_rules,
        "sources": [sources[source_id] for source_id in sorted(used_source_ids) if source_id in sources],
    }


def build_retriever(tables: ProjectTables | None = None) -> OcrRagRetriever:
    return OcrRagRetriever(tables or load_processed_tables())


if __name__ == "__main__":
    loaded_tables = load_processed_tables()
    build_kg(loaded_tables)
    retriever = build_retriever(loaded_tables)
    for image_row in loaded_tables["product_images"]:
        top_result = retriever.search(image_row["image_id"], image_row["ocr_text"], top_k=1)[0]
        print(image_row["image_id"], top_result.item_code, f"{top_result.score:.4f}")
