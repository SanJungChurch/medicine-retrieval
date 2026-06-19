"""CSV schema definitions and lightweight loaders for the mock experiment data."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROCESSED_TABLES = {
    "drug_items": ["item_code", "name_kr", "manufacturer", "dosage_form", "strength"],
    "ingredients": ["ingredient_id", "name_kr", "name_en"],
    "drug_item_ingredients": ["item_code", "ingredient_id", "amount", "unit"],
    "evidence_sources": ["source_id", "name", "publisher", "url", "version", "retrieved_at"],
    "dur_rules": ["rule_id", "ref_type", "ref_id", "rule_type", "severity", "advice_text", "source_id"],
    "food_rules": ["food_rule_id", "item_code", "ingredient_id", "food_name", "effect_type", "severity", "advice_text", "source_id"],
    "product_images": ["image_id", "item_code", "image_path", "split", "difficulty", "ocr_text", "ocr_confidence"],
    "image_embeddings": ["image_id", "embedding_model", "embedding"],
    "evaluation_labels": ["image_id", "gold_item_code", "gold_name", "gold_manufacturer", "gold_ingredients", "gold_dur_rules", "gold_food_rules", "gold_sources"],
}


@dataclass(frozen=True)
class ProjectTables:
    """Container for loaded CSV rows keyed by table name."""

    tables: dict[str, list[dict[str, str]]]

    def __getitem__(self, table_name: str) -> list[dict[str, str]]:
        return self.tables[table_name]


def project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward until README.md and data/ exist."""

    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").exists():
            return candidate
    return current


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_columns(table_name: str, rows: Iterable[dict[str, str]]) -> None:
    expected = PROCESSED_TABLES[table_name]
    rows = list(rows)
    if not rows:
        raise ValueError(f"{table_name}.csv has no rows.")

    actual = list(rows[0].keys())
    missing = [column for column in expected if column not in actual]
    if missing:
        raise ValueError(f"{table_name}.csv is missing columns: {missing}")


def load_processed_tables(data_dir: Path | None = None) -> ProjectTables:
    root = project_root()
    processed_dir = data_dir or root / "data" / "processed"
    loaded: dict[str, list[dict[str, str]]] = {}

    for table_name in PROCESSED_TABLES:
        rows = read_csv_rows(processed_dir / f"{table_name}.csv")
        validate_columns(table_name, rows)
        loaded[table_name] = rows

    return ProjectTables(loaded)
