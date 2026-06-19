"""Build CLIP package image embeddings, NPY storage, and a FAISS index."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from .faiss_store import FaissMetaRecord, build_inner_product_index, save_index, write_meta
from .image_embedder import ClipImageEmbedder
from .schemas import load_processed_tables, project_root


def resolve_image_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def rows_from_processed_tables() -> list[dict[str, str]]:
    tables = load_processed_tables()
    return [
        {
            "image_id": row["image_id"],
            "item_code": row["item_code"],
            "image_path": row["image_path"],
            "role": row.get("split", ""),
        }
        for row in tables["product_images"]
    ]


def rows_from_manifest(path: Path, item_code_column: str) -> list[dict[str, str]]:
    rows = read_manifest_rows(path)
    if not rows:
        raise SystemExit(f"Manifest has no rows: {path}")
    required = {"image_id", "image_path", item_code_column}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise SystemExit(f"Manifest is missing columns: {missing}")
    return [
        {
            "image_id": row["image_id"],
            "item_code": row[item_code_column],
            "image_path": row["image_path"],
            "role": row.get("role", ""),
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build package image embedding artifacts.")
    parser.add_argument("--manifest-csv", type=Path)
    parser.add_argument("--item-code-column", default="item_seq")
    parser.add_argument("--role", choices=["all", "gallery", "query"], default="all")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-prefix", default="image_embedding")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    if args.manifest_csv:
        image_rows = rows_from_manifest(args.manifest_csv, args.item_code_column)
    else:
        image_rows = rows_from_processed_tables()

    if args.role != "all":
        image_rows = [row for row in image_rows if row.get("role") == args.role]

    if args.limit:
        image_rows = image_rows[: args.limit]

    selected_rows = []
    image_paths: list[Path] = []
    missing_paths: list[Path] = []
    for row in image_rows:
        image_path = resolve_image_path(root, row["image_path"])
        if image_path.exists():
            selected_rows.append(row)
            image_paths.append(image_path)
        else:
            missing_paths.append(image_path)

    if missing_paths and not args.allow_missing:
        print("Missing image files:")
        for path in missing_paths[:20]:
            print(f"- {path}")
        if len(missing_paths) > 20:
            print(f"... and {len(missing_paths) - 20} more")
        raise SystemExit(
            "Put package images under data/raw/package or run with --allow-missing "
            "to embed only existing files."
        )

    if not image_paths:
        raise SystemExit("No image files were found to embed.")

    print(
        f"Embedding {len(image_paths)} images "
        f"(role={args.role}, model={args.model_name}, batch_size={args.batch_size})"
    )
    embedder = ClipImageEmbedder(model_name=args.model_name)
    embeddings = embedder.embed_paths(image_paths, batch_size=args.batch_size)

    output_dir = args.output_dir or root / "data" / "processed"
    embeddings_path = output_dir / f"{args.output_prefix}_embeddings.npy"
    meta_path = output_dir / f"{args.output_prefix}_meta.csv"
    index_path = output_dir / f"{args.output_prefix}_index.faiss"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, embeddings.astype(np.float32))
    meta_records = [
        FaissMetaRecord(
            row_id=index,
            image_id=row["image_id"],
            item_code=row["item_code"],
            image_path=row["image_path"],
            embedding_model=args.model_name,
        )
        for index, row in enumerate(selected_rows)
    ]
    write_meta(meta_path, meta_records)

    index = build_inner_product_index(embeddings)
    save_index(index, index_path)

    print("Built image embedding artifacts")
    print(f"embeddings: {embeddings_path}")
    print(f"metadata:   {meta_path}")
    print(f"faiss:      {index_path}")
    print(f"shape:      {embeddings.shape}")


if __name__ == "__main__":
    main()
