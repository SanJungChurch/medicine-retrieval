"""Build a multi-gallery evaluation subset from selected DUR-positive products."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--products-csv", type=Path, required=True)
    parser.add_argument("--rules-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products", type=int, default=50)
    parser.add_argument("--gallery-per-product", type=int, default=2)
    parser.add_argument("--query-per-product", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260618)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def product_id(category: str, product_name: str) -> str:
    digest = hashlib.sha1(f"{category}|{product_name}".encode("utf-8")).hexdigest()[:12]
    return f"PROD_{digest}"


def joined_ocr_text(data: dict) -> str:
    texts: list[str] = []
    for annotation in data.get("annotations", []):
        for polygon in annotation.get("polygons", []):
            text = str(polygon.get("text", "")).strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def load_grouped_labels(
    labels_zip_path: Path,
    selected_keys: set[tuple[str, str]],
) -> dict[tuple[str, str], list[tuple[str, dict]]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    with zipfile.ZipFile(labels_zip_path) as labels_zip:
        for member in labels_zip.namelist():
            if not (
                member.startswith("result/medicine/annotations/")
                and member.endswith(".json")
            ):
                continue
            data = json.loads(labels_zip.read(member))
            image_info = data["images"][0]
            key = (
                str(image_info.get("product_category", "")).strip(),
                str(image_info.get("product_name", "")).strip(),
            )
            if key in selected_keys:
                grouped[key].append((member, data))
    return grouped


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")

    total_images_per_product = args.gallery_per_product + args.query_per_product
    rng = random.Random(args.seed)

    products = read_csv(args.products_csv)
    rules = read_csv(args.rules_csv)
    product_by_key = {
        (row["aihub_category"], row["aihub_product_name"]): row
        for row in products
    }

    print("Reading AI Hub validation labels...", flush=True)
    grouped = load_grouped_labels(args.labels_zip, set(product_by_key))
    eligible_keys = [
        key
        for key, records in grouped.items()
        if len(records) >= total_images_per_product
    ]
    if len(eligible_keys) < args.products:
        raise SystemExit(
            f"Only {len(eligible_keys)} products have at least "
            f"{total_images_per_product} images; requested {args.products}."
        )

    eligible_keys = sorted(eligible_keys, key=lambda key: (key[0], key[1]))
    selected_keys = rng.sample(eligible_keys, args.products)

    images_dir = args.output_dir / "images"
    annotations_dir = args.output_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, str]] = []
    selected_products = [product_by_key[key] for key in selected_keys]
    selected_product_ids = {row["product_id"] for row in selected_products}
    selected_rules = [row for row in rules if row["product_id"] in selected_product_ids]

    with (
        zipfile.ZipFile(args.labels_zip) as labels_zip,
        zipfile.ZipFile(args.images_zip) as images_zip,
    ):
        image_member_by_name = {
            PurePosixPath(member).name: member
            for member in images_zip.namelist()
            if member.startswith("result/medicine/images/")
            and member.lower().endswith((".jpg", ".jpeg", ".png"))
        }

        for key in sorted(selected_keys, key=lambda value: product_by_key[value]["product_id"]):
            product = product_by_key[key]
            records = sorted(grouped[key], key=lambda value: value[1]["Identifier"])
            chosen = rng.sample(records, total_images_per_product)
            role_values = (
                ["gallery"] * args.gallery_per_product
                + ["query"] * args.query_per_product
            )
            for role_index, (role, (label_member, data)) in enumerate(
                zip(role_values, chosen),
                start=1,
            ):
                image_info = data["images"][0]
                image_name = data["name"]
                image_member = image_member_by_name.get(image_name)
                if image_member is None:
                    raise RuntimeError(f"Missing image for {label_member}: {image_name}")
                identifier = data["Identifier"]
                extension = PurePosixPath(image_name).suffix.lower()
                image_path = images_dir / f"{identifier}{extension}"
                annotation_path = annotations_dir / f"{identifier}.json"
                with images_zip.open(image_member) as source, image_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                annotation_path.write_bytes(labels_zip.read(label_member))
                manifest_rows.append(
                    {
                        "image_id": identifier,
                        "product_id": product["product_id"],
                        "item_seq": product["item_seq"],
                        "product_name": product["aihub_product_name"],
                        "matched_product_name": product["matched_product_name"],
                        "product_category": product["aihub_category"],
                        "role": role,
                        "role_index": str(role_index),
                        "image_path": str(image_path.resolve()),
                        "annotation_path": str(annotation_path.resolve()),
                        "width": str(image_info.get("width", "")),
                        "height": str(image_info.get("height", "")),
                        "shooting_env": str(image_info.get("shooting_env", "")),
                        "ocr_ground_truth": joined_ocr_text(data),
                    }
                )

    write_csv(
        args.output_dir / "products.csv",
        sorted(selected_products, key=lambda row: row["product_id"]),
    )
    write_csv(
        args.output_dir / "manifest.csv",
        sorted(manifest_rows, key=lambda row: (row["product_id"], row["role"], row["image_id"])),
    )
    write_csv(args.output_dir / "item_dur_rules.csv", selected_rules)

    image_counts = Counter(
        (row["product_category"], row["product_name"]) for row in manifest_rows
    )
    summary = {
        "seed": args.seed,
        "source_products_csv": str(args.products_csv),
        "selection_rule": (
            f"Products from the DUR-positive 100 subset with at least "
            f"{total_images_per_product} original AI Hub validation images."
        ),
        "total_products": len(selected_products),
        "professional_products": sum(
            row["aihub_category"] == "의약품_전문의약품" for row in selected_products
        ),
        "general_products": sum(
            row["aihub_category"] == "의약품_일반의약품" for row in selected_products
        ),
        "gallery_per_product": args.gallery_per_product,
        "query_per_product": args.query_per_product,
        "total_images": len(manifest_rows),
        "gallery_images": sum(row["role"] == "gallery" for row in manifest_rows),
        "query_images": sum(row["role"] == "query" for row in manifest_rows),
        "unique_item_seq": len({row["item_seq"] for row in selected_products}),
        "dur_rule_rows": len(selected_rules),
        "image_count_histogram": dict(sorted(Counter(image_counts.values()).items())),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
