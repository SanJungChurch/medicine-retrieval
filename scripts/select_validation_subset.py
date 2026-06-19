"""Select and extract a reproducible AI Hub medicine validation subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath


PROFESSIONAL = "의약품_전문의약품"
GENERAL = "의약품_일반의약품"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--products-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    images_dir = output_dir / "images"
    labels_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    grouped: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    with zipfile.ZipFile(args.labels_zip) as labels_zip:
        label_members = sorted(
            member
            for member in labels_zip.namelist()
            if member.startswith("result/medicine/annotations/") and member.endswith(".json")
        )
        for member in label_members:
            data = json.loads(labels_zip.read(member))
            image_info = data["images"][0]
            category = image_info.get("product_category", "")
            if category not in {PROFESSIONAL, GENERAL}:
                continue
            name = image_info.get("product_name", "").strip()
            if name:
                grouped[(category, name)].append((member, data))

    eligible: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (category, name), records in grouped.items():
        if len(records) >= 2:
            eligible[category].append((category, name))

    rng = random.Random(args.seed)
    selected_products: list[tuple[str, str]] = []
    for category in (PROFESSIONAL, GENERAL):
        candidates = sorted(eligible[category], key=lambda value: value[1])
        if len(candidates) < args.products_per_category:
            raise SystemExit(
                f"Not enough eligible products for {category}: "
                f"{len(candidates)} < {args.products_per_category}"
            )
        selected_products.extend(rng.sample(candidates, args.products_per_category))

    manifest_rows: list[dict[str, str]] = []
    product_rows: list[dict[str, str]] = []
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

        for category, name in sorted(selected_products):
            records = sorted(grouped[(category, name)], key=lambda value: value[1]["Identifier"])
            chosen = rng.sample(records, 2)
            pid = product_id(category, name)
            product_rows.append(
                {
                    "product_id": pid,
                    "product_name": name,
                    "product_category": category,
                    "available_images": str(len(records)),
                }
            )

            for role, (label_member, data) in zip(("gallery", "query"), chosen):
                image_info = data["images"][0]
                image_name = data["name"]
                image_member = image_member_by_name.get(image_name)
                if image_member is None:
                    raise SystemExit(f"Missing image for label {label_member}: {image_name}")

                identifier = data["Identifier"]
                extension = PurePosixPath(image_name).suffix.lower()
                output_image_name = f"{identifier}{extension}"
                output_label_name = f"{identifier}.json"
                output_image_path = images_dir / output_image_name
                output_label_path = labels_dir / output_label_name

                with images_zip.open(image_member) as source, output_image_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
                output_label_path.write_bytes(labels_zip.read(label_member))

                manifest_rows.append(
                    {
                        "image_id": identifier,
                        "product_id": pid,
                        "product_name": name,
                        "product_category": category,
                        "role": role,
                        "image_path": str(output_image_path),
                        "annotation_path": str(output_label_path),
                        "source_image_member": image_member,
                        "source_label_member": label_member,
                        "width": str(image_info.get("width", "")),
                        "height": str(image_info.get("height", "")),
                        "shooting_env": str(image_info.get("shooting_env", "")),
                        "ocr_ground_truth": joined_ocr_text(data),
                    }
                )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(manifest_rows, key=lambda row: (row["product_id"], row["role"])))

    products_path = output_dir / "products.csv"
    with products_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(product_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(product_rows, key=lambda row: row["product_id"]))

    summary = {
        "seed": args.seed,
        "products_per_category": args.products_per_category,
        "total_products": len(product_rows),
        "total_images": len(manifest_rows),
        "gallery_images": sum(row["role"] == "gallery" for row in manifest_rows),
        "query_images": sum(row["role"] == "query" for row in manifest_rows),
        "categories": {
            PROFESSIONAL: sum(row["product_category"] == PROFESSIONAL for row in product_rows),
            GENERAL: sum(row["product_category"] == GENERAL for row in product_rows),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
