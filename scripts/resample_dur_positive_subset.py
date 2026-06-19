"""Build a reproducible 100-product AI Hub subset with product-level DUR rules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import requests

from audit_dur_coverage import parse_detail, search_candidates


PROFESSIONAL = "의약품_전문의약품"
GENERAL = "의약품_일반의약품"
CATEGORIES = (PROFESSIONAL, GENERAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--prior-rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--professional-products", type=int, default=70)
    parser.add_argument("--general-products", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--max-new-audits", type=int, default=500)
    return parser.parse_args()


def product_id(category: str, product_name: str) -> str:
    digest = hashlib.sha1(f"{category}|{product_name}".encode("utf-8")).hexdigest()[:12]
    return f"PROD_{digest}"


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


def joined_ocr_text(data: dict) -> str:
    texts: list[str] = []
    for annotation in data.get("annotations", []):
        for polygon in annotation.get("polygons", []):
            text = str(polygon.get("text", "")).strip()
            if text:
                texts.append(text)
    return " ".join(texts)


def load_products(
    labels_zip_path: Path,
) -> dict[tuple[str, str], list[tuple[str, dict]]]:
    grouped: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    with zipfile.ZipFile(labels_zip_path) as labels_zip:
        members = sorted(
            member
            for member in labels_zip.namelist()
            if member.startswith("result/medicine/annotations/") and member.endswith(".json")
        )
        for member in members:
            data = json.loads(labels_zip.read(member))
            image_info = data["images"][0]
            category = str(image_info.get("product_category", "")).strip()
            name = str(image_info.get("product_name", "")).strip()
            if category in CATEGORIES and name:
                grouped[(category, name)].append((member, data))
    return {
        key: records
        for key, records in grouped.items()
        if len(records) >= 2
    }


def eligible_audit(row: dict[str, str]) -> bool:
    return (
        row.get("match_status") == "exact"
        and row.get("dur_positive") == "Y"
        and not row.get("license_status", "").strip()
        and row.get("record_status") == "active"
        and bool(row.get("item_seq", "").strip())
    )


def audit_product(
    session: requests.Session,
    category: str,
    name: str,
    checked_at: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    pid = product_id(category, name)
    candidates = search_candidates(session, name)
    best = candidates[0] if candidates else None
    if best is None:
        return {
            "product_id": pid,
            "aihub_product_name": name,
            "aihub_category": category,
            "match_status": "unmatched",
            "match_score": "",
            "matched_product_name": "",
            "search_query_used": "",
            "item_seq": "",
            "record_status": "",
            "manufacturer": "",
            "classification": "",
            "permit_date": "",
            "standard_code": "",
            "license_status": "",
            "license_status_date": "",
            "ingredient_count": "0",
            "ingredient_names": "",
            "dur_positive": "N",
            "dur_rule_count": "0",
            "dur_rule_types": "",
            "detail_url": "",
            "checked_at": checked_at,
            "error": "",
        }, []

    detail = parse_detail(session, best) if best.match_score >= 0.88 else None
    match_status = (
        "low_confidence"
        if detail is None
        else ("exact" if best.match_score == 1.0 else "fuzzy")
    )
    ingredients = detail["ingredients"] if detail else []
    rules = detail["dur_rules"] if detail else []
    item_seq = detail["item_seq"] if detail else best.item_seq
    audit_row = {
        "product_id": pid,
        "aihub_product_name": name,
        "aihub_category": category,
        "match_status": match_status,
        "match_score": f"{best.match_score:.4f}",
        "matched_product_name": detail["official_name"] if detail else best.name,
        "search_query_used": best.search_query_used,
        "item_seq": item_seq,
        "record_status": best.record_status,
        "manufacturer": detail["manufacturer"] if detail else "",
        "classification": detail["classification"] if detail else "",
        "permit_date": detail["permit_date"] if detail else "",
        "standard_code": detail["standard_code"] if detail else "",
        "license_status": detail["license_status"] if detail else "",
        "license_status_date": detail["license_status_date"] if detail else "",
        "ingredient_count": str(len(ingredients)),
        "ingredient_names": "|".join(row["ingredient_name"] for row in ingredients),
        "dur_positive": "Y" if rules else "N",
        "dur_rule_count": str(len(rules)),
        "dur_rule_types": "|".join(sorted({row["dur_type"] for row in rules})),
        "detail_url": detail["detail_url"] if detail else best.detail_url,
        "checked_at": checked_at,
        "error": "",
    }
    rule_rows = [
        {
            "product_id": pid,
            "item_seq": item_seq,
            "rule_index": str(index),
            **rule,
        }
        for index, rule in enumerate(rules, start=1)
    ]
    return audit_row, rule_rows


def choose_products(
    grouped: dict[tuple[str, str], list[tuple[str, dict]]],
    audit_by_id: dict[str, dict[str, str]],
    targets: dict[str, int],
    rng: random.Random,
    session: requests.Session,
    checked_at: str,
    checkpoint_dir: Path,
    all_rules: list[dict[str, str]],
    delay: float,
    max_new_audits: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_item_seq: set[str] = set()

    for category in CATEGORIES:
        seeded = sorted(
            (
                row
                for row in audit_by_id.values()
                if row.get("aihub_category") == category
                and eligible_audit(row)
                and (category, row["aihub_product_name"]) in grouped
            ),
            key=lambda row: row["aihub_product_name"],
        )
        rng.shuffle(seeded)
        for row in seeded:
            if len([item for item in selected if item["aihub_category"] == category]) >= targets[category]:
                break
            if row["item_seq"] not in used_item_seq:
                selected.append(row)
                used_item_seq.add(row["item_seq"])

    new_audits = 0
    category_order = [PROFESSIONAL, GENERAL]
    candidate_queues: dict[str, list[tuple[str, str]]] = {}
    for category in CATEGORIES:
        candidates = [
            (category, name)
            for candidate_category, name in grouped
            if candidate_category == category
            and product_id(category, name) not in audit_by_id
        ]
        rng.shuffle(candidates)
        candidate_queues[category] = candidates

    while any(
        sum(row["aihub_category"] == category for row in selected) < targets[category]
        for category in CATEGORIES
    ):
        progressed = False
        for category in category_order:
            current = sum(row["aihub_category"] == category for row in selected)
            if current >= targets[category]:
                continue
            if not candidate_queues[category]:
                raise RuntimeError(f"No candidates remain for category: {category}")
            if new_audits >= max_new_audits:
                raise RuntimeError(
                    f"Reached --max-new-audits={max_new_audits} before filling targets"
                )

            _, name = candidate_queues[category].pop()
            pid = product_id(category, name)
            new_audits += 1
            print(
                f"[audit {new_audits}] {category} / {name} "
                f"(selected {len(selected)}/{sum(targets.values())})",
                flush=True,
            )
            try:
                row, rule_rows = audit_product(session, category, name, checked_at)
            except Exception as error:
                row = {
                    "product_id": pid,
                    "aihub_product_name": name,
                    "aihub_category": category,
                    "match_status": "error",
                    "match_score": "",
                    "matched_product_name": "",
                    "search_query_used": "",
                    "item_seq": "",
                    "record_status": "",
                    "manufacturer": "",
                    "classification": "",
                    "permit_date": "",
                    "standard_code": "",
                    "license_status": "",
                    "license_status_date": "",
                    "ingredient_count": "0",
                    "ingredient_names": "",
                    "dur_positive": "N",
                    "dur_rule_count": "0",
                    "dur_rule_types": "",
                    "detail_url": "",
                    "checked_at": checked_at,
                    "error": repr(error),
                }
                rule_rows = []

            audit_by_id[pid] = row
            all_rules.extend(rule_rows)
            if eligible_audit(row) and row["item_seq"] not in used_item_seq:
                selected.append(row)
                used_item_seq.add(row["item_seq"])
                print(f"  accepted: {row['item_seq']} / {row['dur_rule_types']}", flush=True)

            write_csv(
                checkpoint_dir / "candidate_audit_checkpoint.csv",
                sorted(audit_by_id.values(), key=lambda value: value["product_id"]),
            )
            write_csv(
                checkpoint_dir / "candidate_dur_rules_checkpoint.csv",
                all_rules,
            )
            time.sleep(delay)
            progressed = True
        if not progressed:
            break

    return selected


def extract_subset(
    grouped: dict[tuple[str, str], list[tuple[str, dict]]],
    selected: list[dict[str, str]],
    labels_zip_path: Path,
    images_zip_path: Path,
    output_dir: Path,
    rng: random.Random,
) -> list[dict[str, str]]:
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []

    with (
        zipfile.ZipFile(labels_zip_path) as labels_zip,
        zipfile.ZipFile(images_zip_path) as images_zip,
    ):
        image_member_by_name = {
            PurePosixPath(member).name: member
            for member in images_zip.namelist()
            if member.startswith("result/medicine/images/")
            and member.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        for row in sorted(selected, key=lambda value: value["product_id"]):
            key = (row["aihub_category"], row["aihub_product_name"])
            records = sorted(grouped[key], key=lambda value: value[1]["Identifier"])
            chosen = rng.sample(records, 2)
            for role, (label_member, data) in zip(("gallery", "query"), chosen):
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
                manifest.append(
                    {
                        "image_id": identifier,
                        "product_id": row["product_id"],
                        "item_seq": row["item_seq"],
                        "product_name": row["aihub_product_name"],
                        "matched_product_name": row["matched_product_name"],
                        "product_category": row["aihub_category"],
                        "role": role,
                        "image_path": str(image_path.resolve()),
                        "annotation_path": str(annotation_path.resolve()),
                        "width": str(image_info.get("width", "")),
                        "height": str(image_info.get("height", "")),
                        "shooting_env": str(image_info.get("shooting_env", "")),
                        "ocr_ground_truth": joined_ocr_text(data),
                    }
                )
    return manifest


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    print("Reading AI Hub validation labels...", flush=True)
    grouped = load_products(args.labels_zip)
    print(f"Eligible AI Hub products with >=2 images: {len(grouped)}", flush=True)

    prior_audit = read_csv(args.prior_audit)
    audit_by_id = {row["product_id"]: row for row in prior_audit}
    all_rules = read_csv(args.prior_rules)
    targets = {
        PROFESSIONAL: args.professional_products,
        GENERAL: args.general_products,
    }
    rng = random.Random(args.seed)
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/137 Safari/537.36"
            )
        }
    )

    selected = choose_products(
        grouped=grouped,
        audit_by_id=audit_by_id,
        targets=targets,
        rng=rng,
        session=session,
        checked_at=checked_at,
        checkpoint_dir=args.output_dir,
        all_rules=all_rules,
        delay=args.delay,
        max_new_audits=args.max_new_audits,
    )
    selected_ids = {row["product_id"] for row in selected}
    selected_rules = [row for row in all_rules if row["product_id"] in selected_ids]
    manifest = extract_subset(
        grouped=grouped,
        selected=selected,
        labels_zip_path=args.labels_zip,
        images_zip_path=args.images_zip,
        output_dir=args.output_dir,
        rng=rng,
    )

    write_csv(
        args.output_dir / "products.csv",
        sorted(selected, key=lambda value: value["product_id"]),
    )
    write_csv(
        args.output_dir / "manifest.csv",
        sorted(manifest, key=lambda value: (value["product_id"], value["role"])),
    )
    write_csv(args.output_dir / "item_dur_rules.csv", selected_rules)

    summary = {
        "created_at": checked_at,
        "seed": args.seed,
        "selection_rule": (
            "AI Hub validation product with at least two images; exact MFDS product-name "
            "match; active record; no cancellation/withdrawal status; at least one "
            "product-level DUR rule; unique item_seq."
        ),
        "total_products": len(selected),
        "professional_products": sum(
            row["aihub_category"] == PROFESSIONAL for row in selected
        ),
        "general_products": sum(row["aihub_category"] == GENERAL for row in selected),
        "total_images": len(manifest),
        "gallery_images": sum(row["role"] == "gallery" for row in manifest),
        "query_images": sum(row["role"] == "query" for row in manifest),
        "unique_item_seq": len({row["item_seq"] for row in selected}),
        "dur_rule_rows": len(selected_rules),
        "source": "AI Hub validation package OCR data + MFDS NEDRUG product detail DUR",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; checkpoint files were preserved.", file=sys.stderr)
        raise
