"""Sample an audited AI Hub medicine subset with DUR-positive/negative labels."""

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

AUDIT_FIELDS = [
    "product_id",
    "aihub_product_name",
    "aihub_category",
    "match_status",
    "match_score",
    "matched_product_name",
    "search_query_used",
    "item_seq",
    "record_status",
    "manufacturer",
    "classification",
    "permit_date",
    "standard_code",
    "license_status",
    "license_status_date",
    "ingredient_count",
    "ingredient_names",
    "dur_positive",
    "dur_rule_count",
    "dur_rule_types",
    "detail_url",
    "checked_at",
    "error",
]

RULE_FIELDS = [
    "product_id",
    "item_seq",
    "rule_index",
    "single_or_combination",
    "dur_ingredient",
    "dur_type",
    "dosage_form",
    "warning",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-zip", type=Path, required=True)
    parser.add_argument("--images-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, action="append", default=[])
    parser.add_argument("--prior-rules", type=Path, action="append", default=[])
    parser.add_argument("--positive-products", type=int, default=700)
    parser.add_argument("--negative-products", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--max-new-audits", type=int, default=2500)
    parser.add_argument("--min-match-score", type=float, default=0.88)
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def product_id(category: str, product_name: str) -> str:
    digest = hashlib.sha1(f"{category}|{product_name}".encode("utf-8")).hexdigest()[:12]
    return f"PROD_{digest}"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
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


def load_grouped_products(labels_zip_path: Path) -> dict[tuple[str, str], list[tuple[str, dict]]]:
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
            category = str(image_info.get("product_category", "")).strip()
            name = str(image_info.get("product_name", "")).strip()
            if category in CATEGORIES and name:
                grouped[(category, name)].append((member, data))
    return {key: records for key, records in grouped.items() if len(records) >= 2}


def normalize_audit_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: row.get(field, "") for field in AUDIT_FIELDS}
    if not normalized["product_id"] and normalized["aihub_category"] and normalized["aihub_product_name"]:
        normalized["product_id"] = product_id(
            normalized["aihub_category"],
            normalized["aihub_product_name"],
        )
    return normalized


def read_prior_audits(paths: list[Path]) -> dict[str, dict[str, str]]:
    audit_by_id: dict[str, dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            normalized = normalize_audit_row(row)
            if normalized["product_id"]:
                audit_by_id[normalized["product_id"]] = normalized
    return audit_by_id


def read_prior_rules(paths: list[Path]) -> dict[tuple[str, str, str, str, str], dict[str, str]]:
    rule_by_key: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            normalized = {field: row.get(field, "") for field in RULE_FIELDS}
            key = (
                normalized["product_id"],
                normalized["item_seq"],
                normalized["dur_ingredient"],
                normalized["dur_type"],
                normalized["warning"],
            )
            if normalized["product_id"] and normalized["dur_type"]:
                rule_by_key[key] = normalized
    return rule_by_key


def is_clean_match(row: dict[str, str]) -> bool:
    return (
        row.get("match_status") == "exact"
        and row.get("record_status") == "active"
        and not row.get("license_status", "").strip()
        and bool(row.get("item_seq", "").strip())
    )


def audit_label(row: dict[str, str]) -> str:
    if not is_clean_match(row):
        return "ineligible"
    return "positive" if row.get("dur_positive") == "Y" else "negative"


def audit_product(
    session: requests.Session,
    category: str,
    name: str,
    checked_at: str,
    min_match_score: float,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    pid = product_id(category, name)
    candidates = search_candidates(session, name)
    best = candidates[0] if candidates else None
    if best is None:
        row = {field: "" for field in AUDIT_FIELDS}
        row.update(
            {
                "product_id": pid,
                "aihub_product_name": name,
                "aihub_category": category,
                "match_status": "unmatched",
                "dur_positive": "N",
                "dur_rule_count": "0",
                "checked_at": checked_at,
            }
        )
        return row, []

    detail = parse_detail(session, best) if best.match_score >= min_match_score else None
    match_status = "low_confidence" if detail is None else ("exact" if best.match_score == 1.0 else "fuzzy")
    ingredients = detail["ingredients"] if detail else []
    rules = detail["dur_rules"] if detail else []
    item_seq = detail["item_seq"] if detail else best.item_seq
    row = {
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
        "ingredient_names": "|".join(item["ingredient_name"] for item in ingredients),
        "dur_positive": "Y" if rules else "N",
        "dur_rule_count": str(len(rules)),
        "dur_rule_types": "|".join(sorted({item["dur_type"] for item in rules})),
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
    return row, rule_rows


def counts_for_selection(selected: list[dict[str, str]]) -> dict[str, int]:
    return {
        "positive": sum(row.get("dur_positive") == "Y" for row in selected),
        "negative": sum(row.get("dur_positive") != "Y" for row in selected),
        "total": len(selected),
    }


def select_from_audits(
    grouped: dict[tuple[str, str], list[tuple[str, dict]]],
    audit_by_id: dict[str, dict[str, str]],
    positive_target: int,
    negative_target: int,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    used_item_seq: set[str] = set()
    for label, target in (("positive", positive_target), ("negative", negative_target)):
        rows = [
            row
            for row in audit_by_id.values()
            if audit_label(row) == label
            and (row["aihub_category"], row["aihub_product_name"]) in grouped
        ]
        rows = sorted(rows, key=lambda row: (row["aihub_category"], row["aihub_product_name"]))
        for row in rows:
            if sum(audit_label(item) == label for item in selected) >= target:
                break
            if row["item_seq"] not in used_item_seq:
                selected.append(row)
                used_item_seq.add(row["item_seq"])
    return selected


def save_checkpoint(
    output_dir: Path,
    audit_by_id: dict[str, dict[str, str]],
    rule_by_key: dict[tuple[str, str, str, str, str], dict[str, str]],
    selected: list[dict[str, str]],
    grouped: dict[tuple[str, str], list[tuple[str, dict]]],
    positive_target: int,
    negative_target: int,
    checked_at: str,
) -> None:
    audit_rows = sorted(audit_by_id.values(), key=lambda row: row["product_id"])
    rule_rows = sorted(rule_by_key.values(), key=lambda row: (row["product_id"], row["rule_index"]))
    write_csv(output_dir / "candidate_audit_checkpoint.csv", audit_rows, AUDIT_FIELDS)
    write_csv(output_dir / "candidate_dur_rules_checkpoint.csv", rule_rows, RULE_FIELDS)
    write_csv(output_dir / "products_checkpoint.csv", selected, AUDIT_FIELDS)
    summary = {
        "checked_at": checked_at,
        "targets": {
            "positive": positive_target,
            "negative": negative_target,
            "total": positive_target + negative_target,
        },
        "selected": counts_for_selection(selected),
        "audited_products": len(audit_rows),
        "eligible_products": sum(audit_label(row) in {"positive", "negative"} for row in audit_rows),
        "eligible_positive": sum(audit_label(row) == "positive" for row in audit_rows),
        "eligible_negative": sum(audit_label(row) == "negative" for row in audit_rows),
        "total_grouped_aihub_products": len(grouped),
    }
    (output_dir / "summary_checkpoint.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def extract_manifest(
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
    manifest_rows: list[dict[str, str]] = []
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
        for row in sorted(selected, key=lambda item: item["product_id"]):
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
                manifest_rows.append(
                    {
                        "image_id": identifier,
                        "product_id": row["product_id"],
                        "item_seq": row["item_seq"],
                        "safety_label": "dur_positive" if row["dur_positive"] == "Y" else "dur_negative",
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
    return manifest_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    checked_at = datetime.now().astimezone().isoformat(timespec="seconds")

    print("Reading AI Hub validation labels...", flush=True)
    grouped = load_grouped_products(args.labels_zip)
    print(f"AI Hub products with >=2 medicine images: {len(grouped)}", flush=True)

    audit_by_id = read_prior_audits(args.prior_audit)
    checkpoint_audit = args.output_dir / "candidate_audit_checkpoint.csv"
    if checkpoint_audit.exists():
        audit_by_id.update(read_prior_audits([checkpoint_audit]))

    rule_by_key = read_prior_rules(args.prior_rules)
    checkpoint_rules = args.output_dir / "candidate_dur_rules_checkpoint.csv"
    if checkpoint_rules.exists():
        rule_by_key.update(read_prior_rules([checkpoint_rules]))

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/137 Safari/537.36"
            )
        }
    )

    candidates = [
        key
        for key in grouped
        if product_id(*key) not in audit_by_id
    ]
    rng.shuffle(candidates)
    new_audits = 0

    selected = select_from_audits(
        grouped,
        audit_by_id,
        args.positive_products,
        args.negative_products,
    )
    print(f"Seed selection: {counts_for_selection(selected)}", flush=True)

    while (
        counts_for_selection(selected)["positive"] < args.positive_products
        or counts_for_selection(selected)["negative"] < args.negative_products
    ):
        if new_audits >= args.max_new_audits or not candidates:
            message = (
                "Could not fill requested targets. "
                f"selected={counts_for_selection(selected)}, "
                f"new_audits={new_audits}, candidates_left={len(candidates)}"
            )
            if not args.allow_partial:
                raise RuntimeError(message)
            print(message, flush=True)
            break

        category, name = candidates.pop()
        pid = product_id(category, name)
        new_audits += 1
        print(
            f"[audit {new_audits}/{args.max_new_audits}] {name} "
            f"selected={counts_for_selection(selected)}",
            flush=True,
        )
        try:
            audit_row, rule_rows = audit_product(
                session,
                category,
                name,
                checked_at,
                args.min_match_score,
            )
        except Exception as exc:
            audit_row = {field: "" for field in AUDIT_FIELDS}
            audit_row.update(
                {
                    "product_id": pid,
                    "aihub_product_name": name,
                    "aihub_category": category,
                    "match_status": "error",
                    "dur_positive": "N",
                    "dur_rule_count": "0",
                    "checked_at": checked_at,
                    "error": repr(exc),
                }
            )
            rule_rows = []

        audit_by_id[pid] = audit_row
        for rule in rule_rows:
            normalized = {field: rule.get(field, "") for field in RULE_FIELDS}
            key = (
                normalized["product_id"],
                normalized["item_seq"],
                normalized["dur_ingredient"],
                normalized["dur_type"],
                normalized["warning"],
            )
            if normalized["product_id"] and normalized["dur_type"]:
                rule_by_key[key] = normalized

        selected = select_from_audits(
            grouped,
            audit_by_id,
            args.positive_products,
            args.negative_products,
        )

        if new_audits % 20 == 0 or audit_label(audit_row) in {"positive", "negative"}:
            save_checkpoint(
                args.output_dir,
                audit_by_id,
                rule_by_key,
                selected,
                grouped,
                args.positive_products,
                args.negative_products,
                checked_at,
            )
        time.sleep(args.delay)

    save_checkpoint(
        args.output_dir,
        audit_by_id,
        rule_by_key,
        selected,
        grouped,
        args.positive_products,
        args.negative_products,
        checked_at,
    )

    selected_ids = {row["product_id"] for row in selected}
    selected_rules = [
        row for row in rule_by_key.values() if row["product_id"] in selected_ids
    ]
    write_csv(args.output_dir / "products.csv", selected, AUDIT_FIELDS)
    write_csv(args.output_dir / "item_dur_rules.csv", selected_rules, RULE_FIELDS)

    manifest_rows: list[dict[str, str]] = []
    if args.extract_images:
        manifest_rows = extract_manifest(
            grouped,
            selected,
            args.labels_zip,
            args.images_zip,
            args.output_dir,
            rng,
        )
        write_csv(args.output_dir / "manifest.csv", manifest_rows)

    summary = {
        "checked_at": checked_at,
        "seed": args.seed,
        "targets": {
            "positive": args.positive_products,
            "negative": args.negative_products,
            "total": args.positive_products + args.negative_products,
        },
        "selected": counts_for_selection(selected),
        "new_audits": new_audits,
        "audited_products": len(audit_by_id),
        "eligible_positive": sum(audit_label(row) == "positive" for row in audit_by_id.values()),
        "eligible_negative": sum(audit_label(row) == "negative" for row in audit_by_id.values()),
        "dur_rule_rows": len(selected_rules),
        "images_extracted": bool(args.extract_images),
        "manifest_rows": len(manifest_rows),
        "selection_rule": (
            "AI Hub validation medicine products with >=2 images; exact MFDS match; "
            "active record; no cancellation/withdrawal status; unique item_seq."
        ),
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
