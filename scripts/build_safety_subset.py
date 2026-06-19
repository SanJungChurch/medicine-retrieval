"""Build the safety-information evaluation subset from DUR audit results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MATCHED_STATUSES = {"exact", "fuzzy"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest)
    audit_rows = read_csv(args.audit)

    eligible_rows = [
        row
        for row in audit_rows
        if row["match_status"] in MATCHED_STATUSES and not row["license_status"]
    ]
    eligible_ids = {row["product_id"] for row in eligible_rows}
    safety_manifest = [
        row for row in manifest_rows if row["product_id"] in eligible_ids
    ]

    safety_products = []
    for row in eligible_rows:
        safety_products.append(
            {
                **row,
                "safety_label": "dur_positive" if row["dur_positive"] == "Y" else "dur_negative",
                "manual_match_review": "Y" if row["match_status"] == "fuzzy" else "N",
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "safety_products.csv",
        list(safety_products[0].keys()),
        safety_products,
    )
    write_csv(
        args.output_dir / "safety_manifest.csv",
        list(safety_manifest[0].keys()),
        safety_manifest,
    )

    summary = {
        "eligible_products": len(safety_products),
        "eligible_images": len(safety_manifest),
        "gallery_images": sum(row["role"] == "gallery" for row in safety_manifest),
        "query_images": sum(row["role"] == "query" for row in safety_manifest),
        "dur_positive_products": sum(row["dur_positive"] == "Y" for row in safety_products),
        "dur_negative_products": sum(row["dur_positive"] == "N" for row in safety_products),
        "exact_matches": sum(row["match_status"] == "exact" for row in safety_products),
        "fuzzy_matches_requiring_review": sum(
            row["match_status"] == "fuzzy" for row in safety_products
        ),
        "exclusion_rule": (
            "Exclude unmatched/low-confidence records and products with a non-empty "
            "MFDS cancellation/withdrawal status."
        ),
    }
    (args.output_dir / "safety_subset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
