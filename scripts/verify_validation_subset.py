"""Verify extracted validation subset files and gallery/query pairing."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.subset_dir.resolve()
    with (root / "manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    roles: dict[str, set[str]] = defaultdict(set)
    errors: list[str] = []
    for row in rows:
        roles[row["product_id"]].add(row["role"])
        try:
            with Image.open(row["image_path"]) as image:
                image.verify()
            json.loads(Path(row["annotation_path"]).read_text(encoding="utf-8"))
        except Exception as error:
            errors.append(f"{row['image_id']}: {error!r}")

    summary = {
        "manifest_rows": len(rows),
        "products": len(roles),
        "complete_gallery_query_pairs": sum(
            role_set == {"gallery", "query"} for role_set in roles.values()
        ),
        "invalid_files": len(errors),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\n".join(errors[:20]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
