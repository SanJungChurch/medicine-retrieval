"""Evaluate DUR rule retrieval from candidate item rankings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DUR rule retrieval from rankings.")
    parser.add_argument("--rankings-csv", type=Path, required=True)
    parser.add_argument("--rules-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5])
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def recall(gold: set[str], predicted: set[str]) -> float:
    if not gold:
        return 1.0
    return len(gold & predicted) / len(gold)


def main() -> None:
    args = parse_args()
    ranking_rows = read_csv(args.rankings_csv)
    rule_rows = read_csv(args.rules_csv)

    rules_by_item: dict[str, set[str]] = defaultdict(set)
    for row in rule_rows:
        item_seq = row.get("item_seq", "")
        dur_type = row.get("dur_type", "")
        if item_seq and dur_type:
            rules_by_item[item_seq].add(dur_type)

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in ranking_rows:
        grouped[(row["method"], row["query_image_id"])].append(row)

    by_method: dict[str, list[tuple[str, list[str]]]] = defaultdict(list)
    labels: dict[str, str] = {}
    for (method, image_id), rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row["rank"]))
        gold = rows[0]["gold_item_code"]
        labels[image_id] = gold
        by_method[method].append((image_id, [row["matched_item_code"] for row in rows]))

    metric_rows: list[dict[str, str]] = []
    metrics_json: dict[str, dict[str, float]] = {}
    for method, method_rows in sorted(by_method.items()):
        total = len(method_rows)
        values: dict[str, float] = {}

        top1_item_hits = 0
        top1_rule_exact = 0
        top1_rule_jaccard_total = 0.0
        top1_rule_recall_total = 0.0
        for image_id, ranked_items in method_rows:
            gold_item = labels[image_id]
            top1_item = ranked_items[0] if ranked_items else ""
            gold_rules = rules_by_item.get(gold_item, set())
            top1_rules = rules_by_item.get(top1_item, set())
            top1_item_hits += int(top1_item == gold_item)
            top1_rule_exact += int(top1_rules == gold_rules)
            top1_rule_jaccard_total += jaccard(gold_rules, top1_rules)
            top1_rule_recall_total += recall(gold_rules, top1_rules)

        values["top1_item_accuracy"] = top1_item_hits / total if total else 0.0
        values["top1_rule_type_exact_match"] = top1_rule_exact / total if total else 0.0
        values["top1_rule_type_jaccard"] = top1_rule_jaccard_total / total if total else 0.0
        values["top1_rule_type_recall"] = top1_rule_recall_total / total if total else 0.0

        for k in args.k_values:
            item_hits = 0
            rule_recall_total = 0.0
            rule_jaccard_total = 0.0
            for image_id, ranked_items in method_rows:
                gold_item = labels[image_id]
                gold_rules = rules_by_item.get(gold_item, set())
                top_items = ranked_items[:k]
                item_hits += int(gold_item in top_items)
                union_rules: set[str] = set()
                for item in top_items:
                    union_rules.update(rules_by_item.get(item, set()))
                rule_recall_total += recall(gold_rules, union_rules)
                rule_jaccard_total += jaccard(gold_rules, union_rules)
            values[f"item_recall_at_{k}"] = item_hits / total if total else 0.0
            values[f"rule_type_recall_at_{k}"] = rule_recall_total / total if total else 0.0
            values[f"rule_type_jaccard_at_{k}"] = rule_jaccard_total / total if total else 0.0

        metrics_json[method] = values
        metric_rows.append(
            {
                "method": method,
                **{key: f"{value:.6f}" for key, value in values.items()},
            }
        )

    write_csv(args.output_csv, metric_rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(metrics_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metric_rows, ensure_ascii=False, indent=2))
    print(f"Saved CSV:  {args.output_csv}")
    print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
