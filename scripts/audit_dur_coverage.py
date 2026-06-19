"""Audit AI Hub medicine products against MFDS NEDRUG item and DUR data."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://nedrug.mfds.go.kr"
SEARCH_URL = f"{BASE_URL}/searchDrug"
DETAIL_PATTERN = re.compile(
    r"/pbp/(?P<section>CCBBB01T?)/getItemDetail\?itemSeq=(?P<item_seq>\d+)"
)


@dataclass(frozen=True)
class Candidate:
    name: str
    item_seq: str
    detail_url: str
    record_status: str
    match_score: float
    search_query_used: str


def compact(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return "".join(character for character in value if character.isalnum() or "가" <= character <= "힣")


def base_product_name(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value)
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\{[^}]*\}", "", value)
    value = re.sub(r"\b\d+\s*(?:정|캡슐|병|포|매)\b$", "", value)
    return compact(value)


def product_match_score(query: str, candidate: str) -> float:
    query_full = normalize_name(query)
    candidate_full = normalize_name(candidate)
    if query_full == candidate_full:
        return 1.0
    if query_full and query_full in candidate_full:
        return 0.97

    full_score = SequenceMatcher(None, query_full, candidate_full).ratio()
    query_base = normalize_name(base_product_name(query))
    candidate_base = normalize_name(base_product_name(candidate))
    base_score = SequenceMatcher(None, query_base, candidate_base).ratio()
    if query_base and query_base == candidate_base:
        base_score = 0.99
    return max(full_score, base_score)


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    timeout: int = 40,
    attempts: int = 3,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


def search_variants(product_name: str) -> list[str]:
    variants = [
        compact(product_name),
        compact(re.sub(r"\([^)]*수출[^)]*\)", "", product_name)),
        compact(re.sub(r"\{[^}]*수출[^}]*\}", "", product_name)),
        compact(re.sub(r"\s+\d+\s*(?:g|그램|정|캡슐|병|포|매|회).*$", "", product_name, flags=re.I)),
        base_product_name(product_name),
    ]
    variants.extend(
        [
            re.sub(r"(\d+(?:\.\d+)?)mg", r"\1밀리그램", value, flags=re.I)
            for value in list(variants)
        ]
    )
    if product_name.upper().endswith("Q"):
        variants.append(product_name[:-1] + "큐")

    unique: list[str] = []
    for value in variants:
        value = compact(value)
        if len(value) >= 2 and value not in unique:
            unique.append(value)
    return unique


def search_candidates(session: requests.Session, product_name: str) -> list[Candidate]:
    by_item_seq: dict[str, Candidate] = {}
    for search_query in search_variants(product_name):
        candidates = search_candidates_once(session, product_name, search_query)
        for candidate in candidates:
            current = by_item_seq.get(candidate.item_seq)
            if current is None or candidate.match_score > current.match_score:
                by_item_seq[candidate.item_seq] = candidate
        if by_item_seq:
            break
    return sorted(by_item_seq.values(), key=lambda candidate: candidate.match_score, reverse=True)


def search_candidates_once(
    session: requests.Session,
    original_product_name: str,
    search_query: str,
) -> list[Candidate]:
    params = {
        "searchYn": "true",
        "searchDivision": "detail",
        "page": "1",
        "itemName": search_query,
    }
    response = get_with_retry(session, f"{SEARCH_URL}?{urlencode(params)}")
    soup = BeautifulSoup(response.text, "html.parser")

    by_item_seq: dict[str, Candidate] = {}
    for link in soup.select('a[href*="getItemDetail?itemSeq="]'):
        name = compact(link.get_text(" ", strip=True))
        href = html.unescape(link.get("href", ""))
        match = DETAIL_PATTERN.search(href)
        if not match or not name:
            continue
        item_seq = match.group("item_seq")
        section = match.group("section")
        original_score = product_match_score(original_product_name, name)
        variant_score = product_match_score(search_query, name)
        candidate = Candidate(
            name=name,
            item_seq=item_seq,
            detail_url=urljoin(BASE_URL, match.group(0)),
            record_status="active" if section == "CCBBB01" else "archived",
            match_score=max(original_score, variant_score * 0.98),
            search_query_used=search_query,
        )
        current = by_item_seq.get(item_seq)
        if current is None or candidate.match_score > current.match_score:
            by_item_seq[item_seq] = candidate

    return sorted(by_item_seq.values(), key=lambda candidate: candidate.match_score, reverse=True)


def direct_cells(row) -> list[str]:
    return [
        compact(cell.get_text(" ", strip=True))
        for cell in row.find_all(["th", "td"], recursive=False)
    ]


def clean_labeled_cell(value: str, label: str) -> str:
    value = compact(value)
    if value.startswith(label):
        return compact(value[len(label):])
    return value


def parse_detail(session: requests.Session, candidate: Candidate) -> dict:
    response = get_with_retry(session, candidate.detail_url)
    soup = BeautifulSoup(response.text, "html.parser")

    info: dict[str, str] = {}
    ingredients: list[dict[str, str]] = []
    dur_rules: list[dict[str, str]] = []

    for table in soup.select("table"):
        table_text = compact(table.get_text(" ", strip=True))
        rows = table.select("tr")

        if "품목기준코드" in table_text and "제품명" in table_text:
            for row in rows:
                cells = direct_cells(row)
                if len(cells) >= 2:
                    info[cells[0]] = cells[1]

        if "성분명" in table_text and "분량" in table_text and "단위" in table_text:
            for row in rows:
                cells = direct_cells(row)
                if len(cells) >= 4 and cells[0].isdigit():
                    ingredients.append(
                        {
                            "ingredient_name": cells[1],
                            "amount": cells[2],
                            "unit": cells[3],
                            "standard": cells[4] if len(cells) > 4 else "",
                        }
                    )

        if "DUR유형" in table_text and "금기 및 주의내용" in table_text:
            for row in rows:
                cells = direct_cells(row)
                if len(cells) < 5 or "DUR유형" in cells[0]:
                    continue
                dur_rules.append(
                    {
                        "single_or_combination": clean_labeled_cell(cells[0], "단일/복합"),
                        "dur_ingredient": clean_labeled_cell(
                            cells[1], "DUR성분(성분1/성분2..[병용성분])"
                        ),
                        "dur_type": clean_labeled_cell(cells[2], "DUR유형"),
                        "dosage_form": clean_labeled_cell(cells[3], "제형"),
                        "warning": clean_labeled_cell(cells[4], "금기 및 주의내용"),
                        "note": clean_labeled_cell(cells[5], "비고") if len(cells) > 5 else "",
                    }
                )

    ingredients = list(
        {
            (
                row["ingredient_name"],
                row["amount"],
                row["unit"],
                row["standard"],
            ): row
            for row in ingredients
        }.values()
    )
    dur_rules = list(
        {
            (
                row["dur_ingredient"],
                row["dur_type"],
                row["dosage_form"],
                row["warning"],
                row["note"],
            ): row
            for row in dur_rules
            if row["dur_type"]
        }.values()
    )
    return {
        "detail_url": response.url,
        "official_name": info.get("제품명", candidate.name),
        "manufacturer": info.get("업체명", ""),
        "classification": info.get("전문/일반", ""),
        "permit_date": info.get("허가일", ""),
        "item_seq": info.get("품목기준코드", candidate.item_seq),
        "standard_code": info.get("표준코드", ""),
        "license_status": info.get("취소/취하구분", ""),
        "license_status_date": info.get("취소/취하일자", ""),
        "ingredients": ingredients,
        "dur_rules": dur_rules,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--products-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--min-match-score", type=float, default=0.88)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    with args.products_csv.open(encoding="utf-8-sig", newline="") as handle:
        products = list(csv.DictReader(handle))
    if args.limit:
        products = products[: args.limit]

    audit_rows: list[dict[str, str]] = []
    ingredient_rows: list[dict[str, str]] = []
    dur_rows: list[dict[str, str]] = []
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

    for index, product in enumerate(products, start=1):
        product_name = product["product_name"]
        print(f"[{index}/{len(products)}] {product_name}", flush=True)
        error_message = ""
        try:
            candidates = search_candidates(session, product_name)
            best = candidates[0] if candidates else None
            if best is None:
                match_status = "unmatched"
                detail = None
            elif best.match_score < args.min_match_score:
                match_status = "low_confidence"
                detail = None
            else:
                match_status = "exact" if best.match_score == 1.0 else "fuzzy"
                detail = parse_detail(session, best)
        except Exception as error:
            best = None
            detail = None
            match_status = "error"
            error_message = repr(error)

        ingredients = detail["ingredients"] if detail else []
        dur_rules = detail["dur_rules"] if detail else []
        item_seq = detail["item_seq"] if detail else (best.item_seq if best else "")

        audit_rows.append(
            {
                "product_id": product["product_id"],
                "aihub_product_name": product_name,
                "aihub_category": product["product_category"],
                "match_status": match_status,
                "match_score": f"{best.match_score:.4f}" if best else "",
                "matched_product_name": detail["official_name"] if detail else (best.name if best else ""),
                "search_query_used": best.search_query_used if best else "",
                "item_seq": item_seq,
                "record_status": best.record_status if best else "",
                "manufacturer": detail["manufacturer"] if detail else "",
                "classification": detail["classification"] if detail else "",
                "permit_date": detail["permit_date"] if detail else "",
                "standard_code": detail["standard_code"] if detail else "",
                "license_status": detail["license_status"] if detail else "",
                "license_status_date": detail["license_status_date"] if detail else "",
                "ingredient_count": str(len(ingredients)),
                "ingredient_names": "|".join(row["ingredient_name"] for row in ingredients),
                "dur_positive": "Y" if dur_rules else "N",
                "dur_rule_count": str(len(dur_rules)),
                "dur_rule_types": "|".join(
                    sorted({row["dur_type"] for row in dur_rules if row["dur_type"]})
                ),
                "detail_url": detail["detail_url"] if detail else (best.detail_url if best else ""),
                "checked_at": checked_at,
                "error": error_message,
            }
        )

        for ingredient in ingredients:
            ingredient_rows.append(
                {
                    "product_id": product["product_id"],
                    "item_seq": item_seq,
                    **ingredient,
                }
            )
        for rule_index, rule in enumerate(dur_rules, start=1):
            dur_rows.append(
                {
                    "product_id": product["product_id"],
                    "item_seq": item_seq,
                    "rule_index": str(rule_index),
                    **rule,
                }
            )

        time.sleep(args.delay)

    output_dir = args.output_dir.resolve()
    write_csv(
        output_dir / "dur_coverage_audit.csv",
        [
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
        ],
        audit_rows,
    )
    write_csv(
        output_dir / "item_ingredients.csv",
        ["product_id", "item_seq", "ingredient_name", "amount", "unit", "standard"],
        ingredient_rows,
    )
    write_csv(
        output_dir / "item_dur_rules.csv",
        [
            "product_id",
            "item_seq",
            "rule_index",
            "single_or_combination",
            "dur_ingredient",
            "dur_type",
            "dosage_form",
            "warning",
            "note",
        ],
        dur_rows,
    )

    matched = [row for row in audit_rows if row["match_status"] in {"exact", "fuzzy"}]
    summary = {
        "checked_at": checked_at,
        "total_products": len(audit_rows),
        "matched_products": len(matched),
        "exact_matches": sum(row["match_status"] == "exact" for row in audit_rows),
        "fuzzy_matches": sum(row["match_status"] == "fuzzy" for row in audit_rows),
        "low_confidence": sum(row["match_status"] == "low_confidence" for row in audit_rows),
        "unmatched": sum(row["match_status"] == "unmatched" for row in audit_rows),
        "errors": sum(row["match_status"] == "error" for row in audit_rows),
        "dur_positive_products": sum(row["dur_positive"] == "Y" for row in matched),
        "dur_negative_products": sum(row["dur_positive"] == "N" for row in matched),
        "source": "MFDS NEDRUG product search and item detail",
    }
    (output_dir / "dur_coverage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
