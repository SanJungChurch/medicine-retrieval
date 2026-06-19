"""Local HTML prototype server for Hybrid OCR + Image-KG retrieval.

The server is intentionally small: it serves a static demo page and exposes a
single JSON API that accepts a base64 image, runs OCR/CLIP retrieval, and asks
an EXAONE-compatible endpoint to write a grounded safety card.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .faiss_store import FaissImageStore, read_meta
from .image_embedder import ClipImageEmbedder
from .run_easyocr_queries import flatten_easyocr_result


DEFAULT_DATA_DIR = Path(r"D:\medicine_data\validation_audited_1000")
SAFETY_SYSTEM_PROMPT = (
    "검색 근거에 충실한 한국어 복약 안전 정보 작성자입니다. "
    "근거 JSON에 없는 의약학 정보, 별칭, 병용 약물, 용량, 임신 관련 설명을 만들지 않습니다."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def product_document(row: dict[str, str]) -> str:
    parts = [
        row.get("aihub_product_name", ""),
        row.get("matched_product_name", ""),
        row.get("manufacturer", ""),
        row.get("classification", ""),
        row.get("standard_code", ""),
        row.get("ingredient_names", "").replace("|", " "),
    ]
    return " ".join(part for part in parts if part)


def minmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    score_min = float(scores.min())
    score_max = float(scores.max())
    if score_max <= score_min:
        return np.zeros_like(scores, dtype=np.float32)
    return (scores - score_min) / (score_max - score_min)


def dynamic_alpha(ocr_confidence: float) -> float:
    return max(0.3, min(0.8, 0.2 + 0.6 * ocr_confidence))


def compact_rules(rules: list[dict[str, str]], limit: int = 4) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    for row in rules[:limit]:
        compacted.append(
            {
                "dur_type": row.get("dur_type", ""),
                "ingredient": row.get("dur_ingredient", ""),
                "warning": row.get("warning", ""),
                "note": row.get("note", ""),
                "kind": row.get("single_or_combination", ""),
            }
        )
    return compacted


def split_ingredient_expression(value: str) -> list[str]:
    normalized = (
        value.replace("·", "/")
        .replace(",", "/")
        .replace("+", "/")
        .replace(" 및 ", "/")
        .replace("와 ", "/")
        .replace("과 ", "/")
    )
    parts = [part.strip() for part in normalized.split("/") if part.strip()]
    return parts or ([value.strip()] if value.strip() else [])


def format_rule(rule: dict[str, Any]) -> str:
    parts = [
        str(rule.get("dur_type") or "").strip(),
        str(rule.get("ingredient") or "").strip(),
        str(rule.get("warning") or "").strip(),
        str(rule.get("note") or "").strip(),
    ]
    return " ".join(part for part in parts if part) or "DUR 세부 정보 없음"


def explain_rule(rule: dict[str, Any]) -> str:
    dur_type = str(rule.get("dur_type") or "").strip()
    ingredient = str(rule.get("ingredient") or "").strip()
    warning = str(rule.get("warning") or "").strip()
    note = str(rule.get("note") or "").strip()
    examples_by_ingredient = rule.get("example_products") or {}
    target = ingredient or "해당 성분"
    example_lines = []
    for name in split_ingredient_expression(ingredient):
        examples = examples_by_ingredient.get(name) or []
        if examples:
            example_lines.append(f"{name}: {', '.join(examples[:3])}")
        else:
            example_lines.append(f"{name}: 데이터셋 내 예시 품목 없음, 성분명으로 확인 필요")
    examples_text = " / ".join(example_lines)

    if "병용금기" in dur_type:
        meaning = "같이 복용하지 않도록 관리되는 조합입니다."
        action = f"약 봉투나 처방전에서 {target} 성분 또는 아래 예시 품목과 같은 계열의 약이 있는지 약사에게 확인하세요."
    elif "임부금기" in dur_type:
        meaning = "임신 중이거나 임신 가능성이 있을 때 특히 확인이 필요한 성분입니다."
        action = "임신 중이거나 임신 가능성이 있으면 복용 전 의사 또는 약사에게 확인하세요."
    elif "용량" in dur_type:
        meaning = "하루 복용량이 기준을 넘지 않도록 확인해야 하는 항목입니다."
        action = "같은 성분이 들어간 다른 진통제나 감기약을 함께 먹고 있는지 확인하세요."
    elif "투여기간" in dur_type:
        meaning = "정해진 기간보다 오래 복용하지 않도록 확인해야 하는 항목입니다."
        action = "며칠째 복용 중인지 확인하고, 장기 복용 전 전문가에게 문의하세요."
    elif "연령" in dur_type or "노인" in dur_type:
        meaning = "나이에 따라 사용 제한이나 주의가 필요한 항목입니다."
        action = "복용자의 나이를 기준으로 약사 또는 의사에게 적절성을 확인하세요."
    else:
        meaning = "DUR에서 복용 전 확인이 필요한 안전 규칙으로 분류된 항목입니다."
        action = "현재 복용 중인 약과 건강 상태를 함께 알려주고 전문가에게 확인하세요."

    details = []
    if dur_type:
        details.append(f"분류: {dur_type}")
    if ingredient:
        details.append(f"관련 성분: {ingredient}")
    if warning:
        details.append(f"경고값: {warning}")
    if note:
        details.append(f"비고: {note}")
    detail_text = ", ".join(details) if details else "세부값 없음"
    examples_block = f"\n  데이터셋 내 예시 품목: {examples_text}" if examples_text else ""
    return f"- {detail_text}{examples_block}\n  의미: {meaning}\n  확인할 점: {action}"


def grounded_context_summary(context: dict[str, Any]) -> str:
    candidates = context.get("candidates") or []
    if not candidates:
        return "검색 후보 없음"
    top = candidates[0]
    rules = top.get("rules") or []
    rule_text = "\n".join(explain_rule(rule) for rule in rules) if rules else "연결된 DUR 규칙 없음"
    return (
        f"Top-1 품목: {top.get('product_name') or ''}\n"
        f"제조사: {top.get('manufacturer') or ''}\n"
        f"성분: {top.get('ingredients') or '성분 정보 없음'}\n"
        f"DUR 규칙: {rule_text}\n"
        "음식 상호작용: 프로토타입 데이터에 없음"
    )


def looks_ungrounded(text: str, evidence_text: str) -> bool:
    risky_terms = [
        "메토트렉세이트",
        "3200",
        "3,200",
        "임신 2기",
        "간손상",
        "심혈관",
        "위장관 출혈",
        "천공",
    ]
    return any(term in text and term not in evidence_text for term in risky_terms)


def missing_required_card_sections(text: str) -> bool:
    required_markers = ["1. 인식 결과", "2. 성분", "3. DUR 경고", "4. 음식 상호작용", "5. 근거와 한계"]
    return any(marker not in text for marker in required_markers)


class ExaoneClient:
    def __init__(self) -> None:
        self.provider = os.environ.get("EXAONE_PROVIDER", "local").strip().lower()
        self.model = os.environ.get("EXAONE_MODEL", "LGAI-EXAONE/EXAONE-4.0-1.2B")
        self.model_path = os.environ.get("EXAONE_MODEL_PATH", "").strip()
        self.base_url = os.environ.get("EXAONE_API_BASE", "http://127.0.0.1:11434").rstrip("/")
        self.api_key = os.environ.get("EXAONE_API_KEY", "")
        self.timeout_sec = float(os.environ.get("EXAONE_TIMEOUT_SEC", "45"))
        self.device_setting = os.environ.get("EXAONE_DEVICE", "").strip() or "auto"
        self._local_tokenizer: Any | None = None
        self._local_model: Any | None = None
        self._local_device = ""

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> tuple[str, str]:
        if os.environ.get("EXAONE_DISABLED", "0") == "1":
            return "", "EXAONE_DISABLED=1"
        try:
            if self.provider in {"local", "transformers", "hf"}:
                return self._generate_local(prompt, max_new_tokens=max_new_tokens), ""
            if self.provider == "openai":
                return self._generate_openai_compatible(prompt, max_new_tokens=max_new_tokens), ""
            return self._generate_ollama(prompt, max_new_tokens=max_new_tokens), ""
        except Exception as exc:  # pragma: no cover - depends on local LLM runtime
            return "", repr(exc)

    def _post_json(self, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))

    def _generate_ollama(self, prompt: str, max_new_tokens: int | None = None) -> str:
        token_budget = max_new_tokens or int(os.environ.get("EXAONE_MAX_NEW_TOKENS", "350"))
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": token_budget},
        }
        result = self._post_json(f"{self.base_url}/api/generate", payload)
        return str(result.get("response", "")).strip()

    def _generate_openai_compatible(self, prompt: str, max_new_tokens: int | None = None) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SAFETY_SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": max_new_tokens or int(os.environ.get("EXAONE_MAX_NEW_TOKENS", "350")),
        }
        result = self._post_json(f"{self.base_url}/chat/completions", payload, headers)
        choices = result.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", "")).strip()

    def _load_local(self) -> tuple[Any, Any, str]:
        if self._local_tokenizer is not None and self._local_model is not None:
            return self._local_tokenizer, self._local_model, self._local_device

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.model_path or self.model
        requested_device = os.environ.get("EXAONE_DEVICE", "").strip().lower()
        device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to(device)
        model.eval()
        self._local_tokenizer = tokenizer
        self._local_model = model
        self._local_device = device
        return tokenizer, model, device

    def _generate_local(self, prompt: str, max_new_tokens: int | None = None) -> str:
        import torch

        tokenizer, model, device = self._load_local()
        messages = [
            {
                "role": "system",
                "content": SAFETY_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            rendered = (
                f"System: {SAFETY_SYSTEM_PROMPT}\n"
                f"User: {prompt}\nAssistant:"
            )
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=4096).to(device)
        token_budget = max_new_tokens or int(os.environ.get("EXAONE_MAX_NEW_TOKENS", "350"))
        do_sample = os.environ.get("EXAONE_DO_SAMPLE", "0") == "1"
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": token_budget,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = float(os.environ.get("EXAONE_TEMPERATURE", "0.2"))
        with torch.no_grad():
            output_ids = model.generate(**generate_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class PrototypeEngine:
    def __init__(self, data_dir: Path, upload_dir: Path, clip_model: str, ocr_gpu: bool) -> None:
        self.data_dir = data_dir
        self.upload_dir = upload_dir
        self.clip_model = clip_model
        self.ocr_gpu = ocr_gpu
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.products = read_csv(data_dir / "products.csv")
        self.rules = read_csv(data_dir / "item_dur_rules.csv")
        self.item_codes = [row["item_seq"] for row in self.products]
        self.products_by_item = {row["item_seq"]: row for row in self.products}
        self.example_products_by_ingredient = self._build_example_products_by_ingredient()
        self.rules_by_item: dict[str, list[dict[str, str]]] = {}
        for row in self.rules:
            self.rules_by_item.setdefault(row["item_seq"], []).append(row)

        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=False)
        self.document_matrix = self.vectorizer.fit_transform([product_document(row) for row in self.products])

        embeddings_dir = data_dir / "embeddings"
        self.gallery_meta = read_meta(embeddings_dir / "clip_gallery_meta.csv")
        self.gallery_store = FaissImageStore.load(
            index_path=embeddings_dir / "clip_gallery_index.faiss",
            meta_path=embeddings_dir / "clip_gallery_meta.csv",
            embeddings_path=embeddings_dir / "clip_gallery_embeddings.npy",
        )
        self.embedder: ClipImageEmbedder | None = None
        self.ocr_reader: Any | None = None
        self.llm = ExaoneClient()

    def _build_example_products_by_ingredient(self) -> dict[str, list[str]]:
        examples: dict[str, list[str]] = {}
        for product in self.products:
            product_name = product.get("aihub_product_name") or product.get("matched_product_name") or ""
            if not product_name:
                continue
            for ingredient in str(product.get("ingredient_names") or "").split("|"):
                ingredient = ingredient.strip()
                if not ingredient:
                    continue
                bucket = examples.setdefault(ingredient, [])
                if product_name not in bucket:
                    bucket.append(product_name)
        return examples

    def _examples_for_ingredient(self, ingredient: str, current_item: str, limit: int = 3) -> list[str]:
        direct = self.example_products_by_ingredient.get(ingredient, [])
        current_name = self.products_by_item.get(current_item, {}).get("aihub_product_name", "")
        values = [name for name in direct if name != current_name]
        if len(values) < limit:
            for product in self.products:
                product_name = product.get("aihub_product_name") or product.get("matched_product_name") or ""
                ingredient_names = product.get("ingredient_names") or ""
                if product_name and product_name != current_name and ingredient in ingredient_names and product_name not in values:
                    values.append(product_name)
                if len(values) >= limit:
                    break
        return values[:limit]

    def compact_rules_for_item(self, item_code: str, rules: list[dict[str, str]], limit: int = 4) -> list[dict[str, Any]]:
        compacted = compact_rules(rules, limit=limit)
        for rule in compacted:
            examples: dict[str, list[str]] = {}
            for ingredient in split_ingredient_expression(str(rule.get("ingredient") or "")):
                examples[ingredient] = self._examples_for_ingredient(ingredient, item_code)
            rule["example_products"] = examples
        return compacted

    def status(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "products": len(self.products),
            "dur_rules": len(self.rules),
            "clip_model": self.clip_model,
            "ocr_gpu": self.ocr_gpu,
            "exaone_provider": self.llm.provider,
            "exaone_model": self.llm.model,
            "exaone_device": self.llm.device_setting,
            "exaone_base_url": self.llm.base_url,
        }

    def _get_embedder(self) -> ClipImageEmbedder:
        if self.embedder is None:
            self.embedder = ClipImageEmbedder(model_name=self.clip_model)
        return self.embedder

    def _get_ocr_reader(self) -> Any:
        if self.ocr_reader is None:
            import easyocr

            model_dir = Path(os.environ.get("EASYOCR_MODEL_DIR", r"D:\medicine_data\easyocr_models"))
            model_dir.mkdir(parents=True, exist_ok=True)
            self.ocr_reader = easyocr.Reader(
                ["ko", "en"],
                gpu=self.ocr_gpu,
                model_storage_directory=str(model_dir),
                verbose=False,
            )
        return self.ocr_reader

    def save_upload(self, payload: dict[str, Any]) -> Path:
        data_url = str(payload.get("image_data", ""))
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("Uploaded image is larger than 16MB.")
        suffix = Path(str(payload.get("filename") or "upload.jpg")).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"
        path = self.upload_dir / f"upload_{uuid.uuid4().hex}{suffix}"
        path.write_bytes(raw)
        return path

    def run_ocr(self, image_path: Path) -> dict[str, Any]:
        started_at = time.perf_counter()
        result = self._get_ocr_reader().readtext(
            str(image_path),
            canvas_size=int(os.environ.get("MEDICINE_OCR_CANVAS_SIZE", "1280")),
            mag_ratio=float(os.environ.get("MEDICINE_OCR_MAG_RATIO", "1.0")),
        )
        lines = flatten_easyocr_result(result)
        confidences = [float(line["confidence"]) for line in lines]
        return {
            "text": " ".join(str(line["text"]) for line in lines),
            "confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "line_count": len(lines),
            "elapsed_sec": time.perf_counter() - started_at,
            "lines": lines[:30],
        }

    def retrieve(self, payload: dict[str, Any]) -> dict[str, Any]:
        top_k = int(payload.get("top_k") or 5)
        top_k = max(1, min(top_k, 10))
        requested_alpha = payload.get("alpha", 0.8)
        use_dynamic = str(payload.get("mode", "")).lower() == "dynamic"
        max_new_tokens = int(payload.get("max_new_tokens") or os.environ.get("EXAONE_MAX_NEW_TOKENS", "350"))
        max_new_tokens = max(120, min(max_new_tokens, 700))

        image_path = self.save_upload(payload)
        ocr = self.run_ocr(image_path)

        ocr_query = self.vectorizer.transform([ocr["text"]])
        ocr_scores = cosine_similarity(ocr_query, self.document_matrix).ravel().astype(np.float32)
        ocr_scores_norm = minmax(ocr_scores)

        image_vector = self._get_embedder().embed_paths([image_path], batch_size=1)[0]
        image_scores_by_item = {item_code: 0.0 for item_code in self.item_codes}
        image_hits = self.gallery_store.search_by_vector(image_vector, top_k=len(self.gallery_meta))
        for hit in image_hits:
            image_scores_by_item[hit.item_code] = max(image_scores_by_item.get(hit.item_code, 0.0), hit.score)
        image_scores = np.asarray([image_scores_by_item[item_code] for item_code in self.item_codes])
        image_scores_norm = minmax(image_scores)

        alpha = dynamic_alpha(float(ocr["confidence"])) if use_dynamic else float(requested_alpha)
        alpha = max(0.0, min(1.0, alpha))
        final_scores = alpha * ocr_scores_norm + (1.0 - alpha) * image_scores_norm
        ranked_indexes = final_scores.argsort()[::-1][:top_k]

        candidates: list[dict[str, Any]] = []
        for rank, index in enumerate(ranked_indexes, start=1):
            item_code = self.item_codes[index]
            product = self.products_by_item[item_code]
            rules = self.rules_by_item.get(item_code, [])
            candidates.append(
                {
                    "rank": rank,
                    "item_seq": item_code,
                    "product_name": product.get("aihub_product_name") or product.get("matched_product_name"),
                    "matched_product_name": product.get("matched_product_name", ""),
                    "manufacturer": product.get("manufacturer", ""),
                    "classification": product.get("classification", ""),
                    "ingredients": product.get("ingredient_names", ""),
            "dur_positive": product.get("dur_positive", ""),
            "dur_rule_types": product.get("dur_rule_types", ""),
            "detail_url": product.get("detail_url", ""),
                    "ocr_score": float(ocr_scores_norm[index]),
                    "image_score": float(image_scores_norm[index]),
                    "final_score": float(final_scores[index]),
                    "rules": self.compact_rules_for_item(item_code, rules),
                }
            )

        safety_card, llm_error = self.generate_safety_card(
            candidates,
            ocr,
            alpha,
            use_dynamic,
            max_new_tokens=max_new_tokens,
        )
        if not safety_card:
            safety_card = self.fallback_safety_card(candidates, alpha, use_dynamic)

        return {
            "alpha": alpha,
            "mode": "dynamic" if use_dynamic else "fixed",
            "ocr": ocr,
            "candidates": candidates,
            "safety_card": safety_card,
            "llm_error": llm_error,
            "max_new_tokens": max_new_tokens,
            "status": self.status(),
        }

    def generate_safety_card(
        self,
        candidates: list[dict[str, Any]],
        ocr: dict[str, Any],
        alpha: float,
        dynamic: bool,
        max_new_tokens: int,
    ) -> tuple[str, str]:
        top = candidates[0] if candidates else {}
        evidence = {
            "hybrid_alpha": alpha,
            "mode": "dynamic" if dynamic else "fixed",
            "ocr_confidence": round(float(ocr["confidence"]), 4),
            "ocr_text_excerpt": str(ocr["text"])[:600],
            "top_candidate": top,
            "other_candidates": candidates[1:3],
        }
        source_card = self.fallback_safety_card(candidates, alpha, dynamic)
        prompt = f"""
아래 원본 카드 초안을 더 읽기 쉽게 다듬으세요. 정보 추가는 금지입니다.

엄격한 규칙:
- 품목명, 제조사, 성분명은 JSON 문자열을 그대로 복사하세요.
- JSON과 원본 카드 초안에 없는 병용금기 상대 약물, 용량 제한, 임신 주수, 부작용, 음식 상호작용을 만들지 마세요.
- rules 배열이 비어 있으면 DUR 경고는 '연결된 DUR 규칙 없음'이라고 쓰세요.
- 음식 상호작용 데이터는 현재 JSON에 없으므로 항상 '프로토타입 데이터에 없음'이라고 쓰세요.
- 사용자가 이해하기 쉬운 말로 풀어 쓰되, 경고의 의학적 원인이나 부작용명은 새로 만들지 마세요.
- 목록만 나열하지 말고 각 섹션마다 한두 문장 설명을 붙이세요.
- DUR 경고에서는 성분명만 쓰지 말고, JSON rules의 example_products에 있는 구체 품목명을 함께 보여주세요.
- example_products가 비어 있으면 '데이터셋 내 예시 품목 없음, 성분명으로 약 봉투/처방전 확인 필요'라고 쓰세요.
- 음식 상호작용은 food rule 데이터가 없으므로 음식 이름을 새로 만들지 마세요.
- 첫 문장에 'JSON 데이터를 기반으로' 같은 메타 설명을 쓰지 마세요.
- Markdown 제목은 쓰지 말고, 원본 카드 초안의 5개 섹션 구조를 유지하세요.

원본 카드 초안:
{source_card}

출력:
1. 인식 결과: 어떤 약으로 인식됐는지, 제조사, final_score를 쉬운 문장으로 설명
2. 성분: JSON ingredients를 그대로 쓰고, 같은 성분의 다른 약과 중복될 수 있음을 일반적으로 안내
3. DUR 경고: rules의 dur_type, ingredient, warning, note, example_products를 경고별로 나누어 의미와 확인할 점 설명
4. 음식 상호작용: 프로토타입 데이터에 없음
5. 근거와 한계: OCR confidence, alpha, 사진 기반 추정, 전문가 확인 권고

검색 근거 JSON:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
""".strip()
        answer, llm_error = self.llm.generate(prompt, max_new_tokens=max_new_tokens)
        evidence_text = json.dumps(evidence, ensure_ascii=False) + source_card
        if answer and looks_ungrounded(answer, evidence_text):
            return source_card, "LLM output failed grounding check; used structured fallback."
        if answer and missing_required_card_sections(answer):
            return source_card, "LLM output missed required card sections; used structured fallback."
        return answer, llm_error

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message is required.")
        context = payload.get("context") or {}
        history = payload.get("history") or []
        max_new_tokens = int(payload.get("max_new_tokens") or os.environ.get("EXAONE_CHAT_MAX_NEW_TOKENS", "260"))
        max_new_tokens = max(80, min(max_new_tokens, 600))

        compact_history = []
        for turn in history[-6:]:
            role = str(turn.get("role", ""))[:20]
            content = str(turn.get("content", ""))[:500]
            if role and content:
                compact_history.append({"role": role, "content": content})

        grounded_summary = grounded_context_summary(context)
        lowered = message.lower()
        if any(keyword in lowered for keyword in ["dur", "경고", "금기", "성분", "음식", "상호작용", "후보", "품목"]):
            answer = grounded_summary
            if "음식" not in lowered and "상호작용" not in lowered:
                answer += "\n실제 복용 여부는 개인 상태와 병용약에 따라 달라질 수 있으므로 약사 또는 의사에게 확인하세요."
            return {
                "answer": answer,
                "llm_error": "",
                "max_new_tokens": max_new_tokens,
                "guardrail": "structured_context",
            }

        prompt = f"""
사용자가 방금 검색한 의약품 후보와 DUR 근거에 대해 후속 질문을 했습니다.
아래 근거 요약, JSON context, 대화 기록만 사용해서 답하세요.

규칙:
- context에 없는 사실은 '현재 프로토타입 근거에는 없음'이라고 답하세요.
- 약 복용 여부를 단정하지 말고, 의사/약사 확인을 권하세요.
- 음식 상호작용은 context에 별도 food rule이 없으면 없다고 말하지 말고 '프로토타입 데이터에 없음'이라고 하세요.
- 답변은 5문장 이내로 간결하게 작성하세요.

근거 요약:
{grounded_summary}

context JSON:
{json.dumps(context, ensure_ascii=False, indent=2)[:5000]}

최근 대화:
{json.dumps(compact_history, ensure_ascii=False, indent=2)}

사용자 질문:
{message}
""".strip()
        answer, llm_error = self.llm.generate(prompt, max_new_tokens=max_new_tokens)
        if not answer:
            answer = "현재 프로토타입 근거만으로는 답변을 생성하지 못했습니다. 검색 후보와 DUR 근거를 확인한 뒤 약사 또는 의사에게 상담하세요."
        return {
            "answer": answer,
            "llm_error": llm_error,
            "max_new_tokens": max_new_tokens,
        }

    def fallback_safety_card(self, candidates: list[dict[str, Any]], alpha: float, dynamic: bool) -> str:
        if not candidates:
            return "검색 후보를 찾지 못했습니다."
        top = candidates[0]
        rules = top.get("rules") or []
        rule_lines = "\n\n".join(explain_rule(rule) for rule in rules) or (
            "- 연결된 DUR 규칙 없음\n"
            "  의미: 이 프로토타입의 DUR 테이블에서는 해당 후보에 연결된 경고를 찾지 못했습니다.\n"
            "  확인할 점: 실제 복용 전에는 처방전, 약 봉투, 기존 복용약을 함께 확인하세요."
        )
        final_score = top.get("final_score")
        final_score_text = f"{float(final_score):.3f}" if isinstance(final_score, (int, float)) else "정보 없음"
        mode_text = "OCR confidence에 따라 자동 조정" if dynamic else f"OCR {alpha:.2f}, 이미지 {1.0 - alpha:.2f}"
        return f"""1. 인식 결과
- 사진 속 약은 '{top.get("product_name")}'일 가능성이 가장 높게 검색되었습니다.
- 제조사는 {top.get("manufacturer") or "정보 없음"}로 확인됩니다.
- Hybrid 점수는 {final_score_text}이며, 이번 검색은 {mode_text} 방식으로 OCR 텍스트와 이미지 유사도를 함께 반영했습니다.

2. 성분
- 검색된 성분 정보: {top.get("ingredients") or "성분 정보 없음"}
- 같은 성분이 들어간 다른 약을 함께 복용하면 중복 복용이 될 수 있으므로, 감기약이나 진통제를 같이 먹고 있다면 성분명을 확인하세요.

3. DUR 경고
{rule_lines}

4. 음식 상호작용
- 프로토타입 데이터에 없음.
- 이 화면은 현재 DUR 규칙 중심의 데모이므로, 음식이나 음료와의 상호작용은 별도 근거가 연결되어 있을 때만 안내할 수 있습니다.

5. 근거와 한계
- OCR, CLIP 이미지 유사도, DUR 규칙 테이블을 결합한 프로토타입 결과입니다.
- 포장 사진 기반 검색이므로 촬영 각도, 흐림, 비슷한 포장 디자인에 따라 후보가 바뀔 수 있습니다.
- 실제 복용 전에는 약사 또는 의사에게 현재 복용 중인 약과 건강 상태를 함께 알려주고 확인하세요."""


class PrototypeHandler(BaseHTTPRequestHandler):
    engine: PrototypeEngine
    html_path: Path

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = self.html_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/demo", "/docs/medicine_demo.html"}:
            self._send_html()
            return
        if self.path == "/api/status":
            self._send_json(self.engine.status())
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/retrieve", "/api/chat"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 24 * 1024 * 1024:
                self._send_json({"error": "Request is too large."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/api/chat":
                self._send_json(self.engine.chat(payload))
            else:
                self._send_json(self.engine.retrieve(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except error.URLError as exc:
            self._send_json({"error": repr(exc)}, HTTPStatus.BAD_GATEWAY)
        except Exception as exc:  # pragma: no cover - local prototype diagnostics
            self._send_json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[prototype] {self.address_string()} - {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the medicine Hybrid GraphRAG HTML prototype.")
    parser.add_argument("--host", default=os.environ.get("MEDICINE_DEMO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MEDICINE_DEMO_PORT", "8008")))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MEDICINE_DATA_DIR", str(DEFAULT_DATA_DIR))))
    parser.add_argument("--clip-model", default=os.environ.get("MEDICINE_CLIP_MODEL", "openai/clip-vit-base-patch32"))
    parser.add_argument("--ocr-gpu", action="store_true", default=os.environ.get("MEDICINE_OCR_GPU", "0") == "1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    upload_dir = project_root / "outputs" / "prototype_uploads"
    html_path = project_root / "docs" / "medicine_demo.html"
    if not html_path.exists():
        raise SystemExit(f"Demo HTML not found: {html_path}")

    PrototypeHandler.engine = PrototypeEngine(args.data_dir, upload_dir, args.clip_model, args.ocr_gpu)
    PrototypeHandler.html_path = html_path
    server = ThreadingHTTPServer((args.host, args.port), PrototypeHandler)
    print(f"Medicine demo server: http://{args.host}:{args.port}")
    print(json.dumps(PrototypeHandler.engine.status(), ensure_ascii=False, indent=2))
    server.serve_forever()


if __name__ == "__main__":
    main()
