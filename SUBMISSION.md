# 제출용 구성 안내

이 저장소는 의약품 포장 이미지 기반 복약 안전 정보 검색에서 OCR-RAG, Image-KG Retrieval, Hybrid GraphRAG를 비교한 실험 코드와 시연용 프로토타입을 포함합니다.

## 제출에 포함할 것

```text
medicine/
  README.md
  SUBMISSION.md
  DATASET.md
  requirements.txt
  requirements-experiment.txt
  requirements-demo.txt
  docs/
    medicine_project_plan.html
    medicine_demo.html
  scripts/
    audit_dur_coverage.py
    build_multigallery_subset.py
    sample_audited_subset.py
    run_prototype_server.cmd
  src/
    medicine_retrieval/
```

## 제출에서 제외할 것

대용량 원천 데이터, 임베딩, OCR 결과, 업로드 이미지, 모델 캐시는 git이나 zip에 넣지 않습니다.

```text
D:\056.의약품, 화장품 패키징 OCR 데이터\
D:\medicine_data\
outputs/prototype_uploads/
outputs/*.log
__pycache__/
*.faiss
*.npy
```

## 핵심 실험 코드

```text
scripts/sample_audited_subset.py
src/medicine_retrieval/build_image_index.py
src/medicine_retrieval/evaluate_image_index.py
src/medicine_retrieval/run_easyocr_queries.py
src/medicine_retrieval/merge_ocr_chunks.py
src/medicine_retrieval/run_ocr_text_experiment.py
src/medicine_retrieval/run_hybrid_real_experiment.py
src/medicine_retrieval/evaluate_dur_rule_retrieval.py
```

## 최종 실험 데이터 위치

```text
D:\medicine_data\validation_audited_1000
```

주요 파일:

```text
products.csv
manifest.csv
item_dur_rules.csv
images/
annotations/
embeddings/
ocr/
reports/
```

## 주요 결과

```text
Image-KG / CLIP
Top-1:    0.189
Recall@5: 0.222
MRR:      0.2034

OCR-RAG / EasyOCR
Top-1:    0.535
Recall@5: 0.714
MRR:      0.6121

Hybrid alpha=0.8
Top-1:    0.527
Recall@5: 0.725
MRR:      0.6118
```

## 데모 실행

```cmd
cd /d C:\VSProject\medicine
scripts\run_prototype_server.cmd
```

브라우저:

```text
http://127.0.0.1:8008/
```

## 압축 제출 예시

Windows cmd 기준:

```cmd
cd /d C:\VSProject
tar -a -c -f medicine_submission.zip ^
  medicine\README.md ^
  medicine\SUBMISSION.md ^
  medicine\DATASET.md ^
  medicine\requirements.txt ^
  medicine\requirements-experiment.txt ^
  medicine\requirements-demo.txt ^
  medicine\docs ^
  medicine\scripts ^
  medicine\src ^
  medicine\data\processed
```

주의: 위 명령은 `outputs/`, `D:\medicine_data`, AI Hub 원천 압축 파일을 포함하지 않습니다.
