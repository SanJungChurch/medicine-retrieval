# Medicine Package OCR-RAG / Image-KG Retrieval

의약품 포장 이미지 기반 복약 안전 정보 검색에서 OCR-RAG, Image-KG Retrieval, Hybrid GraphRAG를 비교하는 실험형 AI 프로젝트입니다.

## 프로젝트 목표

약 포장 이미지를 입력했을 때 다음 정보를 근거 기반으로 검색합니다.

```text
품목명
성분
DUR 경고
음식 상호작용
복약 안전 안내
근거/출처/버전
```

본 구현은 전체 모바일 앱보다 핵심 검색 문제에 집중합니다.

```text
OCR-RAG
Image-KG Retrieval
Hybrid GraphRAG
```

## 최종 실험 요약

실험셋:

```text
AI Hub 의약품 패키징 OCR validation
1000품목
gallery 1000장
query 1000장
DUR 양성 700품목
DUR 음성 300품목
```

주요 결과:

```text
OCR-RAG / EasyOCR
Top-1:    0.535
Recall@5: 0.714
MRR:      0.6121

Image-KG / CLIP
Top-1:    0.189
Recall@5: 0.222
MRR:      0.2034

Hybrid alpha=0.8
Top-1:    0.527
Recall@5: 0.725
MRR:      0.6118
```

해석:

```text
OCR-RAG가 가장 안정적인 품목 식별 baseline이었습니다.
CLIP 이미지 단독 검색은 1000품목 규모에서 유사 포장 디자인 때문에 성능이 낮았습니다.
Hybrid는 Top-1에서는 OCR-RAG를 넘지 못했지만 Recall@5를 일부 개선했습니다.
```

## 저장소 구조

```text
medicine/
  data/processed/                 # 작은 mock/sample CSV
  docs/
    medicine_project_plan.html    # 프로젝트 기획서
    medicine_demo.html            # 시연용 HTML UI
  scripts/
    sample_audited_subset.py      # 1000품목 샘플링
    run_prototype_server.cmd      # demo 실행
  src/medicine_retrieval/
    build_image_index.py
    evaluate_image_index.py
    run_easyocr_queries.py
    run_ocr_text_experiment.py
    run_hybrid_real_experiment.py
    evaluate_dur_rule_retrieval.py
    prototype_server.py
  requirements.txt
  requirements-experiment.txt
  requirements-demo.txt
  DATASET.md
  SUBMISSION.md
```

## 환경

```text
Windows
conda env: medicine
Python 3.11
```

전체 설치:

```cmd
cd /d C:\VSProject\medicine
conda activate medicine
pip install -r requirements.txt
```

실험만 재현할 경우:

```cmd
pip install -r requirements-experiment.txt
```

데모만 실행할 경우:

```cmd
pip install -r requirements-demo.txt
```

## 데모 실행

로컬 EXAONE 모델을 사용합니다.

```cmd
cd /d C:\VSProject\medicine
scripts\run_prototype_server.cmd
```

브라우저:

```text
http://127.0.0.1:8008/
```

기본값:

```text
MEDICINE_DATA_DIR=D:\medicine_data\validation_audited_1000
EXAONE_PROVIDER=local
EXAONE_MODEL=LGAI-EXAONE/EXAONE-4.0-1.2B
EXAONE_DEVICE=cpu
```

## 데이터셋

자세한 데이터셋 경로와 재생성 명령은 [DATASET.md](DATASET.md)를 참고하세요.

제출/압축 대상과 제외 대상은 [SUBMISSION.md](SUBMISSION.md)를 참고하세요.

## 주의

이 프로젝트는 연구 및 교육용 프로토타입입니다. 실제 복약 판단은 의사, 약사, 공식 의약품 정보와 함께 확인해야 합니다.
