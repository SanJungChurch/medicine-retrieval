# 데이터셋 정리

## 원천 데이터

AI Hub 의약품 패키징 OCR validation 데이터를 사용했습니다.

```text
라벨 zip:
D:\056.의약품, 화장품 패키징 OCR 데이터\01.데이터\2. Validation\라벨링데이터\VL1.zip

이미지 zip:
D:\056.의약품, 화장품 패키징 OCR 데이터\01.데이터\2. Validation\원천데이터\VS1.zip
```

원천 데이터는 용량이 크고 배포 권한이 AI Hub 정책에 따르므로 저장소에 포함하지 않습니다.

## 최종 실험 샘플

```text
D:\medicine_data\validation_audited_1000
```

구성:

```text
총 품목: 1000
DUR 양성 품목: 700
DUR 음성 품목: 300
이미지: 2000장
gallery: 1000장
query: 1000장
DUR rule rows: 2632
```

선별 기준:

```text
- AI Hub validation 의약품 패키징 이미지
- 품목당 이미지 2장 이상
- MFDS 품목 정확 매칭
- 정상 품목
- 취소/취하 품목 제외
- item_seq 고유
```

## 주요 산출물

```text
products.csv       # 1000개 품목 메타데이터, MFDS 매칭 결과, DUR 양성 여부
manifest.csv       # gallery/query 이미지 경로, 품목 코드, OCR ground truth
item_dur_rules.csv # 품목별 DUR rule 연결
embeddings/        # CLIP embeddings 및 FAISS index
ocr/               # EasyOCR query OCR 결과
reports/           # OCR-RAG, Image-KG, Hybrid, DUR 평가 결과
```

## 재생성 명령

1000품목 샘플링:

```cmd
cd /d C:\VSProject\medicine
set PYTHONPATH=src
C:\Anaconda3\envs\medicine\python.exe scripts\sample_audited_subset.py --labels-zip "D:\056.의약품, 화장품 패키징 OCR 데이터\01.데이터\2. Validation\라벨링데이터\VL1.zip" --images-zip "D:\056.의약품, 화장품 패키징 OCR 데이터\01.데이터\2. Validation\원천데이터\VS1.zip" --output-dir "D:\medicine_data\validation_audited_1000" --positive-products 700 --negative-products 300 --seed 20260619 --delay 0.15 --extract-images
```

CLIP 임베딩 생성:

```cmd
cd /d C:\VSProject\medicine
set PYTHONPATH=src
C:\Anaconda3\envs\medicine\python.exe -m medicine_retrieval.build_image_index --manifest-csv "D:\medicine_data\validation_audited_1000\manifest.csv" --role gallery --model-name openai/clip-vit-base-patch32 --batch-size 16 --output-dir "D:\medicine_data\validation_audited_1000\embeddings" --output-prefix clip_gallery
C:\Anaconda3\envs\medicine\python.exe -m medicine_retrieval.build_image_index --manifest-csv "D:\medicine_data\validation_audited_1000\manifest.csv" --role query --model-name openai/clip-vit-base-patch32 --batch-size 16 --output-dir "D:\medicine_data\validation_audited_1000\embeddings" --output-prefix clip_query
```

OCR 결과는 노트북 안정성을 위해 10장 단위 chunk로 실행했습니다.
