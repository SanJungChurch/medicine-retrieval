# Overleaf 사용 방법

이 폴더는 ACM 더블컬럼 conference 형식의 LaTeX 초안입니다.

## 파일

```text
acm_medicine_retrieval.tex  # main LaTeX file
references.bib             # BibTeX references
```

## Overleaf 업로드

1. Overleaf에서 새 프로젝트를 생성합니다.
2. `paper/` 폴더 안의 파일 두 개를 업로드합니다.
3. `acm_medicine_retrieval.tex`를 main file로 지정합니다.
4. Compiler는 `pdfLaTeX`로 둡니다.

## ACM 양식

현재 문서는 다음 클래스를 사용합니다.

```latex
\documentclass[sigconf,screen,nonacm]{acmart}
```

제출용 ACM 저작권/메타데이터가 필요하면 학회 지침에 맞게 `nonacm` 옵션과 `\settopmatter` 설정을 수정하면 됩니다.

## 수정하면 좋은 부분

- `Author Name`, `Institution Name`, `email@example.com`
- 실제 학회명 또는 과목명
- 관련 연구 인용 추가
- 음식 상호작용 평가가 추가되면 실험 결과 표 확장
- 그림을 실제 시스템 아키텍처 이미지로 교체
