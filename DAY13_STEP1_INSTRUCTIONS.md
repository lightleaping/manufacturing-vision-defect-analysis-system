# Day 13 Step 1 — 저장소 구조·체크포인트 점검

이 묶음은 기존 FastAPI·Classification·Streamlit·OpenCV 구조를 추측하지 않고
Day 13 Detection API 구현 위치와 호출 관계를 확인하기 위한 안전한 선행 단계다.

## 포함 파일

- `scripts/inspect_day13_integration_prerequisites.py`
  - FastAPI 앱·Router·Endpoint
  - Pydantic Schema
  - Model Service·Checkpoint 관련 코드
  - 이미지 검증·Exception Handler
  - FastAPI 테스트
  - Streamlit 페이지·API Client·Session State·UI Helper
  - OpenCV 관련 파일
  - Dependency 버전
  - Detection Best Checkpoint 존재·크기·SHA-256·메타데이터
  - README Day 12 Marker
  - pytest cache 상태
  - JSON Artifact 생성

- `scripts/export_day13_repository_context.py`
  - 실제 구현에 필요한 텍스트 소스만 ZIP으로 묶는다.
  - 모델 바이너리와 Dataset은 포함하지 않는다.

- `tests/test_inspect_day13_integration_prerequisites.py`
  - Inspector의 Endpoint·Schema·README·pytest cache 탐지 로직을 검증한다.

## 적용

ZIP을 프로젝트 루트에 풀면 기존 파일을 수정하지 않고 위 세 파일만 추가한다.

## 실행 순서

```powershell
Set-Location `
    C:\Users\kflow\Downloads\manufacturing-vision-defect-analysis-system

.\.venv\Scripts\python.exe `
    -m pytest `
    .\tests\test_inspect_day13_integration_prerequisites.py `
    -q

.\.venv\Scripts\python.exe `
    -m scripts.inspect_day13_integration_prerequisites `
    --project-root .

.\.venv\Scripts\python.exe `
    -m scripts.export_day13_repository_context `
    --project-root .
```

`--run-collect-only`를 추가하면 테스트 실행 없이 현재 수집 가능한 테스트 목록만 확인한다.

```powershell
.\.venv\Scripts\python.exe `
    -m scripts.inspect_day13_integration_prerequisites `
    --project-root . `
    --run-collect-only
```

## 생성 결과

```text
reports/artifacts/day13_integration_prerequisites.json
reports/artifacts/day13_repository_context.zip
```

다음 단계에서는 Console 출력과 위 두 Artifact를 기준으로 기존 Naming·Router Prefix·Schema
스타일에 맞춘 Detection Schema·Checkpoint Loader·Inference Service·Endpoint·테스트 전체 코드를
구현한다.
