# Day 14 Final Integration Prerequisites Inspector

## 포함 파일

- `scripts/inspect_day14_final_integration_prerequisites.py`
- `tests/test_inspect_day14_final_integration_prerequisites.py`

## 목적

README를 바로 수정하기 전에 실제 저장소의 구조, Marker, Report, Artifact,
Figure, Endpoint, Dashboard 경계, 실행 경로, 문서 표현을 자동 점검한다.

이 단계에서는 기존 애플리케이션 코드와 README를 변경하지 않는다.

## 권장 실행 순서

```powershell
Set-Location `
    C:\Users\kflow\Downloads\manufacturing-vision-defect-analysis-system

.\.venv\Scripts\python.exe `
    -m pytest `
    .\tests\test_inspect_day14_final_integration_prerequisites.py `
    -q

.\.venv\Scripts\python.exe `
    -m scripts.inspect_day14_final_integration_prerequisites `
    --project-root .
```

생성 Artifact:

```text
reports/artifacts/day14_final_integration_prerequisites_inspection.json
```

## 주의

- `WARN`은 즉시 수정이 필요한 오류가 아니라 최종 README 작성 전에 사람이 확인할 항목이다.
- Endpoint 수집은 정적 decorator 검색이다.
- 실행 Command 검증은 필요한 파일의 존재 여부만 확인한다.
- 실제 FastAPI, Streamlit, Checkpoint 추론은 다음 단계에서 별도로 검증한다.
- Day 13 browser check는 실제 수동 확인 전까지 `not_recorded`를 유지한다.
