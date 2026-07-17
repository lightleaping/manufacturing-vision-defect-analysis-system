# Day 14 Final Integration Evidence Bundle

## 목적

README를 직접 수정하기 전에 현재 저장소에서 다음 근거를 자동 수집한다.

- README Heading과 Day Marker
- FastAPI Endpoint
- 주요 Checkpoint·Dataset·Dashboard 경로
- Report·Artifact·Figure Inventory
- Day 4·12·13 JSON Artifact 후보와 실제 Metric Key
- 실행 Command 정적 경로
- Architecture Diagram
- End-to-End User Flow
- README 최종 구조와 수정 전 Gate

## 포함 파일

- `scripts/collect_day14_final_integration_evidence.py`
- `tests/test_collect_day14_final_integration_evidence.py`

## 생성 파일

- `reports/artifacts/day14_final_integration_evidence.json`
- `reports/day14_final_integration_readme_architecture_plan.md`

README와 기존 Application Source는 수정하지 않는다.

## 테스트

```powershell
.\.venv\Scripts\python.exe `
    -m pytest `
    .\tests\test_collect_day14_final_integration_evidence.py `
    -q
```

## 실행

```powershell
.\.venv\Scripts\python.exe `
    -m scripts.collect_day14_final_integration_evidence `
    --project-root .
```
