# Day 14 Final Documentation Bundle

## 목적

Day 4·12·13·14의 실제 JSON Artifact와 Checkpoint 경로를 검증한 뒤 다음을 생성한다.

- README Day 14 최종 통합 Section
- Architecture·User Flow
- 성능·Failure Analysis
- 실행 방법
- Portfolio Summary
- Interview Guide
- Limitations·Future Improvements
- Day 14 Summary JSON

## 포함 파일

- `scripts/create_day14_docs.py`
- `tests/test_create_day14_docs.py`

## 생성 파일

- `reports/day14_final_integration_portfolio_interview_summary.md`
- `reports/artifacts/day14_final_integration_summary.json`
- `README.md`의 `DAY14_FINAL_INTEGRATION` Marker Block
- `reports/artifacts/backups/README.md.before_day14_final_docs`

## 실행 순서

1. 대상 테스트
2. 전체 회귀 테스트
3. 실제 테스트 수·Warning·Runtime으로 문서 생성
4. Day 14 점검 및 전체 회귀 테스트 재실행

## 문서 생성 명령 예시

실제 출력값으로 숫자를 교체한다.

```powershell
.\.venv\Scripts\python.exe `
    -m scripts.create_day14_docs `
    --project-root . `
    --targeted-test-count 15 `
    --regression-test-count 1700 `
    --warning-count 1 `
    --runtime-seconds 100.00
```
