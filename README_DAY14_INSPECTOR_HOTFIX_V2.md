# Day 14 Inspector Hotfix V2

## 변경 내용

1. FastAPI Endpoint 탐지를 정규식에서 Python AST 방식으로 변경
   - `app`, `router` 이외의 decorator 객체 이름 지원
   - 같은 파일 내부 문자열 상수, 문자열 결합, f-string 경로 지원

2. README Day 1~3 Marker 부재를 경고가 아닌 정보로 분리
   - Marker 부재만으로 본문 누락을 의미하지 않음
   - START/END 중복 또는 불일치는 계속 오류로 처리

3. 문자 깨짐 점검 범위 분리
   - README, Markdown 보고서, 실행 소스는 경고 대상
   - `reports/artifacts/*.txt` 원시 Context Dump는 정보 항목으로 분리
   - 생성된 Day 14 JSON이 자기 자신을 다시 탐지하는 문제 방지

4. Inspector Schema Version을 2로 변경

## 포함 파일

- `scripts/inspect_day14_final_integration_prerequisites.py`
- `tests/test_inspect_day14_final_integration_prerequisites.py`

## 검증 결과

```text
15 passed
```

이 수치는 Hotfix Inspector 단위 테스트 결과이며 프로젝트 전체 회귀 테스트 수가 아니다.
