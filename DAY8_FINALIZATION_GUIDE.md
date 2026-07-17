# Day 8 Finalization Bundle

추가 파일:

```text
scripts/finalize_day8_visual_validation.py
scripts/create_day8_docs.py
tests/test_finalize_day8_visual_validation.py
tests/test_create_day8_docs.py
```

순서:

1. 두 Screenshot을 지정 경로에 저장한다.
2. 육안 검증 확정 Script를 실행한다.
3. 새 테스트를 실행한다.
4. 전체 회귀 테스트를 실행해 최종 passed·warning 수를 확정한다.
5. 확정된 수치로 `scripts.create_day8_docs`를 실행한다.
