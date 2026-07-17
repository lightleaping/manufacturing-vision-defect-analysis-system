# Day 14 Final Documentation Hotfix V2

## 원인

Day 4 Classification 평가 Artifact는 실제 계산 정밀도 또는 백분율
단위 값을 저장한다. 최종 README는 사람이 읽기 쉽도록 소수 둘째 자리
백분율로 표시하므로 다음 값은 같은 결과다.

```text
Artifact ratio      : 0.9734265734
Artifact percentage : 97.34265734
README              : 97.34%
```

기존 생성기는 README용 반올림 비율 `0.9734`를 Artifact 안에서
`5e-6` 오차로 직접 찾았기 때문에 정상 근거를 누락으로 판단했다.

## 변경 내용

- Ratio와 Percentage 표현을 자동 정규화
- README 반올림을 고려한 절대 허용 오차 `5e-5`
- Integer Count는 단위 변환 없이 검증
- Percentage Artifact 회귀 테스트 추가
- Day 14 Summary Schema Version 2

## 안전성

이 오류는 문서 쓰기 전에 발생했으므로 README·보고서·Summary는
변경되지 않았다.
