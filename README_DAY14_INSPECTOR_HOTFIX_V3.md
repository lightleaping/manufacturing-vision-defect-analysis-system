# Day 14 Inspector Hotfix V3

## 원인

V2의 문구 품질 점검이 `scripts` 디렉터리를 검사하면서,
Inspector 내부에 탐지 패턴으로 선언된 금지 문구와 문자 깨짐 토큰까지
실제 문서 문제로 다시 탐지했다.

## 변경 내용

1. Inspector 자기 파일을 사용자 노출 문구 검사에서 제외
2. 제외 파일 경로를 JSON에 명시
3. Schema Version을 3으로 변경
4. README에 실제 과장 표현이 있으면 계속 WARN이 발생하는지 검증
5. 자기 참조 패턴만 제외되는지 검증

## 검증 결과

```text
17 passed
```

이는 Inspector 대상 단위 테스트 결과이며 프로젝트 전체 회귀 테스트 수가 아니다.

## 예상 실제 점검 상태

```text
Overall status               : PASS
Text-quality findings        : 0
Errors                       : 0
Warnings                     : 0
Information                  : 2
```

Day 1~3 Marker 부재와 원시 진단 TXT 문자 깨짐은 INFO로 유지된다.
