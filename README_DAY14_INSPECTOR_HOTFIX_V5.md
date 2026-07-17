# Day 14 Inspector Hotfix V5

## 원인

Day 14 최종 문서 생성기에는 README와 보고서에 들어가면 안 되는 표현을
차단하기 위한 `FORBIDDEN_OVERCLAIMS` 상수 6개가 있다.

Inspector는 `scripts/create_day14_docs.py`를 일반 실행 Source로 읽으면서
그 금지 표현 상수 자체를 실제 문서 문제로 다시 탐지했다.

## 변경 내용

사용자 노출 문구 검사에서 다음 자기 참조 파일을 제외한다.

- `scripts/inspect_day14_final_integration_prerequisites.py`
- `scripts/rebuild_day13_api_core_context.py`
- `scripts/create_day14_docs.py`

README, Markdown 보고서, 일반 `src`·`scripts` 파일의 실제 과장 표현 탐지는
그대로 유지한다.

Schema Version은 5다.

## 예상 최종 상태

```text
Overall status               : PASS
Text-quality findings        : 0
Errors                       : 0
Warnings                     : 0
Information                  : 1
```
