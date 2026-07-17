# Day 14 Inspector Hotfix V4

## 확인된 원인

Day 13 API Core Context 복구 후 나타난 5개 경고는 복구 Script 내부의
문자 깨짐 탐지 토큰 5개를 Inspector가 다시 탐지한 자기 참조 오탐이다.

남은 진단 TXT 1개는 복구 전 손상 원본을 보존한 다음 경로의 Backup이다.

```text
reports/artifacts/backups/
```

활성 `day13_api_core_context.txt`가 다시 깨진 것이 아니다.

## 변경 내용

1. 다음 두 자기 참조 Script를 사용자 노출 문구 검사에서 제외
   - `scripts/inspect_day14_final_integration_prerequisites.py`
   - `scripts/rebuild_day13_api_core_context.py`

2. `reports/artifacts/backups/**/*.txt`를 활성 진단 Artifact 검사에서 제외

3. 제외한 Backup 목록과 개수를 JSON에 별도로 기록

4. README·보고서·일반 실행 소스의 실제 문자 깨짐과 과장 표현 검사는 유지

5. Schema Version을 4로 변경

## 예상 결과

```text
Overall status               : PASS
Text-quality findings        : 0
Errors                       : 0
Warnings                     : 0
Information                  : 1
```

남는 Information 1개는 Day 1~3 README Marker 부재 정보다.
