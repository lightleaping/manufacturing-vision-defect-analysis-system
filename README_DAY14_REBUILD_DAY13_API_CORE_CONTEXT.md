# Day 14 — Day 13 API Core Context UTF-8 Rebuild

## 판단

현재 저장소의 `scripts`, `src`, `tests`에서는
`day13_api_core_context.txt`를 생성하는 기존 Script가 확인되지 않았다.
따라서 손상된 Artifact를 재인코딩하지 않고, 파일 내부 `FILE:` Header에
기록된 원본 경로 목록을 이용해 현재 정상 Source에서 다시 생성한다.

## 포함 파일

- `scripts/rebuild_day13_api_core_context.py`
- `tests/test_rebuild_day13_api_core_context.py`

## 안전 정책

- 기존 Application Source는 수정하지 않는다.
- 먼저 `--dry-run`으로 모든 Header와 Source를 검증한다.
- 손상 Artifact 원본 Byte를 SHA-256 이름의 Backup으로 보존한다.
- Backup 생성 후에만 Artifact를 원자적으로 교체한다.
- UTF-8로 읽히지 않는 Source, 누락 Source, 프로젝트 밖 경로가 있으면 중단한다.
- 정상 Source에서 문자 깨짐 토큰이 발견되면 중단한다.
- 재생성 결과와 Source 목록·Hash를 JSON Artifact로 기록한다.

## 생성 파일

```text
reports/artifacts/day13_api_core_context.txt
reports/artifacts/backups/day13_api_core_context.before_utf8_rebuild.<hash>.txt
reports/artifacts/day14_day13_api_core_context_rebuild.json
```

## 테스트

```powershell
.\.venv\Scripts\python.exe `
    -m pytest `
    .\tests\test_rebuild_day13_api_core_context.py `
    -q
```

## Dry Run

```powershell
.\.venv\Scripts\python.exe `
    -m scripts.rebuild_day13_api_core_context `
    --project-root . `
    --dry-run
```

## 실제 재생성

```powershell
.\.venv\Scripts\python.exe `
    -m scripts.rebuild_day13_api_core_context `
    --project-root .
```
