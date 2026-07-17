# Day 8 — Streamlit Image Inference Dashboard: First Implementation Bundle

## 적용 범위

이 Bundle은 다음 첫 구현 묶음을 포함합니다.

- `src/dashboard/__init__.py`
- `src/dashboard/config.py`
- `src/dashboard/api_client.py`
- `src/dashboard/session_state.py`
- `src/dashboard/styles.py`
- `src/dashboard/ui_helpers.py`
- `src/dashboard/app.py`
- `scripts/inspect_day8_dashboard_prerequisites.py`
- `scripts/update_day8_requirements.py`
- `scripts/run_day8_dashboard_validation.py`
- Dashboard 단위·AppTest·통합 검증 테스트

Day 8 보고서, README Marker, 실제 브라우저 Screenshot은 실제 UI 육안 검증 후 다음 단계에서 생성합니다.

## 설계

```text
Browser
→ Streamlit app.py
→ DashboardApiClient
→ GET /api/v1/health
→ POST /api/v1/predictions
→ Day 7 FastAPI
→ ResNet18Transfer
→ JSON
→ Streamlit Session State
→ Prediction Card / Metrics / Metadata
```

Streamlit은 Checkpoint, PyTorch Model, Test Transform을 직접 불러오지 않습니다.

## 적용

프로젝트 루트에서 ZIP 내용을 직접 풀어 실제 경로에 배치합니다. 이 Bundle은 기존 `src/dashboard/__init__.py`를 빈 파일에서 Package 설명 파일로 갱신하지만 Day 7 API 파일은 수정하지 않습니다.

## 실행 순서

1. 사전 점검

```powershell
python -m scripts.inspect_day8_dashboard_prerequisites
```

2. requirements.txt 갱신

```powershell
python -m scripts.update_day8_requirements
```

3. Dependency 설치·확인

```powershell
python -m pip install `
    -r .\requirements.txt

python -m pip show `
    streamlit `
    httpx `
    Pillow

python -m pip check
```

4. 첫 구현 테스트

```powershell
python -m pytest `
    .\tests\test_dashboard_config.py `
    .\tests\test_dashboard_api_client.py `
    .\tests\test_dashboard_session_state.py `
    .\tests\test_dashboard_ui_helpers.py `
    .\tests\test_dashboard_styles.py `
    .\tests\test_dashboard_app.py `
    .\tests\test_day8_dashboard_validation.py `
    .\tests\test_update_day8_requirements.py `
    -q
```

5. Terminal 1 — FastAPI

```powershell
python -m uvicorn `
    src.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

6. Terminal 2 — API Client 실제 통합 검증

```powershell
python -m scripts.run_day8_dashboard_validation
```

7. Terminal 2 — Streamlit

```powershell
python -m streamlit run `
    .\src\dashboard\app.py
```

8. 브라우저에서 실제 NORMAL·DEFECT 이미지를 각각 업로드해 결과와 화면을 확인합니다.

## 환경변수

기본 FastAPI URL은 `http://127.0.0.1:8000`입니다.

```powershell
$env:MVDA_API_BASE_URL = "http://127.0.0.1:8000"
```

필요 시 Timeout을 변경할 수 있습니다.

```powershell
$env:MVDA_API_CONNECT_TIMEOUT_SECONDS = "2"
$env:MVDA_API_READ_TIMEOUT_SECONDS = "30"
```

UI 구조 AppTest에서만 Health 요청을 끄는 설정도 지원합니다.

```powershell
$env:MVDA_DASHBOARD_HEALTH_CHECK_ENABLED = "false"
```

일반 실행에서는 기본값 `true`를 사용합니다.
