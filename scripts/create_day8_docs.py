"""Day 8 Streamlit Dashboard 보고서와 README 구간을 생성한다.

실행 예:
    python -m scripts.create_day8_docs --regression-test-count 1315 --warning-count 1

실제 FastAPI 통합 검증, 육안 검증 완료 상태, 두 Screenshot Artifact를 모두
확인한 뒤에만 문서를 생성한다. 최종 전체 회귀 테스트 수는 실행 인자로 받아
문서 테스트 추가 후 수치가 달라지는 문제를 방지한다.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN = "제조 비전 결함 분석 시스템"
DAY8_TITLE = "Day 8 — Streamlit Image Inference Dashboard"

VALIDATION_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day8_streamlit_dashboard_validation.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day8_streamlit_image_inference_dashboard_summary.md"
)
README_PATH = PROJECT_ROOT / "README.md"
README_START_MARKER = "<!-- DAY8_STREAMLIT_DASHBOARD_START -->"
README_END_MARKER = "<!-- DAY8_STREAMLIT_DASHBOARD_END -->"

EXPECTED_RUN_NAME = "day8_streamlit_dashboard_validation"
EXPECTED_MODEL_NAME = "ResNet18Transfer"
EXPECTED_MODEL_VERSION = "resnet18_transfer_best"
EXPECTED_THRESHOLD = 0.5
EXPECTED_POSITIVE_CLASS = "DEFECT"
EXPECTED_SCREENSHOT_LABELS = ("NORMAL", "DEFECT")
INITIAL_DAY8_TEST_COUNT = 45


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Validation artifact does not exist: {path}")
    with path.open(mode="r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise TypeError("Validation artifact top-level value must be an object.")
    return payload


def _require_mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a Mapping.")
    return value


def _require_non_empty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value.strip()


def _require_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")
    return numeric_value


def _project_path(path_value: object, *, name: str) -> Path:
    text = _require_non_empty_string(path_value, name=name)
    candidate = Path(text)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _validate_png(path: Path, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise ValueError(f"{name} must decode as PNG.")
            if image.width <= 0 or image.height <= 0:
                raise ValueError(f"{name} dimensions must be positive.")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{name} must be a readable PNG.") from exc


def _validate_prediction(
    response: Mapping[str, Any],
    *,
    expected_prediction: int,
    expected_class_name: str,
    name: str,
) -> None:
    required_keys = {
        "prediction",
        "prediction_class_name",
        "defect_probability",
        "normal_probability",
        "raw_logit",
        "classification_threshold",
        "model_name",
        "model_version",
        "positive_class",
        "original_filename",
        "image_width",
        "image_height",
        "image_mode",
        "inference_time_ms",
    }
    missing = required_keys - set(response)
    if missing:
        raise KeyError(f"{name} is missing keys: {sorted(missing)}")

    if response["prediction"] != expected_prediction:
        raise ValueError(f"{name}.prediction is unexpected.")
    if response["prediction_class_name"] != expected_class_name:
        raise ValueError(f"{name}.prediction_class_name is unexpected.")
    if response["model_name"] != EXPECTED_MODEL_NAME:
        raise ValueError(f"{name}.model_name is unexpected.")
    if response["model_version"] != EXPECTED_MODEL_VERSION:
        raise ValueError(f"{name}.model_version is unexpected.")
    if response["positive_class"] != EXPECTED_POSITIVE_CLASS:
        raise ValueError(f"{name}.positive_class is unexpected.")

    defect_probability = _require_finite_float(
        response["defect_probability"],
        name=f"{name}.defect_probability",
    )
    normal_probability = _require_finite_float(
        response["normal_probability"],
        name=f"{name}.normal_probability",
    )
    raw_logit = _require_finite_float(response["raw_logit"], name=f"{name}.raw_logit")
    threshold = _require_finite_float(
        response["classification_threshold"],
        name=f"{name}.classification_threshold",
    )
    inference_time_ms = _require_finite_float(
        response["inference_time_ms"],
        name=f"{name}.inference_time_ms",
    )

    if not math.isclose(threshold, EXPECTED_THRESHOLD, abs_tol=1e-12):
        raise ValueError(f"{name}.classification_threshold is unexpected.")
    if not 0.0 <= defect_probability <= 1.0:
        raise ValueError(f"{name}.defect_probability must be in [0, 1].")
    if not 0.0 <= normal_probability <= 1.0:
        raise ValueError(f"{name}.normal_probability must be in [0, 1].")
    if not math.isclose(
        defect_probability + normal_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name} probabilities must sum to 1.")
    if (1 if defect_probability >= threshold else 0) != expected_prediction:
        raise ValueError(f"{name} prediction does not match threshold policy.")

    sigmoid_probability = 1.0 / (1.0 + math.exp(-raw_logit))
    if not math.isclose(
        defect_probability,
        sigmoid_probability,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(f"{name}.defect_probability does not match raw_logit.")
    if inference_time_ms < 0.0:
        raise ValueError(f"{name}.inference_time_ms must be non-negative.")


def validate_day8_payload(payload: Mapping[str, Any]) -> None:
    """통합 검증과 육안 검증이 모두 완료된 Day 8 Artifact를 검증한다."""

    if payload.get("project") != PROJECT_NAME:
        raise ValueError(f"project must be {PROJECT_NAME}.")
    if payload.get("run_name") != EXPECTED_RUN_NAME:
        raise ValueError(f"run_name must be {EXPECTED_RUN_NAME}.")

    base_url = _require_non_empty_string(payload.get("base_url"), name="base_url")
    if not base_url.startswith("http://127.0.0.1:"):
        raise ValueError("base_url must use local 127.0.0.1 HTTP address.")

    health = _require_mapping(payload.get("health"), name="health")
    if health.get("status") != "ok":
        raise ValueError("health.status must be ok.")
    if health.get("model_loaded") is not True:
        raise ValueError("health.model_loaded must be true.")
    if health.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("health.model_name is unexpected.")
    if health.get("device") != "cpu":
        raise ValueError("health.device must be cpu.")

    normal_image = _require_mapping(payload.get("normal_image"), name="normal_image")
    defect_image = _require_mapping(payload.get("defect_image"), name="defect_image")
    normal_response = _require_mapping(
        normal_image.get("response"),
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        defect_image.get("response"),
        name="defect_image.response",
    )
    _validate_prediction(
        normal_response,
        expected_prediction=0,
        expected_class_name="NORMAL",
        name="normal_image.response",
    )
    _validate_prediction(
        defect_response,
        expected_prediction=1,
        expected_class_name="DEFECT",
        name="defect_image.response",
    )

    runtime_seconds = _require_finite_float(
        payload.get("runtime_seconds"),
        name="runtime_seconds",
    )
    if runtime_seconds <= 0.0:
        raise ValueError("runtime_seconds must be greater than 0.")

    if payload.get("ui_visual_validation_completed") is not True:
        raise ValueError("ui_visual_validation_completed must be true.")
    if payload.get("ui_visual_validation_result") != "PASS":
        raise ValueError("ui_visual_validation_result must be PASS.")

    screenshot_artifacts = payload.get("screenshot_artifacts")
    if not isinstance(screenshot_artifacts, list) or len(screenshot_artifacts) != 2:
        raise ValueError("Exactly two screenshot_artifacts are required.")

    labels: list[str] = []
    for index, artifact_value in enumerate(screenshot_artifacts):
        artifact = _require_mapping(
            artifact_value,
            name=f"screenshot_artifacts[{index}]",
        )
        label = _require_non_empty_string(
            artifact.get("label"),
            name=f"screenshot_artifacts[{index}].label",
        )
        labels.append(label)
        screenshot_path = _project_path(
            artifact.get("path"),
            name=f"screenshot_artifacts[{index}].path",
        )
        _validate_png(screenshot_path, name=f"screenshot_artifacts[{index}]")

    if tuple(labels) != EXPECTED_SCREENSHOT_LABELS:
        raise ValueError("Screenshot labels must be NORMAL then DEFECT.")


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _format_probability(value: object) -> str:
    return f"{float(value):.12f}"


def _format_percent(value: object) -> str:
    return f"{float(value) * 100.0:.2f}%"


def build_day8_report(
    payload: Mapping[str, Any],
    *,
    regression_test_count: int,
    warning_count: int,
) -> str:
    validate_day8_payload(payload)
    if regression_test_count <= 0:
        raise ValueError("regression_test_count must be positive.")
    if warning_count < 0:
        raise ValueError("warning_count must be non-negative.")

    health = _require_mapping(payload["health"], name="health")
    normal_image = _require_mapping(payload["normal_image"], name="normal_image")
    defect_image = _require_mapping(payload["defect_image"], name="defect_image")
    normal_response = _require_mapping(
        normal_image["response"], name="normal_image.response"
    )
    defect_response = _require_mapping(
        defect_image["response"], name="defect_image.response"
    )
    screenshot_artifacts = payload["screenshot_artifacts"]
    normal_screenshot = _require_mapping(
        screenshot_artifacts[0], name="normal_screenshot"
    )
    defect_screenshot = _require_mapping(
        screenshot_artifacts[1], name="defect_screenshot"
    )

    warning_text = (
        "없음"
        if warning_count == 0
        else f"{warning_count}건 — 실패와 분리하여 기술부채로 추적"
    )

    return f"""# {DAY8_TITLE}

## 1. 프로젝트

```text
영문: {PROJECT_NAME}
한글: {PROJECT_NAME_KOREAN}
```

Day 8의 목표는 사용자가 브라우저에서 제조 이미지를 업로드하고, Day 7
FastAPI를 통해 `NORMAL` 또는 `DEFECT` 추론 결과를 확인할 수 있는 단일 페이지
Streamlit Dashboard를 구현하는 것이다.

---

## 2. 책임 분리

```text
사용자 브라우저
→ Streamlit Dashboard
→ DashboardApiClient
→ Day 7 FastAPI
→ ResNet18Transfer
→ Prediction JSON
→ Streamlit Session State
→ 결과 표시
```

FastAPI는 모델 로딩·이미지 검증·전처리·추론·오류 응답을 담당한다.
Streamlit은 이미지 업로드·Preview·HTTP 요청·결과 표시·사용자 경험만 담당한다.
Streamlit 내부에서는 Checkpoint를 직접 로딩하거나 확률을 다시 계산하지 않는다.

---

## 3. 구현 파일

```text
src/dashboard/__init__.py
src/dashboard/config.py
src/dashboard/api_client.py
src/dashboard/session_state.py
src/dashboard/styles.py
src/dashboard/ui_helpers.py
src/dashboard/app.py

scripts/inspect_day8_dashboard_prerequisites.py
scripts/update_day8_requirements.py
scripts/run_day8_dashboard_validation.py
scripts/finalize_day8_visual_validation.py
scripts/create_day8_docs.py
```

핵심 책임:

- 환경변수 기반 FastAPI Base URL과 Timeout 설정
- Dataclass 기반 Health·Prediction 결과
- `httpx.Client` 기반 Health·multipart Prediction 요청
- FastAPI 공통 오류 Schema의 안전한 해석
- 5초 TTL Health Cache
- Streamlit rerun을 고려한 Session State
- 업로드 이미지 Preview와 Metadata
- Prediction Card·확률 Metric·Progress·모델 Metadata
- 내부 예외 문자열과 Stack Trace 비노출

---

## 4. Dashboard 화면

```text
상단 프로젝트 설명
FastAPI 연결·Model Loaded·Device 상태
JPEG·PNG 이미지 업로더
업로드 이미지 Preview
파일명·크기·Mode Metadata
이미지 분석 실행 버튼
Prediction Card
P(DEFECT)·P(NORMAL)
Raw Logit·Inference Time
DEFECT Probability Progress
Model·Image Metadata
모델 한계와 Grad-CAM 분리 안내
```

버튼을 누르기 전에는 Prediction 요청을 전송하지 않는다. Health 요청만 짧게
Cache하며, Prediction 결과와 오류는 Session State에 저장해 Widget rerun 중에도
불필요하게 사라지지 않도록 했다.

---

## 5. API Client와 오류 정책

Client 검증 범위:

```text
Health 200
Prediction 200 NORMAL
Prediction 200 DEFECT
Timeout
Connection Error
FastAPI 오류 Schema
잘못된 JSON
필수 Key 누락
NaN·Infinity
확률 합 오류
Prediction·Class Name 불일치
Threshold 정책 불일치
```

Dashboard 오류 코드:

```text
API_CONNECTION_ERROR
API_TIMEOUT
API_INVALID_RESPONSE
API_REQUEST_ERROR
MODEL_NOT_READY
INVALID_IMAGE
UNSUPPORTED_FILE_TYPE
FILE_TOO_LARGE
IMAGE_TOO_LARGE
INFERENCE_FAILED
```

FastAPI가 반환한 내부 예외 문자열은 화면에 그대로 노출하지 않고, 사전에 정의한
안전한 사용자 메시지로 변환한다.

---

## 6. 실제 FastAPI 통합 검증

Health:

```text
Status       : {health['status']}
Model Loaded : {health['model_loaded']}
Model        : {health['model_name']}
Device       : {health['device']}
Base URL     : {payload['base_url']}
```

NORMAL 이미지:

```text
File         : {normal_response['original_filename']}
Prediction   : {normal_response['prediction_class_name']}
P(DEFECT)    : {_format_probability(normal_response['defect_probability'])}
P(NORMAL)    : {_format_probability(normal_response['normal_probability'])}
Display      : {_format_percent(normal_response['defect_probability'])} DEFECT
Raw Logit    : {float(normal_response['raw_logit']):.12f}
Inference    : {float(normal_response['inference_time_ms']):.2f} ms
```

DEFECT 이미지:

```text
File         : {defect_response['original_filename']}
Prediction   : {defect_response['prediction_class_name']}
P(DEFECT)    : {_format_probability(defect_response['defect_probability'])}
P(NORMAL)    : {_format_probability(defect_response['normal_probability'])}
Display      : {_format_percent(defect_response['defect_probability'])} DEFECT
Raw Logit    : {float(defect_response['raw_logit']):.12f}
Inference    : {float(defect_response['inference_time_ms']):.2f} ms
```

통합 검증 Runtime:

```text
{float(payload['runtime_seconds']):.2f} seconds
```

Artifact:

```text
reports/artifacts/day8_streamlit_dashboard_validation.json
```

---

## 7. 브라우저 육안 검증과 Screenshot

```text
UI Visual Validation : PASS
NORMAL Screenshot     : {normal_screenshot['path']}
NORMAL Screenshot Size: {normal_screenshot['width']} × {normal_screenshot['height']}
DEFECT Screenshot     : {defect_screenshot['path']}
DEFECT Screenshot Size: {defect_screenshot['width']} × {defect_screenshot['height']}
```

![Day 8 NORMAL Dashboard](figures/day8_streamlit_dashboard_normal.png)

![Day 8 DEFECT Dashboard](figures/day8_streamlit_dashboard_defect.png)

Screenshot은 실제 Streamlit 브라우저 화면에서 이미지 Preview, FastAPI 상태,
Prediction 결과와 확률이 함께 표시되는지 확인한 뒤 저장했다.

---

## 8. Dependency

```text
Streamlit : {_package_version('streamlit')}
httpx     : {_package_version('httpx')}
Pillow    : {_package_version('Pillow')}
FastAPI   : {_package_version('fastapi')}
Pydantic  : {_package_version('pydantic')}
PyTorch   : {_package_version('torch')}
```

---

## 9. 테스트

초기 Dashboard 구현 테스트:

```text
{INITIAL_DAY8_TEST_COUNT} passed
```

최종 전체 회귀 테스트:

```text
{regression_test_count} passed
Warnings: {warning_text}
```

테스트는 실제 FastAPI 서버나 Checkpoint에 의존하지 않도록 `httpx.MockTransport`와
Streamlit `AppTest`를 사용했다. 실제 모델과 Dataset 이미지는 별도의 통합 검증
Script에서만 사용했다.

---

## 10. 실행 방법

Terminal 1 — FastAPI:

```powershell
python -m uvicorn `
    src.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

Terminal 2 — Streamlit:

```powershell
python -m streamlit run `
    .\\src\\dashboard\\app.py
```

실제 Client 통합 검증:

```powershell
python -m scripts.run_day8_dashboard_validation
```

육안 검증 Artifact 확정:

```powershell
python -m scripts.finalize_day8_visual_validation
```

---

## 11. 현재 범위와 향후 확장

현재 구현:

- 단일 이미지 정상·불량 분류 UI
- FastAPI Client 기반 책임 분리
- JPEG·PNG Preview
- Health·Model Loaded 상태
- Prediction·확률·Raw Logit·Metadata 표시
- Session State
- 안전한 오류 메시지
- NORMAL·DEFECT 브라우저 Screenshot

향후 확장 가능 범위:

- 별도 Grad-CAM Explain Endpoint와 Dashboard 연결
- Batch Upload
- Prediction History
- 모델 버전 선택
- 운영 Metric과 구조화 Logging
- 배포 환경별 Base URL 관리

Day 8에서는 일반 Prediction과 Grad-CAM을 합치지 않았다. Grad-CAM은 Gradient와
Hook이 필요한 별도 설명 가능성 작업이므로, 빠른 기본 추론 화면과 분리하는 것이
현재 범위에 적합하다.

---

## 12. 실무 포인트

1. UI와 모델 추론 서버의 책임을 분리했다.
2. Dashboard가 FastAPI 응답을 신뢰하기 전에 Schema 일관성을 검증한다.
3. 확률을 UI에서 재계산하지 않아 Backend와 표시 결과의 불일치를 방지한다.
4. Health 요청만 짧게 Cache하고 Prediction은 버튼 클릭 시에만 실행한다.
5. Session State로 Streamlit rerun 중 결과를 유지한다.
6. 단위 테스트는 Mock Transport를 사용하고 실제 모델은 통합 검증에서만 사용한다.
7. 브라우저 Screenshot까지 확보해 코드뿐 아니라 실제 동작 화면을 증명한다.
8. 모델 결과가 실제 생산 공정의 최종 판정을 대체하지 않는다는 한계를 명시한다.
"""


def build_readme_section(
    payload: Mapping[str, Any],
    *,
    regression_test_count: int,
    warning_count: int,
) -> str:
    validate_day8_payload(payload)
    if regression_test_count <= 0:
        raise ValueError("regression_test_count must be positive.")
    if warning_count < 0:
        raise ValueError("warning_count must be non-negative.")

    health = _require_mapping(payload["health"], name="health")
    normal_response = _require_mapping(
        _require_mapping(payload["normal_image"], name="normal_image")["response"],
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        _require_mapping(payload["defect_image"], name="defect_image")["response"],
        name="defect_image.response",
    )
    warnings = "none" if warning_count == 0 else str(warning_count)

    return f"""{README_START_MARKER}
## {DAY8_TITLE}

Day 7 FastAPI를 추론 Backend로 유지하고, Streamlit은 이미지 업로드·Preview·
HTTP 요청·결과 표시를 담당하는 단일 페이지 Dashboard로 구현했습니다.

### Architecture

```text
Browser
→ Streamlit Dashboard
→ DashboardApiClient
→ FastAPI /api/v1/predictions
→ ResNet18Transfer
→ JSON Response
→ Session State
→ Prediction Card
```

Streamlit에서는 Checkpoint를 직접 로딩하지 않으며 확률도 다시 계산하지 않습니다.
FastAPI가 반환한 Prediction, 확률, Raw Logit과 Metadata를 검증한 뒤 표시합니다.

### Dashboard Features

```text
Image Upload and Preview
FastAPI Health and Model Loaded Status
NORMAL / DEFECT Prediction Card
P(DEFECT) and P(NORMAL)
Raw Logit and Inference Time
Model and Image Metadata
Session State
Safe API Error Messages
```

### Real Validation

```text
Health Model Loaded : {health['model_loaded']}
Model               : {health['model_name']}
Device              : {health['device']}

NORMAL Image        : {normal_response['original_filename']}
NORMAL Prediction   : {normal_response['prediction_class_name']}
NORMAL P(DEFECT)    : {_format_probability(normal_response['defect_probability'])}

DEFECT Image        : {defect_response['original_filename']}
DEFECT Prediction   : {defect_response['prediction_class_name']}
DEFECT P(DEFECT)    : {_format_probability(defect_response['defect_probability'])}

UI Visual Validation: PASS
```

Screenshots:

```text
reports/figures/day8_streamlit_dashboard_normal.png
reports/figures/day8_streamlit_dashboard_defect.png
```

![Day 8 NORMAL Dashboard](reports/figures/day8_streamlit_dashboard_normal.png)

![Day 8 DEFECT Dashboard](reports/figures/day8_streamlit_dashboard_defect.png)

Artifact and Report:

```text
reports/artifacts/day8_streamlit_dashboard_validation.json
reports/day8_streamlit_image_inference_dashboard_summary.md
```

Tests:

```text
Initial Day 8 Dashboard Tests: {INITIAL_DAY8_TEST_COUNT} passed
Full Regression Tests        : {regression_test_count} passed
Warnings                     : {warnings}
```

Grad-CAM은 Day 6에서 수행한 별도 설명 가능성 분석입니다. Day 8 기본 Dashboard는
빠른 Prediction 결과를 우선 제공하며, 실제 생산 공정의 최종 품질 판정을
대체하지 않습니다.
<!-- DAY8_STREAMLIT_DASHBOARD_END -->"""


def upsert_marked_section(
    *,
    original_text: str,
    section_text: str,
    start_marker: str,
    end_marker: str,
) -> str:
    if original_text.count(start_marker) > 1:
        raise ValueError(f"Duplicate start marker: {start_marker}")
    if original_text.count(end_marker) > 1:
        raise ValueError(f"Duplicate end marker: {end_marker}")

    start_index = original_text.find(start_marker)
    end_index = original_text.find(end_marker)
    if (start_index == -1) != (end_index == -1):
        raise ValueError("README contains only one Day 8 marker.")

    normalized_section = section_text.strip()
    if start_index == -1:
        stripped_original = original_text.rstrip()
        if stripped_original:
            return f"{stripped_original}\n\n---\n\n{normalized_section}\n"
        return f"{normalized_section}\n"

    if end_index < start_index:
        raise ValueError("README Day 8 markers are in the wrong order.")

    end_index += len(end_marker)
    prefix = original_text[:start_index].rstrip()
    suffix = original_text[end_index:].lstrip()

    if prefix and suffix:
        return f"{prefix}\n\n{normalized_section}\n\n{suffix}"
    if prefix:
        return f"{prefix}\n\n{normalized_section}\n"
    if suffix:
        return f"{normalized_section}\n\n{suffix}"
    return f"{normalized_section}\n"


def write_text_atomically(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8", newline="\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-test-count", type=int, required=True)
    parser.add_argument("--warning-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.regression_test_count <= 0:
        raise ValueError("--regression-test-count must be positive.")
    if args.warning_count < 0:
        raise ValueError("--warning-count must be non-negative.")

    payload = read_json_object(VALIDATION_ARTIFACT_PATH)
    validate_day8_payload(payload)

    report_text = build_day8_report(
        payload,
        regression_test_count=args.regression_test_count,
        warning_count=args.warning_count,
    )
    write_text_atomically(path=REPORT_PATH, content=report_text)

    if not README_PATH.is_file():
        raise FileNotFoundError(f"README does not exist: {README_PATH}")
    readme_text = README_PATH.read_text(encoding="utf-8-sig")
    readme_section = build_readme_section(
        payload,
        regression_test_count=args.regression_test_count,
        warning_count=args.warning_count,
    )
    updated_readme = upsert_marked_section(
        original_text=readme_text,
        section_text=readme_section,
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )
    write_text_atomically(path=README_PATH, content=updated_readme)

    final_readme = README_PATH.read_text(encoding="utf-8")
    if final_readme.count(README_START_MARKER) != 1:
        raise ValueError("README Day 8 start marker count must be 1.")
    if final_readme.count(README_END_MARKER) != 1:
        raise ValueError("README Day 8 end marker count must be 1.")

    print("[PASS] Day 8 report created")
    print("[PASS] README Day 8 section added")
    print(f"[REPORT] {REPORT_PATH}")
    print(f"[README] {README_PATH}")
    print(f"[REGRESSION] {args.regression_test_count} passed")
    print(f"[WARNINGS] {args.warning_count}")


if __name__ == "__main__":
    main()
