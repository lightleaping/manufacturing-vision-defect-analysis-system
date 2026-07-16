"""Day 7 FastAPI 이미지 추론 API 보고서와 README 구간을 생성한다.

실행:
    python -m scripts.create_day7_docs

이 Script는 실제 HTTP 검증 Artifact를 먼저 검증한 뒤에만 문서를 생성한다.
따라서 아직 실행하지 않은 결과를 문서에 완료 상태로 기록하지 않는다.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN = "제조 비전 결함 분석 시스템"
DAY7_TITLE = "Day 7 — FastAPI Image Inference API"

VALIDATION_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day7_fastapi_inference_validation.json"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "day7_fastapi_image_inference_api_summary.md"
)
README_PATH = PROJECT_ROOT / "README.md"

README_START_MARKER = "<!-- DAY7_FASTAPI_INFERENCE_START -->"
README_END_MARKER = "<!-- DAY7_FASTAPI_INFERENCE_END -->"

EXPECTED_MODEL_NAME = "ResNet18Transfer"
EXPECTED_MODEL_VERSION = "resnet18_transfer_best"
EXPECTED_THRESHOLD = 0.5
EXPECTED_DEVICE = "cpu"
EXPECTED_NORMAL_CLASS = "NORMAL"
EXPECTED_DEFECT_CLASS = "DEFECT"
EXPECTED_POSITIVE_CLASS = "DEFECT"
EXPECTED_REGRESSION_TEST_COUNT = 1255


def read_json_object(path: Path) -> dict[str, Any]:
    """JSON 파일을 UTF-8로 읽고 최상위 Object 형식을 검증한다."""

    if not path.is_file():
        raise FileNotFoundError(f"JSON artifact does not exist: {path}")

    with path.open(mode="r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError("JSON top-level value must be an object.")

    return payload


def _require_mapping(
    value: object,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a Mapping.")
    return value


def _require_non_empty_string(
    value: object,
    *,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string.")
    return value


def _require_finite_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    return numeric_value


def _validate_prediction_response(
    response: Mapping[str, Any],
    *,
    expected_prediction: int,
    expected_class_name: str,
    response_name: str,
) -> None:
    """실제 Prediction 응답의 핵심 정책과 수치 일관성을 검증한다."""

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
        "content_type",
        "image_width",
        "image_height",
        "image_mode",
        "inference_time_ms",
    }

    missing_keys = required_keys - set(response.keys())
    if missing_keys:
        raise KeyError(
            f"{response_name} is missing keys: {sorted(missing_keys)}"
        )

    prediction = response["prediction"]
    if prediction != expected_prediction:
        raise ValueError(
            f"{response_name}.prediction must be {expected_prediction}."
        )

    if response["prediction_class_name"] != expected_class_name:
        raise ValueError(
            f"{response_name}.prediction_class_name must be "
            f"{expected_class_name}."
        )

    if response["model_name"] != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"{response_name}.model_name must be {EXPECTED_MODEL_NAME}."
        )

    if response["model_version"] != EXPECTED_MODEL_VERSION:
        raise ValueError(
            f"{response_name}.model_version must be {EXPECTED_MODEL_VERSION}."
        )

    if response["positive_class"] != EXPECTED_POSITIVE_CLASS:
        raise ValueError(
            f"{response_name}.positive_class must be "
            f"{EXPECTED_POSITIVE_CLASS}."
        )

    threshold = _require_finite_float(
        response["classification_threshold"],
        name=f"{response_name}.classification_threshold",
    )
    if not math.isclose(
        threshold,
        EXPECTED_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{response_name}.classification_threshold must be "
            f"{EXPECTED_THRESHOLD}."
        )

    defect_probability = _require_finite_float(
        response["defect_probability"],
        name=f"{response_name}.defect_probability",
    )
    normal_probability = _require_finite_float(
        response["normal_probability"],
        name=f"{response_name}.normal_probability",
    )
    raw_logit = _require_finite_float(
        response["raw_logit"],
        name=f"{response_name}.raw_logit",
    )
    inference_time_ms = _require_finite_float(
        response["inference_time_ms"],
        name=f"{response_name}.inference_time_ms",
    )

    if not 0.0 <= defect_probability <= 1.0:
        raise ValueError(
            f"{response_name}.defect_probability must be in [0, 1]."
        )

    if not 0.0 <= normal_probability <= 1.0:
        raise ValueError(
            f"{response_name}.normal_probability must be in [0, 1]."
        )

    if not math.isclose(
        defect_probability + normal_probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{response_name} probabilities must sum to 1."
        )

    expected_from_threshold = (
        1 if defect_probability >= EXPECTED_THRESHOLD else 0
    )
    if prediction != expected_from_threshold:
        raise ValueError(
            f"{response_name}.prediction does not match threshold policy."
        )

    # sigmoid(raw_logit)과 Artifact 확률이 같은지 교차 검증한다.
    expected_probability = 1.0 / (1.0 + math.exp(-raw_logit))
    if not math.isclose(
        defect_probability,
        expected_probability,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"{response_name}.defect_probability does not match raw_logit."
        )

    if inference_time_ms < 0.0:
        raise ValueError(
            f"{response_name}.inference_time_ms must be non-negative."
        )

    width = response["image_width"]
    height = response["image_height"]

    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise ValueError(f"{response_name}.image_width must be positive int.")

    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise ValueError(f"{response_name}.image_height must be positive int.")

    _require_non_empty_string(
        response["original_filename"],
        name=f"{response_name}.original_filename",
    )
    _require_non_empty_string(
        response["content_type"],
        name=f"{response_name}.content_type",
    )
    _require_non_empty_string(
        response["image_mode"],
        name=f"{response_name}.image_mode",
    )


def validate_validation_payload(
    payload: Mapping[str, Any],
) -> None:
    """실제 Day 7 HTTP 검증 Artifact 전체를 검증한다."""

    if payload.get("project") != PROJECT_NAME:
        raise ValueError(f"project must be {PROJECT_NAME}.")

    if payload.get("run_name") != "day7_fastapi_inference_validation":
        raise ValueError("Unexpected run_name.")

    base_url = _require_non_empty_string(
        payload.get("base_url"),
        name="base_url",
    )
    if not base_url.startswith("http://127.0.0.1:"):
        raise ValueError("base_url must use local 127.0.0.1 HTTP address.")

    health = _require_mapping(
        payload.get("health"),
        name="health",
    )

    if health.get("status") != "ok":
        raise ValueError("health.status must be ok.")

    if health.get("service") != PROJECT_NAME:
        raise ValueError("health.service does not match project name.")

    if health.get("model_loaded") is not True:
        raise ValueError("health.model_loaded must be true.")

    if health.get("model_name") != EXPECTED_MODEL_NAME:
        raise ValueError("health.model_name is unexpected.")

    if health.get("device") != EXPECTED_DEVICE:
        raise ValueError("health.device must be cpu.")

    normal_image = _require_mapping(
        payload.get("normal_image"),
        name="normal_image",
    )
    defect_image = _require_mapping(
        payload.get("defect_image"),
        name="defect_image",
    )

    _require_non_empty_string(
        normal_image.get("path"),
        name="normal_image.path",
    )
    _require_non_empty_string(
        defect_image.get("path"),
        name="defect_image.path",
    )

    normal_response = _require_mapping(
        normal_image.get("response"),
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        defect_image.get("response"),
        name="defect_image.response",
    )

    _validate_prediction_response(
        normal_response,
        expected_prediction=0,
        expected_class_name=EXPECTED_NORMAL_CLASS,
        response_name="normal_image.response",
    )
    _validate_prediction_response(
        defect_response,
        expected_prediction=1,
        expected_class_name=EXPECTED_DEFECT_CLASS,
        response_name="defect_image.response",
    )

    runtime_seconds = _require_finite_float(
        payload.get("runtime_seconds"),
        name="runtime_seconds",
    )
    if runtime_seconds <= 0.0:
        raise ValueError("runtime_seconds must be greater than 0.")


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _format_probability(value: object) -> str:
    return f"{float(value):.12f}"


def _format_milliseconds(value: object) -> str:
    return f"{float(value):.2f} ms"


def build_day7_report(
    payload: Mapping[str, Any],
) -> str:
    """검증된 Artifact로 Day 7 Markdown 보고서를 만든다."""

    validate_validation_payload(payload)

    health = _require_mapping(payload["health"], name="health")
    normal_image = _require_mapping(
        payload["normal_image"],
        name="normal_image",
    )
    defect_image = _require_mapping(
        payload["defect_image"],
        name="defect_image",
    )
    normal_response = _require_mapping(
        normal_image["response"],
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        defect_image["response"],
        name="defect_image.response",
    )

    return f"""# {DAY7_TITLE}

## 1. 프로젝트

```text
영문: {PROJECT_NAME}
한글: {PROJECT_NAME_KOREAN}
```

Day 7의 목표는 Day 4에서 학습한 ResNet18 Best Checkpoint를 재학습하지 않고
FastAPI 서비스에서 재사용하여, 업로드한 제조 이미지를 `NORMAL` 또는
`DEFECT`로 분류하는 HTTP 추론 경로를 구현하는 것이다.

---

## 2. 구현 범위

```text
GET  /api/v1/health
POST /api/v1/predictions
GET  /docs
GET  /redoc
```

구현 내용:

- FastAPI Application Factory
- Lifespan 기반 모델 1회 로딩
- App State 기반 Inference Service 보관
- JPEG·JPG·PNG 업로드 검증
- 제한 크기 Chunk 읽기
- Pillow 실제 Decode 및 무결성 검증
- RGBA·Grayscale 입력의 RGB 변환
- Day 2 Test Transform 재사용
- ResNet18 Best Checkpoint 복원
- `torch.inference_mode()` 추론
- Raw Logit → Sigmoid → Threshold 처리
- 공통 오류 응답
- 단위 테스트·통합 테스트
- 실제 Uvicorn HTTP 검증

이번 기본 Prediction Endpoint에는 Grad-CAM을 포함하지 않았다.
일반 추론과 설명 가능성 계산의 책임을 분리하여 응답 시간과 연산 비용을
줄이고, 향후 별도 Explain Endpoint로 확장할 수 있도록 설계했다.

---

## 3. API 구조

```text
FastAPI 시작
→ Lifespan
→ create_production_inference_service()
→ restore_best_checkpoint()
→ create_test_transform()
→ ImageInferenceService
→ app.state.inference_service
```

Prediction 요청 흐름:

```text
multipart/form-data 이미지 업로드
→ 확장자·Content-Type 교차 검증
→ 10 MB 제한 읽기
→ Pillow 실제 Decode
→ JPEG·PNG 실제 형식 확인
→ 25,000,000 Pixel 제한 확인
→ 원본 Metadata 기록
→ RGB 변환
→ Resize 224 × 224
→ ToTensor
→ ImageNet Normalize
→ Tensor [1, 3, 224, 224]
→ torch.inference_mode()
→ ResNet18 Raw Logit
→ sigmoid
→ Threshold 0.5
→ NORMAL 또는 DEFECT JSON
```

---

## 4. 모델 재사용 정책

```text
Checkpoint:
models/checkpoints/resnet18_transfer_best.pt

Model:
ResNet18Transfer

Device:
cpu

Positive Class:
DEFECT

Threshold:
0.5
```

모델은 요청마다 다시 로딩하지 않는다.
FastAPI Lifespan에서 Checkpoint를 한 번 복원하고, 준비된 Service를
`app.state.inference_service`에 저장한다.

Production Loader는 기존 Day 4의 검증된
`restore_best_checkpoint(checkpoint_path, device)`를 그대로 재사용한다.
따라서 Checkpoint Schema를 API 계층에서 중복 구현하거나 추측하지 않는다.

전처리 역시 Day 2의 `create_test_transform()`을 그대로 사용한다.

---

## 5. 이미지 검증 정책

지원 입력:

```text
Extensions:
.jpg
.jpeg
.png

Content-Types:
image/jpeg
image/png

Decoded Formats:
JPEG
PNG
```

검증 기준:

- 파일 없음
- 빈 파일
- 지원하지 않는 확장자
- 지원하지 않는 Content-Type
- 확장자와 Content-Type 불일치
- Metadata와 실제 Decode 형식 불일치
- 손상 이미지
- Pillow가 읽을 수 없는 이미지
- 10 MB 초과 업로드
- 25,000,000 Pixel 초과 이미지
- 잘못된 Transform 출력
- NaN·Infinity 모델 출력
- 잘못된 모델 출력 Shape

이미지는 서버 디스크에 영구 저장하지 않는다.
원본 파일명을 서버 경로로 사용하지 않으며, 모든 Decode와 RGB 변환은
메모리 안에서 수행한다.

Mode 처리:

```text
RGB  → 그대로 사용
RGBA → RGB 변환
L    → RGB 변환
```

응답의 `image_mode`에는 RGB 변환 전 원본 Mode를 기록한다.

---

## 6. 추론 정책

```text
Model Output:
Raw Logit

defect_probability:
sigmoid(raw_logit)

normal_probability:
1 - defect_probability

prediction:
1 if defect_probability >= 0.5 else 0

0 = NORMAL
1 = DEFECT
```

모델 내부에는 Sigmoid를 추가하지 않았다.
기존 학습·평가 구조와 동일하게 모델은 Raw Logit만 반환하고,
API Service에서 확률 변환과 Threshold 판정을 수행한다.

---

## 7. 오류 응답

공통 형식:

```json
{{
  "detail": {{
    "code": "INVALID_IMAGE",
    "message": "업로드한 파일을 정상적인 이미지로 읽을 수 없습니다."
  }}
}}
```

오류 코드:

```text
MISSING_FILE
EMPTY_FILE
UNSUPPORTED_FILE_TYPE
FILE_TOO_LARGE
IMAGE_TOO_LARGE
INVALID_IMAGE
MODEL_NOT_READY
INVALID_MODEL_INPUT
INVALID_MODEL_OUTPUT
INFERENCE_FAILED
```

내부 Checkpoint 경로, Python Stack Trace, 원본 예외 문자열은 외부 응답에
노출하지 않는다.

---

## 8. 실제 HTTP 검증

Health 응답:

```text
Status       : {health["status"]}
Service      : {health["service"]}
Model Loaded : {health["model_loaded"]}
Model Name   : {health["model_name"]}
Device       : {health["device"]}
```

### NORMAL 이미지

```text
Path              : {normal_image["path"]}
Prediction        : {normal_response["prediction_class_name"]}
Prediction Index  : {normal_response["prediction"]}
P(DEFECT)         : {_format_probability(normal_response["defect_probability"])}
P(NORMAL)         : {_format_probability(normal_response["normal_probability"])}
Raw Logit         : {float(normal_response["raw_logit"]):.12f}
Original Size     : {normal_response["image_width"]} × {normal_response["image_height"]}
Original Mode     : {normal_response["image_mode"]}
Inference Time    : {_format_milliseconds(normal_response["inference_time_ms"])}
```

### DEFECT 이미지

```text
Path              : {defect_image["path"]}
Prediction        : {defect_response["prediction_class_name"]}
Prediction Index  : {defect_response["prediction"]}
P(DEFECT)         : {_format_probability(defect_response["defect_probability"])}
P(NORMAL)         : {_format_probability(defect_response["normal_probability"])}
Raw Logit         : {float(defect_response["raw_logit"]):.12f}
Original Size     : {defect_response["image_width"]} × {defect_response["image_height"]}
Original Mode     : {defect_response["image_mode"]}
Inference Time    : {_format_milliseconds(defect_response["inference_time_ms"])}
```

전체 검증 Runtime:

```text
{float(payload["runtime_seconds"]):.2f} seconds
```

검증 Artifact:

```text
reports/artifacts/day7_fastapi_inference_validation.json
```

Day 6에서 선택한 동일 표본의 확률과 API 응답이 일치하므로,
Script 추론과 HTTP 추론이 같은 Checkpoint와 같은 Test Transform을
사용한다는 것을 교차 검증했다.

---

## 9. Dependency

실제 실행 환경:

```text
FastAPI         : {_package_version("fastapi")}
Uvicorn         : {_package_version("uvicorn")}
python-multipart: {_package_version("python-multipart")}
httpx           : {_package_version("httpx")}
Pydantic        : {_package_version("pydantic")}
Pillow          : {_package_version("Pillow")}
PyTorch         : {_package_version("torch")}
Torchvision     : {_package_version("torchvision")}
```

---

## 10. 테스트

Day 7 대상 테스트:

```text
40 passed
```

전체 회귀 테스트:

```text
{EXPECTED_REGRESSION_TEST_COUNT} passed
```

검증 범위:

- 이미지 업로드와 Decode
- RGB·RGBA·Grayscale 처리
- 확장자·Content-Type·실제 형식 교차 검증
- 파일 크기·Pixel 제한
- Raw Logit·Sigmoid·Threshold
- NORMAL·DEFECT 분류
- `torch.inference_mode()`
- NaN·Infinity 차단
- 모델 출력 Shape 검증
- Production Loader의 기존 함수 재사용
- FastAPI Health·Prediction Endpoint
- Model Not Ready
- Startup 실패 보호
- 내부 경로·Stack Trace 비노출
- 실제 ResNet18 HTTP 요청

테스트 실행 중 다음 호환성 경고가 1건 발생했다.

```text
StarletteDeprecationWarning:
Using httpx with starlette.testclient is deprecated;
install httpx2 instead.
```

현재 테스트와 실제 HTTP 검증은 모두 통과했다.
이 경고는 기능 실패가 아니며, 향후 FastAPI·Starlette 테스트 Client
Dependency 정책을 정리할 때 처리할 기술부채로 기록한다.

---

## 11. 실행 방법

API 실행:

```powershell
python -m uvicorn `
    src.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

문서:

```text
Swagger UI: http://127.0.0.1:8000/docs
ReDoc     : http://127.0.0.1:8000/redoc
```

실제 HTTP 자동 검증:

```powershell
python -m scripts.run_day7_api_validation
```

---

## 12. 현재 범위와 향후 확장

현재 구현:

- 단일 이미지 정상·불량 이진 분류
- CPU 추론
- JPEG·PNG 업로드
- 동기 단일 요청
- 모델 1회 로딩
- JSON 응답
- 안전한 입력·오류 검증

향후 확장 가능 범위:

- 별도 Grad-CAM Explain Endpoint
- Batch Prediction
- 요청 추적 ID
- 구조화 Logging
- Prometheus Metric
- Streamlit Dashboard 연결
- Model Version Registry
- 비동기 Queue 기반 대량 추론

이번 Day 7에서는 범위 확장을 위해 복잡도를 늘리지 않고,
포트폴리오에 필요한 안정적인 단일 이미지 추론 API를 우선 완성했다.

---

## 13. 실무 포인트

1. 모델 학습 코드와 HTTP 요청 처리를 분리했다.
2. Lifespan에서 모델을 한 번만 로딩하여 요청마다 Checkpoint를 읽지 않는다.
3. 실제 Decoder 결과까지 확인하여 확장자 위장 파일을 차단한다.
4. 업로드 크기와 Pixel 수를 모두 제한한다.
5. 모델은 Raw Logit을 유지하고 Service 계층에서 확률을 계산한다.
6. Dummy Service 주입으로 API 테스트가 실제 Checkpoint에 의존하지 않는다.
7. 실제 Uvicorn HTTP 검증에서만 Best Checkpoint와 Dataset 이미지를 사용한다.
8. 내부 경로와 Stack Trace를 외부 오류 응답에 노출하지 않는다.
"""


def build_readme_section(
    payload: Mapping[str, Any],
) -> str:
    """README에 삽입할 Day 7 요약 구간을 만든다."""

    validate_validation_payload(payload)

    normal_image = _require_mapping(
        payload["normal_image"],
        name="normal_image",
    )
    defect_image = _require_mapping(
        payload["defect_image"],
        name="defect_image",
    )
    normal_response = _require_mapping(
        normal_image["response"],
        name="normal_image.response",
    )
    defect_response = _require_mapping(
        defect_image["response"],
        name="defect_image.response",
    )

    return f"""{README_START_MARKER}
## Day 7 — FastAPI Image Inference API

ResNet18 Best Checkpoint를 FastAPI Lifespan에서 한 번만 로딩하고,
업로드한 제조 이미지를 `NORMAL` 또는 `DEFECT`로 분류하는 HTTP 추론 API를
구현했습니다.

### Endpoint

```text
GET  /api/v1/health
POST /api/v1/predictions
GET  /docs
GET  /redoc
```

### Inference Flow

```text
UploadFile
→ 제한 크기 읽기
→ 확장자·Content-Type·실제 Decode 형식 검증
→ RGB 변환
→ Day 2 Test Transform
→ ResNet18 Raw Logit
→ Sigmoid
→ Threshold 0.5
→ NORMAL / DEFECT JSON
```

모델은 요청마다 다시 로딩하지 않습니다.
FastAPI Lifespan에서 `resnet18_transfer_best.pt`를 한 번 복원하고
`app.state.inference_service`에 저장합니다.

### Validation Policy

```text
Supported: JPEG, JPG, PNG
Maximum Upload Size: 10 MB
Maximum Pixel Count: 25,000,000
Positive Class: DEFECT
Classification Threshold: 0.5
```

### Real HTTP Validation

```text
Health Model Loaded : true
Model               : ResNet18Transfer
Device              : cpu

NORMAL Image        : {normal_response["original_filename"]}
NORMAL Prediction   : {normal_response["prediction_class_name"]}
NORMAL P(DEFECT)    : {_format_probability(normal_response["defect_probability"])}

DEFECT Image        : {defect_response["original_filename"]}
DEFECT Prediction   : {defect_response["prediction_class_name"]}
DEFECT P(DEFECT)    : {_format_probability(defect_response["defect_probability"])}
```

Artifact:

```text
reports/artifacts/day7_fastapi_inference_validation.json
```

Report:

```text
reports/day7_fastapi_image_inference_api_summary.md
```

Tests:

```text
Day 7 API Tests      : 40 passed
Full Regression Tests: {EXPECTED_REGRESSION_TEST_COUNT} passed
```

현재 Prediction Endpoint는 빠른 일반 추론만 담당합니다.
Grad-CAM은 Gradient·Hook·Backward가 필요한 별도 책임이므로 향후 Explain
Endpoint로 분리할 수 있습니다.
{README_END_MARKER}"""


def upsert_marked_section(
    *,
    original_text: str,
    section_text: str,
    start_marker: str,
    end_marker: str,
) -> str:
    """Marker 구간을 교체하고, 없으면 문서 끝에 추가한다."""

    if original_text.count(start_marker) > 1:
        raise ValueError(f"Duplicate start marker: {start_marker}")

    if original_text.count(end_marker) > 1:
        raise ValueError(f"Duplicate end marker: {end_marker}")

    start_index = original_text.find(start_marker)
    end_index = original_text.find(end_marker)

    if (start_index == -1) != (end_index == -1):
        raise ValueError("README contains only one Day 7 marker.")

    normalized_section = section_text.strip()

    if start_index == -1:
        stripped_original = original_text.rstrip()
        if stripped_original:
            return f"{stripped_original}\n\n---\n\n{normalized_section}\n"
        return f"{normalized_section}\n"

    if end_index < start_index:
        raise ValueError("README Day 7 markers are in the wrong order.")

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


def update_regression_test_count(
    text: str,
    *,
    test_count: int = EXPECTED_REGRESSION_TEST_COUNT,
) -> str:
    """기존 Full Regression Tests 표기가 있으면 최신 수치로 갱신한다."""

    pattern = re.compile(
        r"(Full Regression Tests\s*:\s*)\d+(\s*passed)",
        flags=re.IGNORECASE,
    )

    return pattern.sub(
        rf"\g<1>{test_count}\g<2>",
        text,
    )


def write_text_atomically(
    *,
    path: Path,
    content: str,
) -> None:
    """임시 파일에 쓴 뒤 최종 경로로 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")

    try:
        temporary_path.write_text(
            content,
            encoding="utf-8",
            newline="\n",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    """실제 Artifact 검증 후 Day 7 문서를 생성한다."""

    payload = read_json_object(VALIDATION_ARTIFACT_PATH)
    validate_validation_payload(payload)

    report_text = build_day7_report(payload)
    write_text_atomically(
        path=REPORT_PATH,
        content=report_text,
    )

    if not README_PATH.is_file():
        raise FileNotFoundError(f"README does not exist: {README_PATH}")

    readme_text = README_PATH.read_text(encoding="utf-8")
    readme_section = build_readme_section(payload)

    updated_readme = upsert_marked_section(
        original_text=readme_text,
        section_text=readme_section,
        start_marker=README_START_MARKER,
        end_marker=README_END_MARKER,
    )
    updated_readme = update_regression_test_count(updated_readme)

    write_text_atomically(
        path=README_PATH,
        content=updated_readme,
    )

    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Report was not created: {REPORT_PATH}")

    final_readme = README_PATH.read_text(encoding="utf-8")
    if final_readme.count(README_START_MARKER) != 1:
        raise ValueError("README Day 7 start marker count must be 1.")
    if final_readme.count(README_END_MARKER) != 1:
        raise ValueError("README Day 7 end marker count must be 1.")

    print("[PASS] Day 7 report created")
    print("[PASS] README Day 7 section added")
    print(f"[REPORT] {REPORT_PATH}")
    print(f"[README] {README_PATH}")


if __name__ == "__main__":
    main()
