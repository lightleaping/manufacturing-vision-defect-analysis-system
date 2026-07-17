"""Day 13 Detection FastAPI·Streamlit 통합 Artifact로 보고서와 README를 생성한다.

[기존 코드 참고]
Day 12 문서 생성기의 Artifact 우선 검증, README Marker 교체, 최초 1회
백업, 원자적 파일 저장 방식을 따른다.

[신규 구현]
- Detection API 사전 점검·실제 Best Checkpoint Inspection·HTTP Smoke Test 검증
- Detection Dashboard 정적 구조·실제 API Client·Overlay Artifact 검증
- 기존 Classification Endpoint와 Dashboard를 유지한 확장 구조 기록
- OpenCV Contour 후보와 Faster R-CNN Prediction의 의미 구분
- 실행하지 않은 Day 14 최종 통합·Portfolio·Interview 작업을 완료로 기록하지 않음
- 브라우저 수동 검증 상태를 자동 검증과 분리하여 기록
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


PREREQUISITES_ARTIFACT = Path(
    "reports/artifacts/day13_integration_prerequisites.json"
)
API_INSPECTION_ARTIFACT = Path(
    "reports/artifacts/day13_detection_api_stage1_inspection.json"
)
API_SMOKE_ARTIFACT = Path(
    "reports/artifacts/day13_detection_api_smoke_test.json"
)
DASHBOARD_INSPECTION_ARTIFACT = Path(
    "reports/artifacts/day13_detection_dashboard_stage2_inspection.json"
)
DASHBOARD_VALIDATION_ARTIFACT = Path(
    "reports/artifacts/day13_detection_dashboard_api_client_validation.json"
)
DASHBOARD_OVERLAY_FIGURE = Path(
    "reports/figures/day13_detection_dashboard_overlay.png"
)
BEST_CHECKPOINT = Path(
    "models/detection/day12_detection_best.pt"
)

REPORT_PATH = Path(
    "reports/day13_detection_fastapi_streamlit_integration_summary.md"
)
SUMMARY_ARTIFACT = Path(
    "reports/artifacts/day13_detection_integration_summary.json"
)
README_PATH = Path("README.md")
README_START = "<!-- DAY13_DETECTION_FASTAPI_STREAMLIT_START -->"
README_END = "<!-- DAY13_DETECTION_FASTAPI_STREAMLIT_END -->"
BACKUP_ROOT = Path("reports/backups/day13_docs")

EXPECTED_ENDPOINT = "/api/v1/detection/predictions"
EXPECTED_ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_320_fpn"
EXPECTED_CHECKPOINT_EPOCH = 3
EXPECTED_VALIDATION_MAP_50 = 0.677418
EXPECTED_DEFAULT_SCORE_THRESHOLD = 0.5
EXPECTED_IOU_THRESHOLD = 0.5

REQUIRED_IMPLEMENTATION_FILES = (
    Path("src/api/detection_config.py"),
    Path("src/api/detection_inference_service.py"),
    Path("src/api/detection_model_loader.py"),
    Path("src/dashboard/detection_api_client.py"),
    Path("src/dashboard/detection_session_state.py"),
    Path("src/dashboard/detection_ui_helpers.py"),
    Path("src/dashboard/detection_page.py"),
    Path("src/dashboard/pages/2_Detection.py"),
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Day 13 Artifact does not exist: {path}."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(
            f"JSON top-level value must be an object: {path}."
        )
    return payload


def _mapping(
    value: object,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return dict(value)


def _boolean(
    value: object,
    *,
    name: str,
) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool.")
    return value


def _integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int.")
    if minimum is not None and value < minimum:
        raise ValueError(
            f"{name} must be greater than or equal to {minimum}."
        )
    return value


def _finite_float(
    value: object,
    *,
    name: str,
) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise TypeError(f"{name} must be numeric.")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _string(
    value: object,
    *,
    name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty str.")
    return value.strip()


def _require_close(
    actual: float,
    expected: float,
    *,
    name: str,
    tolerance: float = 1e-6,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(
            f"{name} must be {expected}, got {actual}."
        )


def _validate_test_counts(
    *,
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
) -> None:
    _integer(
        targeted_test_count,
        name="targeted_test_count",
        minimum=1,
    )
    _integer(
        regression_test_count,
        name="regression_test_count",
        minimum=targeted_test_count,
    )
    _integer(
        warning_count,
        name="warning_count",
        minimum=0,
    )


def collect_day13_evidence(
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Day 13 구현·실행 Artifact를 읽고 상호 일관성을 검증한다."""

    root = project_root.resolve()
    prerequisites = _read_json_object(
        root / PREREQUISITES_ARTIFACT
    )
    api_inspection = _read_json_object(
        root / API_INSPECTION_ARTIFACT
    )
    api_smoke = _read_json_object(
        root / API_SMOKE_ARTIFACT
    )
    dashboard_inspection = _read_json_object(
        root / DASHBOARD_INSPECTION_ARTIFACT
    )
    dashboard_validation = _read_json_object(
        root / DASHBOARD_VALIDATION_ARTIFACT
    )

    required_files = {
        path.as_posix(): (root / path).is_file()
        for path in REQUIRED_IMPLEMENTATION_FILES
    }
    if not all(required_files.values()):
        missing = [
            path
            for path, exists in required_files.items()
            if not exists
        ]
        raise FileNotFoundError(
            f"Day 13 implementation files are missing: {missing}."
        )

    checkpoint_path = root / BEST_CHECKPOINT
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Best Detection Checkpoint does not exist: {checkpoint_path}."
        )

    overlay_path = root / DASHBOARD_OVERLAY_FIGURE
    if not overlay_path.is_file():
        raise FileNotFoundError(
            "Dashboard Overlay Figure does not exist. "
            "Run scripts.run_day13_detection_dashboard_validation first."
        )

    prerequisites_validation = _mapping(
        prerequisites.get("validation"),
        name="prerequisites.validation",
    )
    if not _boolean(
        prerequisites_validation.get("all_passed"),
        name="prerequisites.validation.all_passed",
    ):
        raise ValueError("Day 13 prerequisites inspection did not pass.")

    api_inspection_passed = _boolean(
        api_inspection.get("validation_passed"),
        name="api_inspection.validation_passed",
    )
    if not api_inspection_passed:
        raise ValueError("Detection API Stage 1 inspection did not pass.")

    api_checkpoint = _mapping(
        api_inspection.get("checkpoint"),
        name="api_inspection.checkpoint",
    )
    checkpoint_epoch = _integer(
        api_checkpoint.get("epoch_number"),
        name="api_inspection.checkpoint.epoch_number",
        minimum=1,
    )
    if checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            "Day 13 API must use the Day 12 Best Checkpoint from Epoch 3."
        )

    checkpoint_metric = _finite_float(
        api_checkpoint.get("best_metric"),
        name="api_inspection.checkpoint.best_metric",
    )
    _require_close(
        checkpoint_metric,
        EXPECTED_VALIDATION_MAP_50,
        name="Best validation mAP@0.50",
    )

    api_policy = _mapping(
        api_inspection.get("api_policy"),
        name="api_inspection.api_policy",
    )
    if _string(
        api_policy.get("endpoint"),
        name="api_policy.endpoint",
    ) != EXPECTED_ENDPOINT:
        raise ValueError("Unexpected Detection Endpoint.")

    score_threshold = _finite_float(
        api_policy.get("default_score_threshold"),
        name="api_policy.default_score_threshold",
    )
    _require_close(
        score_threshold,
        EXPECTED_DEFAULT_SCORE_THRESHOLD,
        name="Default Score Threshold",
    )

    iou_threshold = _finite_float(
        api_policy.get("iou_threshold"),
        name="api_policy.iou_threshold",
    )
    _require_close(
        iou_threshold,
        EXPECTED_IOU_THRESHOLD,
        name="IoU Threshold",
    )

    if not _boolean(
        api_smoke.get("validation_passed"),
        name="api_smoke.validation_passed",
    ):
        raise ValueError("Actual Detection API Smoke Test did not pass.")

    smoke_request = _mapping(
        api_smoke.get("request"),
        name="api_smoke.request",
    )
    smoke_response = _mapping(
        api_smoke.get("response"),
        name="api_smoke.response",
    )
    smoke_payload = _mapping(
        smoke_response.get("payload"),
        name="api_smoke.response.payload",
    )

    if _integer(
        smoke_response.get("status_code"),
        name="api_smoke.response.status_code",
    ) != 200:
        raise ValueError("Detection API Smoke Test HTTP status must be 200.")

    if _string(
        smoke_request.get("endpoint"),
        name="api_smoke.request.endpoint",
    ) != EXPECTED_ENDPOINT:
        raise ValueError("Smoke Test used an unexpected endpoint.")

    smoke_threshold = _finite_float(
        smoke_request.get("score_threshold"),
        name="api_smoke.request.score_threshold",
    )
    _require_close(
        smoke_threshold,
        EXPECTED_DEFAULT_SCORE_THRESHOLD,
        name="Smoke Test Score Threshold",
    )

    if _integer(
        smoke_payload.get("checkpoint_epoch"),
        name="api_smoke.payload.checkpoint_epoch",
        minimum=1,
    ) != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("Smoke Test used an unexpected checkpoint epoch.")

    if _string(
        smoke_payload.get("architecture"),
        name="api_smoke.payload.architecture",
    ) != EXPECTED_ARCHITECTURE:
        raise ValueError("Smoke Test used an unexpected architecture.")

    smoke_detections = smoke_payload.get("detections")
    if not isinstance(smoke_detections, list):
        raise TypeError("api_smoke.payload.detections must be a list.")

    smoke_detection_count = _integer(
        smoke_payload.get("detection_count"),
        name="api_smoke.payload.detection_count",
        minimum=0,
    )
    if smoke_detection_count != len(smoke_detections):
        raise ValueError(
            "Smoke Test detection_count does not match detections length."
        )

    smoke_inference_time_ms = _finite_float(
        smoke_payload.get("inference_time_ms"),
        name="api_smoke.payload.inference_time_ms",
    )
    if smoke_inference_time_ms < 0.0:
        raise ValueError("Smoke Test inference time must be non-negative.")

    if not _boolean(
        dashboard_inspection.get("validation_passed"),
        name="dashboard_inspection.validation_passed",
    ):
        raise ValueError(
            "Detection Dashboard Stage 2 inspection did not pass."
        )

    dashboard_checks = _mapping(
        dashboard_inspection.get("checks"),
        name="dashboard_inspection.checks",
    )
    required_dashboard_checks = (
        "required_files_exist",
        "detection_endpoint_present",
        "default_threshold_is_0_5",
        "threshold_range_present",
        "api_client_only",
        "prediction_overlay_present",
        "prediction_table_present",
        "opencv_distinction_present",
        "ground_truth_warning_present",
    )
    for check_name in required_dashboard_checks:
        if not _boolean(
            dashboard_checks.get(check_name),
            name=f"dashboard_inspection.checks.{check_name}",
        ):
            raise ValueError(
                f"Dashboard inspection check failed: {check_name}."
            )

    forbidden_imports = dashboard_inspection.get(
        "forbidden_imports"
    )
    if not isinstance(forbidden_imports, list):
        raise TypeError(
            "dashboard_inspection.forbidden_imports must be a list."
        )
    if forbidden_imports:
        raise ValueError(
            "Streamlit Detection page must not import model runtime modules."
        )

    if not _boolean(
        dashboard_validation.get("validation_passed"),
        name="dashboard_validation.validation_passed",
    ):
        raise ValueError(
            "Actual Dashboard API Client validation did not pass."
        )

    if _string(
        dashboard_validation.get("endpoint"),
        name="dashboard_validation.endpoint",
    ) != EXPECTED_ENDPOINT:
        raise ValueError(
            "Dashboard Client validation used an unexpected endpoint."
        )

    dashboard_threshold = _finite_float(
        dashboard_validation.get("score_threshold"),
        name="dashboard_validation.score_threshold",
    )
    _require_close(
        dashboard_threshold,
        EXPECTED_DEFAULT_SCORE_THRESHOLD,
        name="Dashboard validation Score Threshold",
    )

    dashboard_checkpoint_epoch = _integer(
        dashboard_validation.get("checkpoint_epoch"),
        name="dashboard_validation.checkpoint_epoch",
        minimum=1,
    )
    if dashboard_checkpoint_epoch != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError(
            "Dashboard Client validation used an unexpected checkpoint epoch."
        )

    dashboard_metric = _finite_float(
        dashboard_validation.get("checkpoint_metric_value"),
        name="dashboard_validation.checkpoint_metric_value",
    )
    _require_close(
        dashboard_metric,
        EXPECTED_VALIDATION_MAP_50,
        name="Dashboard validation checkpoint metric",
    )

    dashboard_detection_count = _integer(
        dashboard_validation.get("detection_count"),
        name="dashboard_validation.detection_count",
        minimum=0,
    )
    dashboard_inference_time_ms = _finite_float(
        dashboard_validation.get("inference_time_ms"),
        name="dashboard_validation.inference_time_ms",
    )
    if dashboard_inference_time_ms < 0.0:
        raise ValueError(
            "Dashboard validation inference time must be non-negative."
        )

    return {
        "required_implementation_files": required_files,
        "checkpoint": {
            "path": BEST_CHECKPOINT.as_posix(),
            "size_bytes": checkpoint_path.stat().st_size,
            "epoch": checkpoint_epoch,
            "best_validation_map_50": checkpoint_metric,
        },
        "api": {
            "endpoint": EXPECTED_ENDPOINT,
            "architecture": EXPECTED_ARCHITECTURE,
            "default_score_threshold": score_threshold,
            "minimum_score_threshold": _finite_float(
                api_policy.get("minimum_score_threshold"),
                name="api_policy.minimum_score_threshold",
            ),
            "maximum_score_threshold": _finite_float(
                api_policy.get("maximum_score_threshold"),
                name="api_policy.maximum_score_threshold",
            ),
            "iou_threshold": iou_threshold,
            "smoke_image_path": _string(
                smoke_request.get("image_path"),
                name="api_smoke.request.image_path",
            ),
            "smoke_detection_count": smoke_detection_count,
            "smoke_inference_time_ms": smoke_inference_time_ms,
        },
        "dashboard": {
            "api_client_only": True,
            "overlay_figure": DASHBOARD_OVERLAY_FIGURE.as_posix(),
            "validation_image_path": _string(
                dashboard_validation.get("image_path"),
                name="dashboard_validation.image_path",
            ),
            "detection_count": dashboard_detection_count,
            "inference_time_ms": dashboard_inference_time_ms,
            "checkpoint_epoch": dashboard_checkpoint_epoch,
            "forbidden_imports": forbidden_imports,
        },
        "artifacts": {
            "prerequisites": PREREQUISITES_ARTIFACT.as_posix(),
            "api_inspection": API_INSPECTION_ARTIFACT.as_posix(),
            "api_smoke": API_SMOKE_ARTIFACT.as_posix(),
            "dashboard_inspection": (
                DASHBOARD_INSPECTION_ARTIFACT.as_posix()
            ),
            "dashboard_validation": (
                DASHBOARD_VALIDATION_ARTIFACT.as_posix()
            ),
        },
    }


def _manual_browser_status_text(
    status: str,
) -> str:
    if status == "pass":
        return (
            "브라우저 수동 시각 검증도 통과한 것으로 기록했다. "
            "원본·Overlay·Prediction Table·Threshold UI와 기존 "
            "Classification 페이지를 직접 확인했다."
        )
    if status == "not_recorded":
        return (
            "브라우저 수동 시각 검증 결과는 이 보고서에 별도로 기록하지 않았다. "
            "완료 사실은 자동 HTTP Client·Overlay Artifact·정적 구조 검증 "
            "범위로 한정한다."
        )
    raise ValueError(
        "manual_browser_check_status must be pass or not_recorded."
    )


def render_day13_report(
    *,
    evidence: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
    manual_browser_check_status: str,
) -> str:
    """검증된 Day 13 Evidence를 상세 Markdown 보고서로 변환한다."""

    _validate_test_counts(
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    data = _mapping(evidence, name="evidence")
    checkpoint = _mapping(
        data.get("checkpoint"),
        name="evidence.checkpoint",
    )
    api = _mapping(
        data.get("api"),
        name="evidence.api",
    )
    dashboard = _mapping(
        data.get("dashboard"),
        name="evidence.dashboard",
    )
    artifacts = _mapping(
        data.get("artifacts"),
        name="evidence.artifacts",
    )

    browser_text = _manual_browser_status_text(
        manual_browser_check_status
    )

    return f"""# Day 13 — Detection FastAPI and Streamlit Integration

## 1. 목표

Day 12에서 선택을 끝낸 Faster R-CNN Best Checkpoint를 기존
Classification 시스템과 충돌하지 않는 Detection API로 연결하고,
Streamlit이 Checkpoint를 직접 로딩하지 않은 채 FastAPI Client를 통해
Class·Score·원본 이미지 좌표 Bounding Box를 표시하도록 구현했다.

Day 13은 최종 Portfolio·Interview·전체 Architecture 정리 단계가 아니다.
해당 작업은 Day 14 범위로 남긴다.

## 2. 최종 호출 흐름

```text
Browser
→ Streamlit Detection Page
→ DetectionDashboardApiClient
→ POST {api["endpoint"]}
→ FastAPI Upload Validation
→ DetectionInferenceService
→ Day 12 Best Checkpoint
→ Detection Response JSON
→ Prediction Overlay·Table
```

기존 Classification 흐름은 유지한다.

```text
POST /api/v1/predictions
→ ResNet18 Classification

POST {api["endpoint"]}
→ Faster R-CNN Object Detection
```

## 3. Detection Model Service

| 항목 | 결과 |
| --- | --- |
| Architecture | `{api["architecture"]}` |
| Device | CPU |
| Checkpoint | `{checkpoint["path"]}` |
| Checkpoint Epoch | {checkpoint["epoch"]} |
| Best Validation mAP@0.50 | {float(checkpoint["best_validation_map_50"]):.6f} |
| Checkpoint Size | {int(checkpoint["size_bytes"]):,} bytes |
| Startup Policy | FastAPI Lifespan 1회 로딩 |
| Inference Context | `torch.inference_mode()` |
| CPU Forward Safety | Process 내부 Lock |
| Network Weight Download | 사용하지 않음 |

Checkpoint 내부 Epoch Index 2는 사람이 읽는 Epoch 3을 의미한다.
Test 결과를 사용해 Checkpoint를 다시 선택하지 않았다.

## 4. Detection API

```text
POST {api["endpoint"]}
```

정상 응답은 다음 정보를 제공한다.

```text
Class ID
Class Name
Score
Bounding Box (xmin, ymin, xmax, ymax)
Detection Count
Score Threshold
IoU Threshold
Model·Checkpoint Metadata
원본 이미지 Metadata
Inference Time
```

Threshold 정책:

| 항목 | 값 |
| --- | ---: |
| 기본 Score Threshold | {float(api["default_score_threshold"]):.2f} |
| 최소 Score Threshold | {float(api["minimum_score_threshold"]):.2f} |
| 최대 Score Threshold | {float(api["maximum_score_threshold"]):.2f} |
| IoU Threshold Metadata | {float(api["iou_threshold"]):.2f} |

Dashboard Slider는 Prediction 탐색용이다. Day 12 공식 Test 지표는
Score Threshold 0.5 결과로 유지하며 `crazing` Recall을 이유로 기본값을
임의로 낮추지 않았다.

## 5. 입력·출력 방어

기존 업로드 이미지 검증을 Detection에서도 재사용했다.

```text
빈 파일
JPEG·PNG 확장자
MIME Type
확장자·MIME·실제 Decode 형식 일치
깨진 이미지
업로드 Byte 제한
Pixel 수 제한
RGB 변환
```

Detection Service는 모델 출력에 대해 다음을 추가 검증한다.

```text
boxes=[N,4]
labels=[N]
scores=[N]
개수 일치
NaN·inf 차단
Label 1~6
Class Mapping 일치
Score 0~1
Threshold 적용
xmin < xmax
ymin < ymax
원본 이미지 좌표 범위
Score 내림차순 정렬
```

Torchvision이 원본 입력 크기로 복원한 Box를 사용하며, API에서
NEU-DET의 200×200 좌표로 임의 변환하지 않는다.

## 6. 실제 Detection API Smoke Test

| 항목 | 결과 |
| --- | --- |
| HTTP Status | 200 |
| Endpoint | `{api["endpoint"]}` |
| Score Threshold | {float(api["default_score_threshold"]):.2f} |
| Detection Count | {int(api["smoke_detection_count"])} |
| Inference Time | {float(api["smoke_inference_time_ms"]):.2f} ms |
| Checkpoint Epoch | {checkpoint["epoch"]} |
| Result | PASS |

검증 이미지는 `{api["smoke_image_path"]}`를 사용했다.

## 7. Streamlit Detection Page

기존 Classification `src/dashboard/app.py`를 유지하고,
Streamlit Multipage 구조에 Detection 전용 페이지를 추가했다.

```text
src/dashboard/pages/2_Detection.py
src/dashboard/detection_page.py
src/dashboard/detection_api_client.py
src/dashboard/detection_session_state.py
src/dashboard/detection_ui_helpers.py
```

주요 화면 기능:

```text
이미지 업로드
Score Threshold Slider
원본 이미지 Preview
Detection 실행
Detection Count
Inference Time
Checkpoint Epoch
원본·Prediction Overlay 좌우 표시
Prediction Table
Class·Score·Box 좌표
빈 Detection 상태
안전한 API 오류 메시지
```

Streamlit Detection Page는 `torch`, `torchvision`, `src.detection`,
Detection Checkpoint Loader를 직접 Import하지 않는다.

## 8. Overlay 정책

Box에는 `P1`, `P2` 같은 짧은 태그만 표시하고 Class·Score·좌표는
이미지 밖 Prediction Table에서 제공한다. 긴 Class 이름과 Box의 겹침을
방지하며 Score 상위 8개까지 Overlay에 표시한다.

자동 Dashboard Client 검증:

| 항목 | 결과 |
| --- | --- |
| Detection Count | {int(dashboard["detection_count"])} |
| Inference Time | {float(dashboard["inference_time_ms"]):.2f} ms |
| Checkpoint Epoch | {int(dashboard["checkpoint_epoch"])} |
| Overlay Figure | `{dashboard["overlay_figure"]}` |
| API Client Only | PASS |
| Result | PASS |

검증 이미지는 `{dashboard["validation_image_path"]}`를 사용했다.

{browser_text}

## 9. OpenCV와 Detection의 구분

```text
OpenCV
→ 이미지 명암·경계·형태 특성 보조 분석
→ Threshold·Morphology 기반 Contour 후보
→ Ground Truth나 Detection Prediction이 아님

Detection
→ 학습된 Faster R-CNN Prediction
→ Class·Score·Bounding Box 제공
→ Ground Truth가 아님
```

Detection 0개는 정상 제품 또는 Ground Truth 결함 없음이라는 뜻이 아니다.
현재 Score Threshold 이상으로 반환된 모델 Prediction이 없다는 뜻이다.

## 10. 오류 처리

API Client는 다음 오류를 안전한 Dashboard 메시지로 변환한다.

```text
Timeout
Connection Error
HTTP 4xx
HTTP 5xx
잘못된 JSON
Schema 누락
잘못된 Class Mapping
범위 밖 Box
NaN·inf
Detection Model Not Ready
```

서버 내부 경로·Checkpoint 경로·Stack Trace는 Dashboard에 노출하지 않는다.

## 11. 검증 결과

| 검증 | 결과 |
| --- | ---: |
| Day 13 Targeted Tests | {targeted_test_count} passed |
| Full Regression Tests | {regression_test_count} passed |
| Warnings | {warning_count} |
| API Prerequisites | PASS |
| Best Checkpoint Inspection | PASS |
| Actual Detection API Smoke Test | PASS |
| Dashboard Static Inspection | PASS |
| Dashboard API Client·Overlay Validation | PASS |

기존 Warning이 있다면 Day 7부터 유지된 Starlette `TestClient`와
`httpx` 관련 기술부채이며 Day 13에서 Dependency를 무리하게 변경하지 않았다.

## 12. 생성 Artifact

```text
{artifacts["prerequisites"]}
{artifacts["api_inspection"]}
{artifacts["api_smoke"]}
{artifacts["dashboard_inspection"]}
{artifacts["dashboard_validation"]}
{dashboard["overlay_figure"]}
{SUMMARY_ARTIFACT.as_posix()}
```

## 13. Day 14 연결 지점

Day 14에서는 다음을 진행한다.

```text
최종 기능 통합 점검
README 전체 구조 정리
최종 Architecture Diagram
Portfolio 문구
Interview 답변
최종 실행·회귀 검증
```

Day 13 보고서에서는 위 작업을 완료했다고 표현하지 않는다.
"""


def render_readme_section(
    *,
    evidence: Mapping[str, Any],
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
    manual_browser_check_status: str,
) -> str:
    """README에 들어갈 간결한 Day 13 요약 Section."""

    _validate_test_counts(
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    data = _mapping(evidence, name="evidence")
    checkpoint = _mapping(
        data.get("checkpoint"),
        name="evidence.checkpoint",
    )
    api = _mapping(
        data.get("api"),
        name="evidence.api",
    )
    dashboard = _mapping(
        data.get("dashboard"),
        name="evidence.dashboard",
    )

    manual_line = (
        "- Browser Visual Check: PASS"
        if manual_browser_check_status == "pass"
        else "- Browser Visual Check: not recorded separately"
    )

    return f"""{README_START}
## Day 13 — Detection FastAPI and Streamlit Integration

Day 12의 Faster R-CNN Best Checkpoint를 기존 Classification과 독립된
Detection Endpoint와 Streamlit Multipage 화면으로 연결했다.

```text
POST {api["endpoint"]}
```

핵심 구현:

```text
FastAPI Lifespan에서 Detection 모델 1회 로딩
CPU map_location·eval()·torch.inference_mode()
Process 내부 Forward Lock
Score Threshold 0.05~0.95, 기본값 0.5
원본 이미지 좌표 Bounding Box 반환
Class ID·Class Name·Score·Inference Time
Detection API Client
Prediction Overlay·Table
빈 Detection·Timeout·4xx·5xx·잘못된 JSON 처리
```

검증 결과:

```text
- Checkpoint Epoch: {checkpoint["epoch"]}
- Best Validation mAP@0.50: {float(checkpoint["best_validation_map_50"]):.6f}
- Actual Detection API Smoke Test: PASS
- Dashboard API Client·Overlay Validation: PASS
- Streamlit Direct Model Loading: 없음
{manual_line}
- Day 13 Tests: {targeted_test_count} passed
- Full Regression: {regression_test_count} passed
- Warnings: {warning_count}
```

OpenCV의 Threshold·Morphology 기반 Contour 후보는 Detection Prediction이나
Ground Truth가 아니다. Detection 0개도 정상 판정이 아니라 현재 Threshold 이상
Prediction이 없다는 의미다.

상세 보고서:

```text
{REPORT_PATH.as_posix()}
```

Day 14의 최종 README·Architecture·Portfolio·Interview 정리는 아직 수행하지 않았다.
{README_END}"""


def update_marker_block(
    original_text: str,
    section: str,
) -> str:
    """Day 13 Marker Block을 중복 없이 추가하거나 교체한다."""

    start_count = original_text.count(README_START)
    end_count = original_text.count(README_END)

    if start_count == 0 and end_count == 0:
        prefix = original_text.rstrip()
        if prefix:
            return f"{prefix}\n\n{section}\n"
        return f"{section}\n"

    if start_count != 1 or end_count != 1:
        raise ValueError(
            "README must contain either zero Day 13 markers or exactly one pair."
        )

    start_index = original_text.index(README_START)
    end_index = (
        original_text.index(
            README_END,
            start_index,
        )
        + len(README_END)
    )

    if start_index >= end_index:
        raise ValueError("README Day 13 marker order is invalid.")

    return (
        original_text[:start_index]
        + section
        + original_text[end_index:]
    )


def _write_text_atomically(
    path: Path,
    text: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        f".{path.name}.tmp"
    )
    try:
        temporary.write_text(
            text,
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_once(
    source: Path,
    destination: Path,
) -> None:
    if not source.is_file() or destination.exists():
        return

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    destination.write_bytes(
        source.read_bytes()
    )


def create_day13_docs(
    *,
    project_root: Path,
    targeted_test_count: int,
    regression_test_count: int,
    warning_count: int,
    manual_browser_check_status: str = "not_recorded",
) -> tuple[Path, Path, Path]:
    """Day 13 보고서·요약 Artifact·README Marker Section을 생성한다."""

    _validate_test_counts(
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
    )
    if manual_browser_check_status not in {
        "pass",
        "not_recorded",
    }:
        raise ValueError(
            "manual_browser_check_status must be pass or not_recorded."
        )

    root = project_root.resolve()
    evidence = collect_day13_evidence(
        project_root=root
    )

    report = render_day13_report(
        evidence=evidence,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
        manual_browser_check_status=manual_browser_check_status,
    )
    readme_section = render_readme_section(
        evidence=evidence,
        targeted_test_count=targeted_test_count,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
        manual_browser_check_status=manual_browser_check_status,
    )

    summary = {
        "stage": "day13_detection_fastapi_streamlit_integration",
        "completion_scope": {
            "detection_api": True,
            "actual_best_checkpoint_smoke_test": True,
            "detection_dashboard_api_client": True,
            "prediction_overlay": True,
            "prediction_table": True,
            "opencv_detection_distinction": True,
            "manual_browser_check_status": manual_browser_check_status,
            "day14_final_integration_completed": False,
            "portfolio_completed": False,
            "interview_answers_completed": False,
        },
        "tests": {
            "targeted_test_count": targeted_test_count,
            "regression_test_count": regression_test_count,
            "warning_count": warning_count,
        },
        "evidence": evidence,
    }

    report_path = root / REPORT_PATH
    summary_path = root / SUMMARY_ARTIFACT
    readme_path = root / README_PATH
    backup_root = root / BACKUP_ROOT

    _backup_once(
        readme_path,
        backup_root / "README.md.before_day13_docs",
    )
    _backup_once(
        report_path,
        backup_root / f"{REPORT_PATH.name}.before_day13_docs",
    )
    _backup_once(
        summary_path,
        backup_root / f"{SUMMARY_ARTIFACT.name}.before_day13_docs",
    )

    original_readme = (
        readme_path.read_text(encoding="utf-8")
        if readme_path.exists()
        else ""
    )
    updated_readme = update_marker_block(
        original_readme,
        readme_section,
    )

    _write_text_atomically(
        report_path,
        report,
    )
    _write_text_atomically(
        summary_path,
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
    )
    _write_text_atomically(
        readme_path,
        updated_readme,
    )

    return (
        report_path,
        summary_path,
        readme_path,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Day 13 Detection FastAPI and Streamlit "
            "integration report and README section."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--targeted-test-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--regression-test-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--warning-count",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--manual-browser-check-status",
        choices=(
            "pass",
            "not_recorded",
        ),
        default="not_recorded",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    report_path, summary_path, readme_path = (
        create_day13_docs(
            project_root=args.project_root,
            targeted_test_count=args.targeted_test_count,
            regression_test_count=args.regression_test_count,
            warning_count=args.warning_count,
            manual_browser_check_status=(
                args.manual_browser_check_status
            ),
        )
    )

    print("[PASS] Day 13 report created")
    print("[PASS] Day 13 summary Artifact created")
    print("[PASS] README Day 13 section added or updated")
    print(f"[REPORT] {report_path}")
    print(f"[ARTIFACT] {summary_path}")
    print(f"[README] {readme_path}")
    print(
        "[TESTS] "
        f"Day 13={args.targeted_test_count} / "
        f"Full regression={args.regression_test_count} / "
        f"Warnings={args.warning_count}"
    )
    print(
        "[BROWSER CHECK] "
        f"{args.manual_browser_check_status}"
    )


if __name__ == "__main__":
    main()
