"""Create Day 14 final-integration README, portfolio, and interview documents.

This script validates the existing Day 4, Day 12, Day 13, and Day 14 evidence
artifacts before writing any final documentation. It does not train models,
run inference, or modify application source/checkpoints.

Outputs:
- reports/day14_final_integration_portfolio_interview_summary.md
- reports/artifacts/day14_final_integration_summary.json
- README.md Day 14 marker block
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KO = "제조 비전 결함 분석 시스템"

README_PATH = Path("README.md")
REPORT_PATH = Path(
    "reports/day14_final_integration_portfolio_interview_summary.md"
)
SUMMARY_PATH = Path(
    "reports/artifacts/day14_final_integration_summary.json"
)
BACKUP_PATH = Path(
    "reports/artifacts/backups/README.md.before_day14_final_docs"
)

CLASSIFICATION_EVALUATION_PATH = Path(
    "reports/artifacts/day4_resnet18_test_evaluation.json"
)
DETECTION_EVALUATION_PATH = Path(
    "reports/artifacts/day12_detection_evaluation.json"
)
DETECTION_FAILURE_PATH = Path(
    "reports/artifacts/day12_detection_failure_analysis.json"
)
DAY13_SUMMARY_PATH = Path(
    "reports/artifacts/day13_detection_integration_summary.json"
)
DAY14_EVIDENCE_PATH = Path(
    "reports/artifacts/day14_final_integration_evidence.json"
)
DAY14_INSPECTION_PATH = Path(
    "reports/artifacts/day14_final_integration_prerequisites_inspection.json"
)
CONTEXT_REBUILD_PATH = Path(
    "reports/artifacts/day14_day13_api_core_context_rebuild.json"
)

README_START_MARKER = "<!-- DAY14_FINAL_INTEGRATION_START -->"
README_END_MARKER = "<!-- DAY14_FINAL_INTEGRATION_END -->"

CLASSIFICATION_CHECKPOINT = Path(
    "models/checkpoints/resnet18_transfer_best.pt"
)
DETECTION_CHECKPOINT = Path(
    "models/detection/day12_detection_best.pt"
)
DETECTION_SPLIT_MANIFEST = Path(
    "data/processed/neu_det/splits.json"
)

EXPECTED_ENDPOINTS = (
    "GET /api/v1/health",
    "POST /api/v1/predictions",
    "POST /api/v1/detection/predictions",
)

# These are the verified project results that the corresponding JSON artifacts
# must contain. They are not estimates and are checked before documentation.
EXPECTED_NUMERIC_EVIDENCE: Mapping[Path, tuple[float, ...]] = {
    CLASSIFICATION_EVALUATION_PATH: (
        0.9734,
        0.9717,
        0.9868,
        0.9792,
        249.0,
        13.0,
        6.0,
        447.0,
    ),
    DETECTION_EVALUATION_PATH: (
        0.812950,
        0.526807,
        0.639321,
        0.752338,
        0.707726,
        0.310533,
        226.0,
        52.0,
        203.0,
        0.841026,
        0.888495,
        0.025316,
        0.048780,
        0.522723,
    ),
    DETECTION_FAILURE_PATH: (
        182.0,
        129.0,
        229.0,
        140.0,
        37.0,
        25.0,
        23.0,
        3.0,
        1.0,
    ),
    DAY13_SUMMARY_PATH: (
        92.0,
        1668.0,
        1.0,
        3.0,
    ),
}

FORBIDDEN_OVERCLAIMS = (
    "실제 생산 환경에서 검증 완료",
    "산업 현장 배포 완료",
    "실시간 생산 시스템 구축",
    "COCO 공식 mAP와 완전히 동일",
    "OpenCV Contour가 실제 결함 위치",
    "Detection Prediction이 Ground Truth",
)


class Day14DocumentationError(RuntimeError):
    """Final documentation cannot be created safely."""


@dataclass(frozen=True)
class ValidationResult:
    targeted_count: int
    regression_count: int
    warning_count: int
    runtime_seconds: float


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Day14DocumentationError(
            f"UTF-8로 읽을 수 없는 파일입니다: {path}"
        ) from exc


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise Day14DocumentationError(
            f"JSON 형식이 잘못됐습니다: {path} "
            f"(line={exc.lineno}, column={exc.colno})"
        ) from exc

    if not isinstance(payload, dict):
        raise Day14DocumentationError(
            f"JSON 최상위 값이 Object가 아닙니다: {path}"
        )
    return payload


def _walk_scalars(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_scalars(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_scalars(child)
    else:
        yield value


def _numeric_values(payload: Any) -> list[float]:
    values: list[float] = []
    for scalar in _walk_scalars(payload):
        if isinstance(scalar, bool):
            continue
        if isinstance(scalar, (int, float)) and math.isfinite(float(scalar)):
            values.append(float(scalar))
    return values


def _normalized_numeric_candidates(
    value: float,
    expected: float,
) -> tuple[float, ...]:
    """Return directly comparable ratio/percentage representations.

    Evaluation artifacts may store a metric as either a ratio such as
    ``0.973426...`` or a percentage such as ``97.3426...``. README values are
    intentionally rounded for presentation, so validation compares both
    representations without changing integer counts.
    """

    candidates = [float(value)]

    if 0.0 <= expected <= 1.0 and 1.0 < value <= 100.0:
        candidates.append(value / 100.0)
    elif 1.0 < expected <= 100.0 and 0.0 <= value <= 1.0:
        candidates.append(value * 100.0)

    return tuple(candidates)


def _contains_numeric(
    values: Sequence[float],
    expected: float,
    *,
    absolute_tolerance: float = 5e-5,
) -> bool:
    """Match exact counts and rounded ratio/percentage metrics safely."""

    for value in values:
        for candidate in _normalized_numeric_candidates(
            float(value),
            float(expected),
        ):
            if math.isclose(
                candidate,
                expected,
                rel_tol=1e-6,
                abs_tol=absolute_tolerance,
            ):
                return True
    return False


def validate_expected_numeric_evidence(
    *,
    root: Path,
    expected: Mapping[Path, Sequence[float]] = EXPECTED_NUMERIC_EVIDENCE,
) -> dict[str, Any]:
    """Validate that authoritative artifacts contain every documented result."""

    rows: list[dict[str, Any]] = []

    for relative_path, expected_values in expected.items():
        path = root / relative_path
        if not path.is_file():
            raise Day14DocumentationError(
                f"필수 근거 Artifact가 없습니다: {relative_path.as_posix()}"
            )

        payload = _read_json_object(path)
        actual_values = _numeric_values(payload)
        missing = [
            value
            for value in expected_values
            if not _contains_numeric(actual_values, float(value))
        ]

        if missing:
            raise Day14DocumentationError(
                "문서에 사용할 검증 수치가 Artifact와 일치하지 않습니다. "
                f"파일={relative_path.as_posix()}, 누락={missing}"
            )

        rows.append(
            {
                "path": relative_path.as_posix(),
                "expected_numeric_count": len(expected_values),
                "all_values_found": True,
            }
        )

    return {
        "status": "PASS",
        "artifact_count": len(rows),
        "artifacts": rows,
    }


def validate_repository_evidence(root: Path) -> dict[str, Any]:
    """Validate paths, evidence status, endpoint status, and browser truth."""

    required_files = (
        README_PATH,
        CLASSIFICATION_EVALUATION_PATH,
        DETECTION_EVALUATION_PATH,
        DETECTION_FAILURE_PATH,
        DAY13_SUMMARY_PATH,
        DAY14_EVIDENCE_PATH,
        DAY14_INSPECTION_PATH,
        CONTEXT_REBUILD_PATH,
        CLASSIFICATION_CHECKPOINT,
        DETECTION_CHECKPOINT,
        DETECTION_SPLIT_MANIFEST,
    )

    missing = [
        path.as_posix()
        for path in required_files
        if not (root / path).is_file()
    ]
    if missing:
        raise Day14DocumentationError(
            "필수 파일이 없습니다: " + ", ".join(missing)
        )

    evidence = _read_json_object(root / DAY14_EVIDENCE_PATH)
    inspection = _read_json_object(root / DAY14_INSPECTION_PATH)
    day13 = _read_json_object(root / DAY13_SUMMARY_PATH)
    rebuild = _read_json_object(root / CONTEXT_REBUILD_PATH)

    evidence_status = (
        evidence.get("status", {}).get("overall")
        if isinstance(evidence.get("status"), dict)
        else None
    )
    inspection_status = (
        inspection.get("status", {}).get("overall")
        if isinstance(inspection.get("status"), dict)
        else None
    )

    if evidence_status != "PASS":
        raise Day14DocumentationError(
            "Day 14 Evidence 상태가 PASS가 아닙니다."
        )
    if inspection_status != "PASS":
        raise Day14DocumentationError(
            "Day 14 사전 점검 상태가 PASS가 아닙니다."
        )
    if rebuild.get("status") != "PASS":
        raise Day14DocumentationError(
            "Day 13 API Context UTF-8 복구 상태가 PASS가 아닙니다."
        )
    if rebuild.get("rebuilt", {}).get("mojibake_tokens") != []:
        raise Day14DocumentationError(
            "복구된 API Context에 문자 깨짐 토큰이 남아 있습니다."
        )

    endpoint_status = evidence.get("fastapi", {}).get(
        "expected_status",
        {},
    )
    missing_endpoints = [
        endpoint
        for endpoint in EXPECTED_ENDPOINTS
        if endpoint_status.get(endpoint) is not True
    ]
    if missing_endpoints:
        raise Day14DocumentationError(
            "검증되지 않은 Endpoint가 있습니다: "
            + ", ".join(missing_endpoints)
        )

    important_paths = evidence.get("important_paths", {})
    for name in (
        "classification_checkpoint",
        "detection_checkpoint_best",
        "detection_split_manifest",
    ):
        row = important_paths.get(name, {})
        if row.get("found_any") is not True:
            raise Day14DocumentationError(
                f"Evidence에서 주요 경로를 확인하지 못했습니다: {name}"
            )

    day13_scalars = list(_walk_scalars(day13))
    browser_statuses = [
        scalar
        for scalar in day13_scalars
        if isinstance(scalar, str)
        and scalar in {"not_recorded", "passed", "failed"}
    ]
    manual_browser_status = (
        "not_recorded"
        if "not_recorded" in browser_statuses
        else (
            browser_statuses[0]
            if browser_statuses
            else "not_recorded"
        )
    )

    numeric_validation = validate_expected_numeric_evidence(root=root)

    return {
        "status": "PASS",
        "evidence_status": evidence_status,
        "inspection_status": inspection_status,
        "context_rebuild_status": rebuild.get("status"),
        "expected_endpoints": {
            endpoint: True
            for endpoint in EXPECTED_ENDPOINTS
        },
        "manual_browser_check_status": manual_browser_status,
        "numeric_validation": numeric_validation,
        "checkpoints": {
            "classification": CLASSIFICATION_CHECKPOINT.as_posix(),
            "detection": DETECTION_CHECKPOINT.as_posix(),
        },
    }


def build_architecture_mermaid() -> str:
    return """```mermaid
flowchart LR
    U[사용자 / Browser]

    subgraph Dashboard["Streamlit Dashboard"]
        CP[Classification Page]
        DP[Detection Page]
    end

    subgraph API["FastAPI Inference Layer"]
        H[GET /api/v1/health]
        CAPI[POST /api/v1/predictions]
        DAPI[POST /api/v1/detection/predictions]
        V[Image Validation]
        LIFE[Lifespan Model Loading]
        LOCK[Inference Lock]
    end

    subgraph Classification["Classification"]
        CTRANS[Test Transform]
        CM[ResNet18 Transfer]
        COUT[NORMAL / DEFECT]
    end

    subgraph Detection["Object Detection"]
        DTRANS[Tensor Conversion]
        DM[Faster R-CNN MobileNetV3 320 FPN]
        DOUT[Class / Score / Bounding Box]
    end

    subgraph OpenCV["OpenCV Auxiliary Analysis"]
        OP[Brightness / Histogram / Edge]
        OM[Threshold / Morphology]
        OC[Contour Candidates]
    end

    U --> CP
    U --> DP
    CP --> CAPI
    DP --> DAPI
    CAPI --> V
    DAPI --> V
    LIFE --> CM
    LIFE --> DM
    V --> LOCK
    LOCK --> CTRANS --> CM --> COUT
    LOCK --> DTRANS --> DM --> DOUT
    U --> OP --> OM --> OC

    COUT --> CP
    DOUT --> DP

    CCHK[ResNet18 Best Checkpoint] --> CM
    DCHK[Detection Best Checkpoint] --> DM

    NOTE["OpenCV Contour는 Ground Truth나 Detection Prediction이 아닌 후보 영역"]
    OC -. 역할 구분 .-> NOTE
```"""


def build_user_flow_mermaid() -> str:
    return """```mermaid
sequenceDiagram
    actor User
    participant ST as Streamlit
    participant API as FastAPI
    participant Service as Inference Service
    participant Model as Loaded Checkpoint Model

    User->>ST: 이미지 업로드 및 분석 요청
    ST->>API: multipart/form-data HTTP 요청
    API->>API: 확장자·MIME·Decode·크기·Pixel 검증
    API->>Service: 검증된 RGB Image 전달
    Service->>Model: Process-local Inference Lock 내부 추론
    Model-->>Service: Classification 또는 Detection 결과
    Service-->>API: 응답 Schema 구성
    API-->>ST: JSON 응답
    ST-->>User: Label·확률 또는 Bounding Box Overlay·표 표시

    Note over ST,Model: Streamlit은 Model·Checkpoint를 직접 로드하지 않음
    Note over API,Model: Model은 FastAPI Lifespan에서 한 번 로드됨
```"""


def build_performance_table() -> str:
    return """| Pipeline | Model / Method | Test Result |
|---|---|---|
| Classification | ResNet18 Transfer | Accuracy 97.34%, Precision 97.17%, Recall 98.68%, F1 97.92% |
| Classification Confusion Matrix | Binary NORMAL/DEFECT | TN 249, FP 13, FN 6, TP 447 |
| Detection | Faster R-CNN MobileNetV3 Large 320 FPN | Precision 0.812950, Recall 0.526807, F1 0.639321 |
| Detection Localization | IoU 0.50 Evaluation | Mean matched IoU 0.752338, mAP@0.50 0.707726 |
| Detection Project AP | Project all-point interpolation | AP 0.50:0.95 0.310533 |
| Detection Best Class | patches | F1 0.841026, AP@0.50 0.888495 |
| Detection Weak Class | crazing | Recall 0.025316, F1 0.048780, AP@0.50 0.522723 |"""


def build_failure_table() -> str:
    return """| Failure Type | Count | Interpretation |
|---|---:|---|
| Low-confidence correct match | 140 | 위치와 Class는 맞지만 운영 Threshold보다 Score가 낮음 |
| False Negative | 37 | Ground Truth와 일치하는 Prediction이 없음 |
| Low IoU | 25 | Class는 맞을 수 있으나 위치 겹침이 기준 미달 |
| False Positive | 23 | 대응되는 Ground Truth가 없는 Prediction |
| Duplicate | 3 | 하나의 Ground Truth에 중복 Prediction |
| Wrong Class | 1 | 위치가 겹치지만 Class가 다름 |"""


def build_run_commands() -> str:
    return r"""### FastAPI

```powershell
.\.venv\Scripts\python.exe `
    -m uvicorn `
    src.api.app:app `
    --host 127.0.0.1 `
    --port 8000
```

### Streamlit

별도 PowerShell 창에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe `
    -m streamlit `
    run `
    .\src\dashboard\app.py
```

### Tests

```powershell
.\.venv\Scripts\python.exe `
    -m pytest `
    -q
```"""


def build_readme_section(
    *,
    test_result: ValidationResult,
    manual_browser_status: str,
) -> str:
    runtime = f"{test_result.runtime_seconds:.2f}"

    return f"""{README_START_MARKER}

## Day 14 — Final Integration, README, Portfolio, and Interview

**Manufacturing Vision Defect Analysis System**은 Day 1~13에서 구현한 **Classification·OpenCV·Object Detection·FastAPI·Streamlit**을 하나의 제조 비전 결함 분석 시스템으로 정리한 프로젝트입니다.

### Final System Scope

- **Classification**: Casting Product Image를 `NORMAL / DEFECT`로 분류
- **OpenCV**: 명암·Histogram·Edge·Threshold·Morphology 기반 보조 분석
- **Detection**: NEU-DET 6개 결함 Class의 위치·Score·Bounding Box 예측
- **FastAPI**: Classification과 Detection 추론 Endpoint 통합
- **Streamlit**: API Client 방식의 Classification·Detection Dashboard
- **Documentation**: Architecture·실행 방법·성능·Failure Analysis·Portfolio·면접 정리

### Architecture

{build_architecture_mermaid()}

### Final Performance

{build_performance_table()}

> Detection의 `AP 0.50:0.95`는 이 프로젝트가 구현한 all-point interpolation 기반 지표이며 공식 COCOeval 결과로 표현하지 않습니다.

### API

| Method | Endpoint | Role |
|---|---|---|
| GET | `/api/v1/health` | Classification·Detection Service 상태 |
| POST | `/api/v1/predictions` | NORMAL / DEFECT Classification |
| POST | `/api/v1/detection/predictions` | 결함 Class·Score·Bounding Box Detection |

### Validation

- Day 14 대상 테스트: **{test_result.targeted_count} passed**
- 전체 회귀 테스트: **{test_result.regression_count} passed**
- Warning: **{test_result.warning_count}**
- 전체 회귀 Runtime: **{runtime} seconds**
- Day 14 구조·문서 사전 점검: **PASS**
- FastAPI 예상 Endpoint: **3/3 PASS**
- Classification Checkpoint: `{CLASSIFICATION_CHECKPOINT.as_posix()}`
- Detection Checkpoint: `{DETECTION_CHECKPOINT.as_posix()}`
- Day 13 수동 Browser 확인: **{manual_browser_status}**

수동 Browser 확인은 자동 HTTP·API Client·Overlay 검증과 구분하며, 실제 기록이 없는 상태를 완료로 변경하지 않습니다.

### Final Documents

- [Day 14 Final Integration·Portfolio·Interview Report]({REPORT_PATH.as_posix()})
- [Day 14 Final Integration Summary]({SUMMARY_PATH.as_posix()})
- [Day 14 Architecture Plan](reports/day14_final_integration_readme_architecture_plan.md)

{README_END_MARKER}"""


def build_report(
    *,
    test_result: ValidationResult,
    manual_browser_status: str,
) -> str:
    runtime = f"{test_result.runtime_seconds:.2f}"

    return f"""# Day 14 — Final Integration, README, Portfolio, and Interview Summary

## 1. Executive Summary

**{PROJECT_NAME}({PROJECT_NAME_KO})**은 제조 이미지에 대해 전체 이미지 Classification, OpenCV 보조 분석, Object Detection을 제공하고 이를 FastAPI와 Streamlit으로 연결한 학습·제출용 프로젝트입니다.

Day 14에서는 새 모델을 추가하지 않고 Day 1~13 결과를 하나의 시스템 관점으로 통합했습니다. README, Architecture, 실행 방법, 성능, Failure Analysis, Portfolio 설명, 면접 답변을 실제 Artifact와 테스트 근거에 맞춰 정리했습니다.

## 2. Problem Situation

제조 이미지 검사에서 단일 결과만으로는 다음 질문을 모두 해결하기 어렵습니다.

1. 이미지 전체가 정상인지 불량인지
2. 밝기·경계·형태 특성은 어떤지
3. 어떤 결함이 어느 위치에 있는지
4. 모델 결과를 API와 Dashboard에서 어떻게 일관되게 제공할지

이 프로젝트는 세 분석 Pipeline을 역할별로 분리하고 하나의 서비스 구조로 연결했습니다.

## 3. Solution Architecture

{build_architecture_mermaid()}

### 호출 관계

1. 사용자가 Streamlit에 이미지를 업로드합니다.
2. Streamlit은 Model이나 Checkpoint를 직접 읽지 않고 FastAPI로 요청합니다.
3. FastAPI는 파일 확장자, MIME, Decode, 이미지 크기와 Pixel을 검증합니다.
4. Lifespan에서 한 번 로드한 Service가 Process-local Lock 내부에서 추론합니다.
5. FastAPI가 검증된 Response Schema를 반환합니다.
6. Streamlit이 Classification 결과 또는 Detection Overlay·표를 표시합니다.

## 4. End-to-End User Flow

{build_user_flow_mermaid()}

## 5. Pipeline Boundaries

### Classification

- Dataset: Casting Product Image Data
- Label: `0=NORMAL`, `1=DEFECT`
- Model: ResNet18 Transfer Learning
- 역할: 이미지 전체의 정상·불량 판정
- 하지 않는 일: 결함 위치나 세부 결함 Class 반환

### OpenCV

- 밝기·표준편차·Histogram·Canny Edge·Threshold·Morphology 분석
- Contour는 Threshold·Morphology 기반 후보 영역
- Contour는 Ground Truth나 Detection Prediction이 아님
- 역할: 모델 결과를 대체하지 않는 보조 해석

### Object Detection

- Dataset: NEU Surface Defect Database
- Class: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- Model: Faster R-CNN MobileNetV3 Large 320 FPN
- 역할: 결함 Class·Score·Bounding Box 예측
- Detection 0개: 현재 Score Threshold 이상 Prediction이 없다는 의미

## 6. Dataset and Training Summary

### Classification

| Split | Samples |
|---|---:|
| Train | 5,306 |
| Validation | 1,327 |
| Test | 715 |

- Best Validation Epoch: 5
- Best Validation Accuracy: 97.0610%
- Best Validation Loss: 0.1579
- Checkpoint: `{CLASSIFICATION_CHECKPOINT.as_posix()}`

### Detection

| Split | Images | Boxes |
|---|---:|---:|
| Train | 1,440 | 3,335 |
| Validation | 178 | 425 |
| Test | 182 | 429 |
| Total | 1,800 | 4,189 |

- Best Checkpoint Epoch: 3
- Validation mAP@0.50: 0.677418
- Checkpoint: `{DETECTION_CHECKPOINT.as_posix()}`
- Split Manifest: `{DETECTION_SPLIT_MANIFEST.as_posix()}`

## 7. Final Performance

{build_performance_table()}

### Metric Boundary

- `mAP@0.50`은 IoU 0.50 기준 Class별 AP 평균입니다.
- `AP 0.50:0.95`는 프로젝트 내부 all-point interpolation 구현입니다.
- 공식 COCOeval 결과로 표현하지 않습니다.
- Classification과 Detection은 Dataset·Label·목적이 다르므로 직접 우열 비교하지 않습니다.

## 8. Detection Failure Analysis

- Test Images: 182
- Failure가 하나 이상 있는 이미지: 129
- 전체 Failure Event: 229

{build_failure_table()}

### 핵심 해석

- 가장 많은 유형은 Low-confidence correct match 140건입니다.
- Score Threshold가 False Negative처럼 보이는 결과에 큰 영향을 줍니다.
- `crazing`은 가늘고 갈라진 형태 때문에 Recall이 특히 낮았습니다.
- 단순 Threshold 하향은 Recall을 높일 수 있지만 False Positive 증가 위험이 있습니다.
- 개선 우선순위는 Data Augmentation, Class 균형, 작은 결함 보존, Class별 Threshold 검토입니다.

## 9. API and Dashboard

| Method | Endpoint | Output |
|---|---|---|
| GET | `/api/v1/health` | Service·Checkpoint 상태 |
| POST | `/api/v1/predictions` | Classification Label·Probability |
| POST | `/api/v1/detection/predictions` | Detection Class·Score·Bounding Box |

### 설계 결정

- FastAPI Lifespan에서 Classification·Detection Service를 한 번 로드
- CPU `map_location`과 Checkpoint Metadata 검증
- Process-local Inference Lock으로 동시 추론 안전성 확보
- Streamlit은 API Client 역할만 담당
- Dashboard가 Torch·Model Factory·Checkpoint를 직접 사용하지 않도록 경계 설정

## 10. Run Guide

{build_run_commands()}

## 11. Validation Result

| Verification | Result |
|---|---|
| Day 14 Targeted Tests | {test_result.targeted_count} passed |
| Full Regression | {test_result.regression_count} passed |
| Warning | {test_result.warning_count} |
| Full Regression Runtime | {runtime} seconds |
| Final Integration Prerequisites | PASS |
| Final Integration Evidence | PASS |
| FastAPI Expected Endpoints | 3/3 PASS |
| Day 13 API Core Context UTF-8 Rebuild | PASS |
| Manual Browser Check | {manual_browser_status} |

수동 Browser 상태는 자동 HTTP·API Client·Overlay 검증과 분리합니다. 실제 수동 기록이 없으므로 `not_recorded`를 유지합니다.

## 12. Key Design Decisions

### 문제 상황

Classification·OpenCV·Detection이 같은 이미지 분석 영역을 다루지만 역할과 출력이 달라 혼동될 수 있었습니다.

### 해결 방안과 고민

- 각 Pipeline의 Dataset·Label·Output을 분리했습니다.
- Streamlit에 Model 로딩 책임을 두지 않고 FastAPI에 추론 책임을 집중했습니다.
- Detection 평가는 Global Metric과 Class별 Metric, Failure Event를 함께 봤습니다.
- 공식 COCOeval이 아닌 내부 AP 구현은 명확하게 구분했습니다.
- 자동 검증과 수동 Browser 확인을 동일하게 취급하지 않았습니다.

### 적용

- Lifespan Model Loading
- Input Validation
- Inference Lock
- API Client Boundary
- Checkpoint Metadata Validation
- JSON Artifact 기반 문서 생성
- Regression Test 기반 완료 기준

### 효과와 의미

- 모델·API·Dashboard의 호출 관계를 설명할 수 있습니다.
- 성능 수치의 출처와 한계를 추적할 수 있습니다.
- Classification·Detection의 실패를 서로 다른 관점으로 분석할 수 있습니다.
- AI 보조 도구로 초안을 만들더라도 실제 실행·테스트·수정·문서화로 결과를 검증하는 개발 방식을 보여줍니다.

## 13. Portfolio Summary

### 프로젝트 설명

제조 이미지의 정상·불량 Classification, OpenCV 기반 이미지 특성 분석, 6개 철강 표면 결함 Object Detection을 구현하고 FastAPI·Streamlit으로 통합한 제조 비전 결함 분석 시스템입니다.

### 주요 구현

- PyTorch Dataset·DataLoader·Transform·Stratified Split
- CNN Baseline과 ResNet18 Transfer Learning
- Accuracy·Precision·Recall·F1·Confusion Matrix 평가
- 오분류 분석과 Grad-CAM
- OpenCV Histogram·Edge·Threshold·Morphology·Contour 분석
- Pascal VOC 기반 NEU-DET Dataset과 Faster R-CNN
- Detection mAP·IoU·Class별 AP·Failure Analysis
- FastAPI Lifespan·Input Validation·Inference Lock
- API Client 전용 Streamlit Dashboard
- Artifact·README·Regression Test 자동 검증

### 성과

- Classification Test F1: 97.92%
- Detection Test mAP@0.50: 0.707726
- Detection Mean matched IoU: 0.752338
- FastAPI 예상 Endpoint 3개 검증
- Day 14 최종 통합 사전 점검·근거 수집 PASS
- 전체 회귀 테스트: {test_result.regression_count} passed

## 14. Interview Guide

### Q1. 왜 Classification과 Detection을 모두 구현했나요?

Classification은 이미지 전체가 정상인지 불량인지 빠르게 판단하지만 위치와 결함 종류를 설명하지 못합니다. Detection은 결함 종류와 위치를 제공하지만 데이터 Annotation과 연산 비용이 더 큽니다. 두 문제를 분리해 구현하면서 제조 Vision 서비스에서 목적에 따라 모델을 선택하는 기준을 학습했습니다.

### Q2. OpenCV Contour와 Detection Bounding Box의 차이는 무엇인가요?

OpenCV Contour는 Threshold와 Morphology 결과에서 얻은 후보 영역입니다. 학습된 Class 의미가 없고 Ground Truth도 아닙니다. Detection Bounding Box는 Annotation을 이용해 학습한 모델의 Class·Score·위치 예측입니다.

### Q3. Streamlit에서 모델을 직접 로드하지 않은 이유는 무엇인가요?

UI와 추론 책임을 분리하기 위해서입니다. FastAPI가 Checkpoint·Transform·추론을 관리하고 Streamlit은 HTTP Client 역할만 담당하면 중복 로딩을 방지하고 API 단독 검증과 다른 Client 확장이 쉬워집니다.

### Q4. Detection Recall이 낮은 이유를 어떻게 분석했나요?

Global Metric만 보지 않고 229개 Failure Event를 유형별로 분해했습니다. Low-confidence correct match가 140건으로 가장 많았고, 특히 `crazing`의 Recall이 0.025316으로 낮았습니다. Threshold, 작은 결함 표현, Class 불균형과 형태 특성을 개선 후보로 정리했습니다.

### Q5. Detection AP 0.50:0.95를 COCO mAP라고 부르지 않는 이유는 무엇인가요?

프로젝트 구현은 여러 IoU Threshold에서 계산한 all-point interpolation AP 평균입니다. COCOeval의 세부 설정과 완전히 동일하다고 검증하지 않았으므로 `Project AP 0.50:0.95`로 구분합니다.

### Q6. Checkpoint 로딩에서 무엇을 검증했나요?

CPU 환경에서 `map_location`을 사용하고 Architecture, Class 수, Class Mapping, Image Size, Score Threshold 같은 Metadata가 현재 설정과 맞는지 검증했습니다. 불일치한 Checkpoint로 조용히 잘못 추론하는 문제를 방지했습니다.

### Q7. 프로젝트를 어떻게 검증했나요?

기능 단위 테스트, API TestClient, 실제 Best Checkpoint Smoke Test, Dashboard API Client·Overlay 검증, 구조 Inspector, Evidence Collector, 전체 Regression Test를 단계적으로 실행했습니다.

### Q8. AI 도구를 어떻게 사용했나요?

초안과 반복 작업에는 AI 보조 도구를 활용했지만, 코드는 직접 실행하고 오류를 수정했으며 Artifact·Checkpoint·응답 Schema·테스트 결과를 기준으로 문서화했습니다. 핵심은 생성된 결과를 그대로 사용하는 것이 아니라 실행·검증·수정·설명 가능한 상태로 만드는 것입니다.

## 15. Limitations

- CPU 환경 중심으로 학습·검증해 대규모 Hyperparameter Search를 수행하지 못했습니다.
- NEU-DET 규모가 작고 Class별 형태 차이가 큽니다.
- Detection Recall과 `crazing` 성능이 낮습니다.
- Project AP 0.50:0.95는 공식 COCOeval 결과가 아닙니다.
- Day 13 수동 Browser 확인 상태는 `{manual_browser_status}`입니다.
- 현재 Inference Lock은 Process-local이므로 Multi-worker 전역 Lock이 아닙니다.
- 실시간 생산 설비·카메라·PLC와 연결한 현장 배포 검증은 범위에 포함하지 않았습니다.

## 16. Future Improvements

1. 작은 결함 보존을 위한 고해상도·Tile 기반 Detection
2. Class-balanced Sampling과 결함별 Augmentation
3. Class별 Confidence Threshold Calibration
4. Precision-Recall Curve 기반 운영 Threshold 선택
5. COCOeval을 이용한 표준 Detection 평가 추가
6. Browser E2E 자동화와 수동 Visual Validation 기록
7. Docker·CI·Model Registry·Monitoring 확장

## 17. Completion Statement

Day 14는 새 기능 추가보다 **전체 흐름의 연결·검증·문서화**에 집중했습니다. Classification·OpenCV·Detection의 역할, FastAPI·Streamlit 호출 경계, 실제 성능과 실패 사례, 지표의 한계를 하나의 설명 가능한 프로젝트로 정리했습니다.
"""


def validate_generated_text(text: str) -> None:
    for phrase in FORBIDDEN_OVERCLAIMS:
        if phrase in text:
            raise Day14DocumentationError(
                f"최종 문서에 금지된 과장·역할 혼동 표현이 있습니다: {phrase}"
            )

    required_phrases = (
        PROJECT_NAME,
        "Classification",
        "OpenCV",
        "Object Detection",
        "FastAPI",
        "Streamlit",
        "not_recorded",
        "공식 COCOeval 결과로 표현하지 않습니다",
    )
    missing = [
        phrase
        for phrase in required_phrases
        if phrase not in text
    ]
    if missing:
        raise Day14DocumentationError(
            "최종 문서 필수 표현이 없습니다: " + ", ".join(missing)
        )


def update_marker_block(
    original: str,
    section: str,
) -> str:
    """Insert or replace exactly one Day 14 marker block."""

    start_count = original.count(README_START_MARKER)
    end_count = original.count(README_END_MARKER)

    if start_count != end_count:
        raise Day14DocumentationError(
            "README Day 14 START/END Marker 수가 다릅니다."
        )
    if start_count > 1:
        raise Day14DocumentationError(
            "README Day 14 Marker가 중복되어 있습니다."
        )

    normalized_section = section.strip() + "\n"

    if start_count == 1:
        start_index = original.index(README_START_MARKER)
        end_index = (
            original.index(README_END_MARKER, start_index)
            + len(README_END_MARKER)
        )
        prefix = original[:start_index].rstrip()
        suffix = original[end_index:].lstrip("\r\n")

        result = prefix + "\n\n" + normalized_section
        if suffix:
            result += "\n" + suffix
        return result.rstrip() + "\n"

    prefix = original.rstrip()
    if prefix:
        return prefix + "\n\n" + normalized_section
    return normalized_section


def _write_text_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            text.rstrip() + "\n",
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _backup_once(source: Path, backup: Path) -> None:
    if not source.is_file() or backup.exists():
        return
    backup.parent.mkdir(parents=True, exist_ok=True)
    temporary = backup.with_name(f".{backup.name}.tmp")
    try:
        temporary.write_bytes(source.read_bytes())
        temporary.replace(backup)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def create_day14_documents(
    *,
    project_root: Path,
    test_result: ValidationResult,
) -> tuple[Path, Path, Path]:
    root = project_root.resolve()
    evidence = validate_repository_evidence(root)
    manual_browser_status = evidence["manual_browser_check_status"]

    if manual_browser_status != "not_recorded":
        raise Day14DocumentationError(
            "현재 확정된 Day 13 수동 Browser 상태와 다릅니다: "
            f"{manual_browser_status}"
        )

    report = build_report(
        test_result=test_result,
        manual_browser_status=manual_browser_status,
    )
    readme_section = build_readme_section(
        test_result=test_result,
        manual_browser_status=manual_browser_status,
    )

    validate_generated_text(report)
    validate_generated_text(readme_section)

    readme_path = root / README_PATH
    report_path = root / REPORT_PATH
    summary_path = root / SUMMARY_PATH
    backup_path = root / BACKUP_PATH

    original_readme = _read_text(readme_path)
    updated_readme = update_marker_block(
        original_readme,
        readme_section,
    )

    summary: dict[str, Any] = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "name": PROJECT_NAME,
            "name_ko": PROJECT_NAME_KO,
        },
        "status": "PASS",
        "evidence": evidence,
        "tests": {
            "targeted_test_count": test_result.targeted_count,
            "regression_test_count": test_result.regression_count,
            "warning_count": test_result.warning_count,
            "runtime_seconds": test_result.runtime_seconds,
        },
        "performance": {
            "classification": {
                "accuracy": 0.9734,
                "precision": 0.9717,
                "recall": 0.9868,
                "f1": 0.9792,
                "confusion_matrix": {
                    "tn": 249,
                    "fp": 13,
                    "fn": 6,
                    "tp": 447,
                },
            },
            "detection": {
                "precision": 0.812950,
                "recall": 0.526807,
                "f1": 0.639321,
                "mean_matched_iou": 0.752338,
                "map_50": 0.707726,
                "project_ap_50_95": 0.310533,
                "project_ap_note": (
                    "Project all-point interpolation metric; "
                    "not claimed as official COCOeval."
                ),
            },
        },
        "manual_browser_check_status": manual_browser_status,
        "outputs": {
            "readme": README_PATH.as_posix(),
            "report": REPORT_PATH.as_posix(),
            "summary": SUMMARY_PATH.as_posix(),
            "readme_backup": BACKUP_PATH.as_posix(),
        },
        "write_policy": {
            "application_source_modified": False,
            "checkpoint_modified": False,
            "training_executed": False,
            "inference_executed": False,
            "readme_modified": True,
            "report_created": True,
            "summary_created": True,
        },
    }

    _backup_once(readme_path, backup_path)
    _write_text_atomically(report_path, report)
    _write_text_atomically(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2),
    )
    _write_text_atomically(readme_path, updated_readme)

    final_readme = _read_text(readme_path)
    if final_readme.count(README_START_MARKER) != 1:
        raise Day14DocumentationError(
            "README Day 14 START Marker가 정확히 1개가 아닙니다."
        )
    if final_readme.count(README_END_MARKER) != 1:
        raise Day14DocumentationError(
            "README Day 14 END Marker가 정확히 1개가 아닙니다."
        )

    saved_summary = _read_json_object(summary_path)
    if saved_summary.get("status") != "PASS":
        raise Day14DocumentationError(
            "생성된 Day 14 Summary 상태가 PASS가 아닙니다."
        )

    return report_path, summary_path, readme_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
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
        "--runtime-seconds",
        type=float,
        required=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.targeted_test_count <= 0:
        print("[FAIL] targeted-test-count는 양수여야 합니다.", file=sys.stderr)
        return 1
    if args.regression_test_count <= 0:
        print("[FAIL] regression-test-count는 양수여야 합니다.", file=sys.stderr)
        return 1
    if args.warning_count < 0:
        print("[FAIL] warning-count는 0 이상이어야 합니다.", file=sys.stderr)
        return 1
    if args.runtime_seconds <= 0:
        print("[FAIL] runtime-seconds는 양수여야 합니다.", file=sys.stderr)
        return 1

    try:
        report_path, summary_path, readme_path = create_day14_documents(
            project_root=args.project_root,
            test_result=ValidationResult(
                targeted_count=args.targeted_test_count,
                regression_count=args.regression_test_count,
                warning_count=args.warning_count,
                runtime_seconds=args.runtime_seconds,
            ),
        )
    except Day14DocumentationError as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1

    print("=" * 100)
    print("DAY 14 - FINAL DOCUMENTATION")
    print("=" * 100)
    print(f"Targeted tests             : {args.targeted_test_count} passed")
    print(f"Full regression            : {args.regression_test_count} passed")
    print(f"Warnings                   : {args.warning_count}")
    print(f"Runtime seconds            : {args.runtime_seconds:.2f}")
    print("[PASS] Day 14 report created")
    print("[PASS] Day 14 summary created")
    print("[PASS] README Day 14 section created or updated")
    print(f"[REPORT] {report_path.resolve()}")
    print(f"[SUMMARY] {summary_path.resolve()}")
    print(f"[README] {readme_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
