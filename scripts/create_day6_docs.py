"""Day 6 Grad-CAM 보고서 생성 및 README 반영 Script.

실행 예시
---------
프로젝트 루트에서 다음 명령으로 실행한다.

    python -m scripts.create_day6_docs

이 Script는 실제 Day 6 Metadata JSON과 PNG를 검증한 뒤 다음을 처리한다.

1. reports/day6_resnet18_gradcam_explainability_summary.md 생성
2. README.md의 Day 6 Marker 구간 생성 또는 교체
3. 생성 결과와 필수 내용 자체 검증

긴 Markdown을 PowerShell Here-String으로 직접 붙여 넣지 않고 Python이
UTF-8로 안전하게 생성하도록 구성했다.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

PROJECT_NAME: Final[str] = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN: Final[str] = "제조 비전 결함 분석 시스템"

DEFAULT_METADATA_PATH: Final[Path] = Path(
    "reports/artifacts/day6_resnet18_gradcam_analysis.json"
)
DEFAULT_REPORT_PATH: Final[Path] = Path(
    "reports/day6_resnet18_gradcam_explainability_summary.md"
)
DEFAULT_README_PATH: Final[Path] = Path("README.md")

README_START_MARKER: Final[str] = "<!-- DAY6_GRADCAM_START -->"
README_END_MARKER: Final[str] = "<!-- DAY6_GRADCAM_END -->"
DAY5_END_MARKER: Final[str] = "<!-- DAY5_MISCLASSIFICATION_END -->"

EXPECTED_TARGET_LAYER: Final[str] = "resnet18.layer4.1.conv2"
EXPECTED_SAMPLE_COUNT: Final[int] = 7


class Day6DocumentationError(RuntimeError):
    """Day 6 문서 생성 또는 검증이 실패했을 때 발생합니다."""


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """명령행 Argument를 해석합니다."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate the Day 6 ResNet18 Grad-CAM report and update README."
        )
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA_PATH,
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
    )
    parser.add_argument(
        "--readme-path",
        type=Path,
        default=DEFAULT_README_PATH,
    )
    parser.add_argument(
        "--regression-test-result",
        default="1204 passed",
        help="Day 6 완료 시점의 전체 회귀 테스트 결과",
    )
    parser.add_argument(
        "--visual-check-result",
        default="이상 없음",
        help="생성된 PNG의 육안 확인 결과",
    )
    return parser.parse_args(arguments)


def resolve_project_path(path: Path) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환합니다."""

    if not isinstance(path, Path):
        raise TypeError("path는 pathlib.Path여야 합니다.")
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def read_json_object(path: Path) -> dict[str, Any]:
    """UTF-8 JSON을 읽고 최상위 Object를 검증합니다."""

    if not path.is_file():
        raise FileNotFoundError(f"Metadata JSON이 없습니다: {path}")

    try:
        with path.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise Day6DocumentationError(
            f"Metadata JSON을 읽을 수 없습니다: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise Day6DocumentationError(
            "Metadata JSON 최상위 값은 Object여야 합니다."
        )

    return payload


def _require_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise Day6DocumentationError(
            f"Metadata의 {key!r} 값은 Object여야 합니다."
        )
    return value


def _require_sequence(
    payload: Mapping[str, object],
    key: str,
) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise Day6DocumentationError(
            f"Metadata의 {key!r} 값은 Array여야 합니다."
        )
    return value


def _validate_nonempty_file(path: Path, *, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 파일이 없습니다: {path}")
    if path.stat().st_size <= 0:
        raise Day6DocumentationError(f"{name} 파일이 비어 있습니다: {path}")


def validate_metadata(
    *,
    payload: Mapping[str, object],
    metadata_path: Path,
) -> dict[str, object]:
    """Day 6 실제 실행 Metadata와 Figure를 문서 생성 전에 검증합니다."""

    project = _require_mapping(payload, "project")
    configuration = _require_mapping(payload, "gradcam_configuration")
    selection_summary = _require_mapping(payload, "selection_summary")
    runtime = _require_mapping(payload, "runtime")
    figures = _require_mapping(payload, "figures")
    samples = _require_sequence(payload, "samples")

    if project.get("name") != PROJECT_NAME:
        raise Day6DocumentationError(
            "Metadata 프로젝트명이 고정 프로젝트명과 다릅니다."
        )
    if project.get("name_korean") != PROJECT_NAME_KOREAN:
        raise Day6DocumentationError(
            "Metadata 한글 프로젝트명이 고정 프로젝트명과 다릅니다."
        )

    target_layer = str(configuration.get("target_layer", ""))
    if target_layer != EXPECTED_TARGET_LAYER:
        raise Day6DocumentationError(
            "Grad-CAM Target Layer가 고정 정책과 다릅니다: "
            f"{target_layer!r}"
        )

    if configuration.get("target_policy") != "predicted_class":
        raise Day6DocumentationError(
            "Grad-CAM Target Policy는 predicted_class여야 합니다."
        )
    if configuration.get("defect_target_score") != "raw_logit":
        raise Day6DocumentationError(
            "DEFECT Target Score는 raw_logit이어야 합니다."
        )
    if configuration.get("normal_target_score") != "negative_raw_logit":
        raise Day6DocumentationError(
            "NORMAL Target Score는 negative_raw_logit이어야 합니다."
        )

    selected_sample_count = int(
        selection_summary.get("selected_sample_count", -1)
    )
    if selected_sample_count != EXPECTED_SAMPLE_COUNT:
        raise Day6DocumentationError(
            "선택 표본 수가 7장이 아닙니다: "
            f"{selected_sample_count}"
        )
    if len(samples) != EXPECTED_SAMPLE_COUNT:
        raise Day6DocumentationError(
            f"Metadata samples 수가 7개가 아닙니다: {len(samples)}"
        )

    seen_sample_indices: set[int] = set()
    maximum_logit_error = 0.0
    maximum_probability_error = 0.0

    for position, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise Day6DocumentationError(
                f"samples[{position}]는 Object여야 합니다."
            )

        sample_index = int(raw_sample.get("sample_index", -1))
        if sample_index < 0:
            raise Day6DocumentationError(
                f"samples[{position}]의 sample_index가 유효하지 않습니다."
            )
        if sample_index in seen_sample_indices:
            raise Day6DocumentationError(
                f"중복 sample_index가 있습니다: {sample_index}"
            )
        seen_sample_indices.add(sample_index)

        reproduction = raw_sample.get("reproduction_check")
        if not isinstance(reproduction, Mapping):
            raise Day6DocumentationError(
                f"sample_index={sample_index}에 reproduction_check가 없습니다."
            )
        if reproduction.get("prediction_matches") is not True:
            raise Day6DocumentationError(
                f"sample_index={sample_index}의 Prediction 재현이 실패했습니다."
            )

        logit_error = float(
            reproduction.get("raw_logit_absolute_error", float("inf"))
        )
        probability_error = float(
            reproduction.get(
                "probability_absolute_error",
                float("inf"),
            )
        )
        if logit_error > 1e-4:
            raise Day6DocumentationError(
                f"sample_index={sample_index}의 Logit 재현 오차가 큽니다."
            )
        if probability_error > 1e-5:
            raise Day6DocumentationError(
                f"sample_index={sample_index}의 Probability 재현 오차가 큽니다."
            )

        maximum_logit_error = max(maximum_logit_error, logit_error)
        maximum_probability_error = max(
            maximum_probability_error,
            probability_error,
        )

    duration_seconds = float(runtime.get("duration_seconds", -1.0))
    if duration_seconds < 0.0:
        raise Day6DocumentationError(
            "Metadata runtime.duration_seconds가 유효하지 않습니다."
        )

    resolved_figures: dict[str, Path] = {}
    for figure_name in (
        "overview",
        "high_confidence_errors",
        "boundary_errors",
    ):
        relative_path = figures.get(figure_name)
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise Day6DocumentationError(
                f"Figure 경로가 없습니다: {figure_name}"
            )

        figure_path = (
            Path(relative_path).resolve()
            if Path(relative_path).is_absolute()
            else (PROJECT_ROOT / relative_path).resolve()
        )
        _validate_nonempty_file(
            figure_path,
            name=f"Day 6 {figure_name}",
        )
        resolved_figures[figure_name] = figure_path

    _validate_nonempty_file(metadata_path, name="Day 6 Metadata")

    return {
        "configuration": dict(configuration),
        "samples": samples,
        "duration_seconds": duration_seconds,
        "maximum_logit_error": maximum_logit_error,
        "maximum_probability_error": maximum_probability_error,
        "figures": resolved_figures,
    }


def _project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _sample_filename(sample: Mapping[str, object]) -> str:
    image_path = str(sample.get("image_path", ""))
    return Path(image_path).name


def _selection_label(selection_type: str) -> str:
    labels = {
        "HIGH_CONFIDENCE_TRUE_NEGATIVE": "고확신 True Negative",
        "HIGH_CONFIDENCE_TRUE_POSITIVE": "고확신 True Positive",
        "HIGH_CONFIDENCE_FALSE_POSITIVE_1": "고확신 False Positive 1",
        "HIGH_CONFIDENCE_FALSE_POSITIVE_2": "고확신 False Positive 2",
        "HIGH_CONFIDENCE_FALSE_NEGATIVE": "고확신 False Negative",
        "BOUNDARY_FALSE_POSITIVE": "결정 경계 False Positive",
        "BOUNDARY_FALSE_NEGATIVE": "결정 경계 False Negative",
    }
    return labels.get(selection_type, selection_type)


def _build_sample_table(samples: Sequence[object]) -> str:
    rows = [
        "| 유형 | Index | 파일 | Ground Truth | Prediction | P(DEFECT) | Target |",
        "|---|---:|---|---|---|---:|---|",
    ]

    for raw_sample in samples:
        if not isinstance(raw_sample, Mapping):
            continue
        gradcam = raw_sample.get("gradcam")
        if not isinstance(gradcam, Mapping):
            raise Day6DocumentationError(
                "Sample에 Grad-CAM Metadata가 없습니다."
            )

        rows.append(
            "| "
            f"{_selection_label(str(raw_sample.get('selection_type', '')))}"
            " | "
            f"{int(raw_sample.get('sample_index', -1))}"
            " | "
            f"`{_sample_filename(raw_sample)}`"
            " | "
            f"{raw_sample.get('ground_truth_class_name')}"
            " | "
            f"{raw_sample.get('prediction_class_name')}"
            " | "
            f"{float(raw_sample.get('defect_probability', 0.0)):.6f}"
            " | "
            f"{gradcam.get('target_class')}"
            " |"
        )

    return "\n".join(rows)


def _build_artifact_table(
    *,
    metadata_path: Path,
    figures: Mapping[str, Path],
) -> str:
    rows = [
        "| Artifact | 경로 | 크기 |",
        "|---|---|---:|",
        (
            "| Metadata JSON | "
            f"`{_project_relative(metadata_path)}` | "
            f"{metadata_path.stat().st_size:,} bytes |"
        ),
    ]

    labels = {
        "overview": "전체 비교 Figure",
        "high_confidence_errors": "고확신 오류 Figure",
        "boundary_errors": "결정 경계 오류 Figure",
    }
    for key in (
        "overview",
        "high_confidence_errors",
        "boundary_errors",
    ):
        path = figures[key]
        rows.append(
            f"| {labels[key]} | `{_project_relative(path)}` | "
            f"{path.stat().st_size:,} bytes |"
        )

    return "\n".join(rows)


def build_report(
    *,
    metadata_path: Path,
    validated: Mapping[str, object],
    regression_test_result: str,
    visual_check_result: str,
) -> str:
    """실제 Day 6 결과를 기반으로 상세 보고서를 만듭니다."""

    configuration = validated["configuration"]
    samples = validated["samples"]
    duration_seconds = float(validated["duration_seconds"])
    maximum_logit_error = float(validated["maximum_logit_error"])
    maximum_probability_error = float(
        validated["maximum_probability_error"]
    )
    figures = validated["figures"]

    if not isinstance(configuration, Mapping):
        raise Day6DocumentationError(
            "검증된 configuration 형식이 올바르지 않습니다."
        )
    if not isinstance(samples, list):
        raise Day6DocumentationError(
            "검증된 samples 형식이 올바르지 않습니다."
        )
    if not isinstance(figures, Mapping):
        raise Day6DocumentationError(
            "검증된 figures 형식이 올바르지 않습니다."
        )

    sample_table = _build_sample_table(samples)
    artifact_table = _build_artifact_table(
        metadata_path=metadata_path,
        figures=figures,
    )

    return f"""# Day 6 - ResNet18 Grad-CAM Explainability Summary

## 1. 프로젝트와 Day 6 목표

프로젝트명:

```text
영문: {PROJECT_NAME}
한글: {PROJECT_NAME_KOREAN}
```

Day 6의 목표는 Day 4에서 학습·평가한 ResNet18 전이학습 모델이
특정 이미지를 `NORMAL` 또는 `DEFECT`로 판단할 때 마지막 Convolution
Layer의 어느 공간 영역을 상대적으로 강하게 사용했는지 Grad-CAM으로
시각화하는 것이다.

Day 5에서 확인한 고확신 오분류와 결정 경계 오분류를 정분류 표본과 함께
비교하여 모델의 판단 근거를 분석할 수 있는 설명 가능성 Artifact를
추가했다.

---

## 2. Grad-CAM이 필요한 이유

분류 지표만으로는 모델이 왜 해당 결과를 출력했는지 확인할 수 없다.

Grad-CAM은 다음 질문을 검토하는 데 사용한다.

```text
모델이 실제 제품 영역을 보고 있는가?
결함으로 보이는 표면 패턴을 보고 있는가?
제품 가장자리·조명·배경을 잘못된 단서로 사용하고 있는가?
정분류와 오분류의 주목 영역에 차이가 있는가?
```

Grad-CAM은 모델 검증과 오류 분석을 보조하지만 실제 결함 위치의
정답 Mask, Bounding Box 또는 Detection 결과는 아니다.

---

## 3. 구현 구조

구현 파일:

```text
src/explainability/gradcam.py
src/explainability/gradcam_sample_selector.py
src/explainability/gradcam_visualization.py
src/explainability/gradcam_pipeline.py
scripts/run_day6_resnet18_gradcam.py
```

테스트 파일:

```text
tests/test_gradcam.py
tests/test_gradcam_sample_selector.py
tests/test_gradcam_visualization.py
tests/test_gradcam_pipeline.py
tests/test_day6_resnet18_gradcam.py
```

호출 흐름:

```text
Day 4 평가 JSON
→ Day 5 오분류 JSON 교차 검증
→ 대표 표본 7장 선택
→ ResNet18 Best Checkpoint 복원
→ Test Transform 적용
→ Batch Size 1 Forward
→ Target Score Backward
→ Activation·Gradient 저장
→ Channel Weight 계산
→ Weighted Activation 합산
→ ReLU 및 0~1 정규화
→ 원본·Heatmap·Overlay 생성
→ JSON·PNG Atomic 저장
```

---

## 4. Grad-CAM 고정 정책

Target Layer:

```text
{configuration.get("target_layer")}
```

Target 정책:

```text
Prediction = DEFECT
→ target_score = raw_logit

Prediction = NORMAL
→ target_score = -raw_logit
```

설정:

```text
Batch Size             = {configuration.get("batch_size")}
Classification Threshold = {float(configuration.get("classification_threshold", 0.5)):.1f}
Target Policy          = {configuration.get("target_policy")}
Channel Weight         = Spatial Mean of Gradients
Weighted Activation    = Channel Weighted Sum
ReLU                   = {configuration.get("relu")}
Normalization          = Min-Max 0~1
Input Size             = {configuration.get("input_image_size")}
Input Normalization    = ImageNet Mean / Standard Deviation
Color Map              = {configuration.get("colormap")}
Overlay Alpha          = {float(configuration.get("overlay_alpha", 0.4)):.2f}
```

마지막 Convolution Layer는 Classification Head에 가까운 고수준 Feature를
가지면서도 공간 정보를 보존한다. Fully Connected Layer 이후에는 공간
정보가 사라지므로 `resnet18.layer4.1.conv2`를 사용했다.

---

## 5. Hook와 Grad-CAM 계산

Forward Hook는 Target Layer의 Activation Feature Map을 저장한다.

Forward 과정에서 얻은 Activation Tensor에 Gradient Hook를 등록하고,
Target Score를 Backward하여 다음 Gradient를 저장한다.

```text
∂target_score / ∂activation
```

Channel Weight:

```python
weights = gradients.mean(
    dim=(2, 3),
    keepdim=True,
)
```

Weighted Activation:

```python
cam = (
    weights * activations
).sum(
    dim=1,
    keepdim=True,
)
```

최종 처리:

```text
ReLU
→ 입력 Tensor 크기로 Bilinear Resize
→ Min-Max 0~1 정규화
→ 원본 이미지 크기로 Resize
→ Color Map
→ Alpha Blending
```

Frozen Backbone에서도 Activation Gradient를 생성할 수 있도록 Grad-CAM
계산용 입력 복사본에만 `requires_grad_(True)`를 적용했다. Optimizer나
학습 Parameter Update는 수행하지 않는다.

---

## 6. 대표 표본 선택 결과

총 7장을 자동 선택했다.

```text
고확신 정분류 NORMAL 1장
고확신 정분류 DEFECT 1장
고확신 False Positive 2장
고확신 False Negative 1장
결정 경계 False Positive 1장
결정 경계 False Negative 1장
```

{sample_table}

정분류는 모델 확신도가 가장 높은 표본을 사용했고, 오분류는 잘못된 예측
확신도와 Threshold 0.5까지의 거리를 기준으로 선택했다. 동일 표본이
여러 기준에 걸릴 경우 먼저 선택한 표본을 유지하고 다음 후보를 선택하여
중복을 제거했다.

---

## 7. 실제 실행 결과

```text
Generated Samples          = {len(samples)}
Correct Prediction Samples = 2
High-Confidence Errors     = 3
Boundary Errors            = 2
Runtime                    = {duration_seconds:.2f} seconds
Target Layer               = {configuration.get("target_layer")}
Device                     = CPU
```

Day 4 저장 결과와 Grad-CAM 재추론 결과도 표본별로 비교했다.

```text
Prediction Match           = 모든 표본 True
Maximum Raw Logit Error    = {maximum_logit_error:.12f}
Maximum Probability Error  = {maximum_probability_error:.12f}
```

이 검증은 잘못된 Checkpoint 복원, 다른 Transform 사용, 이미지 경로 오류,
모델 Architecture 불일치를 Artifact 생성 전에 차단하기 위한 것이다.

---

## 8. 생성 Artifact

{artifact_table}

Figure 구성:

```text
Overview
→ 정분류 2장과 오분류 5장 전체 비교

High-Confidence Errors
→ 고확신 False Positive 2장과 False Negative 1장

Boundary Errors
→ 결정 경계 False Positive 1장과 False Negative 1장
```

모든 JSON과 PNG는 임시 파일에 먼저 기록한 뒤 `os.replace()`를 사용해
최종 경로로 교체했다.

---

## 9. 테스트 및 검증

Day 6 단위·통합 테스트:

```text
40 passed
```

전체 회귀 테스트:

```text
{regression_test_result}
```

PNG 육안 확인:

```text
{visual_check_result}
```

검증 범위:

```text
Target Layer 탐색
Forward Hook 등록
Activation Gradient 저장
NORMAL·DEFECT Target Score 방향
CAM Shape와 0~1 범위
NaN·Infinity 차단
Zero CAM 차단
Batch Size 1 정책
Hook 해제
대표 표본 선택
Day 4·Day 5 교차 검증
Day 4 예측 재현
이미지 로딩과 손상 검사
Heatmap·Overlay 생성
Windows 환경 Figure 메모리 사용 개선
JSON·PNG Atomic 저장
실제 ResNet18 Best Checkpoint 실행
전체 회귀 테스트
```

---

## 10. 오류 처리와 자원 정리

다음 상황을 명시적인 예외로 처리한다.

```text
Target Layer가 존재하지 않음
Activation 또는 Gradient가 저장되지 않음
Activation·Gradient Shape 불일치
Batch Size가 1이 아님
입력 Tensor에 NaN·Infinity 포함
CAM에 NaN·Infinity 발생
CAM 전체 값이 0
이미지 파일 누락 또는 손상
Day 4·Day 5 Artifact 불일치
재추론 Logit·Probability·Prediction 불일치
빈 시각화 표본
Figure 저장 실패
```

Grad-CAM Context 종료 시 Hook를 반드시 해제한다. Figure도 성공과 실패에
관계없이 `clear()`와 `close()`를 수행해 Windows CPU 환경에서 자원이
누적되지 않도록 했다.

---

## 11. 실무 해석 원칙

Grad-CAM 결과는 다음과 같이 사용한다.

```text
정분류에서 제품의 결함 관련 영역을 일관되게 보는지 확인
False Positive에서 정상 패턴을 결함으로 오인한 영역 확인
False Negative에서 실제 결함을 놓치고 다른 영역을 본 가능성 확인
결정 경계 오류에서 Heatmap이 분산되거나 애매한지 확인
배경·가장자리·조명 편향 여부 확인
```

다만 Heatmap의 붉은 영역을 실제 결함 위치라고 단정하지 않는다. Grad-CAM은
모델의 상대적인 주목 영역을 보여주는 설명 보조 수단이며, 정밀한 결함
위치 검증에는 별도의 Annotation과 Detection·Segmentation 평가가 필요하다.

---

## 12. 면접 설명

### Q1. Binary Classification에서 Class별 Logit이 하나뿐인데 NORMAL Grad-CAM은 어떻게 계산했나요?

DEFECT는 Raw Logit이 증가하는 방향이므로 `raw_logit`을 Target Score로
사용했다. NORMAL은 Raw Logit이 감소하는 방향이므로 `-raw_logit`을
Target Score로 사용했다.

### Q2. 왜 마지막 Convolution Layer를 선택했나요?

Classification Head에 가장 가까워 결함 판별에 필요한 고수준 Feature를
포함하면서도 위치 정보를 보존하기 때문이다. Fully Connected Layer는
공간 정보가 없어 Heatmap 생성에 적합하지 않다.

### Q3. Frozen Backbone인데 Gradient를 계산할 수 있나요?

Parameter 학습 Gradient와 Grad-CAM 설명 Gradient는 목적이 다르다.
Optimizer나 Parameter Update 없이 Target Score를 Activation까지
Backward하면 설명용 Gradient를 구할 수 있다. Frozen Parameter와 입력이
모두 Gradient를 요구하지 않는 경우를 방지하기 위해 Grad-CAM 전용 입력
복사본에만 `requires_grad_(True)`를 적용했다.

### Q4. Grad-CAM으로 결함 위치를 검출했다고 말할 수 있나요?

말할 수 없다. Grad-CAM은 모델이 예측에 상대적으로 사용한 영역을
시각화할 뿐이며 실제 결함 위치의 정답 Mask나 Detection 결과가 아니다.

### Q5. 왜 전체 Test Dataset 715장에 Grad-CAM을 생성하지 않았나요?

Day 6의 목적은 대표 오류 분석과 포트폴리오 설명 가능성 확보다. CPU
환경에서 전체 표본을 처리하는 것보다 정분류 2장, 고확신 오류 3장,
결정 경계 오류 2장을 선택해 비교하는 것이 효율적이고 분석 목적에도
적합하다.

---

## 13. Day 6 결론

Day 6에서는 외부 Grad-CAM 라이브러리에 의존하지 않고 PyTorch Hook 기반
Grad-CAM 계산 과정을 직접 구현했다.

ResNet18 Best Checkpoint와 Day 4·Day 5 Artifact를 연결하고, 대표 표본
7장에 대해 모델이 실제로 선택한 예측 Class 관점의 Heatmap을 생성했다.
재추론 결과를 기존 평가 JSON과 다시 비교했으며, Metadata JSON과
세 종류의 PNG Figure를 Atomic 방식으로 저장했다.

최종적으로 {regression_test_result}와 PNG 육안 확인 `{visual_check_result}`을
통과하여 Day 6 ResNet18 Grad-CAM Explainability 구현과 검증을 완료했다.
"""


def build_readme_section(
    *,
    metadata_path: Path,
    validated: Mapping[str, object],
    regression_test_result: str,
    visual_check_result: str,
) -> str:
    """README에 넣을 간결한 Day 6 섹션을 만든다."""

    configuration = validated["configuration"]
    samples = validated["samples"]
    duration_seconds = float(validated["duration_seconds"])
    figures = validated["figures"]

    if not isinstance(configuration, Mapping):
        raise Day6DocumentationError(
            "검증된 configuration 형식이 올바르지 않습니다."
        )
    if not isinstance(samples, list):
        raise Day6DocumentationError(
            "검증된 samples 형식이 올바르지 않습니다."
        )
    if not isinstance(figures, Mapping):
        raise Day6DocumentationError(
            "검증된 figures 형식이 올바르지 않습니다."
        )

    return f"""{README_START_MARKER}
## Day 6 — ResNet18 Grad-CAM Explainability

ResNet18이 특정 이미지를 `NORMAL` 또는 `DEFECT`로 판단할 때 마지막
Convolution Layer에서 상대적으로 주목한 영역을 확인하기 위해 PyTorch
Hook 기반 Grad-CAM을 직접 구현했다.

### 핵심 설계

```text
Target Layer  : {configuration.get("target_layer")}
Target Policy : Predicted Class
DEFECT Score  : raw_logit
NORMAL Score  : -raw_logit
Batch Size    : 1
Input         : 224 × 224, ImageNet Normalize
Output        : Original / Heatmap / Overlay
```

Day 4 평가 JSON과 Day 5 오분류 JSON을 교차 검증한 뒤 다음 대표 표본
{len(samples)}장을 자동 선택했다.

```text
고확신 True Negative 1장
고확신 True Positive 1장
고확신 False Positive 2장
고확신 False Negative 1장
결정 경계 False Positive 1장
결정 경계 False Negative 1장
```

실제 실행 결과:

```text
Generated Samples      : {len(samples)}
Runtime                 : {duration_seconds:.2f} seconds
Day 6 Tests             : 40 passed
Full Regression Tests   : {regression_test_result}
PNG Visual Check        : {visual_check_result}
```

Artifact:

```text
{_project_relative(metadata_path)}
{_project_relative(figures["overview"])}
{_project_relative(figures["high_confidence_errors"])}
{_project_relative(figures["boundary_errors"])}
reports/day6_resnet18_gradcam_explainability_summary.md
```

> Grad-CAM Heatmap은 모델이 예측에 상대적으로 사용한 영역을 보여주는 설명
> 보조 수단이다. 실제 결함 위치의 정답 Mask나 Detection 결과로 해석하지 않는다.

{README_END_MARKER}"""


def _detect_utf8_bom(path: Path) -> bool:
    try:
        return path.read_bytes().startswith(b"\xef\xbb\xbf")
    except OSError as exc:
        raise Day6DocumentationError(
            f"파일 Encoding을 확인할 수 없습니다: {path}"
        ) from exc


def _read_text(path: Path) -> tuple[str, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    has_bom = _detect_utf8_bom(path)
    encoding = "utf-8-sig" if has_bom else "utf-8"

    try:
        return path.read_text(encoding=encoding), has_bom
    except (OSError, UnicodeError) as exc:
        raise Day6DocumentationError(
            f"UTF-8 텍스트를 읽을 수 없습니다: {path}"
        ) from exc


def write_text_atomically(
    *,
    path: Path,
    text: str,
    include_utf8_bom: bool = False,
) -> Path:
    """텍스트를 같은 Directory의 임시 파일에 저장 후 교체한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    encoding = "utf-8-sig" if include_utf8_bom else "utf-8"

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding=encoding,
            newline="\n",
            suffix=".tmp",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(text.rstrip())
            temporary_file.write("\n")

        os.replace(temporary_path, path)
    except OSError as exc:
        raise Day6DocumentationError(
            f"텍스트 파일 저장에 실패했습니다: {path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if not path.is_file() or path.stat().st_size <= 0:
        raise Day6DocumentationError(
            f"저장된 텍스트 파일이 비어 있습니다: {path}"
        )

    return path


def update_readme_content(
    *,
    readme_content: str,
    day6_section: str,
) -> str:
    """Day 6 Marker 구간을 교체하거나 Day 5 다음에 추가한다."""

    start_count = readme_content.count(README_START_MARKER)
    end_count = readme_content.count(README_END_MARKER)

    if start_count != end_count:
        raise Day6DocumentationError(
            "README의 Day 6 Marker 시작·종료 개수가 다릅니다."
        )
    if start_count > 1:
        raise Day6DocumentationError(
            "README에 Day 6 Marker가 중복되어 있습니다."
        )

    if start_count == 1:
        start_index = readme_content.index(README_START_MARKER)
        end_index = (
            readme_content.index(README_END_MARKER)
            + len(README_END_MARKER)
        )
        return (
            readme_content[:start_index].rstrip()
            + "\n\n"
            + day6_section.strip()
            + "\n"
            + readme_content[end_index:].lstrip("\n")
        )

    if DAY5_END_MARKER in readme_content:
        insertion_index = (
            readme_content.index(DAY5_END_MARKER)
            + len(DAY5_END_MARKER)
        )
        return (
            readme_content[:insertion_index].rstrip()
            + "\n\n"
            + day6_section.strip()
            + "\n"
            + readme_content[insertion_index:].lstrip("\n")
        )

    return readme_content.rstrip() + "\n\n" + day6_section.strip() + "\n"


def validate_generated_documents(
    *,
    report_path: Path,
    readme_path: Path,
    regression_test_result: str,
) -> None:
    """생성된 Report와 README의 핵심 내용을 검증한다."""

    report_text, _ = _read_text(report_path)
    readme_text, _ = _read_text(readme_path)

    required_report_text = (
        "Day 6 - ResNet18 Grad-CAM Explainability Summary",
        EXPECTED_TARGET_LAYER,
        "target_score = raw_logit",
        "target_score = -raw_logit",
        "1204 passed" if regression_test_result == "1204 passed" else regression_test_result,
        "Grad-CAM은 모델이 예측에 상대적으로 사용한 영역",
        "실제 결함 위치의 정답 Mask",
    )
    for required_text in required_report_text:
        if required_text not in report_text:
            raise Day6DocumentationError(
                f"Day 6 Report에 필수 내용이 없습니다: {required_text!r}"
            )

    if readme_text.count(README_START_MARKER) != 1:
        raise Day6DocumentationError(
            "README Day 6 시작 Marker가 정확히 한 개가 아닙니다."
        )
    if readme_text.count(README_END_MARKER) != 1:
        raise Day6DocumentationError(
            "README Day 6 종료 Marker가 정확히 한 개가 아닙니다."
        )

    for required_text in (
        "Day 6 — ResNet18 Grad-CAM Explainability",
        EXPECTED_TARGET_LAYER,
        regression_test_result,
        "day6_resnet18_gradcam_analysis.json",
        "day6_resnet18_gradcam_overview.png",
    ):
        if required_text not in readme_text:
            raise Day6DocumentationError(
                f"README Day 6 구간에 필수 내용이 없습니다: {required_text!r}"
            )


def main(arguments: Sequence[str] | None = None) -> None:
    """Day 6 문서 생성 Entry Point."""

    parsed = parse_arguments(arguments)

    metadata_path = resolve_project_path(parsed.metadata_path)
    report_path = resolve_project_path(parsed.report_path)
    readme_path = resolve_project_path(parsed.readme_path)

    payload = read_json_object(metadata_path)
    validated = validate_metadata(
        payload=payload,
        metadata_path=metadata_path,
    )

    report_text = build_report(
        metadata_path=metadata_path,
        validated=validated,
        regression_test_result=parsed.regression_test_result,
        visual_check_result=parsed.visual_check_result,
    )
    day6_readme_section = build_readme_section(
        metadata_path=metadata_path,
        validated=validated,
        regression_test_result=parsed.regression_test_result,
        visual_check_result=parsed.visual_check_result,
    )

    readme_content, readme_has_bom = _read_text(readme_path)
    updated_readme = update_readme_content(
        readme_content=readme_content,
        day6_section=day6_readme_section,
    )

    write_text_atomically(
        path=report_path,
        text=report_text,
        include_utf8_bom=False,
    )
    write_text_atomically(
        path=readme_path,
        text=updated_readme,
        include_utf8_bom=readme_has_bom,
    )

    validate_generated_documents(
        report_path=report_path,
        readme_path=readme_path,
        regression_test_result=parsed.regression_test_result,
    )

    print("[PASS] Day 6 report created")
    print("[PASS] README Day 6 section added")
    print(f"[REPORT] {report_path}")
    print(f"[README] {readme_path}")


if __name__ == "__main__":
    main()
