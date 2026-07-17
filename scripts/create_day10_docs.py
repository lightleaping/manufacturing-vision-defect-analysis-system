"""Day 10 OpenCV 보고서와 README 섹션을 생성한다.

필수 선행 조건:
- reports/artifacts/day10_opencv_image_analysis.json
- reports/artifacts/day10_opencv_visual_validation.json
- 세 Figure 파일
- 육안 검증 validation_passed == True

이 Script는 실제 Artifact의 Config·샘플 지표·파일 경로를 읽어 문서를 만든다.
OpenCV Contour를 Ground Truth나 객체 탐지 Bounding Box로 표현하지 않는다.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KO = "제조 비전 결함 분석 시스템"

DEFAULT_ANALYSIS_PATH = (
    "reports/artifacts/day10_opencv_image_analysis.json"
)
DEFAULT_VISUAL_VALIDATION_PATH = (
    "reports/artifacts/day10_opencv_visual_validation.json"
)
DEFAULT_REPORT_PATH = (
    "reports/day10_opencv_image_analysis_pipeline_summary.md"
)
DEFAULT_README_PATH = "README.md"

README_START_MARKER = "<!-- DAY10_OPENCV_IMAGE_ANALYSIS_START -->"
README_END_MARKER = "<!-- DAY10_OPENCV_IMAGE_ANALYSIS_END -->"

EXPECTED_SAMPLE_IDS = {
    "casting_normal",
    "casting_defect",
    "neu_det_crazing",
}
REQUIRED_FIGURE_KEYS = {
    "pipeline_overview_figure",
    "histogram_and_metrics_figure",
    "contour_analysis_figure",
}


@dataclass(frozen=True, slots=True)
class Day10DocumentationInputs:
    project_root: Path
    analysis: dict[str, Any]
    visual_validation: dict[str, Any]
    regression_test_count: int
    warning_count: int
    regression_runtime_seconds: float
    day10_test_count: int


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}") from error

    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _validate_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")


def _validate_nonnegative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to 0")


def _validate_runtime(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("regression_runtime_seconds must be a number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(
            "regression_runtime_seconds must be finite and greater than 0"
        )


def _validate_analysis(
    analysis: dict[str, Any],
    *,
    project_root: Path,
) -> None:
    if analysis.get("project_name") != PROJECT_NAME:
        raise ValueError("analysis artifact project_name mismatch")
    if analysis.get("project_name_ko") != PROJECT_NAME_KO:
        raise ValueError("analysis artifact project_name_ko mismatch")
    if analysis.get("day") != 10:
        raise ValueError("analysis artifact must be a Day 10 result")
    if analysis.get("sample_count") != 3:
        raise ValueError("analysis artifact must contain exactly 3 samples")

    samples = analysis.get("samples")
    if not isinstance(samples, list) or len(samples) != 3:
        raise ValueError("analysis samples must be a list with 3 records")

    sample_ids = {
        sample.get("sample_id")
        for sample in samples
        if isinstance(sample, dict)
    }
    if sample_ids != EXPECTED_SAMPLE_IDS:
        raise ValueError(
            "analysis samples must contain casting_normal, casting_defect "
            "and neu_det_crazing"
        )

    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("each analysis sample must be an object")
        metrics = sample.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("each analysis sample must contain metrics")
        required_metrics = {
            "width",
            "height",
            "channels",
            "mean_brightness",
            "brightness_standard_deviation",
            "edge_pixel_ratio",
            "threshold_foreground_ratio",
            "contour_count",
            "largest_contour_area_ratio",
            "average_contour_area_ratio",
        }
        if not required_metrics.issubset(metrics):
            missing = sorted(required_metrics - set(metrics))
            raise ValueError(f"sample metrics are missing: {missing}")

    if not isinstance(analysis.get("config"), dict):
        raise ValueError("analysis artifact must contain config")
    if not isinstance(analysis.get("dependency_versions"), dict):
        raise ValueError("analysis artifact must contain dependency_versions")

    artifacts = analysis.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("analysis artifact must contain artifacts")

    if not REQUIRED_FIGURE_KEYS.issubset(artifacts):
        missing = sorted(REQUIRED_FIGURE_KEYS - set(artifacts))
        raise ValueError(f"analysis artifact paths are missing: {missing}")

    for key in sorted(REQUIRED_FIGURE_KEYS):
        figure_path = _resolve(project_root, str(artifacts[key]))
        if not figure_path.is_file():
            raise FileNotFoundError(f"Day 10 figure not found: {figure_path}")
        if figure_path.stat().st_size <= 0:
            raise ValueError(f"Day 10 figure is empty: {figure_path}")


def _validate_visual_validation(
    visual_validation: dict[str, Any],
) -> None:
    if visual_validation.get("project_name") != PROJECT_NAME:
        raise ValueError("visual validation project_name mismatch")
    if visual_validation.get("project_name_ko") != PROJECT_NAME_KO:
        raise ValueError("visual validation project_name_ko mismatch")
    if visual_validation.get("day") != 10:
        raise ValueError("visual validation must be a Day 10 result")

    for key in (
        "automated_checks_passed",
        "manual_checks_passed",
        "validation_passed",
    ):
        if visual_validation.get(key) is not True:
            raise ValueError(f"visual validation requires {key}=True")

    manual_checks = visual_validation.get("manual_checks")
    if not isinstance(manual_checks, dict) or not manual_checks:
        raise ValueError("visual validation must contain manual_checks")
    if any(status != "pass" for status in manual_checks.values()):
        raise ValueError("every Day 10 manual visual check must be pass")


def load_documentation_inputs(
    *,
    project_root: str | Path,
    analysis_path: str | Path = DEFAULT_ANALYSIS_PATH,
    visual_validation_path: str | Path = DEFAULT_VISUAL_VALIDATION_PATH,
    regression_test_count: int,
    warning_count: int,
    regression_runtime_seconds: float,
    day10_test_count: int = 62,
) -> Day10DocumentationInputs:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project root not found: {root}")

    _validate_positive_integer("regression_test_count", regression_test_count)
    _validate_nonnegative_integer("warning_count", warning_count)
    _validate_positive_integer("day10_test_count", day10_test_count)
    _validate_runtime(regression_runtime_seconds)

    analysis = _load_json(
        _resolve(root, analysis_path),
        label="Day 10 analysis artifact",
    )
    visual_validation = _load_json(
        _resolve(root, visual_validation_path),
        label="Day 10 visual validation artifact",
    )

    _validate_analysis(analysis, project_root=root)
    _validate_visual_validation(visual_validation)

    return Day10DocumentationInputs(
        project_root=root,
        analysis=analysis,
        visual_validation=visual_validation,
        regression_test_count=regression_test_count,
        warning_count=warning_count,
        regression_runtime_seconds=float(regression_runtime_seconds),
        day10_test_count=day10_test_count,
    )


def _markdown_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return " × ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _config_rows(config: dict[str, Any]) -> str:
    labels = {
        "clahe_clip_limit": "CLAHE clip limit",
        "clahe_tile_grid_size": "CLAHE tile grid",
        "gaussian_kernel_size": "Gaussian kernel",
        "gaussian_sigma_x": "Gaussian sigma X",
        "canny_low_threshold": "Canny low threshold",
        "canny_high_threshold": "Canny high threshold",
        "adaptive_threshold_block_size": "Adaptive threshold block",
        "adaptive_threshold_c": "Adaptive threshold C",
        "adaptive_threshold_invert": "Adaptive threshold invert",
        "morphology_kernel_size": "Morphology kernel",
        "morphology_open_iterations": "Morphology opening iterations",
        "morphology_close_iterations": "Morphology closing iterations",
        "min_contour_area_ratio": "Minimum contour area ratio",
        "max_contours": "Maximum contours",
        "contour_line_thickness": "Contour line thickness",
    }

    rows = []
    for key, value in config.items():
        rows.append(
            f"| {labels.get(key, key)} | `{_markdown_value(value)}` |"
        )
    return "\n".join(rows)


def _sample_rows(samples: list[dict[str, Any]]) -> str:
    rows = []
    for sample in samples:
        metrics = sample["metrics"]
        rows.append(
            "| {role} | `{filename}` | {width}×{height} | {mean:.3f} | "
            "{contrast:.3f} | {edge:.6f} | {foreground:.6f} | "
            "{contours} | {largest:.6f} |".format(
                role=sample["semantic_role"],
                filename=sample["filename"],
                width=metrics["width"],
                height=metrics["height"],
                mean=float(metrics["mean_brightness"]),
                contrast=float(
                    metrics["brightness_standard_deviation"]
                ),
                edge=float(metrics["edge_pixel_ratio"]),
                foreground=float(
                    metrics["threshold_foreground_ratio"]
                ),
                contours=int(metrics["contour_count"]),
                largest=float(
                    metrics["largest_contour_area_ratio"]
                ),
            )
        )
    return "\n".join(rows)


def _artifact_rows(
    artifacts: dict[str, Any],
    visual_validation_path: str,
) -> str:
    ordered = (
        ("analysis_json", "OpenCV analysis JSON"),
        ("pipeline_overview_figure", "Pipeline overview"),
        ("histogram_and_metrics_figure", "Histogram and metrics"),
        ("contour_analysis_figure", "Contour candidate analysis"),
    )
    rows = [
        f"| {label} | `{artifacts[key]}` |"
        for key, label in ordered
        if key in artifacts
    ]
    rows.append(
        f"| Visual validation JSON | `{visual_validation_path}` |"
    )
    return "\n".join(rows)


def build_day10_report(inputs: Day10DocumentationInputs) -> str:
    analysis = inputs.analysis
    visual = inputs.visual_validation
    dependencies = analysis["dependency_versions"]
    artifacts = analysis["artifacts"]
    visual_path = str(
        visual.get(
            "analysis_artifact",
            DEFAULT_VISUAL_VALIDATION_PATH,
        )
    )
    # analysis_artifact는 입력 분석 JSON을 가리키므로 실제 검증 JSON 경로는
    # 고정된 Day 10 Artifact 이름으로 문서에 기록한다.
    visual_path = DEFAULT_VISUAL_VALIDATION_PATH

    return f"""# Day 10 — OpenCV Image Analysis Pipeline

## 1. 완료 상태

```text
Project              : {PROJECT_NAME}
한글명               : {PROJECT_NAME_KO}
Day                  : 10
Target tests         : {inputs.day10_test_count} passed
Full regression      : {inputs.regression_test_count} passed
Warnings             : {inputs.warning_count}
Regression runtime   : {inputs.regression_runtime_seconds:.2f} seconds
Visual validation    : PASS
```

Day 10에서는 학습 모델이 아닌 **OpenCV 기반 이미지 명암·경계·형태 특성 보조 분석 파이프라인**을 구현했다. OpenCV 결과는 Classification이나 Object Detection을 대체하지 않는다.

## 2. 기능 구분

| 기능 | 역할 | 출력 |
|---|---|---|
| Classification | 이미지 전체가 NORMAL 또는 DEFECT인지 판단 | Class, probability |
| OpenCV Analysis | 명암·경계·Threshold·형태 계산 결과를 사람이 확인 | Metrics, masks, edges, contour candidates |
| Object Detection | 학습된 모델이 결함 종류와 위치를 예측 | Class, bounding box, confidence |

Contour는 Adaptive Threshold와 Morphology 결과에서 계산된 **후보 외곽선**이다. 실제 결함 Ground Truth, 객체 탐지 Bounding Box 또는 Detection Prediction으로 해석하지 않는다.

## 3. 처리 파이프라인

```text
Pillow Image
→ RGB 정규화
→ OpenCV BGR
→ Grayscale
→ Histogram
→ CLAHE
→ Gaussian Blur
→ Canny Edge
→ Adaptive Threshold
→ Morphology Opening·Closing
→ Minimum Area Ratio Filter
→ Contour Candidate Overlay
→ JSON Metrics + PNG Figures
```

각 단계의 목적은 다음과 같다.

| 단계 | 목적 |
|---|---|
| Grayscale | 색상 채널을 단일 명암 채널로 변환 |
| Histogram | 0~255 명암 분포와 Peak 확인 |
| CLAHE | 국소 영역 대비를 제한적으로 향상 |
| Gaussian Blur | 작은 Noise와 급격한 픽셀 변화를 완화 |
| Canny Edge | 명암 변화가 큰 경계 픽셀 추출 |
| Adaptive Threshold | 영역별 밝기 기준으로 전경과 배경 분리 |
| Morphology | 작은 전경 Noise 제거와 끊어진 영역 연결 |
| Contour Overlay | Threshold·Morphology 기반 후보 형태 시각화 |

## 4. 결정론적 Config

| Parameter | Value |
|---|---:|
{_config_rows(analysis["config"])}

파라미터를 불변 Dataclass로 분리해 동일 입력과 동일 설정에서 같은 처리 결과를 만들고, 실행 Artifact에 실제 설정을 남겼다. Kernel 홀수 여부, Canny Threshold 순서, Contour 면적 비율 범위 등의 오류는 실행 전에 검증한다.

## 5. 이미지 변환 정책

```text
Pillow RGB : R, G, B
OpenCV BGR : B, G, R
Grayscale  : Height × Width
dtype      : uint8
```

Pillow 입력은 RGB로 정규화한 뒤 OpenCV BGR로 변환한다. Figure에 표시할 때는 다시 RGB로 변환한다. 빈 배열, 잘못된 Shape, 지원하지 않는 dtype과 이미지 확장자는 명시적으로 거부한다.

## 6. 실제 이미지 분석

전체 Dataset을 반복 분석하지 않고 재현 가능한 고정 샘플 3장만 사용했다.

| Sample | File | Shape | Mean | Contrast | Edge ratio | Foreground ratio | Contours | Largest contour ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{_sample_rows(analysis["samples"])}

Casting과 NEU-DET는 Dataset의 의미와 촬영 조건이 다르므로 위 수치를 모델 성능이나 Dataset 우열 비교로 해석하지 않는다.

## 7. 정량 지표

| Metric | 의미 |
|---|---|
| Mean Brightness | Grayscale 픽셀 평균 밝기 |
| Brightness Standard Deviation | 이미지 내 명암 변화 정도 |
| Histogram Peak | 가장 많은 픽셀이 분포한 명암값 |
| Otsu Threshold | 전체 Histogram 기반 참고 Threshold |
| Edge Pixel Ratio | 전체 픽셀 중 Canny Edge 비율 |
| Threshold Foreground Ratio | Morphology 결과의 전경 픽셀 비율 |
| Contour Count | 최소 면적 기준을 통과한 후보 Contour 수 |
| Largest Contour Area Ratio | 가장 큰 후보 Contour 면적 비율 |
| Average Contour Area Ratio | 후보 Contour 평균 면적 비율 |

이 값들은 이미지의 계산 특성을 설명하는 보조 지표이며 실제 결함 검출 성능 지표가 아니다.

## 8. 생성 Artifact

| Artifact | Path |
|---|---|
{_artifact_rows(artifacts, visual_path)}

세 Figure는 Pillow Decode, PNG 형식, Width·Height 및 파일 크기를 자동 검증했다. 이후 Layout, RGB 표시, Metrics 가독성, Contour 후보 주의 문구를 직접 확인했고 육안 검증 결과를 별도 JSON으로 기록했다.

## 9. Dependency

| Dependency | Version |
|---|---:|
| opencv-python | `{dependencies.get("opencv_python", "unknown")}` |
| cv2 | `{dependencies.get("cv2", "unknown")}` |
| NumPy | `{dependencies.get("numpy", "unknown")}` |
| Pillow | `{dependencies.get("pillow", "unknown")}` |
| Matplotlib | `{dependencies.get("matplotlib", "unknown")}` |

기존 NumPy와의 호환성을 유지하면서 기본 `opencv-python`만 추가했다. `opencv-contrib-python`, `scikit-image`, YOLO Framework 등은 Day 10 범위에 필요하지 않아 추가하지 않았다.

## 10. 테스트

```text
Synthetic-image unit and integration tests : {inputs.day10_test_count} passed
Full project regression tests              : {inputs.regression_test_count} passed
Warnings                                   : {inputs.warning_count}
Runtime                                    : {inputs.regression_runtime_seconds:.2f} seconds
```

검정·흰색·Gradient·사각형·Noise 합성 이미지를 사용해 Config, 채널 변환, 단계별 Shape·dtype, 입력 불변성, 결정론, Metrics, PNG 생성, 실제 샘플 실행 Script 및 육안 검증 Script를 확인했다.

현재 Warning은 기존 Starlette TestClient와 httpx 조합에서 발생하는 기술부채이며 Day 10 기능 실패가 아니다. OpenCV 추가 과정에서 기존 API Dependency를 무리하게 변경하지 않았다.

## 11. 한계와 해석 정책

- Contour는 Threshold·Morphology 파라미터에 반응한 후보 형태다.
- 조명, 그림자, 제품 외곽선과 정상 Texture도 Contour가 될 수 있다.
- Contour에는 결함 Class와 Confidence가 없다.
- Contour를 Pascal VOC XML 대신 Detection Target으로 사용하지 않는다.
- Casting과 NEU-DET Metric을 모델 성능 비교로 사용하지 않는다.
- Day 10에서는 Detection 학습, mAP 평가, Checkpoint, Detection API를 구현하지 않았다.

## 12. Day 11 연결

Day 11에서는 Day 9의 Pascal VOC XML과 Split Artifact를 사용해 Torchvision Detection Dataset을 구현한다. Day 10 OpenCV 결과는 이미지 입력 품질과 보조 시각화를 확인하는 용도로만 연결하며, Detection Ground Truth는 다음 좌표 변환 정책을 유지한다.

```python
(xmin - 1, ymin - 1, xmax, ymax)
```

OpenCV Contour를 Detection Label이나 Bounding Box로 자동 변환하지 않는다.
"""


def build_readme_section(inputs: Day10DocumentationInputs) -> str:
    samples = inputs.analysis["samples"]
    sample_names = ", ".join(
        sample["semantic_role"] for sample in samples
    )
    return f"""{README_START_MARKER}
## Day 10 — OpenCV Image Analysis Pipeline

OpenCV 기반으로 이미지의 명암·경계·Threshold·형태 특성을 계산하는 독립 보조 분석 파이프라인을 구현했습니다.

```text
Pipeline:
Original → Grayscale → Histogram → CLAHE → Gaussian Blur
→ Canny Edge → Adaptive Threshold → Morphology
→ Contour Candidate Overlay
```

- 실제 분석 샘플: {sample_names}
- 결과 분리: JSON 직렬화 가능한 Metrics / PNG 표시용 이미지
- Figure: Pipeline Overview, Histogram·Metrics, Contour Candidate Analysis
- Day 10 대상 테스트: **{inputs.day10_test_count} passed**
- 전체 회귀 테스트: **{inputs.regression_test_count} passed, {inputs.warning_count} warning**
- 보고서: `reports/day10_opencv_image_analysis_pipeline_summary.md`

> OpenCV Contour는 Threshold·Morphology 기반 후보 형태이며 실제 결함 Ground Truth나 객체 탐지 Bounding Box가 아닙니다.
{README_END_MARKER}"""


def _read_utf8_preserving_bom(path: Path) -> tuple[str, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    return text, has_bom


def _atomic_write_text(
    path: Path,
    text: str,
    *,
    write_bom: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.rstrip() + "\n"
    data = normalized.encode("utf-8")
    if write_bom:
        data = b"\xef\xbb\xbf" + data

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
            temp_path = Path(file.name)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def update_readme_content(readme_text: str, section: str) -> str:
    start_count = readme_text.count(README_START_MARKER)
    end_count = readme_text.count(README_END_MARKER)

    if start_count == 0 and end_count == 0:
        return readme_text.rstrip() + "\n\n" + section.strip() + "\n"

    if start_count != 1 or end_count != 1:
        raise ValueError(
            "README Day 10 markers must both be absent or appear exactly once"
        )

    start_index = readme_text.index(README_START_MARKER)
    end_index = readme_text.index(README_END_MARKER)
    if start_index >= end_index:
        raise ValueError("README Day 10 markers are in the wrong order")

    end_index += len(README_END_MARKER)
    return (
        readme_text[:start_index].rstrip()
        + "\n\n"
        + section.strip()
        + "\n\n"
        + readme_text[end_index:].lstrip()
    ).rstrip() + "\n"


def create_day10_documentation(
    inputs: Day10DocumentationInputs,
    *,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    readme_path: str | Path = DEFAULT_README_PATH,
) -> tuple[Path, Path]:
    report_output = _resolve(inputs.project_root, report_path)
    readme_output = _resolve(inputs.project_root, readme_path)

    readme_text, readme_has_bom = _read_utf8_preserving_bom(readme_output)
    report = build_day10_report(inputs)
    readme_section = build_readme_section(inputs)
    updated_readme = update_readme_content(readme_text, readme_section)

    _atomic_write_text(report_output, report)
    _atomic_write_text(
        readme_output,
        updated_readme,
        write_bom=readme_has_bom,
    )
    return report_output, readme_output


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]

    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--analysis-path", default=DEFAULT_ANALYSIS_PATH)
    parser.add_argument(
        "--visual-validation-path",
        default=DEFAULT_VISUAL_VALIDATION_PATH,
    )
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--readme-path", default=DEFAULT_README_PATH)
    parser.add_argument("--regression-test-count", type=int, required=True)
    parser.add_argument("--warning-count", type=int, required=True)
    parser.add_argument(
        "--regression-runtime-seconds",
        type=float,
        required=True,
    )
    parser.add_argument("--day10-test-count", type=int, default=62)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    inputs = load_documentation_inputs(
        project_root=parsed.project_root,
        analysis_path=parsed.analysis_path,
        visual_validation_path=parsed.visual_validation_path,
        regression_test_count=parsed.regression_test_count,
        warning_count=parsed.warning_count,
        regression_runtime_seconds=parsed.regression_runtime_seconds,
        day10_test_count=parsed.day10_test_count,
    )
    report_path, readme_path = create_day10_documentation(
        inputs,
        report_path=parsed.report_path,
        readme_path=parsed.readme_path,
    )

    print("[PASS] Day 10 report created")
    print("[PASS] README Day 10 section added or updated")
    print(f"[REPORT] {_project_relative(report_path, inputs.project_root)}")
    print(f"[README] {_project_relative(readme_path, inputs.project_root)}")
    print(
        "[TESTS] "
        f"{inputs.regression_test_count} passed, "
        f"{inputs.warning_count} warning(s), "
        f"{inputs.regression_runtime_seconds:.2f} seconds"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
