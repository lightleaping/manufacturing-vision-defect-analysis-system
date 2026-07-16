"""Day 6 ResNet18 Grad-CAM 실제 Artifact 생성 Pipeline.

이 모듈은 모델별 Hook 계산 자체를 담당하는 ``gradcam.py``와
표본 선택·시각화 모듈을 연결한다.

핵심 책임
---------
1. Day 4·Day 5 JSON을 읽고 서로 일치하는지 검증한다.
2. Day 4 평가 결과에서 대표 표본 7장을 선택한다.
3. Day 2 Test 전처리와 같은 224×224·ImageNet Normalize를 적용한다.
4. 예측 Class 관점의 Grad-CAM을 표본별 Batch Size 1로 계산한다.
5. 재추론 Logit·Probability·Prediction을 Day 4 JSON과 대조한다.
6. Metadata JSON과 Overview PNG를 Atomic 방식으로 저장한다.

Grad-CAM은 모델이 상대적으로 주목한 영역을 보여주는 설명 보조 수단이며,
실제 결함 위치의 정답 Mask나 Detection 결과가 아니다.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torchvision import transforms

from src.explainability.gradcam import GradCAM, GradCAMResult
from src.explainability.gradcam_sample_selector import (
    SelectedGradCAMSample,
    select_gradcam_samples,
)
from src.explainability.gradcam_visualization import (
    GradCAMVisualizationRecord,
    save_gradcam_overview,
)

PROJECT_NAME: Final[str] = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN: Final[str] = "제조 비전 결함 분석 시스템"
RUN_NAME: Final[str] = "day6_resnet18_gradcam_explainability"

IMAGE_SIZE: Final[tuple[int, int]] = (224, 224)
IMAGENET_MEAN: Final[tuple[float, float, float]] = (
    0.485,
    0.456,
    0.406,
)
IMAGENET_STD: Final[tuple[float, float, float]] = (
    0.229,
    0.224,
    0.225,
)
CLASSIFICATION_THRESHOLD: Final[float] = 0.5
DEFAULT_TARGET_LAYER_NAME: Final[str] = "resnet18.layer4.1.conv2"
DEFAULT_ALPHA: Final[float] = 0.40
DEFAULT_COLORMAP_NAME: Final[str] = "jet"

HIGH_CONFIDENCE_ERROR_TYPES: Final[frozenset[str]] = frozenset(
    {
        "HIGH_CONFIDENCE_FALSE_POSITIVE_1",
        "HIGH_CONFIDENCE_FALSE_POSITIVE_2",
        "HIGH_CONFIDENCE_FALSE_NEGATIVE",
    }
)
BOUNDARY_ERROR_TYPES: Final[frozenset[str]] = frozenset(
    {
        "BOUNDARY_FALSE_POSITIVE",
        "BOUNDARY_FALSE_NEGATIVE",
    }
)

ImageTransform = Callable[[Image.Image], Tensor]


class GradCAMPipelineError(RuntimeError):
    """Day 6 실제 연결·검증·Artifact 저장 과정의 공통 예외입니다."""


@dataclass(frozen=True)
class GradCAMArtifactPaths:
    """Day 6이 생성할 JSON과 PNG 경로입니다."""

    metadata_path: Path
    overview_figure_path: Path
    high_confidence_figure_path: Path
    boundary_figure_path: Path

    def all_paths(self) -> tuple[Path, ...]:
        return (
            self.metadata_path,
            self.overview_figure_path,
            self.high_confidence_figure_path,
            self.boundary_figure_path,
        )


@dataclass(frozen=True)
class GradCAMGeneratedSample:
    """표본 선택 정보, Grad-CAM 결과와 시각화 입력을 묶은 값입니다."""

    selected_sample: SelectedGradCAMSample
    source_sample: Mapping[str, object]
    gradcam_result: GradCAMResult
    visualization_record: GradCAMVisualizationRecord
    raw_logit_absolute_error: float
    probability_absolute_error: float

    def to_metadata_dict(self, *, project_root: Path) -> dict[str, object]:
        source_image_path = _format_project_relative_path(
            Path(self.visualization_record.image_path),
            project_root=project_root,
        )

        return {
            **self.selected_sample.to_dict(),
            "image_path": source_image_path,
            "source_evaluation": {
                "raw_logit": float(self.source_sample["raw_logit"]),
                "defect_probability": float(
                    self.source_sample["defect_probability"]
                ),
                "prediction": int(self.source_sample["prediction"]),
            },
            "gradcam": self.gradcam_result.to_metadata_dict(),
            "reproduction_check": {
                "raw_logit_absolute_error": self.raw_logit_absolute_error,
                "probability_absolute_error": self.probability_absolute_error,
                "prediction_matches": (
                    self.gradcam_result.prediction
                    == int(self.source_sample["prediction"])
                ),
            },
        }


@dataclass(frozen=True)
class GradCAMAnalysisResult:
    """Day 6 실제 Pipeline 완료 결과입니다."""

    generated_samples: tuple[GradCAMGeneratedSample, ...]
    artifact_paths: GradCAMArtifactPaths
    duration_seconds: float


# =============================================================================
# JSON·Path Utility
# =============================================================================


def read_json_object(path: Path) -> dict[str, Any]:
    """UTF-8 JSON을 읽고 최상위 Object 형식을 검증합니다."""

    if not isinstance(path, Path):
        raise TypeError("path는 pathlib.Path여야 합니다.")
    if not path.is_file():
        raise FileNotFoundError(f"JSON 파일이 존재하지 않습니다: {path}")

    try:
        with path.open(mode="r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise GradCAMPipelineError(
            f"JSON 파일을 읽을 수 없습니다: {path}"
        ) from exc

    if not isinstance(payload, dict):
        raise GradCAMPipelineError(
            "JSON 최상위 값은 Object여야 합니다. "
            f"실제 타입={type(payload).__name__}"
        )

    return payload


def write_json_atomically(*, payload: Mapping[str, object], output_path: Path) -> Path:
    """JSON을 임시 파일에 기록한 뒤 ``os.replace``로 교체합니다."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload는 Mapping이어야 합니다.")
    if output_path.suffix.lower() != ".json":
        raise ValueError("output_path는 .json 확장자를 사용해야 합니다.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".json.tmp",
            prefix=f".{output_path.stem}_",
            dir=output_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                dict(payload),
                temporary_file,
                ensure_ascii=False,
                indent=2,
            )
            temporary_file.write("\n")

        os.replace(temporary_path, output_path)
    except (OSError, TypeError, ValueError) as exc:
        raise GradCAMPipelineError(
            f"Grad-CAM Metadata JSON 저장에 실패했습니다: {output_path}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    if not output_path.is_file() or output_path.stat().st_size <= 0:
        raise GradCAMPipelineError(
            f"저장된 Metadata JSON이 비어 있습니다: {output_path}"
        )

    return output_path


def _format_project_relative_path(path: Path, *, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_image_path(*, project_root: Path, image_path: str) -> Path:
    """Day 4 JSON의 상대·절대 이미지 경로를 실제 절대 경로로 변환합니다."""

    if not isinstance(image_path, str) or not image_path.strip():
        raise GradCAMPipelineError("image_path는 비어 있지 않은 문자열이어야 합니다.")

    candidate = Path(image_path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )

    if not resolved.is_file():
        raise FileNotFoundError(f"Grad-CAM 대상 이미지가 없습니다: {resolved}")

    return resolved


# =============================================================================
# Transform·Image Loading
# =============================================================================


def create_day6_test_transform() -> transforms.Compose:
    """Day 2 Validation·Test와 같은 결정적 전처리를 생성합니다.

    Resize 224×224 → ToTensor → ImageNet Normalize 순서를 사용합니다.
    Train 전용 Random Augmentation은 Grad-CAM 재현 단계에서 사용하지 않습니다.
    """

    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def load_model_input_tensor(
    *,
    image_path: Path,
    image_transform: ImageTransform,
    device: torch.device,
) -> Tensor:
    """이미지를 검증·RGB 변환하고 [1, 3, 224, 224] Tensor로 만듭니다."""

    if not image_path.is_file():
        raise FileNotFoundError(f"이미지 파일이 존재하지 않습니다: {image_path}")
    if not callable(image_transform):
        raise TypeError("image_transform은 호출 가능 객체여야 합니다.")
    if not isinstance(device, torch.device):
        raise TypeError("device는 torch.device여야 합니다.")

    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            transformed = image_transform(rgb_image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise GradCAMPipelineError(
            f"Grad-CAM 입력 이미지를 읽거나 변환할 수 없습니다: {image_path}"
        ) from exc

    if not isinstance(transformed, Tensor):
        raise GradCAMPipelineError(
            "image_transform 출력은 torch.Tensor여야 합니다. "
            f"실제 타입={type(transformed).__name__}"
        )
    if transformed.ndim != 3 or tuple(transformed.shape) != (3, 224, 224):
        raise GradCAMPipelineError(
            "전처리 출력은 [3, 224, 224]여야 합니다. "
            f"실제 Shape={tuple(transformed.shape)}"
        )
    if not torch.is_floating_point(transformed):
        raise GradCAMPipelineError("전처리 출력 Tensor는 실수형이어야 합니다.")
    if not torch.isfinite(transformed).all():
        raise GradCAMPipelineError(
            "전처리 출력 Tensor에 NaN 또는 Infinity가 포함되어 있습니다."
        )

    return transformed.unsqueeze(0).to(device)


# =============================================================================
# Source Artifact Validation
# =============================================================================


def extract_day4_sample_results(
    day4_payload: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Day 4 JSON의 ``sample_results`` 배열을 검증해 반환합니다."""

    sample_results = day4_payload.get("sample_results")
    if not isinstance(sample_results, list) or not sample_results:
        raise GradCAMPipelineError(
            "Day 4 JSON의 sample_results는 비어 있지 않은 List여야 합니다."
        )
    if not all(isinstance(item, Mapping) for item in sample_results):
        raise GradCAMPipelineError(
            "Day 4 sample_results의 모든 항목은 Object여야 합니다."
        )

    return list(sample_results)


def validate_day4_day5_cross_reference(
    *,
    day4_sample_results: Sequence[Mapping[str, object]],
    day5_payload: Mapping[str, object],
) -> None:
    """Day 4 오분류와 Day 5 ``misclassifications``가 같은지 검증합니다."""

    day5_misclassifications = day5_payload.get("misclassifications")
    if not isinstance(day5_misclassifications, list):
        raise GradCAMPipelineError(
            "Day 5 JSON의 misclassifications는 List여야 합니다."
        )
    if not all(isinstance(item, Mapping) for item in day5_misclassifications):
        raise GradCAMPipelineError(
            "Day 5 misclassifications의 모든 항목은 Object여야 합니다."
        )

    day4_incorrect = {
        int(sample["sample_index"]): sample
        for sample in day4_sample_results
        if sample.get("correct") is False
    }

    day5_lookup: dict[int, Mapping[str, object]] = {}
    for item in day5_misclassifications:
        if "sample_index" not in item:
            raise GradCAMPipelineError(
                "Day 5 오분류 항목에 sample_index가 없습니다."
            )
        sample_index = int(item["sample_index"])
        if sample_index in day5_lookup:
            raise GradCAMPipelineError(
                f"Day 5 오분류에 중복 sample_index가 있습니다: {sample_index}"
            )
        day5_lookup[sample_index] = item

    if set(day4_incorrect) != set(day5_lookup):
        missing = sorted(set(day4_incorrect) - set(day5_lookup))
        unexpected = sorted(set(day5_lookup) - set(day4_incorrect))
        raise GradCAMPipelineError(
            "Day 4와 Day 5 오분류 sample_index가 일치하지 않습니다. "
            f"Day5 누락={missing}, Day5 추가={unexpected}"
        )

    for sample_index, day4_sample in day4_incorrect.items():
        day5_item = day5_lookup[sample_index]
        for key in (
            "image_path",
            "ground_truth_label",
            "prediction",
        ):
            if key in day5_item and day5_item[key] != day4_sample.get(key):
                raise GradCAMPipelineError(
                    "Day 4와 Day 5 오분류 값이 다릅니다. "
                    f"sample_index={sample_index}, key={key}"
                )

        if "defect_probability" in day5_item and not math.isclose(
            float(day5_item["defect_probability"]),
            float(day4_sample["defect_probability"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise GradCAMPipelineError(
                "Day 4와 Day 5 defect_probability가 다릅니다. "
                f"sample_index={sample_index}"
            )


def build_sample_lookup(
    sample_results: Sequence[Mapping[str, object]],
) -> dict[int, Mapping[str, object]]:
    """sample_index 중복을 차단하고 빠른 조회 Dictionary를 만듭니다."""

    lookup: dict[int, Mapping[str, object]] = {}
    for sample in sample_results:
        if "sample_index" not in sample:
            raise GradCAMPipelineError("Day 4 표본에 sample_index가 없습니다.")
        sample_index = int(sample["sample_index"])
        if sample_index in lookup:
            raise GradCAMPipelineError(
                f"Day 4 표본에 중복 sample_index가 있습니다: {sample_index}"
            )
        lookup[sample_index] = sample

    return lookup


# =============================================================================
# Grad-CAM Generation
# =============================================================================


def validate_reproduced_prediction(
    *,
    source_sample: Mapping[str, object],
    gradcam_result: GradCAMResult,
    raw_logit_tolerance: float = 1e-4,
    probability_tolerance: float = 1e-5,
) -> tuple[float, float]:
    """Grad-CAM Forward 결과가 Day 4 저장 결과와 같은지 검증합니다."""

    if raw_logit_tolerance < 0.0 or probability_tolerance < 0.0:
        raise ValueError("재현 오차 허용값은 0 이상이어야 합니다.")

    required_keys = {
        "raw_logit",
        "defect_probability",
        "prediction",
    }
    missing_keys = required_keys - set(source_sample)
    if missing_keys:
        raise GradCAMPipelineError(
            f"Day 4 표본에 재현 검증 Key가 없습니다: {sorted(missing_keys)}"
        )

    source_logit = float(source_sample["raw_logit"])
    source_probability = float(source_sample["defect_probability"])
    source_prediction = int(source_sample["prediction"])

    raw_logit_absolute_error = abs(gradcam_result.raw_logit - source_logit)
    probability_absolute_error = abs(
        gradcam_result.defect_probability - source_probability
    )

    if raw_logit_absolute_error > raw_logit_tolerance:
        raise GradCAMPipelineError(
            "Grad-CAM 재추론 Logit이 Day 4 JSON과 다릅니다. "
            f"sample_index={source_sample.get('sample_index')}, "
            f"absolute_error={raw_logit_absolute_error:.12f}"
        )
    if probability_absolute_error > probability_tolerance:
        raise GradCAMPipelineError(
            "Grad-CAM 재추론 Probability가 Day 4 JSON과 다릅니다. "
            f"sample_index={source_sample.get('sample_index')}, "
            f"absolute_error={probability_absolute_error:.12f}"
        )
    if gradcam_result.prediction != source_prediction:
        raise GradCAMPipelineError(
            "Grad-CAM 재추론 Prediction이 Day 4 JSON과 다릅니다. "
            f"sample_index={source_sample.get('sample_index')}"
        )

    return raw_logit_absolute_error, probability_absolute_error


def generate_gradcam_samples(
    *,
    model: nn.Module,
    target_layer_name: str,
    selected_samples: Sequence[SelectedGradCAMSample],
    source_sample_lookup: Mapping[int, Mapping[str, object]],
    project_root: Path,
    device: torch.device,
    image_transform: ImageTransform,
    raw_logit_tolerance: float = 1e-4,
    probability_tolerance: float = 1e-5,
) -> tuple[GradCAMGeneratedSample, ...]:
    """선택된 표본을 Batch Size 1로 순서대로 Grad-CAM 처리합니다."""

    if not isinstance(model, nn.Module):
        raise TypeError("model은 nn.Module이어야 합니다.")
    if not selected_samples:
        raise GradCAMPipelineError("Grad-CAM으로 처리할 선택 표본이 없습니다.")
    if not isinstance(device, torch.device):
        raise TypeError("device는 torch.device여야 합니다.")

    model.to(device)
    model.eval()
    generated_samples: list[GradCAMGeneratedSample] = []

    with torch.enable_grad():
        with GradCAM(
            model=model,
            target_layer_name=target_layer_name,
        ) as gradcam:
            for selected_sample in selected_samples:
                source_sample = source_sample_lookup.get(
                    selected_sample.sample_index
                )
                if source_sample is None:
                    raise GradCAMPipelineError(
                        "선택 표본을 Day 4 source lookup에서 찾지 못했습니다. "
                        f"sample_index={selected_sample.sample_index}"
                    )

                absolute_image_path = resolve_image_path(
                    project_root=project_root,
                    image_path=selected_sample.image_path,
                )
                input_tensor = load_model_input_tensor(
                    image_path=absolute_image_path,
                    image_transform=image_transform,
                    device=device,
                )

                # target_class=None: 모델이 실제로 선택한 예측 Class를 설명한다.
                gradcam_result = gradcam.generate(
                    input_tensor=input_tensor,
                    target_class=None,
                    resize_to_input=True,
                )

                raw_logit_error, probability_error = (
                    validate_reproduced_prediction(
                        source_sample=source_sample,
                        gradcam_result=gradcam_result,
                        raw_logit_tolerance=raw_logit_tolerance,
                        probability_tolerance=probability_tolerance,
                    )
                )

                visualization_record = GradCAMVisualizationRecord(
                    sample_index=selected_sample.sample_index,
                    image_path=str(absolute_image_path),
                    selection_type=selected_sample.selection_type,
                    ground_truth_class_name=(
                        selected_sample.ground_truth_class_name
                    ),
                    prediction_class_name=(
                        selected_sample.prediction_class_name
                    ),
                    defect_probability=gradcam_result.defect_probability,
                    target_class=gradcam_result.target_class,
                    target_layer_name=gradcam_result.target_layer_name,
                    cam=gradcam_result.cam.detach().cpu().numpy(),
                )

                generated_samples.append(
                    GradCAMGeneratedSample(
                        selected_sample=selected_sample,
                        source_sample=source_sample,
                        gradcam_result=gradcam_result,
                        visualization_record=visualization_record,
                        raw_logit_absolute_error=raw_logit_error,
                        probability_absolute_error=probability_error,
                    )
                )

        # Context 종료 뒤 Forward·Gradient Hook가 반드시 해제되어야 한다.
        if not gradcam.is_closed:
            raise GradCAMPipelineError("Grad-CAM Hook가 정상적으로 해제되지 않았습니다.")

    return tuple(generated_samples)


# =============================================================================
# Figure·Metadata
# =============================================================================


def _select_visualization_records(
    generated_samples: Sequence[GradCAMGeneratedSample],
    *,
    selection_types: frozenset[str] | None,
) -> list[GradCAMVisualizationRecord]:
    records = [
        item.visualization_record
        for item in generated_samples
        if selection_types is None
        or item.selected_sample.selection_type in selection_types
    ]
    if not records:
        raise GradCAMPipelineError("선택 조건에 맞는 Grad-CAM Figure 표본이 없습니다.")
    return records


def save_gradcam_figures(
    *,
    generated_samples: Sequence[GradCAMGeneratedSample],
    artifact_paths: GradCAMArtifactPaths,
    alpha: float = DEFAULT_ALPHA,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> None:
    """전체·고확신 오류·경계 오류 Figure 세 개를 저장합니다."""

    save_gradcam_overview(
        records=_select_visualization_records(
            generated_samples,
            selection_types=None,
        ),
        output_path=artifact_paths.overview_figure_path,
        alpha=alpha,
        colormap_name=colormap_name,
        figure_title="Day 6 - ResNet18 Grad-CAM Overview",
    )
    save_gradcam_overview(
        records=_select_visualization_records(
            generated_samples,
            selection_types=HIGH_CONFIDENCE_ERROR_TYPES,
        ),
        output_path=artifact_paths.high_confidence_figure_path,
        alpha=alpha,
        colormap_name=colormap_name,
        figure_title="Day 6 - High-Confidence Misclassification Grad-CAM",
    )
    save_gradcam_overview(
        records=_select_visualization_records(
            generated_samples,
            selection_types=BOUNDARY_ERROR_TYPES,
        ),
        output_path=artifact_paths.boundary_figure_path,
        alpha=alpha,
        colormap_name=colormap_name,
        figure_title="Day 6 - Decision-Boundary Error Grad-CAM",
    )


def build_metadata_payload(
    *,
    generated_samples: Sequence[GradCAMGeneratedSample],
    artifact_paths: GradCAMArtifactPaths,
    project_root: Path,
    checkpoint_path: Path,
    day4_evaluation_path: Path,
    day5_analysis_path: Path,
    device: torch.device,
    target_layer_name: str,
    duration_seconds: float,
    alpha: float = DEFAULT_ALPHA,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> dict[str, object]:
    """Day 6 JSON Artifact의 직렬화 가능한 전체 구조를 만듭니다."""

    if not generated_samples:
        raise GradCAMPipelineError("Metadata에 기록할 Grad-CAM 표본이 없습니다.")

    return {
        "project": {
            "name": PROJECT_NAME,
            "name_korean": PROJECT_NAME_KOREAN,
            "run_name": RUN_NAME,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "source_artifacts": {
            "checkpoint_path": _format_project_relative_path(
                checkpoint_path,
                project_root=project_root,
            ),
            "day4_evaluation_path": _format_project_relative_path(
                day4_evaluation_path,
                project_root=project_root,
            ),
            "day5_analysis_path": _format_project_relative_path(
                day5_analysis_path,
                project_root=project_root,
            ),
        },
        "gradcam_configuration": {
            "target_layer": target_layer_name,
            "batch_size": 1,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
            "target_policy": "predicted_class",
            "defect_target_score": "raw_logit",
            "normal_target_score": "negative_raw_logit",
            "channel_weight": "spatial_mean_of_gradients",
            "weighted_activation": "channel_weighted_sum",
            "relu": True,
            "normalization": "min_max_0_to_1",
            "input_image_size": list(IMAGE_SIZE),
            "input_normalization": "ImageNet mean/std",
            "overlay_alpha": alpha,
            "colormap": colormap_name,
        },
        "selection_summary": {
            "selected_sample_count": len(generated_samples),
            "selection_types": [
                item.selected_sample.selection_type
                for item in generated_samples
            ],
        },
        "samples": [
            item.to_metadata_dict(project_root=project_root)
            for item in generated_samples
        ],
        "runtime": {
            "duration_seconds": duration_seconds,
        },
        "figures": {
            "overview": _format_project_relative_path(
                artifact_paths.overview_figure_path,
                project_root=project_root,
            ),
            "high_confidence_errors": _format_project_relative_path(
                artifact_paths.high_confidence_figure_path,
                project_root=project_root,
            ),
            "boundary_errors": _format_project_relative_path(
                artifact_paths.boundary_figure_path,
                project_root=project_root,
            ),
        },
        "limitations": [
            "Grad-CAM은 모델이 상대적으로 주목한 영역을 보여주는 설명 보조 수단이다.",
            "Heatmap은 실제 결함 위치의 정답 Mask 또는 Detection 결과가 아니다.",
            "강한 활성 영역만으로 예측 오류의 인과관계를 단정할 수 없다.",
        ],
    }


# =============================================================================
# Public Pipeline
# =============================================================================


def run_gradcam_analysis(
    *,
    model: nn.Module,
    project_root: Path,
    checkpoint_path: Path,
    day4_evaluation_path: Path,
    day5_analysis_path: Path,
    artifact_paths: GradCAMArtifactPaths,
    device: torch.device,
    target_layer_name: str = DEFAULT_TARGET_LAYER_NAME,
    image_transform: ImageTransform | None = None,
    alpha: float = DEFAULT_ALPHA,
    colormap_name: str = DEFAULT_COLORMAP_NAME,
) -> GradCAMAnalysisResult:
    """실제 Day 4·5 Artifact와 복원 모델로 Day 6 결과를 생성합니다."""

    started_at = time.perf_counter()

    day4_payload = read_json_object(day4_evaluation_path)
    day5_payload = read_json_object(day5_analysis_path)
    sample_results = extract_day4_sample_results(day4_payload)

    validate_day4_day5_cross_reference(
        day4_sample_results=sample_results,
        day5_payload=day5_payload,
    )

    selected_samples = select_gradcam_samples(
        sample_results,
        threshold=CLASSIFICATION_THRESHOLD,
    )
    sample_lookup = build_sample_lookup(sample_results)

    generated_samples = generate_gradcam_samples(
        model=model,
        target_layer_name=target_layer_name,
        selected_samples=selected_samples,
        source_sample_lookup=sample_lookup,
        project_root=project_root,
        device=device,
        image_transform=(
            image_transform
            if image_transform is not None
            else create_day6_test_transform()
        ),
    )

    save_gradcam_figures(
        generated_samples=generated_samples,
        artifact_paths=artifact_paths,
        alpha=alpha,
        colormap_name=colormap_name,
    )

    duration_seconds = time.perf_counter() - started_at
    metadata_payload = build_metadata_payload(
        generated_samples=generated_samples,
        artifact_paths=artifact_paths,
        project_root=project_root,
        checkpoint_path=checkpoint_path,
        day4_evaluation_path=day4_evaluation_path,
        day5_analysis_path=day5_analysis_path,
        device=device,
        target_layer_name=target_layer_name,
        duration_seconds=duration_seconds,
        alpha=alpha,
        colormap_name=colormap_name,
    )
    write_json_atomically(
        payload=metadata_payload,
        output_path=artifact_paths.metadata_path,
    )

    for path in artifact_paths.all_paths():
        if not path.is_file() or path.stat().st_size <= 0:
            raise GradCAMPipelineError(
                f"Day 6 Artifact가 생성되지 않았거나 비어 있습니다: {path}"
            )

    return GradCAMAnalysisResult(
        generated_samples=generated_samples,
        artifact_paths=artifact_paths,
        duration_seconds=duration_seconds,
    )
