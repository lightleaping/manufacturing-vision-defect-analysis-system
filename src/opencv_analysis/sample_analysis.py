"""실제 이미지 한 장을 OpenCV 파이프라인으로 안전하게 분석한다.

이 모듈은 Dataset 전체를 순회하지 않는다. 호출자가 명시한 이미지 경로만 열고,
OpenCV 처리 결과와 JSON 직렬화 가능한 메타데이터·지표를 함께 반환한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

from .config import OpenCVAnalysisConfig
from .image_conversion import pillow_to_bgr
from .metrics import OpenCVAnalysisMetrics, calculate_metrics
from .pipeline import OpenCVPipelineResult, run_opencv_pipeline


SUPPORTED_IMAGE_SUFFIXES: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
_SAMPLE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class ImageSampleSpec:
    """Day 10 실제 분석 대상 이미지의 의미와 상대 경로."""

    sample_id: str
    dataset_name: str
    semantic_role: str
    class_name: str
    relative_path: str

    def __post_init__(self) -> None:
        for name, value in (
            ("sample_id", self.sample_id),
            ("dataset_name", self.dataset_name),
            ("semantic_role", self.semantic_role),
            ("class_name", self.class_name),
            ("relative_path", self.relative_path),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")

        if _SAMPLE_ID_PATTERN.fullmatch(self.sample_id) is None:
            raise ValueError(
                "sample_id must contain lowercase letters, numbers, '_' or '-'"
            )

        normalized = self.relative_path.replace("\\", "/")
        pure_path = PurePosixPath(normalized)
        if pure_path.is_absolute():
            raise ValueError("relative_path must be relative")
        if ".." in pure_path.parts:
            raise ValueError("relative_path must not contain '..'")
        if pure_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
            raise ValueError(
                "relative_path must use one of: .jpg, .jpeg, .png"
            )

    def normalized_relative_path(self) -> str:
        """Artifact에서 운영체제와 무관한 POSIX 상대 경로를 사용한다."""
        return PurePosixPath(self.relative_path.replace("\\", "/")).as_posix()


@dataclass(frozen=True, slots=True)
class AnalyzedImageSample:
    """한 이미지의 원본 정보, OpenCV 배열 결과, 보조 지표."""

    spec: ImageSampleSpec
    absolute_path: Path
    file_size_bytes: int
    sha256: str
    source_format: str
    source_mode: str
    source_width: int
    source_height: int
    pipeline_result: OpenCVPipelineResult
    metrics: OpenCVAnalysisMetrics

    def to_artifact_record(self) -> dict[str, Any]:
        """ndarray를 제외하고 JSON으로 저장 가능한 필드만 반환한다."""
        return {
            "sample_id": self.spec.sample_id,
            "dataset_name": self.spec.dataset_name,
            "semantic_role": self.spec.semantic_role,
            "class_name": self.spec.class_name,
            "image_path": self.spec.normalized_relative_path(),
            "filename": self.absolute_path.name,
            "file_size_bytes": self.file_size_bytes,
            "sha256": self.sha256,
            "source_image": {
                "format": self.source_format,
                "mode": self.source_mode,
                "width": self.source_width,
                "height": self.source_height,
            },
            "metrics": self.metrics.to_dict(),
            "interpretation": (
                "OpenCV statistics and threshold-based contour candidates; "
                "not object-detection predictions or ground truth."
            ),
        }


def _resolve_project_image_path(project_root: Path, relative_path: str) -> Path:
    root = project_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"project_root not found: {root}")

    normalized = PurePosixPath(relative_path.replace("\\", "/"))
    candidate = root.joinpath(*normalized.parts).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("image path must stay inside project_root") from error

    if not candidate.is_file():
        raise FileNotFoundError(f"image file not found: {candidate}")
    if candidate.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ValueError(f"unsupported image extension: {candidate.suffix}")
    if candidate.stat().st_size <= 0:
        raise ValueError(f"image file is empty: {candidate}")
    return candidate


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_image_sample(
    project_root: str | Path,
    spec: ImageSampleSpec,
    config: OpenCVAnalysisConfig | None = None,
) -> AnalyzedImageSample:
    """명시된 JPEG·PNG 한 장을 열어 OpenCV 분석과 Metrics를 계산한다."""
    if not isinstance(spec, ImageSampleSpec):
        raise TypeError("spec must be an ImageSampleSpec")

    root = Path(project_root)
    image_path = _resolve_project_image_path(
        root,
        spec.normalized_relative_path(),
    )

    try:
        with Image.open(image_path) as image:
            source_format = str(image.format or "")
            source_mode = str(image.mode)
            source_width = int(image.width)
            source_height = int(image.height)
            image.load()
            bgr = pillow_to_bgr(image)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"failed to decode image: {image_path}") from error

    pipeline_result = run_opencv_pipeline(bgr, config)
    metrics = calculate_metrics(pipeline_result)

    return AnalyzedImageSample(
        spec=spec,
        absolute_path=image_path,
        file_size_bytes=int(image_path.stat().st_size),
        sha256=_sha256_file(image_path),
        source_format=source_format,
        source_mode=source_mode,
        source_width=source_width,
        source_height=source_height,
        pipeline_result=pipeline_result,
        metrics=metrics,
    )


def analyze_image_samples(
    project_root: str | Path,
    specs: Iterable[ImageSampleSpec],
    config: OpenCVAnalysisConfig | None = None,
) -> tuple[AnalyzedImageSample, ...]:
    """명시된 소수 샘플만 입력 순서대로 분석한다."""
    resolved_specs = tuple(specs)
    if not resolved_specs:
        raise ValueError("specs must contain at least one sample")
    if not all(isinstance(spec, ImageSampleSpec) for spec in resolved_specs):
        raise TypeError("every item in specs must be an ImageSampleSpec")

    sample_ids = [spec.sample_id for spec in resolved_specs]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample_id values must be unique")

    return tuple(
        analyze_image_sample(project_root, spec, config)
        for spec in resolved_specs
    )
