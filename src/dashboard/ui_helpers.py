"""Streamlit 표시용 순수 Helper.

확률 재계산이나 모델 판정은 수행하지 않고 FastAPI 응답을 보기 좋은 문자열로만
변환한다. 순수 함수로 분리하여 Streamlit 실행 없이 단위 테스트할 수 있다.
"""

from __future__ import annotations

import math
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.dashboard.api_client import DashboardApiError, DashboardPrediction


@dataclass(frozen=True, slots=True)
class UploadedImageMetadata:
    filename: str
    byte_size: int
    width: int
    height: int
    mode: str
    image_format: str


def format_probability(probability: float, *, digits: int = 2) -> str:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be a finite value between 0 and 1")
    return f"{probability * 100.0:.{digits}f}%"


def format_inference_time(inference_time_ms: float) -> str:
    if not math.isfinite(inference_time_ms) or inference_time_ms < 0.0:
        raise ValueError("inference_time_ms must be a finite non-negative value")
    return f"{inference_time_ms:.2f} ms"


def format_file_size(byte_size: int) -> str:
    if byte_size < 0:
        raise ValueError("byte_size must be non-negative")
    if byte_size < 1024:
        return f"{byte_size} B"
    if byte_size < 1024 * 1024:
        return f"{byte_size / 1024.0:.2f} KB"
    return f"{byte_size / (1024.0 * 1024.0):.2f} MB"


def build_prediction_message(prediction: DashboardPrediction) -> str:
    if prediction.prediction_class_name == "DEFECT":
        return "모델이 업로드 이미지를 불량으로 분류했습니다."
    if prediction.prediction_class_name == "NORMAL":
        return "모델이 업로드 이미지를 정상으로 분류했습니다."
    raise ValueError("prediction_class_name must be NORMAL or DEFECT")


def build_error_message(error: DashboardApiError) -> str:
    return f"[{error.code}] {error.message}"


def resolve_content_type(*, filename: str, declared_content_type: str | None) -> str:
    """Streamlit Upload metadata가 비어 있을 때 확장자로 MIME Type을 보완한다."""

    if declared_content_type in {"image/jpeg", "image/png"}:
        return declared_content_type

    guessed_type, _ = mimetypes.guess_type(filename)
    if guessed_type in {"image/jpeg", "image/png"}:
        return guessed_type
    return declared_content_type or "application/octet-stream"


def inspect_uploaded_image(
    *,
    filename: str,
    image_bytes: bytes,
) -> UploadedImageMetadata:
    """Preview 표시용 이미지 Metadata를 메모리에서 읽는다.

    최종 보안 검증과 판정 책임은 FastAPI에 있으며, 이 함수는 사용자 Preview만
    돕는다.
    """

    if not image_bytes:
        raise ValueError("image_bytes must not be empty")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = image.format or "UNKNOWN"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("uploaded file cannot be previewed as an image") from exc

    return UploadedImageMetadata(
        filename=Path(filename.replace("\\", "/")).name,
        byte_size=len(image_bytes),
        width=width,
        height=height,
        mode=mode,
        image_format=image_format,
    )


def build_image_metadata_caption(metadata: UploadedImageMetadata) -> str:
    return (
        f"파일: {metadata.filename} | "
        f"크기: {format_file_size(metadata.byte_size)} | "
        f"해상도: {metadata.width}×{metadata.height} | "
        f"Mode: {metadata.mode} | "
        f"Format: {metadata.image_format}"
    )
