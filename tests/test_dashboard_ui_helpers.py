from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from src.dashboard.api_client import DashboardApiError, DashboardPrediction
from src.dashboard.ui_helpers import (
    build_error_message,
    build_image_metadata_caption,
    build_prediction_message,
    format_file_size,
    format_inference_time,
    format_probability,
    inspect_uploaded_image,
    resolve_content_type,
)


def _prediction(*, class_name: str) -> DashboardPrediction:
    prediction = 1 if class_name == "DEFECT" else 0
    probability = 0.9 if prediction == 1 else 0.1
    return DashboardPrediction(
        prediction=prediction,
        prediction_class_name=class_name,
        defect_probability=probability,
        normal_probability=1.0 - probability,
        raw_logit=2.1972246 if prediction == 1 else -2.1972246,
        classification_threshold=0.5,
        model_name="ResNet18Transfer",
        model_version="resnet18_transfer_best",
        positive_class="DEFECT",
        original_filename="sample.png",
        content_type="image/png",
        image_width=300,
        image_height=300,
        image_mode="RGB",
        inference_time_ms=12.3456,
    )


def _png_bytes(*, mode: str = "RGB") -> bytes:
    color = (10, 20, 30) if mode == "RGB" else 100
    image = Image.new(mode, (32, 24), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


def test_probability_and_inference_time_formatting() -> None:
    assert format_probability(0.999903678894) == "99.99%"
    assert format_probability(0.013476800174) == "1.35%"
    assert format_inference_time(54.8322) == "54.83 ms"


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        format_probability(1.1)


@pytest.mark.parametrize(
    ("byte_size", "expected"),
    [
        (512, "512 B"),
        (2048, "2.00 KB"),
        (2 * 1024 * 1024, "2.00 MB"),
    ],
)
def test_file_size_formatting(byte_size: int, expected: str) -> None:
    assert format_file_size(byte_size) == expected


def test_prediction_messages_distinguish_normal_and_defect() -> None:
    assert "불량" in build_prediction_message(_prediction(class_name="DEFECT"))
    assert "정상" in build_prediction_message(_prediction(class_name="NORMAL"))


def test_error_message_contains_only_safe_dashboard_message() -> None:
    error = DashboardApiError(
        code="INVALID_IMAGE",
        message=r"C:\secret\image.png",
    )

    message = build_error_message(error)

    assert "INVALID_IMAGE" in message
    assert "secret" not in message


def test_content_type_falls_back_to_filename_extension() -> None:
    assert (
        resolve_content_type(filename="sample.jpeg", declared_content_type=None)
        == "image/jpeg"
    )
    assert (
        resolve_content_type(filename="sample.png", declared_content_type="")
        == "image/png"
    )


def test_uploaded_image_metadata_is_read_in_memory() -> None:
    metadata = inspect_uploaded_image(
        filename=r"C:\fake\sample.png",
        image_bytes=_png_bytes(),
    )

    assert metadata.filename == "sample.png"
    assert metadata.width == 32
    assert metadata.height == 24
    assert metadata.mode == "RGB"
    assert metadata.image_format == "PNG"
    assert "32×24" in build_image_metadata_caption(metadata)


def test_invalid_preview_image_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be previewed"):
        inspect_uploaded_image(filename="broken.png", image_bytes=b"broken")
