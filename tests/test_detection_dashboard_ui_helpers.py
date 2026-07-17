from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

from src.dashboard.detection_api_client import (
    DashboardDetection,
    DashboardDetectionBox,
    DashboardDetectionPrediction,
)
from src.dashboard.detection_ui_helpers import (
    build_detection_summary_message,
    build_detection_table_rows,
    format_detection_score,
    render_detection_overlay,
)


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new(
        "RGB",
        (20, 16),
        color=(255, 255, 255),
    ).save(buffer, format="PNG")
    return buffer.getvalue()


def prediction(*, detections) -> DashboardDetectionPrediction:
    return DashboardDetectionPrediction(
        detections=tuple(detections),
        detection_count=len(detections),
        score_threshold=0.5,
        iou_threshold=0.5,
        model_name="model",
        model_version="version",
        architecture="architecture",
        device="cpu",
        checkpoint_epoch=3,
        checkpoint_metric_name="map_50",
        checkpoint_metric_value=0.677418,
        original_filename="sample.png",
        content_type="image/png",
        image_width=20,
        image_height=16,
        image_mode="RGB",
        model_input_mode="RGB",
        inference_time_ms=10.0,
    )


def sample_detection() -> DashboardDetection:
    return DashboardDetection(
        label_id=3,
        label_name="patches",
        score=0.91,
        box=DashboardDetectionBox(
            xmin=2.0,
            ymin=3.0,
            xmax=15.0,
            ymax=12.0,
        ),
    )


def test_overlay_preserves_size_and_changes_pixels() -> None:
    raw = png_bytes()
    result = render_detection_overlay(
        image_bytes=raw,
        prediction=prediction(
            detections=[sample_detection()]
        ),
    )

    assert result.size == (20, 16)

    original = Image.open(BytesIO(raw)).convert("RGB")
    assert result.tobytes() != original.tobytes()


def test_overlay_does_not_modify_input_bytes() -> None:
    raw = png_bytes()
    snapshot = bytes(raw)

    render_detection_overlay(
        image_bytes=raw,
        prediction=prediction(
            detections=[sample_detection()]
        ),
    )

    assert raw == snapshot


def test_overlay_rejects_dimension_mismatch() -> None:
    value = prediction(
        detections=[sample_detection()]
    )
    object.__setattr__(value, "image_width", 21)

    with pytest.raises(ValueError, match="dimensions"):
        render_detection_overlay(
            image_bytes=png_bytes(),
            prediction=value,
        )


def test_detection_table_has_short_tag_and_box() -> None:
    rows = build_detection_table_rows(
        prediction(detections=[sample_detection()])
    )

    assert rows == [
        {
            "Rank": 1,
            "Tag": "P1",
            "Class ID": 3,
            "Class": "patches",
            "Score": 0.91,
            "xmin": 2.0,
            "ymin": 3.0,
            "xmax": 15.0,
            "ymax": 12.0,
        }
    ]


def test_empty_summary_is_not_ground_truth_claim() -> None:
    message = build_detection_summary_message(
        prediction(detections=[])
    )

    assert "Prediction" in message
    assert "Ground Truth" not in message


def test_format_detection_score() -> None:
    assert format_detection_score(0.91234) == "91.23%"
