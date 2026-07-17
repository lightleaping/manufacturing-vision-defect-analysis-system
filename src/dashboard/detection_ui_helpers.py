"""Day 13 Detection 결과 표시용 순수 Helper.

이 모듈은 모델을 로딩하거나 추론하지 않는다. FastAPI가 반환한 원본 이미지
좌표의 Box를 Preview 이미지에 표시하고, Table용 행으로 변환한다.
"""

from __future__ import annotations

from io import BytesIO
import math
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from src.dashboard.detection_api_client import (
    DashboardDetectionPrediction,
    DetectionDashboardApiError,
)


_OVERLAY_COLORS: tuple[str, ...] = (
    "#E53935",
    "#1E88E5",
    "#43A047",
    "#FB8C00",
    "#8E24AA",
    "#00897B",
)


def build_detection_error_message(
    error: DetectionDashboardApiError,
) -> str:
    return f"[{error.code}] {error.message}"


def format_detection_score(
    score: float,
) -> str:
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError(
            "score must be a finite value between 0 and 1"
        )
    return f"{float(score) * 100.0:.2f}%"


def build_detection_summary_message(
    prediction: DashboardDetectionPrediction,
) -> str:
    if prediction.detection_count == 0:
        return (
            "현재 Score Threshold 이상인 결함 Prediction이 없습니다."
        )
    return (
        f"현재 Score Threshold 이상인 결함 Prediction "
        f"{prediction.detection_count}개를 찾았습니다."
    )


def build_detection_table_rows(
    prediction: DashboardDetectionPrediction,
) -> list[dict[str, Any]]:
    """Prediction Table에 전달할 행을 Score 순서대로 만든다."""

    rows: list[dict[str, Any]] = []
    for rank, detection in enumerate(
        prediction.detections,
        start=1,
    ):
        rows.append(
            {
                "Rank": rank,
                "Tag": f"P{rank}",
                "Class ID": detection.label_id,
                "Class": detection.label_name,
                "Score": round(detection.score, 6),
                "xmin": round(detection.box.xmin, 2),
                "ymin": round(detection.box.ymin, 2),
                "xmax": round(detection.box.xmax, 2),
                "ymax": round(detection.box.ymax, 2),
            }
        )
    return rows


def render_detection_overlay(
    *,
    image_bytes: bytes,
    prediction: DashboardDetectionPrediction,
    maximum_boxes: int = 8,
) -> Image.Image:
    """원본 이미지에 ``P1`` 같은 짧은 태그와 Box만 표시한다.

    Class·Score·좌표의 자세한 값은 이미지 밖 Prediction Table에서 보여준다.
    이 방식은 긴 Class 이름이 Box와 겹치거나 이미지 경계를 벗어나는 문제를 줄인다.
    """

    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    if not image_bytes:
        raise ValueError("image_bytes must not be empty")
    if (
        not isinstance(maximum_boxes, int)
        or isinstance(maximum_boxes, bool)
        or maximum_boxes <= 0
    ):
        raise ValueError("maximum_boxes must be a positive int")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            source.load()
            image = source.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ValueError(
            "image_bytes cannot be decoded for overlay"
        ) from error

    if image.size != (
        prediction.image_width,
        prediction.image_height,
    ):
        raise ValueError(
            "API image dimensions do not match the uploaded preview"
        )

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(
        2,
        min(image.size) // 100,
    )

    for rank, detection in enumerate(
        prediction.detections[:maximum_boxes],
        start=1,
    ):
        color = _OVERLAY_COLORS[
            (detection.label_id - 1)
            % len(_OVERLAY_COLORS)
        ]
        coordinates = (
            detection.box.xmin,
            detection.box.ymin,
            detection.box.xmax,
            detection.box.ymax,
        )
        draw.rectangle(
            coordinates,
            outline=color,
            width=line_width,
        )

        tag = f"P{rank}"
        tag_bbox = draw.textbbox(
            (0, 0),
            tag,
            font=font,
        )
        tag_width = tag_bbox[2] - tag_bbox[0]
        tag_height = tag_bbox[3] - tag_bbox[1]
        tag_x = min(
            max(0.0, detection.box.xmin),
            max(0.0, image.width - tag_width - 6),
        )
        tag_y = max(
            0.0,
            detection.box.ymin - tag_height - 6,
        )

        draw.rectangle(
            (
                tag_x,
                tag_y,
                tag_x + tag_width + 6,
                tag_y + tag_height + 4,
            ),
            fill=color,
        )
        draw.text(
            (
                tag_x + 3,
                tag_y + 2,
            ),
            tag,
            fill="white",
            font=font,
        )

    return image
