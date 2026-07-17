"""실제 FastAPI를 호출해 Detection Dashboard Client와 Overlay를 검증한다."""

from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any

from src.dashboard.config import DashboardSettings
from src.dashboard.detection_api_client import (
    DetectionDashboardApiClient,
)
from src.dashboard.detection_ui_helpers import (
    build_detection_table_rows,
    render_detection_overlay,
)


DEFAULT_ARTIFACT = Path(
    "reports/artifacts/day13_detection_dashboard_api_client_validation.json"
)
DEFAULT_FIGURE = Path(
    "reports/figures/day13_detection_dashboard_overlay.png"
)


def run_validation(
    *,
    project_root: Path,
    image_path: Path,
    api_base_url: str,
    score_threshold: float,
    artifact_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    image = (
        image_path
        if image_path.is_absolute()
        else root / image_path
    ).resolve()

    if not image.is_file():
        raise FileNotFoundError(
            f"Validation image not found: {image}"
        )

    image_bytes = image.read_bytes()
    content_type, _ = mimetypes.guess_type(
        image.name
    )
    if content_type not in {
        "image/jpeg",
        "image/png",
    }:
        raise ValueError(
            "Validation image must be JPEG or PNG."
        )

    settings = DashboardSettings(
        api_base_url=api_base_url,
    )
    with DetectionDashboardApiClient(
        settings
    ) as client:
        prediction = client.detect_image(
            filename=image.name,
            content_type=content_type,
            image_bytes=image_bytes,
            score_threshold=score_threshold,
        )

    overlay = render_detection_overlay(
        image_bytes=image_bytes,
        prediction=prediction,
        maximum_boxes=8,
    )

    output_figure = (
        figure_path
        if figure_path.is_absolute()
        else root / figure_path
    )
    output_figure.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    overlay.save(output_figure)

    result: dict[str, Any] = {
        "stage": "day13_detection_dashboard_api_client_validation",
        "api_base_url": settings.api_base_url,
        "endpoint": "/api/v1/detection/predictions",
        "image_path": str(image),
        "score_threshold": score_threshold,
        "detection_count": prediction.detection_count,
        "inference_time_ms": prediction.inference_time_ms,
        "checkpoint_epoch": prediction.checkpoint_epoch,
        "checkpoint_metric_name": prediction.checkpoint_metric_name,
        "checkpoint_metric_value": prediction.checkpoint_metric_value,
        "detections": build_detection_table_rows(
            prediction
        ),
        "overlay_path": str(
            output_figure.resolve()
        ),
        "validation_passed": True,
    }

    output_artifact = (
        artifact_path
        if artifact_path.is_absolute()
        else root / artifact_path
    )
    output_artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_artifact.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 100)
    print("DAY 13 - DETECTION DASHBOARD API CLIENT VALIDATION")
    print("=" * 100)
    print(f"API Base URL             : {settings.api_base_url}")
    print(f"Image                    : {image}")
    print(f"Score threshold          : {score_threshold:.2f}")
    print(f"Detection count          : {prediction.detection_count}")
    print(f"Inference time (ms)      : {prediction.inference_time_ms:.2f}")
    print(f"Checkpoint epoch         : {prediction.checkpoint_epoch}")
    print(f"Overlay                  : {output_figure}")
    print(f"Artifact                 : {output_artifact}")
    print("[RESULT]                 : PASS")
    print("=" * 100)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--api-base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=DEFAULT_ARTIFACT,
    )
    parser.add_argument(
        "--figure-path",
        type=Path,
        default=DEFAULT_FIGURE,
    )
    args = parser.parse_args()

    run_validation(
        project_root=args.project_root,
        image_path=args.image_path,
        api_base_url=args.api_base_url,
        score_threshold=args.score_threshold,
        artifact_path=args.artifact_path,
        figure_path=args.figure_path,
    )


if __name__ == "__main__":
    main()
