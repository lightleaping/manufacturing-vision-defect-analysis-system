"""실제 Day 7 FastAPI와 Day 8 Dashboard API Client의 통합 검증.

FastAPI 서버를 먼저 실행한 뒤 이 Script를 실행한다. Streamlit 브라우저 UI와
Screenshot 검증은 이 Script 이후 사용자가 직접 수행한다.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.dashboard.api_client import (
    DashboardApiClient,
    DashboardHealth,
    DashboardPrediction,
)
from src.dashboard.config import DashboardSettings
from src.dashboard.ui_helpers import resolve_content_type

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day8_streamlit_dashboard_validation.json"
)
DEFAULT_NORMAL_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "casting_product_images"
    / "casting_data"
    / "casting_data"
    / "test"
    / "ok_front"
    / "cast_ok_0_7631.jpeg"
)
DEFAULT_DEFECT_IMAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "casting_product_images"
    / "casting_data"
    / "casting_data"
    / "test"
    / "def_front"
    / "cast_def_0_1414.jpeg"
)


def validate_health(health: DashboardHealth) -> None:
    if health.status != "ok":
        raise ValueError("health status must be ok")
    if not health.model_loaded:
        raise ValueError("health model_loaded must be true")
    if not health.model_name:
        raise ValueError("health model_name must not be empty")


def validate_prediction(
    prediction: DashboardPrediction,
    *,
    expected_prediction: int,
    expected_class_name: str,
) -> None:
    if prediction.prediction != expected_prediction:
        raise ValueError(
            f"prediction mismatch: expected {expected_prediction}, "
            f"received {prediction.prediction}"
        )
    if prediction.prediction_class_name != expected_class_name:
        raise ValueError(
            "prediction_class_name mismatch: "
            f"expected {expected_class_name}, "
            f"received {prediction.prediction_class_name}"
        )
    if prediction.positive_class != "DEFECT":
        raise ValueError("positive_class must be DEFECT")


def _predict_path(
    client: DashboardApiClient,
    *,
    image_path: Path,
) -> DashboardPrediction:
    if not image_path.is_file():
        raise FileNotFoundError(f"image does not exist: {image_path}")

    image_bytes = image_path.read_bytes()
    content_type = resolve_content_type(
        filename=image_path.name,
        declared_content_type=None,
    )
    return client.predict_image(
        filename=image_path.name,
        content_type=content_type,
        image_bytes=image_bytes,
    )



def _artifact_path_value(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def build_validation_payload(
    *,
    base_url: str,
    health: DashboardHealth,
    normal_image_path: Path,
    normal_prediction: DashboardPrediction,
    defect_image_path: Path,
    defect_prediction: DashboardPrediction,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "project": "Manufacturing Vision Defect Analysis System",
        "run_name": "day8_streamlit_dashboard_validation",
        "base_url": base_url,
        "health": health.to_dict(),
        "normal_image": {
            "path": _artifact_path_value(normal_image_path),
            "response": normal_prediction.to_dict(),
        },
        "defect_image": {
            "path": _artifact_path_value(defect_image_path),
            "response": defect_prediction.to_dict(),
        },
        "runtime_seconds": runtime_seconds,
        "ui_visual_validation_completed": False,
        "screenshot_artifacts": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--normal-image", type=Path, default=DEFAULT_NORMAL_IMAGE_PATH)
    parser.add_argument("--defect-image", type=Path, default=DEFAULT_DEFECT_IMAGE_PATH)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    settings = DashboardSettings(api_base_url=args.base_url)

    print("=" * 92)
    print("DAY 8 - STREAMLIT DASHBOARD FASTAPI CLIENT VALIDATION")
    print("=" * 92)

    with DashboardApiClient(settings) as client:
        health = client.get_health()
        validate_health(health)
        print("[PASS] FastAPI health and model readiness")

        normal_prediction = _predict_path(client, image_path=args.normal_image)
        validate_prediction(
            normal_prediction,
            expected_prediction=0,
            expected_class_name="NORMAL",
        )
        print(
            "[PASS] NORMAL image -> "
            f"{normal_prediction.prediction_class_name} "
            f"P(DEFECT)={normal_prediction.defect_probability:.12f}"
        )

        defect_prediction = _predict_path(client, image_path=args.defect_image)
        validate_prediction(
            defect_prediction,
            expected_prediction=1,
            expected_class_name="DEFECT",
        )
        print(
            "[PASS] DEFECT image -> "
            f"{defect_prediction.prediction_class_name} "
            f"P(DEFECT)={defect_prediction.defect_probability:.12f}"
        )

    runtime_seconds = time.perf_counter() - started_at
    payload = build_validation_payload(
        base_url=settings.api_base_url,
        health=health,
        normal_image_path=args.normal_image.resolve(),
        normal_prediction=normal_prediction,
        defect_image_path=args.defect_image.resolve(),
        defect_prediction=defect_prediction,
        runtime_seconds=runtime_seconds,
    )

    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[PASS] Validation artifact created: {args.artifact}")
    print(f"[INFO] Runtime: {runtime_seconds:.2f} seconds")
    print("[NEXT] Run Streamlit, verify both images in the browser, and save screenshots.")


if __name__ == "__main__":
    main()
