from __future__ import annotations

import pytest

from scripts.run_day8_dashboard_validation import (
    validate_health,
    validate_prediction,
)
from src.dashboard.api_client import DashboardHealth, DashboardPrediction


def _health(*, model_loaded: bool = True) -> DashboardHealth:
    return DashboardHealth(
        status="ok",
        service="Manufacturing Vision Defect Analysis System",
        model_loaded=model_loaded,
        model_name="ResNet18Transfer",
        device="cpu",
    )


def _prediction(
    *,
    prediction: int,
    class_name: str,
    defect_probability: float,
) -> DashboardPrediction:
    return DashboardPrediction(
        prediction=prediction,
        prediction_class_name=class_name,
        defect_probability=defect_probability,
        normal_probability=1.0 - defect_probability,
        raw_logit=2.0 if prediction == 1 else -2.0,
        classification_threshold=0.5,
        model_name="ResNet18Transfer",
        model_version="resnet18_transfer_best",
        positive_class="DEFECT",
        original_filename="sample.jpeg",
        content_type="image/jpeg",
        image_width=300,
        image_height=300,
        image_mode="RGB",
        inference_time_ms=10.0,
    )


def test_health_validation_accepts_ready_model() -> None:
    validate_health(_health())


def test_health_validation_rejects_unloaded_model() -> None:
    with pytest.raises(ValueError, match="model_loaded"):
        validate_health(_health(model_loaded=False))


def test_prediction_validation_accepts_normal_and_defect() -> None:
    validate_prediction(
        _prediction(
            prediction=0,
            class_name="NORMAL",
            defect_probability=0.1,
        ),
        expected_prediction=0,
        expected_class_name="NORMAL",
    )
    validate_prediction(
        _prediction(
            prediction=1,
            class_name="DEFECT",
            defect_probability=0.9,
        ),
        expected_prediction=1,
        expected_class_name="DEFECT",
    )


def test_prediction_validation_rejects_wrong_class() -> None:
    with pytest.raises(ValueError, match="prediction_class_name"):
        validate_prediction(
            _prediction(
                prediction=1,
                class_name="NORMAL",
                defect_probability=0.9,
            ),
            expected_prediction=1,
            expected_class_name="DEFECT",
        )
