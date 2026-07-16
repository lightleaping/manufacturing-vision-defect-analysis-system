from __future__ import annotations

import pytest

from scripts.run_day7_api_validation import validate_prediction_payload


def _valid_payload() -> dict[str, object]:
    return {
        "prediction": 1,
        "prediction_class_name": "DEFECT",
        "defect_probability": 0.9,
        "normal_probability": 0.1,
        "raw_logit": 2.1972246,
        "classification_threshold": 0.5,
        "model_name": "ResNet18Transfer",
        "model_version": "resnet18_transfer_best",
        "positive_class": "DEFECT",
        "original_filename": "sample.jpeg",
        "content_type": "image/jpeg",
        "image_width": 300,
        "image_height": 300,
        "image_mode": "RGB",
        "inference_time_ms": 10.0,
    }


def test_validation_payload_accepts_expected_response() -> None:
    validate_prediction_payload(
        _valid_payload(),
        expected_prediction=1,
        expected_class_name="DEFECT",
    )


def test_validation_payload_rejects_missing_key() -> None:
    payload = _valid_payload()
    del payload["raw_logit"]

    with pytest.raises(KeyError, match="missing keys"):
        validate_prediction_payload(
            payload,
            expected_prediction=1,
            expected_class_name="DEFECT",
        )


def test_validation_payload_rejects_probability_sum_error() -> None:
    payload = _valid_payload()
    payload["normal_probability"] = 0.2

    with pytest.raises(ValueError, match="sum to 1"):
        validate_prediction_payload(
            payload,
            expected_prediction=1,
            expected_class_name="DEFECT",
        )
