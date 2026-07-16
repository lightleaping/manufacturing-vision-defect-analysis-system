from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from src.explainability.gradcam import GradCAM
from src.explainability.gradcam_pipeline import (
    GradCAMArtifactPaths,
    GradCAMPipelineError,
    build_metadata_payload,
    build_sample_lookup,
    create_day6_test_transform,
    extract_day4_sample_results,
    load_model_input_tensor,
    read_json_object,
    resolve_image_path,
    validate_day4_day5_cross_reference,
    validate_reproduced_prediction,
    write_json_atomically,
)


class DummyBinaryCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(4, 1, bias=False)

        with torch.no_grad():
            self.features[0].weight.fill_(0.10)
            self.classifier.weight.fill_(1.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = self.pool(features).flatten(start_dim=1)
        return self.classifier(pooled).squeeze(dim=1)


def _write_rgb(path: Path, value: int = 128) -> None:
    Image.new("RGB", (32, 24), color=(value, value, value)).save(path)


def _day4_samples() -> list[dict[str, object]]:
    return [
        {
            "sample_index": 0,
            "image_path": "normal.jpeg",
            "ground_truth_label": 0,
            "ground_truth_class_name": "NORMAL",
            "raw_logit": -1.0,
            "defect_probability": 0.2,
            "prediction": 0,
            "prediction_class_name": "NORMAL",
            "correct": True,
        },
        {
            "sample_index": 1,
            "image_path": "fp.jpeg",
            "ground_truth_label": 0,
            "ground_truth_class_name": "NORMAL",
            "raw_logit": 1.0,
            "defect_probability": 0.8,
            "prediction": 1,
            "prediction_class_name": "DEFECT",
            "correct": False,
        },
    ]


def test_create_day6_test_transform_returns_expected_shape(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.jpeg"
    _write_rgb(image_path)

    tensor = load_model_input_tensor(
        image_path=image_path,
        image_transform=create_day6_test_transform(),
        device=torch.device("cpu"),
    )

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_resolve_image_path_supports_project_relative_path(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "sample.jpeg"
    _write_rgb(image_path)

    resolved = resolve_image_path(
        project_root=tmp_path,
        image_path="sample.jpeg",
    )

    assert resolved == image_path.resolve()


def test_read_and_write_json_atomically(tmp_path: Path) -> None:
    output_path = tmp_path / "artifact.json"

    write_json_atomically(
        payload={"status": "ok", "count": 7},
        output_path=output_path,
    )
    payload = read_json_object(output_path)

    assert payload == {"status": "ok", "count": 7}
    assert not list(tmp_path.glob("*.tmp"))


def test_extract_day4_sample_results_rejects_missing_array() -> None:
    with pytest.raises(GradCAMPipelineError, match="sample_results"):
        extract_day4_sample_results({"metrics": {}})


def test_validate_day4_day5_cross_reference_accepts_same_indices() -> None:
    validate_day4_day5_cross_reference(
        day4_sample_results=_day4_samples(),
        day5_payload={
            "misclassifications": [
                {
                    "sample_index": 1,
                    "image_path": "fp.jpeg",
                    "ground_truth_label": 0,
                    "prediction": 1,
                    "defect_probability": 0.8,
                }
            ]
        },
    )


def test_validate_day4_day5_cross_reference_rejects_missing_index() -> None:
    with pytest.raises(GradCAMPipelineError, match="일치하지"):
        validate_day4_day5_cross_reference(
            day4_sample_results=_day4_samples(),
            day5_payload={"misclassifications": []},
        )


def test_build_sample_lookup_rejects_duplicate_index() -> None:
    samples = _day4_samples()
    samples.append(dict(samples[0]))

    with pytest.raises(GradCAMPipelineError, match="중복"):
        build_sample_lookup(samples)


def test_validate_reproduced_prediction_accepts_matching_result() -> None:
    model = DummyBinaryCNN()
    input_tensor = torch.full((1, 3, 16, 16), 0.5)

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(input_tensor=input_tensor)

    logit_error, probability_error = validate_reproduced_prediction(
        source_sample={
            "sample_index": 10,
            "raw_logit": result.raw_logit,
            "defect_probability": result.defect_probability,
            "prediction": result.prediction,
        },
        gradcam_result=result,
    )

    assert logit_error == pytest.approx(0.0)
    assert probability_error == pytest.approx(0.0)


def test_validate_reproduced_prediction_rejects_large_logit_error() -> None:
    model = DummyBinaryCNN()
    input_tensor = torch.full((1, 3, 16, 16), 0.5)

    with GradCAM(model=model, target_layer_name="features.0") as gradcam:
        result = gradcam.generate(input_tensor=input_tensor)

    with pytest.raises(GradCAMPipelineError, match="Logit"):
        validate_reproduced_prediction(
            source_sample={
                "sample_index": 10,
                "raw_logit": result.raw_logit + 1.0,
                "defect_probability": result.defect_probability,
                "prediction": result.prediction,
            },
            gradcam_result=result,
        )


def test_build_metadata_payload_rejects_empty_samples(tmp_path: Path) -> None:
    paths = GradCAMArtifactPaths(
        metadata_path=tmp_path / "metadata.json",
        overview_figure_path=tmp_path / "overview.png",
        high_confidence_figure_path=tmp_path / "confidence.png",
        boundary_figure_path=tmp_path / "boundary.png",
    )

    with pytest.raises(GradCAMPipelineError, match="표본"):
        build_metadata_payload(
            generated_samples=[],
            artifact_paths=paths,
            project_root=tmp_path,
            checkpoint_path=tmp_path / "model.pt",
            day4_evaluation_path=tmp_path / "day4.json",
            day5_analysis_path=tmp_path / "day5.json",
            device=torch.device("cpu"),
            target_layer_name="features.0",
            duration_seconds=1.0,
        )
