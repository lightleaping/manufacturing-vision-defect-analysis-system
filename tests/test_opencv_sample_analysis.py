from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.opencv_analysis.sample_analysis import (
    ImageSampleSpec,
    analyze_image_sample,
    analyze_image_samples,
)


def _write_rgb_image(path: Path, *, size: tuple[int, int] = (32, 24)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[4:18, 8:24] = (220, 180, 40)
    Image.fromarray(array, mode="RGB").save(path)


def _spec(relative_path: str = "data/sample.png") -> ImageSampleSpec:
    return ImageSampleSpec(
        sample_id="sample_1",
        dataset_name="Synthetic",
        semantic_role="Synthetic Sample",
        class_name="RECTANGLE",
        relative_path=relative_path,
    )


def test_image_sample_spec_normalizes_windows_path() -> None:
    spec = _spec(r"data\sample.png")
    assert spec.normalized_relative_path() == "data/sample.png"


@pytest.mark.parametrize(
    "value",
    ["", "UpperCase", "space value", "../escape"],
)
def test_image_sample_spec_rejects_invalid_sample_id(value: str) -> None:
    with pytest.raises(ValueError):
        ImageSampleSpec(
            sample_id=value,
            dataset_name="Synthetic",
            semantic_role="Role",
            class_name="Class",
            relative_path="data/sample.png",
        )


def test_image_sample_spec_rejects_parent_traversal() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        _spec("../sample.png")


def test_image_sample_spec_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="must use one of"):
        _spec("data/sample.bmp")


def test_analyze_image_sample_returns_metadata_and_metrics(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "sample.png"
    _write_rgb_image(image_path)

    analyzed = analyze_image_sample(tmp_path, _spec())

    assert analyzed.absolute_path == image_path.resolve()
    assert analyzed.file_size_bytes > 0
    assert len(analyzed.sha256) == 64
    assert analyzed.source_format == "PNG"
    assert analyzed.source_mode == "RGB"
    assert analyzed.source_width == 32
    assert analyzed.source_height == 24
    assert analyzed.metrics.width == 32
    assert analyzed.metrics.height == 24


def test_artifact_record_is_json_serializable(tmp_path: Path) -> None:
    _write_rgb_image(tmp_path / "data" / "sample.png")
    analyzed = analyze_image_sample(tmp_path, _spec())

    text = json.dumps(analyzed.to_artifact_record())

    assert "Synthetic Sample" in text
    assert "not object-detection predictions" in text


def test_analyze_image_sample_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        analyze_image_sample(tmp_path, _spec())


def test_analyze_image_sample_rejects_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "data" / "sample.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match="failed to decode"):
        analyze_image_sample(tmp_path, _spec())


def test_analyze_image_samples_preserves_input_order(tmp_path: Path) -> None:
    _write_rgb_image(tmp_path / "data" / "first.png")
    _write_rgb_image(tmp_path / "data" / "second.jpg")

    specs = (
        ImageSampleSpec(
            sample_id="first",
            dataset_name="Synthetic",
            semantic_role="First",
            class_name="A",
            relative_path="data/first.png",
        ),
        ImageSampleSpec(
            sample_id="second",
            dataset_name="Synthetic",
            semantic_role="Second",
            class_name="B",
            relative_path="data/second.jpg",
        ),
    )

    result = analyze_image_samples(tmp_path, specs)

    assert [item.spec.sample_id for item in result] == ["first", "second"]


def test_analyze_image_samples_rejects_duplicate_ids(tmp_path: Path) -> None:
    specs = (_spec(), _spec())
    with pytest.raises(ValueError, match="unique"):
        analyze_image_samples(tmp_path, specs)
