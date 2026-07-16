from __future__ import annotations

from pathlib import Path

import pytest

from scripts.run_day6_resnet18_gradcam import (
    DEFAULT_BOUNDARY_OUTPUT_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DAY4_EVALUATION_PATH,
    DEFAULT_DAY5_ANALYSIS_PATH,
    DEFAULT_HIGH_CONFIDENCE_OUTPUT_PATH,
    DEFAULT_METADATA_OUTPUT_PATH,
    DEFAULT_OVERVIEW_OUTPUT_PATH,
    parse_arguments,
    validate_execution_paths,
)
from src.explainability.gradcam_pipeline import GradCAMArtifactPaths


def test_parse_arguments_uses_day6_default_paths() -> None:
    arguments = parse_arguments([])

    assert arguments.checkpoint_path == DEFAULT_CHECKPOINT_PATH
    assert arguments.day4_evaluation_path == DEFAULT_DAY4_EVALUATION_PATH
    assert arguments.day5_analysis_path == DEFAULT_DAY5_ANALYSIS_PATH
    assert arguments.metadata_output_path == DEFAULT_METADATA_OUTPUT_PATH
    assert arguments.overview_output_path == DEFAULT_OVERVIEW_OUTPUT_PATH
    assert (
        arguments.high_confidence_output_path
        == DEFAULT_HIGH_CONFIDENCE_OUTPUT_PATH
    )
    assert arguments.boundary_output_path == DEFAULT_BOUNDARY_OUTPUT_PATH
    assert arguments.validate_only is False


def test_parse_arguments_supports_validate_only() -> None:
    arguments = parse_arguments(["--validate-only"])

    assert arguments.validate_only is True


def test_validate_execution_paths_accepts_expected_extensions(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    day4_path = tmp_path / "day4.json"
    day5_path = tmp_path / "day5.json"
    checkpoint_path.write_bytes(b"checkpoint")
    day4_path.write_text("{}", encoding="utf-8")
    day5_path.write_text("{}", encoding="utf-8")

    validate_execution_paths(
        checkpoint_path=checkpoint_path,
        day4_evaluation_path=day4_path,
        day5_analysis_path=day5_path,
        artifact_paths=GradCAMArtifactPaths(
            metadata_path=tmp_path / "metadata.json",
            overview_figure_path=tmp_path / "overview.png",
            high_confidence_figure_path=tmp_path / "confidence.png",
            boundary_figure_path=tmp_path / "boundary.png",
        ),
    )


def test_validate_execution_paths_rejects_duplicate_outputs(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "model.pt"
    day4_path = tmp_path / "day4.json"
    day5_path = tmp_path / "day5.json"
    checkpoint_path.write_bytes(b"checkpoint")
    day4_path.write_text("{}", encoding="utf-8")
    day5_path.write_text("{}", encoding="utf-8")
    duplicate_path = tmp_path / "duplicate.png"

    with pytest.raises(ValueError, match="서로 달라"):
        validate_execution_paths(
            checkpoint_path=checkpoint_path,
            day4_evaluation_path=day4_path,
            day5_analysis_path=day5_path,
            artifact_paths=GradCAMArtifactPaths(
                metadata_path=tmp_path / "metadata.json",
                overview_figure_path=duplicate_path,
                high_confidence_figure_path=duplicate_path,
                boundary_figure_path=tmp_path / "boundary.png",
            ),
        )


def test_validate_execution_paths_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    day4_path = tmp_path / "day4.json"
    day5_path = tmp_path / "day5.json"
    day4_path.write_text("{}", encoding="utf-8")
    day5_path.write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="checkpoint"):
        validate_execution_paths(
            checkpoint_path=tmp_path / "missing.pt",
            day4_evaluation_path=day4_path,
            day5_analysis_path=day5_path,
            artifact_paths=GradCAMArtifactPaths(
                metadata_path=tmp_path / "metadata.json",
                overview_figure_path=tmp_path / "overview.png",
                high_confidence_figure_path=tmp_path / "confidence.png",
                boundary_figure_path=tmp_path / "boundary.png",
            ),
        )
