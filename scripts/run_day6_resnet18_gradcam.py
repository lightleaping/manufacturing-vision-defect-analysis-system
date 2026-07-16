"""Day 6 ResNet18 Grad-CAM 실제 실행 Script.

실행 예시
---------
구조·입력 Artifact·Checkpoint 복원·표본 선택만 검증:

    python -m scripts.run_day6_resnet18_gradcam --validate-only

실제 Grad-CAM JSON·PNG 생성:

    python -m scripts.run_day6_resnet18_gradcam

생성 Artifact
-------------
1. reports/artifacts/day6_resnet18_gradcam_analysis.json
2. reports/figures/day6_resnet18_gradcam_overview.png
3. reports/figures/day6_resnet18_gradcam_high_confidence_errors.png
4. reports/figures/day6_resnet18_gradcam_boundary_errors.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from src.explainability.gradcam_pipeline import (
    BOUNDARY_ERROR_TYPES,
    CLASSIFICATION_THRESHOLD,
    DEFAULT_TARGET_LAYER_NAME,
    HIGH_CONFIDENCE_ERROR_TYPES,
    GradCAMArtifactPaths,
    build_sample_lookup,
    extract_day4_sample_results,
    read_json_object,
    run_gradcam_analysis,
    validate_day4_day5_cross_reference,
)
from src.explainability.gradcam_sample_selector import select_gradcam_samples
from src.explainability.gradcam import resolve_target_layer
from src.reproducibility import set_global_random_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_PATH = Path(
    "models/checkpoints/resnet18_transfer_best.pt"
)
DEFAULT_DAY4_EVALUATION_PATH = Path(
    "reports/artifacts/day4_resnet18_test_evaluation.json"
)
DEFAULT_DAY5_ANALYSIS_PATH = Path(
    "reports/artifacts/day5_resnet18_misclassification_analysis.json"
)
DEFAULT_METADATA_OUTPUT_PATH = Path(
    "reports/artifacts/day6_resnet18_gradcam_analysis.json"
)
DEFAULT_OVERVIEW_OUTPUT_PATH = Path(
    "reports/figures/day6_resnet18_gradcam_overview.png"
)
DEFAULT_HIGH_CONFIDENCE_OUTPUT_PATH = Path(
    "reports/figures/day6_resnet18_gradcam_high_confidence_errors.png"
)
DEFAULT_BOUNDARY_OUTPUT_PATH = Path(
    "reports/figures/day6_resnet18_gradcam_boundary_errors.png"
)
DEFAULT_RANDOM_SEED = 42


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Day 6 명령행 Argument를 해석합니다."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate prediction-class Grad-CAM explanations from the "
            "Day 4 ResNet18 checkpoint and Day 4/5 artifacts."
        )
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--day4-evaluation-path",
        type=Path,
        default=DEFAULT_DAY4_EVALUATION_PATH,
    )
    parser.add_argument(
        "--day5-analysis-path",
        type=Path,
        default=DEFAULT_DAY5_ANALYSIS_PATH,
    )
    parser.add_argument(
        "--metadata-output-path",
        type=Path,
        default=DEFAULT_METADATA_OUTPUT_PATH,
    )
    parser.add_argument(
        "--overview-output-path",
        type=Path,
        default=DEFAULT_OVERVIEW_OUTPUT_PATH,
    )
    parser.add_argument(
        "--high-confidence-output-path",
        type=Path,
        default=DEFAULT_HIGH_CONFIDENCE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--boundary-output-path",
        type=Path,
        default=DEFAULT_BOUNDARY_OUTPUT_PATH,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate checkpoint restoration, source artifacts, target "
            "layer, and sample selection without generating Grad-CAM files."
        ),
    )
    return parser.parse_args(arguments)


def resolve_project_path(path: Path) -> Path:
    """상대 경로를 프로젝트 루트 기준 절대 경로로 변환합니다."""

    if not isinstance(path, Path):
        raise TypeError("path는 pathlib.Path여야 합니다.")
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def validate_execution_paths(
    *,
    checkpoint_path: Path,
    day4_evaluation_path: Path,
    day5_analysis_path: Path,
    artifact_paths: GradCAMArtifactPaths,
) -> None:
    """실행 전에 입력·출력 경로와 확장자를 검증합니다."""

    for name, path in {
        "checkpoint": checkpoint_path,
        "day4_evaluation": day4_evaluation_path,
        "day5_analysis": day5_analysis_path,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} 입력 파일이 없습니다: {path}")

    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("Checkpoint는 .pt 또는 .pth 확장자여야 합니다.")
    if day4_evaluation_path.suffix.lower() != ".json":
        raise ValueError("Day 4 평가 Artifact는 .json이어야 합니다.")
    if day5_analysis_path.suffix.lower() != ".json":
        raise ValueError("Day 5 분석 Artifact는 .json이어야 합니다.")
    if artifact_paths.metadata_path.suffix.lower() != ".json":
        raise ValueError("Day 6 Metadata 출력은 .json이어야 합니다.")

    png_paths = artifact_paths.all_paths()[1:]
    if any(path.suffix.lower() != ".png" for path in png_paths):
        raise ValueError("Day 6 Figure 출력은 모두 .png여야 합니다.")
    if len(set(artifact_paths.all_paths())) != 4:
        raise ValueError("Day 6 출력 경로 네 개는 서로 달라야 합니다.")


def restore_resnet18_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """Day 4에서 검증한 Best Checkpoint 복원 함수를 그대로 재사용합니다."""

    from scripts.run_day4_resnet18_training import restore_best_checkpoint

    return restore_best_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )


def validate_source_structure(
    *,
    day4_evaluation_path: Path,
    day5_analysis_path: Path,
) -> list[object]:
    """Day 4·5 교차 검증 후 대표 표본 7장을 반환합니다."""

    day4_payload = read_json_object(day4_evaluation_path)
    day5_payload = read_json_object(day5_analysis_path)
    sample_results = extract_day4_sample_results(day4_payload)

    validate_day4_day5_cross_reference(
        day4_sample_results=sample_results,
        day5_payload=day5_payload,
    )
    selected_samples = select_gradcam_samples(
        sample_results,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    # sample_index 전체가 Day 4 lookup에 존재하는지 추가 확인한다.
    sample_lookup = build_sample_lookup(sample_results)
    for sample in selected_samples:
        if sample.sample_index not in sample_lookup:
            raise RuntimeError(
                "선택 표본이 Day 4 lookup에 없습니다: "
                f"sample_index={sample.sample_index}"
            )

    return list(selected_samples)


def print_configuration(
    *,
    model: nn.Module,
    device: torch.device,
    checkpoint_path: Path,
    day4_evaluation_path: Path,
    day5_analysis_path: Path,
    artifact_paths: GradCAMArtifactPaths,
    selected_samples: Sequence[object],
    validate_only: bool,
) -> None:
    """실행 설정과 선택 표본을 읽기 쉬운 형태로 출력합니다."""

    print("=" * 100)
    print("DAY 6 - RESNET18 GRAD-CAM EXPLAINABILITY")
    print("=" * 100)
    print()
    print("[MODEL]")
    print(f"Model                      : {model.__class__.__name__}")
    print(f"Device                     : {device}")
    print(f"Evaluation mode            : {not model.training}")
    print(f"Target layer               : {DEFAULT_TARGET_LAYER_NAME}")
    print("Target policy              : predicted class")
    print("DEFECT target score        : raw_logit")
    print("NORMAL target score        : -raw_logit")
    print("Batch size                 : 1")
    print()
    print("[INPUT ARTIFACTS]")
    print(f"Best checkpoint            : {checkpoint_path}")
    print(f"Day 4 evaluation           : {day4_evaluation_path}")
    print(f"Day 5 analysis             : {day5_analysis_path}")
    print()
    print("[SELECTED SAMPLES]")
    for sample in selected_samples:
        print(
            f"{sample.selection_type:<34} "
            f"index={sample.sample_index:<4} "
            f"P(DEFECT)={sample.defect_probability:.6f} "
            f"file={Path(sample.image_path).name}"
        )
    print()
    print("[OUTPUT ARTIFACTS]")
    print(f"Metadata JSON              : {artifact_paths.metadata_path}")
    print(f"Overview Figure            : {artifact_paths.overview_figure_path}")
    print(
        "High-confidence Figure     : "
        f"{artifact_paths.high_confidence_figure_path}"
    )
    print(f"Boundary Figure            : {artifact_paths.boundary_figure_path}")
    print(f"Validate only              : {validate_only}")


def main(arguments: Sequence[str] | None = None) -> None:
    """Day 6 실제 실행 Entry Point입니다."""

    parsed = parse_arguments(arguments)

    checkpoint_path = resolve_project_path(parsed.checkpoint_path)
    day4_evaluation_path = resolve_project_path(
        parsed.day4_evaluation_path
    )
    day5_analysis_path = resolve_project_path(parsed.day5_analysis_path)
    artifact_paths = GradCAMArtifactPaths(
        metadata_path=resolve_project_path(parsed.metadata_output_path),
        overview_figure_path=resolve_project_path(
            parsed.overview_output_path
        ),
        high_confidence_figure_path=resolve_project_path(
            parsed.high_confidence_output_path
        ),
        boundary_figure_path=resolve_project_path(
            parsed.boundary_output_path
        ),
    )

    validate_execution_paths(
        checkpoint_path=checkpoint_path,
        day4_evaluation_path=day4_evaluation_path,
        day5_analysis_path=day5_analysis_path,
        artifact_paths=artifact_paths,
    )

    settings = set_global_random_seed(
        seed=DEFAULT_RANDOM_SEED,
        deterministic_algorithms=False,
    )
    device = torch.device(settings.device)

    selected_samples = validate_source_structure(
        day4_evaluation_path=day4_evaluation_path,
        day5_analysis_path=day5_analysis_path,
    )
    model = restore_resnet18_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.eval()

    _ = resolve_target_layer(model, DEFAULT_TARGET_LAYER_NAME)

    print_configuration(
        model=model,
        device=device,
        checkpoint_path=checkpoint_path,
        day4_evaluation_path=day4_evaluation_path,
        day5_analysis_path=day5_analysis_path,
        artifact_paths=artifact_paths,
        selected_samples=selected_samples,
        validate_only=parsed.validate_only,
    )

    if parsed.validate_only:
        print()
        print("=" * 100)
        print("[PASS] Day 6 Grad-CAM structure validation")
        print("=" * 100)
        return

    result = run_gradcam_analysis(
        model=model,
        project_root=PROJECT_ROOT,
        checkpoint_path=checkpoint_path,
        day4_evaluation_path=day4_evaluation_path,
        day5_analysis_path=day5_analysis_path,
        artifact_paths=artifact_paths,
        device=device,
        target_layer_name=DEFAULT_TARGET_LAYER_NAME,
    )

    high_confidence_count = sum(
        item.selected_sample.selection_type
        in HIGH_CONFIDENCE_ERROR_TYPES
        for item in result.generated_samples
    )
    boundary_count = sum(
        item.selected_sample.selection_type in BOUNDARY_ERROR_TYPES
        for item in result.generated_samples
    )

    print()
    print("=" * 100)
    print("DAY 6 - RESNET18 GRAD-CAM COMPLETED")
    print("=" * 100)
    print()
    print("[RESULT]")
    print(f"Generated samples           : {len(result.generated_samples)}")
    print(f"Correct prediction samples  : 2")
    print(f"High-confidence errors      : {high_confidence_count}")
    print(f"Boundary errors             : {boundary_count}")
    print(f"Runtime seconds             : {result.duration_seconds:.2f}")
    print()
    print("[ARTIFACTS]")
    for path in result.artifact_paths.all_paths():
        print(f"{path} | exists={path.is_file()} | bytes={path.stat().st_size}")
    print()
    print("[PASS] Day 6 ResNet18 Grad-CAM explainability artifacts generated")


if __name__ == "__main__":
    main()
