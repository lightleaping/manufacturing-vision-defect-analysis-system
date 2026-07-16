"""Day 5 ResNet18 오분류 분석과 Figure 생성을 실행한다."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.evaluation.misclassification_analysis import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    assert_expected_analysis_counts,
    build_misclassification_analysis,
    load_day4_evaluation_samples,
    save_json_atomic,
)
from src.evaluation.misclassification_visualization import (
    create_day5_misclassification_figures,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DAY4_EVALUATION_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day4_resnet18_test_evaluation.json"
)

DAY5_ANALYSIS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day5_resnet18_misclassification_analysis.json"
)

DAY5_FALSE_POSITIVE_FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "day5_resnet18_false_positives.png"
)

DAY5_FALSE_NEGATIVE_FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "day5_resnet18_false_negatives.png"
)

DAY5_ALL_MISCLASSIFICATIONS_FIGURE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "figures"
    / "day5_resnet18_all_misclassifications.png"
)


def run_day5_misclassification_analysis(
    *,
    input_artifact_path: str | Path = DAY4_EVALUATION_PATH,
    analysis_output_path: str | Path = DAY5_ANALYSIS_OUTPUT_PATH,
    false_positive_figure_path: str | Path = (
        DAY5_FALSE_POSITIVE_FIGURE_PATH
    ),
    false_negative_figure_path: str | Path = (
        DAY5_FALSE_NEGATIVE_FIGURE_PATH
    ),
    all_misclassifications_figure_path: str | Path = (
        DAY5_ALL_MISCLASSIFICATIONS_FIGURE_PATH
    ),
    project_root: str | Path = PROJECT_ROOT,
    classification_threshold: float = (
        DEFAULT_CLASSIFICATION_THRESHOLD
    ),
    expected_total_samples: int | None = 715,
    expected_false_positive_count: int | None = 13,
    expected_false_negative_count: int | None = 6,
    expected_misclassified_count: int | None = 19,
) -> dict[str, Any]:
    """Day 5 분석 전체 흐름을 실행하고 생성 결과를 반환한다."""

    resolved_input_path = Path(input_artifact_path)

    samples = load_day4_evaluation_samples(
        resolved_input_path
    )

    analysis = build_misclassification_analysis(
        samples,
        classification_threshold=classification_threshold,
        source_artifact=resolved_input_path,
        ranking_limit=5,
    )

    # Day 4 실제 결과와 Count가 다르면 Figure를 만들기 전에 중단한다.
    assert_expected_analysis_counts(
        analysis,
        expected_total_samples=expected_total_samples,
        expected_false_positive_count=(
            expected_false_positive_count
        ),
        expected_false_negative_count=(
            expected_false_negative_count
        ),
        expected_misclassified_count=(
            expected_misclassified_count
        ),
    )

    figure_paths = create_day5_misclassification_figures(
        analysis,
        project_root=project_root,
        false_positive_output_path=(
            false_positive_figure_path
        ),
        false_negative_output_path=(
            false_negative_figure_path
        ),
        all_misclassifications_output_path=(
            all_misclassifications_figure_path
        ),
        max_columns=4,
        dpi=180,
    )

    # 이미지 경로와 파일 상태까지 확인된 이후 분석 JSON을 저장한다.
    saved_analysis_path = save_json_atomic(
        analysis,
        analysis_output_path,
    )

    return {
        "analysis": analysis,
        "analysis_output_path": saved_analysis_path,
        "figure_paths": figure_paths,
    }


def _print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def main() -> None:
    _print_header(
        "DAY 5 - RESNET18 MISCLASSIFIED IMAGE ANALYSIS "
        "AND VISUALIZATION"
    )

    print()
    print("[INPUT]")
    print(f"Evaluation artifact       : {DAY4_EVALUATION_PATH}")
    print(f"Classification threshold : {DEFAULT_CLASSIFICATION_THRESHOLD}")

    result = run_day5_misclassification_analysis()

    analysis = result["analysis"]
    summary = analysis["summary"]
    rankings = analysis["rankings"]

    _print_header("MISCLASSIFICATION SUMMARY")

    print(f"Total test samples       : {summary['total_samples']}")
    print(f"Correct samples          : {summary['correct_samples']}")
    print(
        "Misclassified samples    : "
        f"{summary['misclassified_samples']}"
    )
    print(
        "False Positive           : "
        f"{summary['false_positive_count']}"
    )
    print(
        "False Negative           : "
        f"{summary['false_negative_count']}"
    )
    print(
        "Error rate               : "
        f"{summary['error_rate'] * 100:.2f}%"
    )

    _print_header("CONFIDENCE SUMMARY")

    distance_summary = summary["threshold_distance"]
    confidence_summary = summary["wrong_prediction_confidence"]

    print(
        "Threshold distance min   : "
        f"{distance_summary['minimum']:.12f}"
    )
    print(
        "Threshold distance max   : "
        f"{distance_summary['maximum']:.12f}"
    )
    print(
        "Threshold distance mean  : "
        f"{distance_summary['mean']:.12f}"
    )
    print(
        "Wrong confidence min     : "
        f"{confidence_summary['minimum']:.12f}"
    )
    print(
        "Wrong confidence max     : "
        f"{confidence_summary['maximum']:.12f}"
    )
    print(
        "Wrong confidence mean    : "
        f"{confidence_summary['mean']:.12f}"
    )

    _print_header("MOST CONFIDENT ERRORS")

    for rank, sample in enumerate(
        rankings["most_confident_errors"],
        start=1,
    ):
        print(
            f"{rank}. "
            f"sample_index={sample['sample_index']} | "
            f"type={sample['error_type']} | "
            f"P(DEFECT)={sample['defect_probability']:.6f} | "
            f"wrong_confidence="
            f"{sample['wrong_prediction_confidence']:.6f} | "
            f"file={sample['image_filename']}"
        )

    _print_header("CLOSEST BOUNDARY ERRORS")

    for rank, sample in enumerate(
        rankings["closest_boundary_errors"],
        start=1,
    ):
        print(
            f"{rank}. "
            f"sample_index={sample['sample_index']} | "
            f"type={sample['error_type']} | "
            f"P(DEFECT)={sample['defect_probability']:.6f} | "
            f"threshold_distance="
            f"{sample['threshold_distance']:.6f} | "
            f"file={sample['image_filename']}"
        )

    _print_header("GENERATED ARTIFACTS")

    print(
        "Analysis JSON            : "
        f"{result['analysis_output_path']}"
    )
    print(
        "False Positive figure    : "
        f"{result['figure_paths']['false_positives']}"
    )
    print(
        "False Negative figure    : "
        f"{result['figure_paths']['false_negatives']}"
    )
    print(
        "All errors figure        : "
        f"{result['figure_paths']['all_misclassifications']}"
    )

    print()
    print("[PASS] Day 5 misclassification analysis completed.")


if __name__ == "__main__":
    main()