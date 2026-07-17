"""Day 9 NEU-DET 전체 분석, Split Manifest, Figure 생성 Script."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import time

from src.detection.dataset_analysis import (
    DatasetAnalysisResult,
    analyze_detection_dataset,
    combine_analysis_results,
    save_analysis_json,
)
from src.detection.dataset_config import (
    SplitRatios,
    build_partition_config,
    discover_neu_det_partitions,
)
from src.detection.dataset_split import (
    build_existing_split_manifest,
    build_source_preserving_split_manifest,
    build_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)
from src.detection.dataset_visualization import (
    create_annotation_overview_figure,
    create_box_statistics_figure,
    create_class_distribution_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NEU-DET 데이터셋 전체 무결성·통계·Split·Figure 분석",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="프로젝트 루트",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="기본값: <project-root>/data/raw/neu_det",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.5,
        help=(
            "원본 validation을 최종 validation/test로 나눌 때 "
            "validation에 둘 비율"
        ),
    )
    return parser.parse_args()


def _records_by_source_split(
    result: DatasetAnalysisResult,
) -> dict[str, list[object]]:
    grouped: dict[str, list[object]] = {}
    for record in result.records:
        grouped.setdefault(record.source_split, []).append(record)
    return grouped


def _build_manifest(
    result: DatasetAnalysisResult,
    *,
    validation_fraction: float,
):
    grouped = _records_by_source_split(result)
    names = set(grouped)

    if {"train", "validation", "test"}.issubset(names):
        return build_existing_split_manifest(
            {
                "train": grouped["train"],
                "validation": grouped["validation"],
                "test": grouped["test"],
            },
            random_seed=42,
        )

    if {"train", "validation"}.issubset(names):
        return build_source_preserving_split_manifest(
            grouped["train"],
            grouped["validation"],
            validation_fraction=validation_fraction,
            random_seed=42,
        )

    return build_split_manifest(
        result.records,
        ratios=SplitRatios(train=0.70, validation=0.15, test=0.15),
        random_seed=42,
    )


def _print_partition_summary(result: DatasetAnalysisResult) -> None:
    summary = result.summary
    source_split = result.config.get("source_split", "unknown")
    print(f"\n[SOURCE PARTITION] {source_split}")
    print(f"Images       : {summary['total_image_files']}")
    print(f"Annotations  : {summary['total_annotation_files']}")
    print(f"Valid records: {summary['valid_record_count']}")
    print(f"Valid boxes  : {summary['total_valid_bounding_boxes']}")
    print(f"Errors       : {summary['error_issue_count']}")
    print(f"Warnings     : {summary['warning_issue_count']}")


def _print_combined_summary(result: DatasetAnalysisResult) -> None:
    summary = result.summary
    print("\n" + "=" * 100)
    print("COMBINED DATASET SUMMARY")
    print("=" * 100)
    print(f"Images              : {summary['total_image_files']}")
    print(f"Annotations         : {summary['total_annotation_files']}")
    print(f"Valid records       : {summary['valid_record_count']}")
    print(f"Valid boxes         : {summary['total_valid_bounding_boxes']}")
    print(f"Classes             : {summary['class_count']}")
    print(f"Image modes         : {summary['image_mode_counts']}")
    print(f"Missing annotations : {summary['missing_annotation_count']}")
    print(f"Missing images      : {summary['missing_image_count']}")
    print(
        "Reconciled pairs    : "
        f"{summary['reconciled_cross_partition_pair_count']}"
    )
    print(f"Corrupted images    : {summary['corrupted_image_count']}")
    print(f"Invalid annotations : {summary['invalid_annotation_count']}")
    print(f"Invalid boxes       : {summary['invalid_box_count']}")
    print(
        "Duplicate hash groups: "
        f"{summary['duplicate_image_hash_group_count']}"
    )
    print(
        "Cross-source duplicates: "
        f"{summary['cross_source_split_duplicate_hash_count']}"
    )
    print(f"Errors              : {summary['error_issue_count']}")
    print(f"Warnings            : {summary['warning_issue_count']}")
    print(f"Issue counts        : {summary['issue_counts_by_code']}")
    print(
        "Coordinate policy   : "
        f"{summary['coordinate_statistics']['inferred_source_coordinate_policy']}"
    )


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    project_root = args.project_root.resolve()
    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else project_root / "data" / "raw" / "neu_det"
    )

    print("=" * 100)
    print("DAY 9 - OBJECT DETECTION DATASET ANALYSIS")
    print("=" * 100)
    print(f"Project root : {project_root}")
    print(f"Dataset root : {dataset_root}")

    layout = discover_neu_det_partitions(dataset_root)
    print(f"Partitions   : {', '.join(layout.partition_names)}")

    partition_results: list[DatasetAnalysisResult] = []
    for partition in layout.partitions:
        config = build_partition_config(
            project_root=project_root,
            dataset_root=dataset_root,
            partition=partition,
        )
        result = analyze_detection_dataset(config)
        partition_results.append(result)
        _print_partition_summary(result)

    provenance = {
        "dataset_name": "NEU Surface Defect Database (NEU-DET)",
        "original_source": "Northeastern University research dataset page",
        "download_mirror": (
            "Kaggle dataset: kaustubhdikshit/neu-surface-defect-database"
        ),
        "downloaded_archive_name": "NEU-DET.zip",
        "archive_documentation_check": {
            "readme_found": False,
            "license_found": False,
            "citation_file_found": False,
        },
        "license_status": (
            "No standardized license file was found inside the downloaded "
            "archive. Original files are not redistributed in the repository."
        ),
        "source_split_status": (
            "The downloaded mirror supplies train and validation directories. "
            "This is recorded as a mirror-provided split, not claimed as an "
            "official NEU split."
        ),
        "data_quality_policy": {
            "raw_files_modified": False,
            "cross_partition_pair": (
                "A unique image/XML stem separated across source partitions is "
                "paired in the analysis manifest and assigned to the image "
                "source partition."
            ),
            "duplicate_image_hash": (
                "Duplicate records are preserved but every identical hash "
                "group must remain inside one final split."
            ),
        },
    }
    combined = combine_analysis_results(
        partition_results,
        dataset_root=dataset_root,
        provenance=provenance,
    )

    artifact_dir = project_root / "reports" / "artifacts"
    figure_dir = project_root / "reports" / "figures"
    processed_dir = project_root / "data" / "processed" / "neu_det"
    analysis_path = artifact_dir / "day9_object_detection_dataset_analysis.json"
    split_artifact_path = (
        artifact_dir / "day9_object_detection_dataset_split.json"
    )
    processed_split_path = processed_dir / "splits.json"

    save_analysis_json(combined, analysis_path)

    # Split 실패 여부와 관계없이 품질 문제를 육안 점검할 Figure는 먼저 만든다.
    class_figure = create_class_distribution_figure(
        combined,
        figure_dir / "day9_detection_class_distribution.png",
    )
    box_figure = create_box_statistics_figure(
        combined,
        figure_dir / "day9_detection_box_statistics.png",
    )
    overview_figure = create_annotation_overview_figure(
        combined,
        dataset_root=dataset_root,
        output_path=figure_dir / "day9_detection_annotation_overview.png",
        max_samples=6,
    )

    _print_combined_summary(combined)
    summary = combined.summary
    if int(summary["error_issue_count"]) > 0:
        print("\n[ARTIFACTS CREATED BEFORE SPLIT]")
        for path in (analysis_path, class_figure, box_figure, overview_figure):
            print(path)
        print(
            "\n[FAIL] 유효한 Dataset 오류가 남아 있어 Split 생성을 "
            "중단했습니다. Traceback 없이 분석 결과를 보존했습니다."
        )
        return 1

    try:
        manifest = _build_manifest(
            combined,
            validation_fraction=args.validation_fraction,
        )
    except (ValueError, RuntimeError) as exc:
        print("\n[ARTIFACTS CREATED BEFORE SPLIT]")
        for path in (analysis_path, class_figure, box_figure, overview_figure):
            print(path)
        print(f"\n[FAIL] Split Manifest 생성 실패: {exc}")
        return 1

    save_split_manifest(manifest, split_artifact_path)
    processed_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(split_artifact_path, processed_split_path)

    validation = validate_split_manifest(
        manifest,
        expected_record_count=len(combined.records),
    )
    runtime_seconds = time.perf_counter() - started_at

    print("\n[SPLIT]")
    print(f"Policy     : {manifest.statistics['split_policy']}")
    print(
        "Duplicate policy: "
        f"{manifest.statistics['duplicate_hash_policy']}"
    )
    for split_name in ("train", "validation", "test"):
        stats = manifest.statistics[split_name]
        print(
            f"{split_name:<10}: images={stats['image_count']}, "
            f"boxes={stats['box_count']}, "
            "duplicate_hash_groups="
            f"{stats['duplicate_image_hash_group_count']}"
        )
    print(f"Validation : {validation}")

    print("\n[ARTIFACTS]")
    for path in (
        analysis_path,
        split_artifact_path,
        processed_split_path,
        class_figure,
        box_figure,
        overview_figure,
    ):
        print(path)
    print(f"\nRuntime: {runtime_seconds:.2f} seconds")

    if not validation["is_valid"]:
        print("\n[FAIL] Split Manifest 검증에 실패했습니다.")
        return 1

    print("\n[PASS] Day 9 실제 Dataset 분석과 Split·Figure 생성 완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
