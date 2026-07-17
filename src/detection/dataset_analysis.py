"""NEU-DET 이미지·Annotation Pair 검사와 통계 분석."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from PIL import Image, UnidentifiedImageError

from .annotation_parser import (
    AnnotationParseError,
    BoundingBox,
    parse_pascal_voc_annotation,
    validate_annotation,
)
from .dataset_config import DetectionDatasetConfig


@dataclass(frozen=True)
class ImageAnnotationRecord:
    """Split과 Visualization에서 재사용할 신뢰 가능한 Pair 레코드."""

    key: str
    image_path: str
    annotation_path: str
    image_width: int
    image_height: int
    image_mode: str
    image_sha256: str
    class_names: tuple[str, ...]
    boxes: tuple[tuple[int, int, int, int], ...]
    source_split: str = "unspecified"

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "image_path": self.image_path,
            "annotation_path": self.annotation_path,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_mode": self.image_mode,
            "image_sha256": self.image_sha256,
            "class_names": list(self.class_names),
            "boxes": [list(box) for box in self.boxes],
            "source_split": self.source_split,
        }


@dataclass(frozen=True)
class DatasetAnalysisResult:
    config: dict[str, object]
    summary: dict[str, object]
    records: tuple[ImageAnnotationRecord, ...]
    issues: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config,
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
            "issues": list(self.issues),
        }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = list(values)
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numbers),
        "mean": float(mean(numbers)),
        "median": float(median(numbers)),
        "min": float(min(numbers)),
        "max": float(max(numbers)),
    }


def _build_unique_stem_map(
    paths: Sequence[Path],
) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[path.stem].append(path)

    unique: dict[str, Path] = {}
    duplicates: dict[str, list[str]] = {}
    for stem, stem_paths in grouped.items():
        if len(stem_paths) == 1:
            unique[stem] = stem_paths[0]
        else:
            duplicates[stem] = [str(path) for path in sorted(stem_paths)]
    return unique, duplicates


def _coordinate_statistics(
    records: Sequence[ImageAnnotationRecord],
) -> dict[str, object]:
    x_mins: list[int] = []
    y_mins: list[int] = []
    x_maxs: list[int] = []
    y_maxs: list[int] = []
    x_max_at_width = 0
    y_max_at_height = 0

    for record in records:
        for x_min, y_min, x_max, y_max in record.boxes:
            x_mins.append(x_min)
            y_mins.append(y_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)
            if x_max == record.image_width:
                x_max_at_width += 1
            if y_max == record.image_height:
                y_max_at_height += 1

    zero_coordinate_count = sum(
        value == 0
        for values in (x_mins, y_mins)
        for value in values
    )
    one_coordinate_count = sum(
        value == 1
        for values in (x_mins, y_mins)
        for value in values
    )

    if not x_mins:
        inferred_policy = "undetermined_no_valid_boxes"
    elif zero_coordinate_count > 0:
        inferred_policy = "zero_based_or_mixed_coordinates_review_required"
    elif (
        min(x_mins) >= 1
        and min(y_mins) >= 1
        and (x_max_at_width > 0 or y_max_at_height > 0)
    ):
        inferred_policy = "pascal_voc_one_based_inclusive_likely"
    else:
        inferred_policy = "no_zero_observed_coordinate_policy_still_requires_review"

    return {
        "x_min": _numeric_summary(x_mins),
        "y_min": _numeric_summary(y_mins),
        "x_max": _numeric_summary(x_maxs),
        "y_max": _numeric_summary(y_maxs),
        "zero_min_coordinate_count": zero_coordinate_count,
        "one_min_coordinate_count": one_coordinate_count,
        "x_max_at_image_width_count": x_max_at_width,
        "y_max_at_image_height_count": y_max_at_height,
        "inferred_source_coordinate_policy": inferred_policy,
        "model_conversion_policy": (
            "deferred_until_full_dataset_review; expected conversion for "
            "one-based inclusive VOC is (xmin-1, ymin-1, xmax, ymax)"
        ),
    }


def _summary_from_records(
    records: Sequence[ImageAnnotationRecord],
    *,
    all_classes: Sequence[str],
) -> dict[str, object]:
    class_image_counts: Counter[str] = Counter()
    class_box_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    source_split_image_counts: Counter[str] = Counter()
    source_split_box_counts: Counter[str] = Counter()
    boxes_per_image: list[int] = []
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_area_ratios: list[float] = []
    box_aspect_ratios: list[float] = []

    for record in records:
        unique_classes = set(record.class_names)
        class_image_counts.update(unique_classes)
        class_box_counts.update(record.class_names)
        mode_counts[record.image_mode] += 1
        source_split_image_counts[record.source_split] += 1
        source_split_box_counts[record.source_split] += len(record.boxes)
        boxes_per_image.append(len(record.boxes))
        image_area = record.image_width * record.image_height
        for x_min, y_min, x_max, y_max in record.boxes:
            width = x_max - x_min
            height = y_max - y_min
            box_widths.append(float(width))
            box_heights.append(float(height))
            if image_area > 0:
                box_area_ratios.append((width * height) / image_area)
            if height > 0:
                box_aspect_ratios.append(width / height)

    return {
        "valid_record_count": len(records),
        "total_valid_bounding_boxes": sum(class_box_counts.values()),
        "class_count": len(all_classes),
        "class_image_counts": {
            class_name: class_image_counts.get(class_name, 0)
            for class_name in all_classes
        },
        "class_box_counts": {
            class_name: class_box_counts.get(class_name, 0)
            for class_name in all_classes
        },
        "image_mode_counts": dict(sorted(mode_counts.items())),
        "source_split_image_counts": dict(sorted(source_split_image_counts.items())),
        "source_split_box_counts": dict(sorted(source_split_box_counts.items())),
        "boxes_per_image": _numeric_summary(boxes_per_image),
        "box_width": _numeric_summary(box_widths),
        "box_height": _numeric_summary(box_heights),
        "box_area_ratio": _numeric_summary(box_area_ratios),
        "box_aspect_ratio": _numeric_summary(box_aspect_ratios),
        "coordinate_statistics": _coordinate_statistics(records),
    }


def analyze_detection_dataset(
    config: DetectionDatasetConfig,
    *,
    allow_empty_annotations: bool = False,
    small_box_area_ratio_threshold: float = 0.0001,
) -> DatasetAnalysisResult:
    """한 원본 Partition을 한 번 순회해 무결성과 통계를 계산한다."""
    if not config.images_dir.exists():
        raise FileNotFoundError(f"Image Directory가 없습니다: {config.images_dir}")
    if not config.annotations_dir.exists():
        raise FileNotFoundError(
            f"Annotation Directory가 없습니다: {config.annotations_dir}"
        )

    all_image_dir_files = [
        path for path in config.images_dir.rglob("*") if path.is_file()
    ]
    all_annotation_dir_files = [
        path for path in config.annotations_dir.rglob("*") if path.is_file()
    ]

    image_files = sorted(
        path
        for path in all_image_dir_files
        if path.suffix.lower() in config.image_extensions
    )
    annotation_files = sorted(
        path
        for path in all_annotation_dir_files
        if path.suffix.lower() in config.annotation_extensions
    )
    supported_extensions = set(
        config.image_extensions + config.annotation_extensions
    )
    unsupported_files = sorted(
        str(path)
        for path in all_image_dir_files + all_annotation_dir_files
        if path.suffix.lower() not in supported_extensions
        and not path.name.startswith(".")
    )

    image_map, duplicate_image_stems = _build_unique_stem_map(image_files)
    annotation_map, duplicate_annotation_stems = _build_unique_stem_map(
        annotation_files
    )

    pair_keys = sorted(set(image_map) & set(annotation_map))
    missing_annotation_keys = sorted(set(image_map) - set(annotation_map))
    missing_image_keys = sorted(set(annotation_map) - set(image_map))

    issues: list[dict[str, object]] = []
    records: list[ImageAnnotationRecord] = []
    corrupted_image_count = 0
    invalid_annotation_count = 0
    duplicate_box_count = 0
    very_small_box_count = 0
    zero_coordinate_box_count = 0
    metadata_mismatch_count = 0
    invalid_box_count = 0

    for stem, paths in duplicate_image_stems.items():
        issues.append(
            {
                "scope": "dataset",
                "source_split": config.source_split,
                "key": stem,
                "code": "duplicate_image_stem",
                "severity": "error",
                "message": f"동일 stem 이미지가 여러 개입니다: {paths}",
            }
        )
    for stem, paths in duplicate_annotation_stems.items():
        issues.append(
            {
                "scope": "dataset",
                "source_split": config.source_split,
                "key": stem,
                "code": "duplicate_annotation_stem",
                "severity": "error",
                "message": f"동일 stem XML이 여러 개입니다: {paths}",
            }
        )
    for key in missing_annotation_keys:
        issues.append(
            {
                "scope": "pair",
                "source_split": config.source_split,
                "key": key,
                "code": "missing_annotation",
                "severity": "error",
                "message": f"이미지에 대응하는 XML이 없습니다: {image_map[key]}",
            }
        )
    for key in missing_image_keys:
        issues.append(
            {
                "scope": "pair",
                "source_split": config.source_split,
                "key": key,
                "code": "missing_image",
                "severity": "error",
                "message": f"XML에 대응하는 이미지가 없습니다: {annotation_map[key]}",
            }
        )

    for key in pair_keys:
        image_path = image_map[key]
        annotation_path = annotation_map[key]

        try:
            with Image.open(image_path) as image:
                image.load()
                actual_width, actual_height = image.size
                image_mode = image.mode
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            corrupted_image_count += 1
            issues.append(
                {
                    "scope": "image",
                    "source_split": config.source_split,
                    "key": key,
                    "code": "corrupted_image",
                    "severity": "error",
                    "message": f"이미지를 Decode할 수 없습니다: {exc}",
                }
            )
            continue

        try:
            annotation = parse_pascal_voc_annotation(annotation_path)
        except AnnotationParseError as exc:
            invalid_annotation_count += 1
            issues.append(
                {
                    "scope": "annotation",
                    "source_split": config.source_split,
                    "key": key,
                    "code": exc.code,
                    "severity": "error",
                    "message": exc.message,
                    "annotation_path": str(annotation_path),
                }
            )
            continue

        validation_issues = validate_annotation(
            annotation,
            actual_image_size=(actual_width, actual_height),
            actual_filename=image_path.name,
            allow_empty=allow_empty_annotations,
            small_box_area_ratio_threshold=small_box_area_ratio_threshold,
        )
        invalid_object_indexes: set[int] = set()
        for validation_issue in validation_issues:
            issue_dict = validation_issue.to_dict()
            issue_dict.update(
                {
                    "scope": "annotation_validation",
                    "source_split": config.source_split,
                    "key": key,
                    "annotation_path": str(annotation_path),
                }
            )
            issues.append(issue_dict)

            if (
                validation_issue.severity == "error"
                and validation_issue.object_index is not None
            ):
                invalid_object_indexes.add(validation_issue.object_index)
            if validation_issue.code == "duplicate_box":
                duplicate_box_count += 1
            elif validation_issue.code == "very_small_box":
                very_small_box_count += 1
            elif validation_issue.code == "zero_coordinate_observed":
                zero_coordinate_box_count += 1
            elif validation_issue.code in {
                "image_size_mismatch",
                "filename_mismatch",
            }:
                metadata_mismatch_count += 1

        invalid_box_count += len(invalid_object_indexes)
        valid_objects: list[tuple[str, BoundingBox]] = []
        for object_index, detection_object in enumerate(annotation.objects):
            if object_index in invalid_object_indexes:
                continue
            valid_objects.append((detection_object.class_name, detection_object.box))

        record_key = (
            f"{config.source_split}/{key}"
            if config.source_split != "unspecified"
            else key
        )
        records.append(
            ImageAnnotationRecord(
                key=record_key,
                image_path=_relative_or_absolute(image_path, config.dataset_root),
                annotation_path=_relative_or_absolute(
                    annotation_path,
                    config.dataset_root,
                ),
                image_width=actual_width,
                image_height=actual_height,
                image_mode=image_mode,
                image_sha256=_sha256_file(image_path),
                class_names=tuple(name for name, _ in valid_objects),
                boxes=tuple(tuple(box.as_list()) for _, box in valid_objects),
                source_split=config.source_split,
            )
        )

    hash_to_images: dict[str, list[str]] = defaultdict(list)
    for record in records:
        hash_to_images[record.image_sha256].append(record.image_path)
    duplicate_hash_groups = {
        digest: paths
        for digest, paths in hash_to_images.items()
        if len(paths) > 1
    }
    for digest, paths in duplicate_hash_groups.items():
        issues.append(
            {
                "scope": "dataset",
                "source_split": config.source_split,
                "key": digest,
                "code": "duplicate_image_hash",
                "severity": "warning",
                "message": (
                    "내용이 동일한 이미지가 여러 개입니다. 원본은 보존하고 "
                    "Split 생성 시 같은 Split에 묶습니다: "
                    f"{paths}"
                ),
            }
        )

    summary = _summary_from_records(
        records,
        all_classes=list(config.source_class_to_index),
    )
    summary.update(
        {
            "total_image_files": len(image_files),
            "total_annotation_files": len(annotation_files),
            "paired_stem_count": len(pair_keys),
            "missing_annotation_count": len(missing_annotation_keys),
            "missing_image_count": len(missing_image_keys),
            "duplicate_image_stem_count": len(duplicate_image_stems),
            "duplicate_annotation_stem_count": len(duplicate_annotation_stems),
            "corrupted_image_count": corrupted_image_count,
            "invalid_annotation_count": invalid_annotation_count,
            "invalid_box_count": invalid_box_count,
            "duplicate_box_count": duplicate_box_count,
            "very_small_box_count": very_small_box_count,
            "zero_coordinate_box_count": zero_coordinate_box_count,
            "metadata_mismatch_count": metadata_mismatch_count,
            "duplicate_image_hash_group_count": len(duplicate_hash_groups),
            "duplicate_image_count": sum(
                len(paths) for paths in duplicate_hash_groups.values()
            ),
            "unsupported_file_count": len(unsupported_files),
            "unsupported_files": unsupported_files,
            "error_issue_count": sum(
                issue.get("severity") == "error" for issue in issues
            ),
            "warning_issue_count": sum(
                issue.get("severity") == "warning" for issue in issues
            ),
        }
    )

    return DatasetAnalysisResult(
        config=config.to_dict(),
        summary=summary,
        records=tuple(records),
        issues=tuple(issues),
    )


def _collect_partition_files(
    results: Sequence[DatasetAnalysisResult],
) -> tuple[
    dict[str, list[tuple[str, Path]]],
    dict[str, list[tuple[str, Path]]],
]:
    """모든 원본 Partition의 이미지·XML을 stem 기준으로 수집한다."""
    images_by_stem: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    annotations_by_stem: dict[str, list[tuple[str, Path]]] = defaultdict(list)

    for result in results:
        source_split = str(result.config.get("source_split", "unspecified"))
        images_dir = Path(str(result.config["images_dir"]))
        annotations_dir = Path(str(result.config["annotations_dir"]))
        image_extensions = {
            str(value).lower()
            for value in result.config.get("image_extensions", [])
        }
        annotation_extensions = {
            str(value).lower()
            for value in result.config.get("annotation_extensions", [])
        }

        for path in images_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in image_extensions:
                images_by_stem[path.stem].append((source_split, path))
        for path in annotations_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in annotation_extensions:
                annotations_by_stem[path.stem].append((source_split, path))

    return images_by_stem, annotations_by_stem


def _build_cross_partition_record(
    *,
    stem: str,
    image_source_split: str,
    image_path: Path,
    annotation_source_split: str,
    annotation_path: Path,
    dataset_root: Path,
) -> tuple[ImageAnnotationRecord | None, list[dict[str, object]]]:
    """서로 다른 Partition에 놓인 이미지·XML을 원본 수정 없이 검증한다."""
    issues: list[dict[str, object]] = []
    try:
        with Image.open(image_path) as image:
            image.load()
            actual_width, actual_height = image.size
            image_mode = image.mode
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        issues.append(
            {
                "scope": "cross_partition_reconciliation",
                "source_split": image_source_split,
                "key": stem,
                "code": "cross_partition_corrupted_image",
                "severity": "error",
                "message": f"교차 Partition 이미지 Decode 실패: {exc}",
            }
        )
        return None, issues

    try:
        annotation = parse_pascal_voc_annotation(annotation_path)
    except AnnotationParseError as exc:
        issues.append(
            {
                "scope": "cross_partition_reconciliation",
                "source_split": image_source_split,
                "key": stem,
                "code": exc.code,
                "severity": "error",
                "message": exc.message,
                "annotation_path": str(annotation_path),
            }
        )
        return None, issues

    validation_issues = validate_annotation(
        annotation,
        actual_image_size=(actual_width, actual_height),
        actual_filename=image_path.name,
        allow_empty=False,
    )
    for validation_issue in validation_issues:
        issue = validation_issue.to_dict()
        issue.update(
            {
                "scope": "cross_partition_reconciliation",
                "source_split": image_source_split,
                "key": stem,
                "image_path": str(image_path),
                "annotation_path": str(annotation_path),
            }
        )
        issues.append(issue)

    if any(issue.get("severity") == "error" for issue in issues):
        return None, issues

    record = ImageAnnotationRecord(
        key=f"{image_source_split}/{stem}",
        image_path=_relative_or_absolute(image_path, dataset_root),
        annotation_path=_relative_or_absolute(annotation_path, dataset_root),
        image_width=actual_width,
        image_height=actual_height,
        image_mode=image_mode,
        image_sha256=_sha256_file(image_path),
        class_names=tuple(obj.class_name for obj in annotation.objects),
        boxes=tuple(tuple(obj.box.as_list()) for obj in annotation.objects),
        source_split=image_source_split,
    )
    issues.append(
        {
            "scope": "cross_partition_reconciliation",
            "source_split": image_source_split,
            "key": stem,
            "code": "cross_partition_pair_reconciled",
            "severity": "warning",
            "message": (
                "이미지와 XML이 서로 다른 원본 Partition에 있어 원본 파일을 "
                "이동하지 않고 Manifest 수준에서 연결했습니다."
            ),
            "image_source_split": image_source_split,
            "annotation_source_split": annotation_source_split,
            "image_path": _relative_or_absolute(image_path, dataset_root),
            "annotation_path": _relative_or_absolute(
                annotation_path,
                dataset_root,
            ),
        }
    )
    return record, issues


def _reconcile_cross_partition_pairs(
    results: Sequence[DatasetAnalysisResult],
    *,
    dataset_root: Path,
) -> tuple[
    list[ImageAnnotationRecord],
    set[str],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """동일 stem의 이미지·XML이 서로 다른 Partition에 있으면 안전하게 연결한다."""
    missing_annotation_stems = {
        str(issue.get("key"))
        for result in results
        for issue in result.issues
        if issue.get("code") == "missing_annotation"
    }
    missing_image_stems = {
        str(issue.get("key"))
        for result in results
        for issue in result.issues
        if issue.get("code") == "missing_image"
    }
    candidate_stems = sorted(
        missing_annotation_stems & missing_image_stems
    )
    images_by_stem, annotations_by_stem = _collect_partition_files(results)

    records: list[ImageAnnotationRecord] = []
    resolved_stems: set[str] = set()
    issues: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    for stem in candidate_stems:
        image_candidates = images_by_stem.get(stem, [])
        annotation_candidates = annotations_by_stem.get(stem, [])
        if len(image_candidates) != 1 or len(annotation_candidates) != 1:
            continue

        image_source_split, image_path = image_candidates[0]
        annotation_source_split, annotation_path = annotation_candidates[0]
        if image_source_split == annotation_source_split:
            continue

        record, pair_issues = _build_cross_partition_record(
            stem=stem,
            image_source_split=image_source_split,
            image_path=image_path,
            annotation_source_split=annotation_source_split,
            annotation_path=annotation_path,
            dataset_root=dataset_root,
        )
        issues.extend(pair_issues)
        if record is None:
            continue

        records.append(record)
        resolved_stems.add(stem)
        details.append(
            {
                "stem": stem,
                "assigned_source_split": image_source_split,
                "image_source_split": image_source_split,
                "annotation_source_split": annotation_source_split,
                "image_path": record.image_path,
                "annotation_path": record.annotation_path,
                "policy": (
                    "preserve_raw_files_and_pair_by_unique_global_stem; "
                    "assign_record_to_image_source_split"
                ),
            }
        )

    return records, resolved_stems, issues, details


def combine_analysis_results(
    results: Sequence[DatasetAnalysisResult],
    *,
    dataset_root: Path,
    provenance: dict[str, object] | None = None,
) -> DatasetAnalysisResult:
    """여러 원본 Partition을 결합하고 교차 Partition Pair를 보정한다."""
    if not results:
        raise ValueError("결합할 Dataset 분석 결과가 없습니다.")

    raw_issues = [dict(issue) for result in results for issue in result.issues]
    raw_error_issue_count = sum(
        issue.get("severity") == "error" for issue in raw_issues
    )
    reconciled_records, resolved_stems, reconciliation_issues, details = (
        _reconcile_cross_partition_pairs(results, dataset_root=dataset_root)
    )

    issues = [
        issue
        for issue in raw_issues
        if not (
            str(issue.get("key")) in resolved_stems
            and issue.get("code") in {"missing_annotation", "missing_image"}
        )
    ]
    issues.extend(reconciliation_issues)

    records = tuple(
        [record for result in results for record in result.records]
        + reconciled_records
    )
    class_names = list(
        results[0].config.get("source_class_to_index", {}).keys()
    )
    summary = _summary_from_records(records, all_classes=class_names)

    raw_summed_fields = (
        "total_image_files",
        "total_annotation_files",
        "paired_stem_count",
        "missing_annotation_count",
        "missing_image_count",
        "duplicate_image_stem_count",
        "duplicate_annotation_stem_count",
        "corrupted_image_count",
        "invalid_annotation_count",
        "invalid_box_count",
        "duplicate_box_count",
        "very_small_box_count",
        "zero_coordinate_box_count",
        "metadata_mismatch_count",
        "unsupported_file_count",
    )
    raw_totals = {
        field: sum(int(result.summary.get(field, 0)) for result in results)
        for field in raw_summed_fields
    }
    summary.update(raw_totals)
    summary["raw_missing_annotation_count"] = raw_totals[
        "missing_annotation_count"
    ]
    summary["raw_missing_image_count"] = raw_totals["missing_image_count"]
    summary["missing_annotation_count"] = max(
        0,
        raw_totals["missing_annotation_count"] - len(resolved_stems),
    )
    summary["missing_image_count"] = max(
        0,
        raw_totals["missing_image_count"] - len(resolved_stems),
    )
    summary["paired_stem_count"] = (
        raw_totals["paired_stem_count"] + len(reconciled_records)
    )

    hash_to_records: dict[str, list[ImageAnnotationRecord]] = defaultdict(list)
    for record in records:
        hash_to_records[record.image_sha256].append(record)
    duplicate_hash_groups = {
        digest: grouped
        for digest, grouped in hash_to_records.items()
        if len(grouped) > 1
    }
    cross_split_duplicate_groups = {
        digest: grouped
        for digest, grouped in duplicate_hash_groups.items()
        if len({record.source_split for record in grouped}) > 1
    }
    for digest, grouped in cross_split_duplicate_groups.items():
        issues.append(
            {
                "scope": "combined_dataset",
                "key": digest,
                "code": "cross_source_split_duplicate_image_hash",
                "severity": "error",
                "message": (
                    "동일 이미지가 여러 원본 Split에 포함되어 있습니다: "
                    f"{[record.image_path for record in grouped]}"
                ),
            }
        )

    issue_counts = Counter(str(issue.get("code", "unknown")) for issue in issues)
    summary.update(
        {
            "duplicate_image_hash_group_count": len(duplicate_hash_groups),
            "duplicate_image_count": sum(
                len(grouped) for grouped in duplicate_hash_groups.values()
            ),
            "cross_source_split_duplicate_hash_count": len(
                cross_split_duplicate_groups
            ),
            "duplicate_image_policy": (
                "preserve_original_records_and_keep_identical_hashes_in_one_"
                "final_split"
            ),
            "source_partition_count": len(results),
            "source_partition_summaries": {
                str(result.config.get("source_split", index)): result.summary
                for index, result in enumerate(results)
            },
            "raw_error_issue_count_before_reconciliation": (
                raw_error_issue_count
            ),
            "reconciled_cross_partition_pair_count": len(
                reconciled_records
            ),
            "reconciled_cross_partition_pairs": details,
            "issue_counts_by_code": dict(sorted(issue_counts.items())),
            "error_issue_count": sum(
                issue.get("severity") == "error" for issue in issues
            ),
            "warning_issue_count": sum(
                issue.get("severity") == "warning" for issue in issues
            ),
        }
    )

    config: dict[str, object] = {
        "dataset_root": str(dataset_root.resolve()),
        "partitions": [result.config for result in results],
        "reconciliation_policy": {
            "raw_files_modified": False,
            "cross_partition_pair_policy": (
                "pair only when exactly one image and one XML share the same "
                "global stem; assign to the image source split"
            ),
            "duplicate_image_policy": (
                "preserve records and prevent cross-split hash leakage"
            ),
        },
    }
    if provenance is not None:
        config["provenance"] = provenance

    return DatasetAnalysisResult(
        config=config,
        summary=summary,
        records=records,
        issues=tuple(issues),
    )

def save_analysis_json(
    result: DatasetAnalysisResult,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
