"""Day 11 실제 NEU-DET Detection Dataset의 런타임 계약을 검증한다.

[기존 코드 참고]
Day 9 ``splits.json``의 실제 Schema와 Class·Box 정보를 기준값으로 사용한다.

[그대로 재사용]
- ``NeuDetDetectionDataset``
- ``create_neu_det_detection_data_loaders``
- Day 9 Class Mapping

[신규 구현]
- 세 Split의 모든 Image·Target을 실제로 로딩한다.
- Manifest Class·Box와 Dataset 변환 결과를 순서까지 포함해 비교한다.
- 정확히 같은 Class·좌표 Duplicate의 보존 상태를 감사한다.
- DataLoader의 가변 Box Batch와 CPU 설정을 검증한다.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import RandomSampler, SequentialSampler

from src.detection.data_loader import (
    DetectionDataLoaderConfig,
    DetectionDataLoaders,
    create_neu_det_detection_data_loaders,
)
from src.detection.dataset_config import DETECTION_MODEL_CLASS_TO_INDEX
from src.detection.detection_dataset import (
    NeuDetDetectionDataset,
    convert_voc_box_to_torchvision,
)
from src.detection.model_config import SUPPORTED_SPLITS


REQUIRED_TARGET_KEYS = frozenset(
    {"boxes", "labels", "image_id", "area", "iscrowd"}
)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path.")
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}.")
    with path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise TypeError("JSON top-level value must be an object.")
    return payload


def _manifest_records(
    payload: Mapping[str, Any],
    split: str,
) -> list[Mapping[str, Any]]:
    splits = payload.get("splits")
    if not isinstance(splits, Mapping):
        raise KeyError("Manifest must contain a 'splits' object.")
    records = splits.get(split)
    if not isinstance(records, list):
        raise KeyError(f"Manifest split {split!r} must be a list.")
    if any(not isinstance(record, Mapping) for record in records):
        raise TypeError(f"Every {split!r} record must be an object.")
    return list(records)


def _validated_record_annotations(
    record: Mapping[str, Any],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    class_names = record.get("class_names")
    boxes = record.get("boxes")
    if not isinstance(class_names, list) or not isinstance(boxes, list):
        raise TypeError("Manifest record class_names and boxes must be lists.")
    if len(class_names) != len(boxes):
        raise ValueError("Manifest class_names and boxes counts must match.")

    annotations: list[tuple[str, tuple[int, int, int, int]]] = []
    for object_index, (class_name, box) in enumerate(zip(class_names, boxes)):
        if not isinstance(class_name, str) or not class_name:
            raise ValueError(
                f"Manifest object {object_index} has an invalid class name."
            )
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in box
            )
        ):
            raise ValueError(
                f"Manifest object {object_index} box must be four integers."
            )
        annotations.append((class_name, tuple(box)))
    return annotations


def _apply_manifest_duplicate_policy(
    annotations: Sequence[tuple[str, tuple[int, int, int, int]]],
    policy: str,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    if policy == "preserve":
        return list(annotations)
    if policy != "remove_exact":
        raise ValueError("Unsupported duplicate policy.")

    output: list[tuple[str, tuple[int, int, int, int]]] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for annotation in annotations:
        if annotation in seen:
            continue
        seen.add(annotation)
        output.append(annotation)
    return output


def _exact_duplicate_count(
    annotations: Sequence[tuple[str, tuple[int, int, int, int]]],
) -> int:
    counts = Counter(annotations)
    return sum(count - 1 for count in counts.values() if count > 1)


def find_manifest_exact_duplicate_records(
    manifest_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """동일 Record 내부에서 Class·좌표가 완전히 같은 Box를 찾는다."""
    results: list[dict[str, Any]] = []
    for split in SUPPORTED_SPLITS:
        for record in _manifest_records(manifest_payload, split):
            annotations = _validated_record_annotations(record)
            counts = Counter(annotations)
            duplicates = [
                {
                    "class_name": class_name,
                    "box": list(box),
                    "occurrence_count": occurrence_count,
                    "duplicate_count": occurrence_count - 1,
                }
                for (class_name, box), occurrence_count in counts.items()
                if occurrence_count > 1
            ]
            if duplicates:
                results.append(
                    {
                        "split": split,
                        "record_id": record.get("key"),
                        "image_path": record.get("image_path"),
                        "annotation_path": record.get("annotation_path"),
                        "duplicates": duplicates,
                    }
                )
    return results


def _summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": round(mean(numeric), 6),
        "median": round(median(numeric), 6),
        "min": round(min(numeric), 6),
        "max": round(max(numeric), 6),
    }


def _validate_tensor_contract(
    *,
    image: Tensor,
    target: Mapping[str, Tensor],
    expected_image_id: int,
    num_classes: int,
) -> list[str]:
    errors: list[str] = []
    if image.dtype != torch.float32:
        errors.append("image dtype is not float32")
    if image.ndim != 3 or int(image.shape[0]) != 3:
        errors.append("image shape is not [3, H, W]")
    if image.numel() == 0 or not bool(torch.isfinite(image).all()):
        errors.append("image is empty or non-finite")
    elif float(image.min()) < 0.0 or float(image.max()) > 1.0:
        errors.append("image range is outside [0, 1]")

    if set(target) != REQUIRED_TARGET_KEYS:
        errors.append("target keys do not match the Detection contract")
        return errors

    boxes = target["boxes"]
    labels = target["labels"]
    image_id = target["image_id"]
    area = target["area"]
    iscrowd = target["iscrowd"]

    if boxes.dtype != torch.float32 or boxes.ndim != 2 or boxes.shape[-1] != 4:
        errors.append("boxes are not FloatTensor[N, 4]")
    if labels.dtype != torch.int64 or labels.ndim != 1:
        errors.append("labels are not Int64Tensor[N]")
    if image_id.dtype != torch.int64 or image_id.tolist() != [expected_image_id]:
        errors.append("image_id is not the stable split index")
    if area.dtype != torch.float32 or area.ndim != 1:
        errors.append("area is not FloatTensor[N]")
    if iscrowd.dtype != torch.int64 or iscrowd.ndim != 1:
        errors.append("iscrowd is not Int64Tensor[N]")

    if not (len(boxes) == len(labels) == len(area) == len(iscrowd)):
        errors.append("target field lengths do not match")
        return errors

    if boxes.numel() > 0:
        height = int(image.shape[-2])
        width = int(image.shape[-1])
        if not bool(torch.isfinite(boxes).all()):
            errors.append("boxes contain non-finite values")
        if not bool(torch.all(boxes[:, 2] > boxes[:, 0])):
            errors.append("a box does not satisfy xmax > xmin")
        if not bool(torch.all(boxes[:, 3] > boxes[:, 1])):
            errors.append("a box does not satisfy ymax > ymin")
        if not bool(torch.all(boxes[:, :2] >= 0)):
            errors.append("a box minimum coordinate is negative")
        if not bool(torch.all(boxes[:, 2] <= width)):
            errors.append("a box xmax exceeds image width")
        if not bool(torch.all(boxes[:, 3] <= height)):
            errors.append("a box ymax exceeds image height")
        if not bool(torch.all((labels >= 1) & (labels < num_classes))):
            errors.append("a label is outside [1, num_classes - 1]")
        expected_area = (
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        )
        if not bool(torch.allclose(area, expected_area)):
            errors.append("area does not match boxes")
    if not bool(torch.all(iscrowd == 0)):
        errors.append("iscrowd contains a non-zero value")
    return errors


def validate_detection_split(
    *,
    dataset: NeuDetDetectionDataset,
    manifest_records: Sequence[Mapping[str, Any]],
    split: str,
    max_recorded_errors: int = 20,
) -> dict[str, Any]:
    """한 Split 전체를 실제로 로딩하고 Manifest와 Target을 비교한다."""
    if len(dataset) != len(manifest_records):
        raise ValueError(
            f"{split} dataset length {len(dataset)} does not match manifest "
            f"length {len(manifest_records)}."
        )

    errors: list[dict[str, Any]] = []
    box_counts: list[int] = []
    class_box_counts: Counter[str] = Counter()
    image_shape_counts: Counter[str] = Counter()
    manifest_box_count = 0
    dataset_box_count = 0
    raw_duplicate_count = 0
    effective_duplicate_count = 0
    compared_record_count = 0

    for index, (sample, record) in enumerate(zip(dataset.samples, manifest_records)):
        record_id = record.get("key")
        try:
            if sample.record_id != record_id:
                raise ValueError(
                    f"Dataset record_id {sample.record_id!r} does not match "
                    f"manifest key {record_id!r}."
                )

            raw_annotations = _validated_record_annotations(record)
            effective_annotations = _apply_manifest_duplicate_policy(
                raw_annotations,
                dataset.duplicate_box_policy,
            )
            raw_duplicate_count += _exact_duplicate_count(raw_annotations)
            effective_duplicate_count += _exact_duplicate_count(
                effective_annotations
            )
            manifest_box_count += len(effective_annotations)

            image, target = dataset[index]
            contract_errors = _validate_tensor_contract(
                image=image,
                target=target,
                expected_image_id=index,
                num_classes=len(dataset.class_names) + 1,
            )
            if contract_errors:
                raise ValueError("; ".join(contract_errors))

            expected_boxes = torch.tensor(
                [
                    convert_voc_box_to_torchvision(
                        box,
                        image_width=int(image.shape[-1]),
                        image_height=int(image.shape[-2]),
                    )
                    for _, box in effective_annotations
                ],
                dtype=torch.float32,
            ).reshape(-1, 4)
            expected_labels = torch.tensor(
                [
                    dataset.class_to_index[class_name]
                    for class_name, _ in effective_annotations
                ],
                dtype=torch.int64,
            )

            if not torch.equal(target["boxes"], expected_boxes):
                raise ValueError("Dataset boxes do not match converted Manifest boxes.")
            if not torch.equal(target["labels"], expected_labels):
                raise ValueError("Dataset labels do not match Manifest class order.")

            box_count = len(target["boxes"])
            box_counts.append(box_count)
            dataset_box_count += box_count
            for label in target["labels"].tolist():
                class_box_counts[dataset.index_to_class[int(label)]] += 1
            image_shape_counts[
                f"{int(image.shape[-1])}x{int(image.shape[-2])}"
            ] += 1
            compared_record_count += 1
        except Exception as error:  # noqa: BLE001 - Artifact에 실제 실패를 기록한다.
            if len(errors) < max_recorded_errors:
                errors.append(
                    {
                        "index": index,
                        "record_id": record_id,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )

    checks = {
        "dataset_length_matches_manifest": len(dataset) == len(manifest_records),
        "all_records_loaded": compared_record_count == len(dataset),
        "manifest_and_dataset_box_counts_match": (
            manifest_box_count == dataset_box_count
        ),
        "no_sample_contract_errors": not errors,
        "all_target_keys_preserved": not errors,
        "all_image_ids_stable": not errors,
    }
    return {
        "split": split,
        "sample_count": len(dataset),
        "compared_record_count": compared_record_count,
        "manifest_box_count_after_policy": manifest_box_count,
        "dataset_box_count": dataset_box_count,
        "class_box_counts": {
            class_name: class_box_counts.get(class_name, 0)
            for class_name in dataset.class_names
        },
        "boxes_per_image": _summary(box_counts),
        "image_shape_counts": dict(sorted(image_shape_counts.items())),
        "duplicate_box_audit": {
            "raw_exact_duplicate_count": raw_duplicate_count,
            "effective_exact_duplicate_count": effective_duplicate_count,
            "policy": dataset.duplicate_box_policy,
        },
        "checks": checks,
        "errors": errors,
        "validation_passed": all(checks.values()),
    }


def _validate_loader(
    *,
    loader: Any,
    dataset_length: int,
    expected_batch_size: int,
    expected_shuffle: bool,
) -> dict[str, Any]:
    images, targets = next(iter(loader))
    actual_batch_size = len(images)
    expected_first_batch_size = min(dataset_length, expected_batch_size)
    sampler_matches = (
        isinstance(loader.sampler, RandomSampler)
        if expected_shuffle
        else isinstance(loader.sampler, SequentialSampler)
    )
    checks = {
        "images_are_tuple": isinstance(images, tuple),
        "targets_are_tuple": isinstance(targets, tuple),
        "image_and_target_counts_match": len(images) == len(targets),
        "first_batch_size_matches": actual_batch_size == expected_first_batch_size,
        "variable_targets_not_stacked": all(isinstance(target, dict) for target in targets),
        "shuffle_policy_matches": sampler_matches,
        "num_workers_is_zero": loader.num_workers == 0,
        "pin_memory_is_false": loader.pin_memory is False,
        "drop_last_is_false": loader.drop_last is False,
    }
    return {
        "first_batch_size": actual_batch_size,
        "image_box_counts": [len(target["boxes"]) for target in targets],
        "sampler_type": type(loader.sampler).__name__,
        "checks": checks,
        "validation_passed": all(checks.values()),
    }


def _path_overlap_report(loaders: DetectionDataLoaders) -> dict[str, Any]:
    split_paths = {
        "train": {str(sample.image_path) for sample in loaders.train_dataset.samples},
        "validation": {
            str(sample.image_path) for sample in loaders.validation_dataset.samples
        },
        "test": {str(sample.image_path) for sample in loaders.test_dataset.samples},
    }
    pair_counts = {
        "train_validation": len(split_paths["train"] & split_paths["validation"]),
        "train_test": len(split_paths["train"] & split_paths["test"]),
        "validation_test": len(split_paths["validation"] & split_paths["test"]),
    }
    return {
        "image_path_overlap_counts": pair_counts,
        "no_split_image_path_overlap": all(count == 0 for count in pair_counts.values()),
    }


def build_day11_detection_dataset_validation(
    *,
    project_root: Path,
    manifest_path: Path,
    loader_config: DetectionDataLoaderConfig | None = None,
    duplicate_box_policy: str = "preserve",
) -> tuple[dict[str, Any], DetectionDataLoaders]:
    """세 Split, DataLoader, Duplicate 정책을 검증할 Artifact Payload를 만든다."""
    resolved_config = loader_config or DetectionDataLoaderConfig()
    manifest_payload = _read_json_object(manifest_path)
    loaders = create_neu_det_detection_data_loaders(
        project_root=project_root,
        manifest_path=manifest_path,
        config=resolved_config,
        duplicate_box_policy=duplicate_box_policy,
        training_horizontal_flip_probability=0.0,
    )

    datasets = {
        "train": loaders.train_dataset,
        "validation": loaders.validation_dataset,
        "test": loaders.test_dataset,
    }
    data_loaders = {
        "train": loaders.train_loader,
        "validation": loaders.validation_loader,
        "test": loaders.test_loader,
    }

    split_results = {
        split: validate_detection_split(
            dataset=datasets[split],
            manifest_records=_manifest_records(manifest_payload, split),
            split=split,
        )
        for split in SUPPORTED_SPLITS
    }
    loader_results = {
        split: _validate_loader(
            loader=data_loaders[split],
            dataset_length=len(datasets[split]),
            expected_batch_size=resolved_config.batch_size,
            expected_shuffle=split == "train",
        )
        for split in SUPPORTED_SPLITS
    }
    overlap = _path_overlap_report(loaders)
    duplicate_records = find_manifest_exact_duplicate_records(manifest_payload)

    total_samples = sum(result["sample_count"] for result in split_results.values())
    total_boxes = sum(result["dataset_box_count"] for result in split_results.values())
    total_raw_duplicates = sum(
        result["duplicate_box_audit"]["raw_exact_duplicate_count"]
        for result in split_results.values()
    )
    checks = {
        "all_split_validations_passed": all(
            result["validation_passed"] for result in split_results.values()
        ),
        "all_loader_validations_passed": all(
            result["validation_passed"] for result in loader_results.values()
        ),
        "no_split_image_path_overlap": overlap["no_split_image_path_overlap"],
        "duplicate_audit_matches_split_counts": total_raw_duplicates
        == sum(
            duplicate["duplicate_count"]
            for record in duplicate_records
            for duplicate in record["duplicates"]
        ),
    }

    try:
        relative_manifest_path = str(manifest_path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        relative_manifest_path = str(manifest_path.resolve())

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "day": 11,
        "title": "Detection Dataset Runtime Validation",
        "project_name": "Manufacturing Vision Defect Analysis System",
        "project_name_ko": "제조 비전 결함 분석 시스템",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "NEU Surface Defect Database (NEU-DET)",
        "manifest_path": relative_manifest_path.replace("\\", "/"),
        "coordinate_conversion_policy": "(xmin - 1, ymin - 1, xmax, ymax)",
        "duplicate_box_policy": duplicate_box_policy,
        "class_mapping": dict(DETECTION_MODEL_CLASS_TO_INDEX),
        "num_classes_including_background": len(DETECTION_MODEL_CLASS_TO_INDEX),
        "data_loader_config": {
            "batch_size": resolved_config.batch_size,
            "num_workers": resolved_config.num_workers,
            "pin_memory": resolved_config.pin_memory,
            "drop_last": resolved_config.drop_last,
            "persistent_workers": resolved_config.persistent_workers,
            "random_seed": resolved_config.random_seed,
            "train_shuffle": True,
            "validation_shuffle": False,
            "test_shuffle": False,
        },
        "totals": {
            "sample_count": total_samples,
            "box_count": total_boxes,
            "raw_exact_duplicate_count": total_raw_duplicates,
        },
        "splits": split_results,
        "data_loader_validation": loader_results,
        "split_overlap_validation": overlap,
        "duplicate_box_records": duplicate_records,
        "checks": checks,
        "validation_passed": all(checks.values()),
    }
    return payload, loaders


def write_validation_artifact(payload: Mapping[str, Any], output_path: Path) -> None:
    """검증 Payload를 UTF-8 JSON Artifact로 저장한다."""
    if not isinstance(output_path, Path):
        raise TypeError("output_path must be pathlib.Path.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
