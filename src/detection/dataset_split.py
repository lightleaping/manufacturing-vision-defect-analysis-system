"""재현 가능한 Detection Dataset Split Manifest 생성."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

from .dataset_analysis import ImageAnnotationRecord
from .dataset_config import SplitRatios


@dataclass(frozen=True)
class SplitManifest:
    """학습·검증·테스트 파일 목록과 통계를 저장한다."""

    dataset_name: str
    random_seed: int
    ratios: dict[str, float]
    splits: dict[str, tuple[dict[str, object], ...]]
    statistics: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "random_seed": self.random_seed,
            "ratios": self.ratios,
            "splits": {
                split_name: list(items)
                for split_name, items in self.splits.items()
            },
            "statistics": self.statistics,
        }


def _record_to_dict(
    record: ImageAnnotationRecord | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(record, ImageAnnotationRecord):
        return record.to_dict()
    return dict(record)


def _allocate_counts(n: int, ratios: SplitRatios) -> dict[str, int]:
    """Largest remainder 방식으로 한 Stratum의 Split 개수를 계산한다."""
    ratio_items = [
        ("train", ratios.train),
        ("validation", ratios.validation),
        ("test", ratios.test),
    ]
    raw = {name: n * ratio for name, ratio in ratio_items}
    counts = {name: int(value) for name, value in raw.items()}
    remainder = n - sum(counts.values())

    order = sorted(
        ratio_items,
        key=lambda item: (raw[item[0]] - counts[item[0]], item[1]),
        reverse=True,
    )
    for index in range(remainder):
        counts[order[index % len(order)][0]] += 1
    return counts


def _stratum_key(record: Mapping[str, object]) -> str:
    class_names = sorted(set(str(name) for name in record["class_names"]))
    return "|".join(class_names) if class_names else "__empty__"


def _unit_stratum_key(unit: Sequence[Mapping[str, object]]) -> str:
    class_names = sorted(
        {
            str(name)
            for item in unit
            for name in item.get("class_names", [])
        }
    )
    return "|".join(class_names) if class_names else "__empty__"


def _split_statistics(
    splits: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    stats: dict[str, object] = {}
    for split_name, items in splits.items():
        class_image_counts: Counter[str] = Counter()
        class_box_counts: Counter[str] = Counter()
        source_split_counts: Counter[str] = Counter()
        hash_counts: Counter[str] = Counter()

        for item in items:
            names = [str(name) for name in item["class_names"]]
            class_image_counts.update(set(names))
            class_box_counts.update(names)
            source_split_counts[str(item.get("source_split", "unspecified"))] += 1
            hash_counts[str(item["image_sha256"])] += 1

        duplicate_hash_groups = {
            digest: count
            for digest, count in hash_counts.items()
            if count > 1
        }
        stats[split_name] = {
            "image_count": len(items),
            "box_count": sum(class_box_counts.values()),
            "class_image_counts": dict(sorted(class_image_counts.items())),
            "class_box_counts": dict(sorted(class_box_counts.items())),
            "source_split_counts": dict(sorted(source_split_counts.items())),
            "duplicate_image_hash_group_count": len(duplicate_hash_groups),
            "duplicate_image_record_count": sum(duplicate_hash_groups.values()),
        }
    return stats


def _validate_unique_paths(items: Sequence[Mapping[str, object]]) -> None:
    image_paths = [str(item["image_path"]) for item in items]
    if len(set(image_paths)) != len(image_paths):
        raise ValueError("동일 image_path가 두 번 이상 입력되었습니다.")


def _duplicate_hashes(
    items: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        grouped[str(item["image_sha256"])].append(dict(item))
    return {
        digest: records
        for digest, records in grouped.items()
        if len(records) > 1
    }


def _hash_units(
    items: Sequence[Mapping[str, object]],
) -> list[list[dict[str, object]]]:
    """동일 이미지 Hash 레코드를 하나의 분할 단위로 묶는다."""
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in items:
        grouped[str(item["image_sha256"])].append(dict(item))
    return [
        sorted(group, key=lambda item: str(item["image_path"]))
        for _, group in sorted(grouped.items())
    ]


def _assign_units_to_targets(
    units: Sequence[Sequence[Mapping[str, object]]],
    *,
    target_counts: Mapping[str, int],
    rng: random.Random,
) -> dict[str, list[dict[str, object]]]:
    """동일 Hash 단위를 쪼개지 않고 목표 이미지 수에 가깝게 배정한다."""
    split_names = tuple(target_counts)
    assigned: dict[str, list[dict[str, object]]] = {
        name: [] for name in split_names
    }
    current = {name: 0 for name in split_names}
    shuffled_units = [list(unit) for unit in units]
    rng.shuffle(shuffled_units)

    # 큰 중복 그룹을 먼저 배치하면 목표 비율 오차가 줄어든다.
    shuffled_units.sort(key=len, reverse=True)
    for unit in shuffled_units:
        size = len(unit)

        def score(name: str) -> tuple[int, int, str]:
            remaining = target_counts[name] - current[name]
            fits = 1 if remaining >= size else 0
            return (fits, remaining, name)

        destination = max(split_names, key=score)
        assigned[destination].extend(dict(item) for item in unit)
        current[destination] += size

    return assigned


def _duplicate_policy_statistics(
    items: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups = _duplicate_hashes(items)
    return {
        "duplicate_hash_policy": (
            "preserve_duplicate_records_and_keep_each_identical_hash_group_"
            "inside_one_final_split"
        ),
        "input_duplicate_image_hash_group_count": len(groups),
        "input_duplicate_image_record_count": sum(
            len(records) for records in groups.values()
        ),
    }


def validate_split_manifest(
    manifest: SplitManifest,
    *,
    expected_record_count: int | None = None,
) -> dict[str, object]:
    """경로 중복·누락·Split 간 동일 Hash 누수를 검사한다."""
    split_names = ("train", "validation", "test")
    path_sets = {
        split_name: {
            str(item["image_path"])
            for item in manifest.splits.get(split_name, ())
        }
        for split_name in split_names
    }

    overlaps: dict[str, list[str]] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            intersection = sorted(path_sets[left] & path_sets[right])
            if intersection:
                overlaps[f"{left}__{right}"] = intersection

    all_items = [
        item
        for split_name in split_names
        for item in manifest.splits.get(split_name, ())
    ]
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    hash_to_paths: dict[str, list[str]] = defaultdict(list)
    for split_name in split_names:
        for item in manifest.splits.get(split_name, ()):
            digest = str(item["image_sha256"])
            hash_to_splits[digest].add(split_name)
            hash_to_paths[digest].append(str(item["image_path"]))

    cross_split_duplicate_hashes = {
        digest: {
            "splits": sorted(names),
            "image_paths": sorted(hash_to_paths[digest]),
        }
        for digest, names in hash_to_splits.items()
        if len(names) > 1
    }

    result = {
        "split_overlap_count": sum(len(values) for values in overlaps.values()),
        "split_overlaps": overlaps,
        "cross_split_duplicate_hash_count": len(
            cross_split_duplicate_hashes
        ),
        "cross_split_duplicate_hashes": cross_split_duplicate_hashes,
        "total_manifest_records": len(all_items),
        "all_records_preserved": (
            expected_record_count is None
            or len(all_items) == expected_record_count
        ),
    }
    result["is_valid"] = (
        result["split_overlap_count"] == 0
        and result["cross_split_duplicate_hash_count"] == 0
        and result["all_records_preserved"]
    )
    return result


def build_split_manifest(
    records: Iterable[ImageAnnotationRecord | Mapping[str, object]],
    *,
    ratios: SplitRatios = SplitRatios(),
    random_seed: int = 42,
    dataset_name: str = "NEU-DET",
    reject_duplicate_hashes: bool = False,
) -> SplitManifest:
    """단일 Pool을 Class 조합별로 나누되 동일 Hash는 한 Split에 둔다."""
    items = [_record_to_dict(record) for record in records]
    _validate_unique_paths(items)
    duplicate_groups = _duplicate_hashes(items)
    if reject_duplicate_hashes and duplicate_groups:
        raise ValueError(
            "동일 이미지 Hash가 있어 명시적 거부 정책으로 중단합니다: "
            f"{list(duplicate_groups)[:5]}"
        )

    units_by_stratum: dict[str, list[list[dict[str, object]]]] = defaultdict(list)
    for unit in _hash_units(items):
        units_by_stratum[_unit_stratum_key(unit)].append(unit)

    rng = random.Random(random_seed)
    split_lists: dict[str, list[dict[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for stratum in sorted(units_by_stratum):
        units = units_by_stratum[stratum]
        record_count = sum(len(unit) for unit in units)
        targets = _allocate_counts(record_count, ratios)
        assigned = _assign_units_to_targets(
            units,
            target_counts=targets,
            rng=rng,
        )
        for split_name in split_lists:
            split_lists[split_name].extend(assigned[split_name])

    for split_name in split_lists:
        rng.shuffle(split_lists[split_name])

    immutable_splits = {
        split_name: tuple(items_for_split)
        for split_name, items_for_split in split_lists.items()
    }
    manifest = SplitManifest(
        dataset_name=dataset_name,
        random_seed=random_seed,
        ratios=ratios.as_dict(),
        splits=immutable_splits,
        statistics={
            **_split_statistics(immutable_splits),
            **_duplicate_policy_statistics(items),
            "split_policy": (
                "random_stratified_hash_group_preserving_from_single_pool"
            ),
        },
    )
    validation = validate_split_manifest(
        manifest,
        expected_record_count=len(items),
    )
    if not validation["is_valid"]:
        raise RuntimeError(f"Split Manifest 검증 실패: {validation}")
    return manifest


def build_source_preserving_split_manifest(
    train_records: Iterable[ImageAnnotationRecord | Mapping[str, object]],
    validation_pool_records: Iterable[
        ImageAnnotationRecord | Mapping[str, object]
    ],
    *,
    validation_fraction: float = 0.5,
    random_seed: int = 42,
    dataset_name: str = "NEU-DET",
) -> SplitManifest:
    """원본 Train을 유지하고 Validation Pool만 Validation·Test로 나눈다.

    동일 Hash 레코드는 한 Split에 묶는다. 동일 Hash가 원본 Train과
    Validation Pool 양쪽에 있으면 원본 Split을 유지한 채 누수를 막을 수
    없으므로 명시적으로 중단한다.
    """
    if validation_fraction <= 0.0 or validation_fraction >= 1.0:
        raise ValueError("validation_fraction은 0보다 크고 1보다 작아야 합니다.")

    train_items = [_record_to_dict(record) for record in train_records]
    validation_pool = [
        _record_to_dict(record) for record in validation_pool_records
    ]
    all_items = train_items + validation_pool
    _validate_unique_paths(all_items)
    if not train_items:
        raise ValueError("원본 Train Record가 없습니다.")
    if not validation_pool:
        raise ValueError("원본 Validation Pool Record가 없습니다.")

    train_hashes = {str(item["image_sha256"]) for item in train_items}
    validation_hashes = {
        str(item["image_sha256"]) for item in validation_pool
    }
    cross_source_hashes = sorted(train_hashes & validation_hashes)
    if cross_source_hashes:
        raise ValueError(
            "동일 이미지 Hash가 원본 Train과 Validation Pool에 동시에 있어 "
            "누수 없이 원본 Split을 보존할 수 없습니다: "
            f"{cross_source_hashes[:5]}"
        )

    units_by_stratum: dict[str, list[list[dict[str, object]]]] = defaultdict(list)
    for unit in _hash_units(validation_pool):
        units_by_stratum[_unit_stratum_key(unit)].append(unit)

    rng = random.Random(random_seed)
    validation_items: list[dict[str, object]] = []
    test_items: list[dict[str, object]] = []
    for stratum in sorted(units_by_stratum):
        units = units_by_stratum[stratum]
        record_count = sum(len(unit) for unit in units)
        validation_count = int(record_count * validation_fraction)
        if record_count >= 2:
            validation_count = max(
                1,
                min(record_count - 1, validation_count),
            )
        assigned = _assign_units_to_targets(
            units,
            target_counts={
                "validation": validation_count,
                "test": record_count - validation_count,
            },
            rng=rng,
        )
        validation_items.extend(assigned["validation"])
        test_items.extend(assigned["test"])

    train_items = sorted(train_items, key=lambda item: str(item["image_path"]))
    rng.shuffle(validation_items)
    rng.shuffle(test_items)

    immutable_splits = {
        "train": tuple(train_items),
        "validation": tuple(validation_items),
        "test": tuple(test_items),
    }
    total = len(all_items)
    actual_ratios = {
        name: len(items) / total
        for name, items in immutable_splits.items()
    }
    statistics = _split_statistics(immutable_splits)
    statistics.update(
        {
            **_duplicate_policy_statistics(all_items),
            "split_policy": (
                "preserve_source_train_and_hash_group_split_source_"
                "validation_pool"
            ),
            "source_train_count": len(train_items),
            "source_validation_pool_count": len(validation_pool),
            "validation_fraction_within_source_validation": validation_fraction,
            "source_split_note": (
                "The downloaded Kaggle mirror supplied train and validation "
                "directories. Train is preserved; validation is divided into "
                "final validation and test with seed 42. Identical image "
                "hashes are kept in one final split."
            ),
        }
    )
    manifest = SplitManifest(
        dataset_name=dataset_name,
        random_seed=random_seed,
        ratios=actual_ratios,
        splits=immutable_splits,
        statistics=statistics,
    )
    validation = validate_split_manifest(
        manifest,
        expected_record_count=total,
    )
    if not validation["is_valid"]:
        raise RuntimeError(f"Split Manifest 검증 실패: {validation}")
    return manifest


def build_existing_split_manifest(
    split_records: Mapping[
        str,
        Iterable[ImageAnnotationRecord | Mapping[str, object]],
    ],
    *,
    random_seed: int = 42,
    dataset_name: str = "NEU-DET",
) -> SplitManifest:
    """원본에 Train·Validation·Test가 모두 있을 때 그대로 보존한다."""
    required = {"train", "validation", "test"}
    if set(split_records) != required:
        raise ValueError(f"필수 Split이 필요합니다: {sorted(required)}")

    converted = {
        name: tuple(_record_to_dict(record) for record in records)
        for name, records in split_records.items()
    }
    all_items = [item for items in converted.values() for item in items]
    _validate_unique_paths(all_items)
    total = len(all_items)
    manifest = SplitManifest(
        dataset_name=dataset_name,
        random_seed=random_seed,
        ratios={name: len(items) / total for name, items in converted.items()},
        splits=converted,
        statistics={
            **_split_statistics(converted),
            **_duplicate_policy_statistics(all_items),
            "split_policy": "preserve_existing_train_validation_test",
        },
    )
    validation = validate_split_manifest(manifest, expected_record_count=total)
    if not validation["is_valid"]:
        raise RuntimeError(f"Split Manifest 검증 실패: {validation}")
    return manifest


def save_split_manifest(
    manifest: SplitManifest,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_split_manifest(path: Path) -> SplitManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitManifest(
        dataset_name=str(payload["dataset_name"]),
        random_seed=int(payload["random_seed"]),
        ratios={
            str(key): float(value)
            for key, value in payload["ratios"].items()
        },
        splits={
            str(split_name): tuple(dict(item) for item in items)
            for split_name, items in payload["splits"].items()
        },
        statistics=dict(payload["statistics"]),
    )
