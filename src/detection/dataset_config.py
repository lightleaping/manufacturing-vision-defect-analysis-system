"""NEU-DET 데이터셋 경로, Class Mapping, 원본 Split 정책 설정."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Mapping


NEU_DET_CANONICAL_CLASSES: tuple[str, ...] = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
)

# 원본 Annotation에는 background 객체가 없다.
# 아래 Source ID는 분석·Manifest에서 사용할 명시적 내부 ID다.
DETECTION_SOURCE_CLASS_TO_INDEX: dict[str, int] = {
    class_name: index
    for index, class_name in enumerate(NEU_DET_CANONICAL_CLASSES)
}

# Torchvision Detection 모델에서는 0을 background로 예약한다.
DETECTION_MODEL_CLASS_TO_INDEX: dict[str, int] = {
    "background": 0,
    **{
        class_name: index
        for index, class_name in enumerate(NEU_DET_CANONICAL_CLASSES, start=1)
    },
}

_CLASS_ALIASES: dict[str, str] = {
    "crazing": "crazing",
    "cr": "crazing",
    "inclusion": "inclusion",
    "inclusions": "inclusion",
    "in": "inclusion",
    "patches": "patches",
    "patch": "patches",
    "pa": "patches",
    "pitted_surface": "pitted_surface",
    "pitted_surfaces": "pitted_surface",
    "pitted": "pitted_surface",
    "ps": "pitted_surface",
    "rolled_in_scale": "rolled_in_scale",
    "rolled_in_scales": "rolled_in_scale",
    "rolled_scale": "rolled_in_scale",
    "rs": "rolled_in_scale",
    "scratches": "scratches",
    "scratch": "scratches",
    "sc": "scratches",
}

SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
)
SUPPORTED_ANNOTATION_EXTENSIONS: tuple[str, ...] = (".xml",)


def _normalization_key(value: str) -> str:
    """대소문자·공백·하이픈 차이를 제거해 이름을 비교한다."""
    normalized = value.strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def normalize_annotation_class_name(raw_name: str) -> str:
    """원본 Annotation Class 이름을 고정 Canonical 이름으로 변환한다.

    알 수 없는 이름을 조용히 새 Class로 추가하지 않는다.
    """
    key = _normalization_key(raw_name)
    try:
        return _CLASS_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"알 수 없는 Detection Class: {raw_name!r}") from exc


def normalize_partition_name(raw_name: str) -> str:
    """압축 파일의 Split 폴더명을 내부 표준 이름으로 변환한다."""
    key = _normalization_key(raw_name)
    aliases = {
        "train": "train",
        "training": "train",
        "validation": "validation",
        "valid": "validation",
        "val": "validation",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(key, key or "all")


@dataclass(frozen=True)
class SplitRatios:
    """Train·Validation·Test 비율."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0.0 or value >= 1.0 for value in values):
            raise ValueError("각 Split 비율은 0보다 크고 1보다 작아야 합니다.")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("Split 비율의 합은 1.0이어야 합니다.")

    def as_dict(self) -> dict[str, float]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class DetectionDatasetLayout:
    """단일 이미지·Annotation 디렉터리 쌍."""

    dataset_root: Path
    images_dir: Path
    annotations_dir: Path


@dataclass(frozen=True)
class DetectionDatasetPartitionLayout:
    """원본 압축에 포함된 한 Split의 이미지·Annotation 경로."""

    name: str
    partition_root: Path
    images_dir: Path
    annotations_dir: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "partition_root": str(self.partition_root),
            "images_dir": str(self.images_dir),
            "annotations_dir": str(self.annotations_dir),
        }


@dataclass(frozen=True)
class DetectionDatasetCollectionLayout:
    """하나 이상의 원본 Split을 포함한 NEU-DET 전체 구조."""

    dataset_root: Path
    partitions: tuple[DetectionDatasetPartitionLayout, ...]

    def get(self, name: str) -> DetectionDatasetPartitionLayout:
        normalized = normalize_partition_name(name)
        for partition in self.partitions:
            if partition.name == normalized:
                return partition
        raise KeyError(f"Dataset Partition을 찾을 수 없습니다: {name!r}")

    @property
    def partition_names(self) -> tuple[str, ...]:
        return tuple(partition.name for partition in self.partitions)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_root": str(self.dataset_root),
            "partitions": [partition.to_dict() for partition in self.partitions],
        }


@dataclass(frozen=True)
class DetectionDatasetConfig:
    """Day 9 분석에서 사용하는 한 Partition의 설정."""

    project_root: Path
    dataset_root: Path
    images_dir: Path
    annotations_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    figures_dir: Path
    source_split: str = "unspecified"
    source_class_to_index: Mapping[str, int] = field(
        default_factory=lambda: dict(DETECTION_SOURCE_CLASS_TO_INDEX)
    )
    model_class_to_index: Mapping[str, int] = field(
        default_factory=lambda: dict(DETECTION_MODEL_CLASS_TO_INDEX)
    )
    image_extensions: tuple[str, ...] = SUPPORTED_IMAGE_EXTENSIONS
    annotation_extensions: tuple[str, ...] = SUPPORTED_ANNOTATION_EXTENSIONS
    split_ratios: SplitRatios = field(default_factory=SplitRatios)
    random_seed: int = 42
    background_is_source_label: bool = False
    coordinate_policy: str = "raw_pascal_voc_coordinates_preserved"

    def __post_init__(self) -> None:
        if self.background_is_source_label:
            raise ValueError(
                "NEU-DET 원본 XML의 background는 객체 Class가 아닙니다."
            )
        if self.model_class_to_index.get("background") != 0:
            raise ValueError("Detection 모델의 background ID는 0이어야 합니다.")
        if "background" in self.source_class_to_index:
            raise ValueError("원본 Annotation Class Mapping에 background를 넣지 않습니다.")
        expected = set(NEU_DET_CANONICAL_CLASSES)
        if set(self.source_class_to_index) != expected:
            raise ValueError("원본 Class Mapping이 고정 Canonical Class와 다릅니다.")
        model_classes = set(self.model_class_to_index) - {"background"}
        if model_classes != expected:
            raise ValueError("모델 Class Mapping이 고정 Canonical Class와 다릅니다.")
        if self.random_seed < 0:
            raise ValueError("Random Seed는 0 이상이어야 합니다.")
        if not self.source_split.strip():
            raise ValueError("source_split은 빈 문자열일 수 없습니다.")

    def to_dict(self) -> dict[str, object]:
        return {
            "project_root": str(self.project_root),
            "dataset_root": str(self.dataset_root),
            "images_dir": str(self.images_dir),
            "annotations_dir": str(self.annotations_dir),
            "processed_dir": str(self.processed_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "figures_dir": str(self.figures_dir),
            "source_split": self.source_split,
            "source_class_to_index": dict(self.source_class_to_index),
            "model_class_to_index": dict(self.model_class_to_index),
            "image_extensions": list(self.image_extensions),
            "annotation_extensions": list(self.annotation_extensions),
            "split_ratios": self.split_ratios.as_dict(),
            "random_seed": self.random_seed,
            "background_is_source_label": self.background_is_source_label,
            "coordinate_policy": self.coordinate_policy,
        }


_IMAGE_DIR_NAMES = ("images", "image", "jpegimages")
_ANNOTATION_DIR_NAMES = ("annotations", "annotation", "labels")
_IMAGE_DIR_KEYS = {_normalization_key(name) for name in _IMAGE_DIR_NAMES}
_ANNOTATION_DIR_KEYS = {
    _normalization_key(name) for name in _ANNOTATION_DIR_NAMES
}


def _contains_supported_file(
    directory: Path,
    extensions: tuple[str, ...],
) -> bool:
    return any(
        path.is_file() and path.suffix.lower() in extensions
        for path in directory.rglob("*")
    )


def discover_neu_det_partitions(
    dataset_root: Path,
) -> DetectionDatasetCollectionLayout:
    """`train/images + train/annotations` 같은 원본 Split 구조를 찾는다.

    Kaggle 미러처럼 여러 쌍이 존재해도 각 쌍을 독립 Partition으로 보존한다.
    형제가 아닌 디렉터리를 임의로 조합하지 않는다.
    """
    dataset_root = dataset_root.resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset Root가 없습니다: {dataset_root}")

    candidate_parents: set[Path] = set()
    directories = [dataset_root]
    directories.extend(
        path
        for path in dataset_root.rglob("*")
        if path.is_dir() and len(path.relative_to(dataset_root).parts) <= 5
    )

    for directory in directories:
        child_keys = {
            _normalization_key(child.name): child
            for child in directory.iterdir()
            if child.is_dir()
        }
        image_dir = next(
            (child_keys[key] for key in _IMAGE_DIR_KEYS if key in child_keys),
            None,
        )
        annotation_dir = next(
            (
                child_keys[key]
                for key in _ANNOTATION_DIR_KEYS
                if key in child_keys
            ),
            None,
        )
        if image_dir is None or annotation_dir is None:
            continue
        if not _contains_supported_file(image_dir, SUPPORTED_IMAGE_EXTENSIONS):
            continue
        if not _contains_supported_file(
            annotation_dir,
            SUPPORTED_ANNOTATION_EXTENSIONS,
        ):
            continue
        candidate_parents.add(directory.resolve())

    if not candidate_parents:
        raise RuntimeError(
            "이미지와 Annotation이 형제 디렉터리인 NEU-DET 구조를 "
            f"찾지 못했습니다: {dataset_root}"
        )

    partitions: list[DetectionDatasetPartitionLayout] = []
    seen_names: set[str] = set()
    for parent in sorted(candidate_parents):
        children = {
            _normalization_key(child.name): child.resolve()
            for child in parent.iterdir()
            if child.is_dir()
        }
        images_dir = next(children[key] for key in _IMAGE_DIR_KEYS if key in children)
        annotations_dir = next(
            children[key] for key in _ANNOTATION_DIR_KEYS if key in children
        )

        raw_name = parent.name
        # 단일 IMAGES/ANNOTATIONS 구조에서 상위 폴더명이 NEU-DET라면
        # Split 이름을 all로 기록한다.
        normalized_parent = _normalization_key(raw_name)
        if normalized_parent in {"neu_det", "neudet", "dataset"}:
            name = "all"
        else:
            name = normalize_partition_name(raw_name)

        if name in seen_names:
            raise RuntimeError(
                "동일한 표준 Split 이름이 둘 이상 발견되었습니다: "
                f"{name!r}"
            )
        seen_names.add(name)
        partitions.append(
            DetectionDatasetPartitionLayout(
                name=name,
                partition_root=parent,
                images_dir=images_dir,
                annotations_dir=annotations_dir,
            )
        )

    order = {"train": 0, "validation": 1, "test": 2, "all": 3}
    partitions.sort(key=lambda item: (order.get(item.name, 99), item.name))
    return DetectionDatasetCollectionLayout(
        dataset_root=dataset_root,
        partitions=tuple(partitions),
    )


def discover_neu_det_layout(dataset_root: Path) -> DetectionDatasetLayout:
    """단일 이미지·Annotation 쌍만 허용하는 기존 호환 함수."""
    collection = discover_neu_det_partitions(dataset_root)
    if len(collection.partitions) != 1:
        raise RuntimeError(
            "NEU-DET 구조가 여러 Partition으로 구성되어 있습니다. "
            "discover_neu_det_partitions()를 사용해야 합니다. "
            f"partitions={collection.partition_names}"
        )
    partition = collection.partitions[0]
    return DetectionDatasetLayout(
        dataset_root=collection.dataset_root,
        images_dir=partition.images_dir,
        annotations_dir=partition.annotations_dir,
    )


def build_partition_config(
    *,
    project_root: Path,
    dataset_root: Path,
    partition: DetectionDatasetPartitionLayout,
) -> DetectionDatasetConfig:
    """발견된 원본 Partition을 분석 Config로 변환한다."""
    resolved_project_root = project_root.resolve()
    resolved_dataset_root = dataset_root.resolve()
    return DetectionDatasetConfig(
        project_root=resolved_project_root,
        dataset_root=resolved_dataset_root,
        images_dir=partition.images_dir,
        annotations_dir=partition.annotations_dir,
        processed_dir=resolved_project_root / "data" / "processed" / "neu_det",
        artifacts_dir=resolved_project_root / "reports" / "artifacts",
        figures_dir=resolved_project_root / "reports" / "figures",
        source_split=partition.name,
    )


def build_default_config(
    *,
    project_root: Path | None = None,
    dataset_root: Path | None = None,
    discover_layout: bool = True,
) -> DetectionDatasetConfig:
    """단일 구조 또는 Fixture를 위한 기본 Config를 생성한다.

    실제 Kaggle NEU-DET처럼 여러 Partition이 있으면
    :func:`discover_neu_det_partitions`와 :func:`build_partition_config`를 쓴다.
    """
    resolved_project_root = (
        project_root.resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved_dataset_root = (
        dataset_root.resolve()
        if dataset_root is not None
        else resolved_project_root / "data" / "raw" / "neu_det"
    )

    if discover_layout and resolved_dataset_root.exists():
        layout = discover_neu_det_layout(resolved_dataset_root)
        images_dir = layout.images_dir
        annotations_dir = layout.annotations_dir
    else:
        # Dataset을 아직 받지 않은 단계에서도 Config와 단위 테스트를 만들 수 있다.
        images_dir = resolved_dataset_root / "IMAGES"
        annotations_dir = resolved_dataset_root / "ANNOTATIONS"

    return DetectionDatasetConfig(
        project_root=resolved_project_root,
        dataset_root=resolved_dataset_root,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        processed_dir=resolved_project_root / "data" / "processed" / "neu_det",
        artifacts_dir=resolved_project_root / "reports" / "artifacts",
        figures_dir=resolved_project_root / "reports" / "figures",
    )
