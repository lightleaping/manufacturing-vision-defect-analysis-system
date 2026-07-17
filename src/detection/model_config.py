"""Day 11 Object Detection 공통 Class·Model 설정.

[그대로 재사용]
Day 9 ``dataset_config.py``에 이미 확정된 NEU-DET Class 순서와
Torchvision 모델 Label Mapping을 이 모듈의 단일 기준으로 사용한다.

이 파일은 모델 Weight를 다운로드하거나 Detection Model을 생성하지 않는다.
Dataset, DataLoader, Model Factory가 공유할 검증된 설정 계약만 제공한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, Sequence, cast

from src.detection.dataset_config import (
    DETECTION_MODEL_CLASS_TO_INDEX,
    NEU_DET_CANONICAL_CLASSES,
)


SUPPORTED_ARCHITECTURE: Final[str] = (
    "fasterrcnn_mobilenet_v3_large_320_fpn"
)
SUPPORTED_SPLITS: Final[tuple[str, ...]] = (
    "train",
    "validation",
    "test",
)
DuplicateBoxPolicy = Literal["preserve", "remove_exact"]


def _find_background_class_name() -> str:
    """Day 9 모델 Mapping에서 Label 0의 Class 이름을 찾는다."""
    matches = [
        class_name
        for class_name, class_index in DETECTION_MODEL_CLASS_TO_INDEX.items()
        if class_index == 0
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "DETECTION_MODEL_CLASS_TO_INDEX must contain exactly one "
            "background class with index 0."
        )
    return matches[0]


BACKGROUND_CLASS_NAME: Final[str] = _find_background_class_name()
DEFAULT_DEFECT_CLASS_NAMES: Final[tuple[str, ...]] = tuple(
    NEU_DET_CANONICAL_CLASSES
)


def validate_defect_class_names(
    class_names: Sequence[str],
) -> tuple[str, ...]:
    """결함 Class 순서를 검증하고 변경 불가능한 Tuple로 반환한다."""
    if isinstance(class_names, (str, bytes)):
        raise TypeError("class_names must be a sequence of strings.")

    normalized = tuple(class_names)
    if not normalized:
        raise ValueError("class_names must not be empty.")
    if any(not isinstance(name, str) for name in normalized):
        raise TypeError("Every class name must be str.")
    if any(not name.strip() for name in normalized):
        raise ValueError("Class names must not be blank.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("class_names must not contain duplicates.")
    if BACKGROUND_CLASS_NAME in normalized:
        raise ValueError(
            "The background class must not be included in defect class names."
        )

    return normalized


def build_detection_label_maps(
    class_names: Sequence[str] = DEFAULT_DEFECT_CLASS_NAMES,
) -> tuple[dict[str, int], dict[int, str]]:
    """Background=0, 실제 결함=1..N인 Detection Mapping을 만든다.

    기본 NEU-DET Class를 사용할 때는 Day 9에서 이미 확정한
    ``DETECTION_MODEL_CLASS_TO_INDEX``를 그대로 복사한다. 테스트 Fixture처럼
    사용자 정의 Class 순서를 받을 때만 같은 규칙으로 새 Mapping을 만든다.
    """
    normalized = validate_defect_class_names(class_names)

    if normalized == DEFAULT_DEFECT_CLASS_NAMES:
        class_to_index = dict(DETECTION_MODEL_CLASS_TO_INDEX)
    else:
        class_to_index = {BACKGROUND_CLASS_NAME: 0}
        class_to_index.update(
            {
                class_name: class_index
                for class_index, class_name in enumerate(normalized, start=1)
            }
        )

    expected_indexes = set(range(len(normalized) + 1))
    if set(class_to_index.values()) != expected_indexes:
        raise RuntimeError(
            "Detection label indexes must be contiguous from 0 to "
            "num_defect_classes."
        )
    if class_to_index.get(BACKGROUND_CLASS_NAME) != 0:
        raise RuntimeError("The background label index must be 0.")

    defect_to_index = {
        class_name: class_to_index[class_name]
        for class_name in normalized
    }
    index_to_class = {
        class_index: class_name
        for class_name, class_index in class_to_index.items()
    }
    return defect_to_index, index_to_class


def validate_split_name(split: str) -> str:
    """Day 9 Manifest가 제공하는 세 Split만 허용한다."""
    if not isinstance(split, str):
        raise TypeError("split must be str.")
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"Unsupported split: {split!r}. Supported: {SUPPORTED_SPLITS}."
        )
    return split


def validate_duplicate_box_policy(policy: str) -> DuplicateBoxPolicy:
    """원본 보존 또는 완전히 같은 Class·좌표만 제거하도록 제한한다."""
    if not isinstance(policy, str):
        raise TypeError("duplicate_box_policy must be str.")
    if policy not in {"preserve", "remove_exact"}:
        raise ValueError(
            "duplicate_box_policy must be 'preserve' or 'remove_exact'."
        )
    return cast(DuplicateBoxPolicy, policy)


@dataclass(frozen=True, slots=True)
class DetectionModelConfig:
    """Day 11·12 Faster R-CNN 공통 설정.

    현재 CPU 환경의 기본 Architecture는 MobileNetV3 Large 320 FPN이다.
    Stage 1에서는 Weight를 사용하지 않으며 실제 생성은 다음 Model Factory가
    담당한다.
    """

    architecture: str = SUPPORTED_ARCHITECTURE
    defect_class_names: tuple[str, ...] = DEFAULT_DEFECT_CLASS_NAMES
    min_size: int = 320
    max_size: int = 320
    use_pretrained_weights: bool = False
    use_pretrained_backbone: bool = False
    progress: bool = False

    def __post_init__(self) -> None:
        if self.architecture != SUPPORTED_ARCHITECTURE:
            raise ValueError(
                f"Unsupported architecture: {self.architecture!r}. "
                f"Supported: {SUPPORTED_ARCHITECTURE!r}."
            )

        normalized = validate_defect_class_names(self.defect_class_names)
        object.__setattr__(self, "defect_class_names", normalized)

        if not isinstance(self.min_size, int) or isinstance(self.min_size, bool):
            raise TypeError("min_size must be int.")
        if not isinstance(self.max_size, int) or isinstance(self.max_size, bool):
            raise TypeError("max_size must be int.")
        if self.min_size <= 0:
            raise ValueError("min_size must be positive.")
        if self.max_size <= 0:
            raise ValueError("max_size must be positive.")
        if self.max_size < self.min_size:
            raise ValueError(
                "max_size must be greater than or equal to min_size."
            )

        for field_name in (
            "use_pretrained_weights",
            "use_pretrained_backbone",
            "progress",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be bool.")

        if self.use_pretrained_weights and self.use_pretrained_backbone:
            raise ValueError(
                "Full pretrained detection weights and pretrained backbone "
                "cannot both be requested explicitly."
            )

    @property
    def num_defect_classes(self) -> int:
        return len(self.defect_class_names)

    @property
    def num_classes(self) -> int:
        """Background 1개를 포함한 Faster R-CNN 출력 Class 수."""
        return self.num_defect_classes + 1

    @property
    def class_to_index(self) -> dict[str, int]:
        return build_detection_label_maps(self.defect_class_names)[0]

    @property
    def index_to_class(self) -> dict[int, str]:
        return build_detection_label_maps(self.defect_class_names)[1]
