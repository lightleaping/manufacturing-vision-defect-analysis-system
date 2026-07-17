"""Day 9 Class Mapping 재사용과 Detection Model Config 테스트."""

from __future__ import annotations

import pytest

from src.detection.dataset_config import (
    DETECTION_MODEL_CLASS_TO_INDEX,
    NEU_DET_CANONICAL_CLASSES,
)
from src.detection.model_config import (
    BACKGROUND_CLASS_NAME,
    DEFAULT_DEFECT_CLASS_NAMES,
    DetectionModelConfig,
    build_detection_label_maps,
    validate_duplicate_box_policy,
    validate_split_name,
)


def test_default_classes_reuse_day9_canonical_order() -> None:
    assert DEFAULT_DEFECT_CLASS_NAMES == tuple(NEU_DET_CANONICAL_CLASSES)


def test_label_mapping_reuses_day9_model_mapping() -> None:
    class_to_index, index_to_class = build_detection_label_maps()

    expected_defect_mapping = {
        class_name: DETECTION_MODEL_CLASS_TO_INDEX[class_name]
        for class_name in NEU_DET_CANONICAL_CLASSES
    }
    assert class_to_index == expected_defect_mapping
    assert index_to_class[0] == BACKGROUND_CLASS_NAME
    assert index_to_class[6] == "scratches"


def test_model_config_includes_background_in_num_classes() -> None:
    config = DetectionModelConfig()

    assert config.num_defect_classes == 6
    assert config.num_classes == 7
    assert config.min_size == 320
    assert config.max_size == 320
    assert config.use_pretrained_weights is False
    assert config.use_pretrained_backbone is False


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_supported_split_names(split: str) -> None:
    assert validate_split_name(split) == split


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported split"):
        validate_split_name("val")


@pytest.mark.parametrize("policy", ["preserve", "remove_exact"])
def test_supported_duplicate_box_policies(policy: str) -> None:
    assert validate_duplicate_box_policy(policy) == policy


def test_invalid_duplicate_box_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate_box_policy"):
        validate_duplicate_box_policy("remove_all")


def test_invalid_architecture_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported architecture"):
        DetectionModelConfig(architecture="fasterrcnn_resnet50_fpn_v2")


def test_pretrained_flags_cannot_both_be_enabled() -> None:
    with pytest.raises(ValueError, match="cannot both"):
        DetectionModelConfig(
            use_pretrained_weights=True,
            use_pretrained_backbone=True,
        )
