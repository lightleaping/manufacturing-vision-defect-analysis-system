"""Day 12 Best Checkpoint를 Day 13 Detection 추론 서비스로 복원."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import torch

from src.api.detection_config import (
    DEFAULT_DETECTION_API_SETTINGS,
    DetectionApiSettings,
)
from src.api.detection_inference_service import DetectionInferenceService
from src.detection.checkpoint import load_detection_checkpoint_payload
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model


class DetectionModelLoadingError(RuntimeError):
    """Detection Production Service를 안전하게 구성하지 못한 경우."""


def create_production_detection_inference_service(
    *,
    settings: DetectionApiSettings = DEFAULT_DETECTION_API_SETTINGS,
) -> DetectionInferenceService:
    """네트워크 다운로드 없이 7-Class 모델을 만들고 Best State를 복원한다."""

    if not isinstance(settings, DetectionApiSettings):
        raise TypeError("settings must be DetectionApiSettings.")

    checkpoint_path = settings.checkpoint_path
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Detection checkpoint does not exist: {checkpoint_path}."
        )

    try:
        payload = load_detection_checkpoint_payload(
            checkpoint_path,
            map_location=settings.device,
        )
        training_config = _mapping(
            payload["training_config"],
            "training_config",
        )

        min_size = _positive_int(
            training_config.get("min_size", 320),
            "min_size",
        )
        max_size = _positive_int(
            training_config.get("max_size", 320),
            "max_size",
        )
        model_config = DetectionModelConfig(
            architecture=settings.architecture,
            min_size=min_size,
            max_size=max_size,
            use_pretrained_weights=False,
            use_pretrained_backbone=False,
            progress=False,
        )

        saved_architecture = training_config.get("architecture")
        if (
            saved_architecture is not None
            and saved_architecture != settings.architecture
        ):
            raise ValueError(
                "Checkpoint training_config architecture does not match "
                "Day 13 settings."
            )

        # Day 11 Model Config는 Day 9 Mapping의 Background 이름을 그대로
        # 사용할 수 있다. Day 12 Checkpoint 저장 정책은 Label 0을
        # ``BACKGROUND``로 정규화하므로, 비교 경계에서도 Label 0만 같은
        # Canonical 이름으로 맞춘다. 결함 Class 이름과 Label 번호는
        # 변경하지 않는다.
        expected_mapping = _canonical_expected_class_mapping(
            model_config.index_to_class
        )
        saved_mapping = {
            str(name): int(index)
            for name, index in _mapping(
                payload["class_mapping"],
                "class_mapping",
            ).items()
        }
        if saved_mapping != expected_mapping:
            raise ValueError(
                "Checkpoint class_mapping does not match the current "
                "NEU-DET project."
            )

        model_result = create_detection_model(
            config=model_config,
            device=settings.device,
            training=False,
            proposal_limits=None,
        )
        if (
            model_result.metadata.get("network_download_requested")
            is not False
        ):
            raise RuntimeError(
                "Day 13 model creation must not request pretrained downloads."
            )

        state_dict = payload["model_state_dict"]
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Checkpoint model_state_dict must be a mapping."
            )

        model_result.model.load_state_dict(
            state_dict,
            strict=True,
        )
        model_result.model.to(torch.device(settings.device))
        model_result.model.eval()

        epoch = payload["epoch"]
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 0
        ):
            raise ValueError(
                "Checkpoint epoch must be a non-negative int."
            )

        best_metric = float(payload["best_metric"])
        if (
            not math.isfinite(best_metric)
            or not 0.0 <= best_metric <= 1.0
        ):
            raise ValueError(
                "Checkpoint best_metric must be finite and in [0, 1]."
            )

        return DetectionInferenceService(
            model=model_result.model,
            checkpoint_epoch_index=epoch,
            checkpoint_metric_value=best_metric,
            class_mapping=saved_mapping,
            settings=settings,
        )

    except FileNotFoundError:
        raise
    except Exception as error:
        raise DetectionModelLoadingError(
            "Could not create the production Detection inference service."
        ) from error


def _canonical_expected_class_mapping(
    index_to_class: Mapping[int, str],
) -> dict[str, int]:
    """Model Config Mapping을 Checkpoint 비교용 Canonical 형식으로 바꾼다.

    Label 0은 Torchvision의 Background이므로 저장된 Checkpoint 정책과
    동일하게 ``BACKGROUND``로 통일한다. Label 1 이상은 기존 NEU-DET
    Canonical Class 이름을 그대로 유지한다.
    """

    if not isinstance(index_to_class, Mapping):
        raise TypeError(
            "index_to_class must be a mapping."
        )
    if not index_to_class:
        raise ValueError(
            "index_to_class must not be empty."
        )

    normalized: dict[str, int] = {}

    for raw_index, raw_name in index_to_class.items():
        if (
            not isinstance(raw_index, int)
            or isinstance(raw_index, bool)
            or raw_index < 0
        ):
            raise ValueError(
                "Every index_to_class key must be a non-negative int."
            )
        if (
            not isinstance(raw_name, str)
            or not raw_name.strip()
        ):
            raise ValueError(
                "Every index_to_class value must be a non-empty str."
            )

        normalized_name = (
            "BACKGROUND"
            if raw_index == 0
            else raw_name
        )
        if normalized_name in normalized:
            raise ValueError(
                "index_to_class contains duplicate canonical class names."
            )
        normalized[normalized_name] = raw_index

    if normalized.get("BACKGROUND") != 0:
        raise ValueError(
            "index_to_class must contain the background label at index 0."
        )
    if set(normalized.values()) != set(range(len(normalized))):
        raise ValueError(
            "index_to_class indexes must be contiguous from 0."
        )

    return normalized


def _mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{name} must be a mapping."
        )
    return dict(value)


def _positive_int(
    value: Any,
    name: str,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive int."
        )
    return value
