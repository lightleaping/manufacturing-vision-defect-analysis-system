"""Torchvision Faster R-CNN 생성과 NEU-DET 7-Class Head 교체.

[기존 코드 참고]
``model_config.py``에서 확정한 Architecture, Class Mapping, 입력 크기를 사용한다.

[신규 구현]
- Weight 다운로드 여부를 명시적으로 분리한다.
- Day 11 Smoke Test에서는 Detection Weight와 Backbone Weight를 모두 끈다.
- Day 12에서는 같은 Factory에 pretrained 설정만 전달해 전이학습 모델을 만든다.
- CPU Smoke Test의 실행 시간을 줄일 때만 Proposal 수 제한을 선택적으로 적용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.detection.model_config import DetectionModelConfig


@dataclass(frozen=True, slots=True)
class DetectionProposalLimits:
    """CPU Smoke Test에서 Proposal 계산량을 제한하는 선택 설정.

    Day 12 실제 학습에서는 이 설정을 전달하지 않아 Torchvision 기본값을 쓴다.
    """

    rpn_pre_nms_top_n_train: int = 40
    rpn_post_nms_top_n_train: int = 20
    rpn_pre_nms_top_n_test: int = 20
    rpn_post_nms_top_n_test: int = 10
    box_detections_per_img: int = 10

    def __post_init__(self) -> None:
        values = {
            field_name: getattr(self, field_name)
            for field_name in (
                "rpn_pre_nms_top_n_train",
                "rpn_post_nms_top_n_train",
                "rpn_pre_nms_top_n_test",
                "rpn_post_nms_top_n_test",
                "box_detections_per_img",
            )
        }
        for field_name, value in values.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be int.")
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")

        if self.rpn_post_nms_top_n_train > self.rpn_pre_nms_top_n_train:
            raise ValueError(
                "rpn_post_nms_top_n_train must not exceed "
                "rpn_pre_nms_top_n_train."
            )
        if self.rpn_post_nms_top_n_test > self.rpn_pre_nms_top_n_test:
            raise ValueError(
                "rpn_post_nms_top_n_test must not exceed "
                "rpn_pre_nms_top_n_test."
            )

    def to_torchvision_kwargs(self) -> dict[str, int]:
        return {
            "rpn_pre_nms_top_n_train": self.rpn_pre_nms_top_n_train,
            "rpn_post_nms_top_n_train": self.rpn_post_nms_top_n_train,
            "rpn_pre_nms_top_n_test": self.rpn_pre_nms_top_n_test,
            "rpn_post_nms_top_n_test": self.rpn_post_nms_top_n_test,
            "box_detections_per_img": self.box_detections_per_img,
        }


@dataclass(frozen=True, slots=True)
class DetectionModelBuildResult:
    """생성된 모델과 재현 가능한 Build Metadata."""

    model: nn.Module
    metadata: dict[str, Any]


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, str):
        resolved = torch.device(device)
    elif isinstance(device, torch.device):
        resolved = device
    else:
        raise TypeError("device must be str or torch.device.")

    if resolved.type != "cpu":
        raise ValueError(
            "Day 11 model factory currently supports the verified CPU path only."
        )
    return resolved


def _resolve_weight_arguments(
    config: DetectionModelConfig,
) -> tuple[
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights | None,
    MobileNet_V3_Large_Weights | None,
]:
    """요청하지 않은 Weight 다운로드가 발생하지 않도록 명시적으로 반환한다."""
    if config.use_pretrained_weights:
        return (
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
            None,
        )
    if config.use_pretrained_backbone:
        return (None, MobileNet_V3_Large_Weights.DEFAULT)
    return (None, None)


def replace_detection_predictor(
    model: nn.Module,
    *,
    num_classes: int,
) -> nn.Module:
    """기존 COCO Predictor를 Background 포함 NEU-DET Class 수로 교체한다."""
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError("num_classes must be int.")
    if num_classes < 2:
        raise ValueError("num_classes must include background and one object class.")

    try:
        current_predictor = model.roi_heads.box_predictor  # type: ignore[attr-defined]
        in_features = current_predictor.cls_score.in_features
    except AttributeError as error:
        raise TypeError(
            "model must expose roi_heads.box_predictor.cls_score.in_features."
        ) from error

    model.roi_heads.box_predictor = FastRCNNPredictor(  # type: ignore[attr-defined]
        in_features,
        num_classes,
    )
    return model


def create_detection_model(
    *,
    config: DetectionModelConfig | None = None,
    device: str | torch.device = "cpu",
    training: bool = False,
    proposal_limits: DetectionProposalLimits | None = None,
) -> DetectionModelBuildResult:
    """MobileNetV3 Faster R-CNN을 생성하고 7-Class Head를 보장한다.

    ``use_pretrained_weights=False``와 ``use_pretrained_backbone=False``이면
    ``weights=None``과 ``weights_backbone=None``을 모두 전달하므로 네트워크
    다운로드가 발생하지 않는다.
    """
    resolved_config = config or DetectionModelConfig()
    if not isinstance(resolved_config, DetectionModelConfig):
        raise TypeError("config must be DetectionModelConfig.")
    if not isinstance(training, bool):
        raise TypeError("training must be bool.")
    if proposal_limits is not None and not isinstance(
        proposal_limits,
        DetectionProposalLimits,
    ):
        raise TypeError("proposal_limits must be DetectionProposalLimits or None.")

    resolved_device = _resolve_device(device)
    detection_weights, backbone_weights = _resolve_weight_arguments(
        resolved_config
    )

    constructor_kwargs: dict[str, Any] = {
        "weights": detection_weights,
        "weights_backbone": backbone_weights,
        "progress": resolved_config.progress,
        "min_size": resolved_config.min_size,
        "max_size": resolved_config.max_size,
    }
    if proposal_limits is not None:
        constructor_kwargs.update(proposal_limits.to_torchvision_kwargs())

    # Full Detection Weight는 COCO Head를 포함하므로 num_classes를 생성자에
    # 전달하지 않고 생성 후 Predictor를 교체한다.
    if detection_weights is None:
        constructor_kwargs["num_classes"] = resolved_config.num_classes

    model = fasterrcnn_mobilenet_v3_large_320_fpn(**constructor_kwargs)

    current_output_classes = int(model.roi_heads.box_predictor.cls_score.out_features)
    if current_output_classes != resolved_config.num_classes:
        replace_detection_predictor(
            model,
            num_classes=resolved_config.num_classes,
        )

    model.to(resolved_device)
    model.train(mode=training)

    output_classes = int(model.roi_heads.box_predictor.cls_score.out_features)
    if output_classes != resolved_config.num_classes:
        raise RuntimeError(
            "Detection predictor output class count does not match config."
        )

    metadata: dict[str, Any] = {
        "architecture": resolved_config.architecture,
        "device": str(resolved_device),
        "training_mode": bool(model.training),
        "num_defect_classes": resolved_config.num_defect_classes,
        "num_classes_with_background": resolved_config.num_classes,
        "class_to_index": resolved_config.class_to_index,
        "min_size": resolved_config.min_size,
        "max_size": resolved_config.max_size,
        "pretrained_detection_weights": (
            None if detection_weights is None else str(detection_weights)
        ),
        "pretrained_backbone_weights": (
            None if backbone_weights is None else str(backbone_weights)
        ),
        "network_download_requested": bool(
            detection_weights is not None or backbone_weights is not None
        ),
        "predictor_output_classes": output_classes,
        "proposal_limits": (
            None
            if proposal_limits is None
            else proposal_limits.to_torchvision_kwargs()
        ),
    }
    return DetectionModelBuildResult(model=model, metadata=metadata)
