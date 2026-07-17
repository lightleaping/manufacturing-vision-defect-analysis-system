"""Detection Model Factory 설정과 7-Class Head를 검증한다."""

from __future__ import annotations

import pytest

from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import (
    DetectionProposalLimits,
    create_detection_model,
)


def test_proposal_limits_validate_order_and_values() -> None:
    limits = DetectionProposalLimits()
    assert limits.to_torchvision_kwargs()["box_detections_per_img"] == 10

    with pytest.raises(ValueError):
        DetectionProposalLimits(
            rpn_pre_nms_top_n_train=10,
            rpn_post_nms_top_n_train=11,
        )
    with pytest.raises(ValueError):
        DetectionProposalLimits(box_detections_per_img=0)


def test_weight_free_model_has_neu_det_predictor_and_cpu_device() -> None:
    config = DetectionModelConfig(min_size=64, max_size=64)

    result = create_detection_model(
        config=config,
        device="cpu",
        training=False,
        proposal_limits=DetectionProposalLimits(),
    )

    assert result.model.training is False
    assert result.model.roi_heads.box_predictor.cls_score.out_features == 7
    assert result.metadata["predictor_output_classes"] == 7
    assert result.metadata["network_download_requested"] is False
    assert result.metadata["pretrained_detection_weights"] is None
    assert result.metadata["pretrained_backbone_weights"] is None
    assert all(parameter.device.type == "cpu" for parameter in result.model.parameters())


def test_factory_can_create_model_in_training_mode() -> None:
    result = create_detection_model(
        config=DetectionModelConfig(min_size=64, max_size=64),
        training=True,
        proposal_limits=DetectionProposalLimits(),
    )
    assert result.model.training is True
    assert result.metadata["training_mode"] is True


def test_factory_rejects_non_cpu_device() -> None:
    with pytest.raises(ValueError, match="CPU"):
        create_detection_model(device="cuda")
