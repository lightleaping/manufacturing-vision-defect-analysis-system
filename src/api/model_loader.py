"""실제 ResNet18 Best Checkpoint를 API 추론 Service로 연결한다.

[그대로 재사용]
- scripts.run_day4_resnet18_training.restore_best_checkpoint
- src.data.image_transforms.create_test_transform
"""

from __future__ import annotations

import torch

from scripts.run_day4_resnet18_training import restore_best_checkpoint
from src.api.config import ApiSettings, DEFAULT_API_SETTINGS
from src.api.inference_service import ImageInferenceService
from src.data.image_transforms import create_test_transform


def create_production_inference_service(
    *,
    settings: ApiSettings = DEFAULT_API_SETTINGS,
) -> ImageInferenceService:
    """Checkpoint와 Test Transform을 한 번 로딩해 Production Service를 만든다."""

    checkpoint_path = settings.checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "ResNet18 Best Checkpoint does not exist."
        )

    device = torch.device(settings.device)

    # 기존 Day 4 복원 함수는 weights=None Model을 만들고 전체 state_dict를
    # 복원하므로 ImageNet Weight를 다시 다운로드하지 않는다.
    model = restore_best_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Day 2에서 검증한 Resize → ToTensor → ImageNet Normalize를 재사용한다.
    transform = create_test_transform()

    return ImageInferenceService(
        model=model,
        transform=transform,
        device=device,
        settings=settings,
    )
