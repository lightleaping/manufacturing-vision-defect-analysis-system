"""Day 4 ResNet18 학습 연결 테스트.

실제 ImageNet Weight 다운로드와 실제 Dataset 전체 학습은 수행하지 않는다.
weights=None과 작은 TensorDataset을 사용해 다음을 검증한다.

1. Optimizer에는 FC Head Parameter만 포함된다.
2. 기존 Training Pipeline이 ResNet18 Wrapper와 호환된다.
3. 학습 중 Frozen Backbone은 변하지 않고 FC Head만 갱신된다.
4. Best Checkpoint가 생성되고 새 Model에 복원된다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_day4_resnet18_training import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    create_training_components,
    restore_best_checkpoint,
    validate_optimizer_targets_only_trainable_parameters,
)
from src.models.resnet18_transfer import ResNet18Transfer
from src.training.training_pipeline import run_training


@pytest.fixture()
def cpu_device() -> torch.device:
    """Test 환경에서 사용할 CPU Device."""
    return torch.device("cpu")


@pytest.fixture()
def components(cpu_device: torch.device):
    """네트워크가 필요 없는 weights=None 학습 구성요소."""
    torch.manual_seed(DEFAULT_RANDOM_SEED)
    return create_training_components(
        device=cpu_device,
        use_pretrained_weights=False,
    )


def test_optimizer_contains_only_classification_head_parameters(
    components,
) -> None:
    """Optimizer Param Group이 FC Weight·Bias만 포함하는지 확인한다."""
    model = components.model
    optimizer = components.optimizer

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    head_parameter_ids = {
        id(parameter)
        for parameter in model.classification_head.parameters()
    }

    assert optimizer_parameter_ids == head_parameter_ids
    assert sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    ) == 513
    assert optimizer.param_groups[0]["lr"] == DEFAULT_LEARNING_RATE


def test_optimizer_validation_rejects_missing_trainable_parameter(
    components,
) -> None:
    """FC Bias가 빠진 Optimizer를 명확한 오류로 차단하는지 확인한다."""
    invalid_optimizer = torch.optim.Adam(
        [components.model.classification_head.weight],
        lr=DEFAULT_LEARNING_RATE,
    )

    with pytest.raises(
        ValueError,
        match="must exactly match model trainable parameters",
    ):
        validate_optimizer_targets_only_trainable_parameters(
            model=components.model,
            optimizer=invalid_optimizer,
        )


def test_shared_training_pipeline_updates_only_head_and_saves_checkpoint(
    components,
    cpu_device: torch.device,
    tmp_path: Path,
) -> None:
    """기존 run_training이 Frozen ResNet18과 그대로 호환되는지 확인한다."""
    model = components.model

    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    checkpoint_path = tmp_path / "resnet18_transfer_best.pt"

    backbone_before = model.resnet18.conv1.weight.detach().clone()
    head_weight_before = (
        model.classification_head.weight.detach().clone()
    )
    head_bias_before = model.classification_head.bias.detach().clone()

    result = run_training(
        model=model,
        train_loader=loader,
        validation_loader=loader,
        loss_function=components.loss_function,
        optimizer=components.optimizer,
        device=cpu_device,
        epoch_count=1,
        classification_threshold=0.5,
        checkpoint_path=checkpoint_path,
        verbose=False,
    )

    assert result.best_epoch_number == 1
    assert result.checkpoint_path == checkpoint_path
    assert checkpoint_path.is_file()

    # Frozen Backbone은 학습 전후가 완전히 같아야 한다.
    assert torch.equal(
        model.resnet18.conv1.weight.detach(),
        backbone_before,
    )

    # 새 FC Head는 Optimizer Step으로 변경돼야 한다.
    head_weight_changed = not torch.equal(
        model.classification_head.weight.detach(),
        head_weight_before,
    )
    head_bias_changed = not torch.equal(
        model.classification_head.bias.detach(),
        head_bias_before,
    )

    assert head_weight_changed or head_bias_changed

    # train_one_epoch 내부 model.train() 호출 뒤에도 BN은 eval 상태다.
    assert model.resnet18.training is False
    assert all(
        not module.training
        for module in model.resnet18.modules()
        if isinstance(module, torch.nn.BatchNorm2d)
    )


def test_best_checkpoint_restores_into_fresh_offline_model(
    components,
    cpu_device: torch.device,
    tmp_path: Path,
) -> None:
    """사전학습 Weight 재다운로드 없이 Checkpoint 전체 복원이 가능한지 확인한다."""
    images = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    loader = DataLoader(
        TensorDataset(images, labels),
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    checkpoint_path = tmp_path / "restorable_resnet18.pt"

    _ = run_training(
        model=components.model,
        train_loader=loader,
        validation_loader=loader,
        loss_function=components.loss_function,
        optimizer=components.optimizer,
        device=cpu_device,
        epoch_count=1,
        classification_threshold=0.5,
        checkpoint_path=checkpoint_path,
        verbose=False,
    )

    restored_model = restore_best_checkpoint(
        checkpoint_path=checkpoint_path,
        device=cpu_device,
    )

    assert isinstance(restored_model, ResNet18Transfer)
    assert restored_model.training is False

    with torch.inference_mode():
        logits = restored_model(images)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
