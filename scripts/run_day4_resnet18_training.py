"""Day 4 ResNet18 전이학습 실제 학습 실행 스크립트.

실행 예시
---------
구조만 검증하고 사전학습 Weight를 다운로드하지 않음:

    python -m scripts.run_day4_resnet18_training \
        --validate-only \
        --no-pretrained-weights

실제 사전학습 ResNet18 학습:

    python -m scripts.run_day4_resnet18_training

설계 핵심
---------
1. 기존 Dataset·DataLoader·Loss·Epoch Runner·Training Pipeline을 재사용한다.
2. ResNet18 Backbone은 동결하고 새 FC Head만 학습한다.
3. Optimizer에는 Model 전체가 아니라 classification_head만 전달한다.
4. CNN Checkpoint와 별도 경로를 사용한다.
5. 학습 후 새 Model에 Best Checkpoint를 strict=True로 복원한다.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.optim import Optimizer

from src.data.data_loader import create_vision_data_loaders
from src.models.resnet18_transfer import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    ResNet18Transfer,
    create_resnet18_transfer_model,
)
from src.reproducibility import set_global_random_seed
from src.training.checkpoint_loader import load_model_checkpoint
from src.training.loss_function import create_binary_classification_loss
from src.training.optimizer import create_optimizer
from src.training.training_pipeline import TrainingResult, run_training


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN = "제조 비전 결함 분석 시스템"
RUN_NAME = "day4_resnet18_transfer_training"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINT_PATH = Path(
    "models/checkpoints/resnet18_transfer_best.pt"
)
DEFAULT_HISTORY_PATH = Path(
    "reports/artifacts/day4_resnet18_training_history.json"
)

DEFAULT_RANDOM_SEED = 42
DEFAULT_EPOCH_COUNT = 5
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 0.0

BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = False
DROP_LAST = False
PERSISTENT_WORKERS = False

BEST_MODEL_SELECTION_METRIC = "validation_loss"


@dataclass(frozen=True)
class TrainingComponents:
    """학습에 함께 사용되는 Model·Loss·Optimizer 묶음."""

    model: ResNet18Transfer
    loss_function: nn.Module
    optimizer: Optimizer


# =============================================================================
# Command Line
# =============================================================================


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Day 4 학습 실행 Argument를 해석한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-backbone ResNet18 transfer model "
            "for binary manufacturing defect classification."
        )
    )

    parser.add_argument(
        "--epoch-count",
        type=int,
        default=DEFAULT_EPOCH_COUNT,
        help="Number of training epochs. Default: 5",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=(
            "Best checkpoint path. Relative paths are resolved "
            "from the project root."
        ),
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help=(
            "Training history JSON path. Relative paths are resolved "
            "from the project root."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the complete training structure without training.",
    )
    parser.add_argument(
        "--no-pretrained-weights",
        action="store_true",
        help=(
            "Use weights=None. This is intended for offline structure "
            "validation and tests, not the final transfer-learning run."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-epoch output from the shared training pipeline.",
    )

    return parser.parse_args(arguments)


# =============================================================================
# Path·Argument Validation
# =============================================================================


def resolve_project_path(path: Path) -> Path:
    """상대 경로를 Project Root 기준 절대 경로로 변환한다."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path.")

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def format_project_relative_path(path: Path) -> str:
    """가능하면 Project Root 기준 POSIX 경로로 표시한다."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def validate_execution_arguments(
    *,
    epoch_count: int,
    checkpoint_path: Path,
    history_path: Path,
) -> None:
    """학습 시작 전에 기본 실행 설정을 검증한다."""
    if not isinstance(epoch_count, int) or isinstance(epoch_count, bool):
        raise TypeError("epoch_count must be int.")

    if epoch_count <= 0:
        raise ValueError("epoch_count must be greater than zero.")

    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(
            "checkpoint_path must use .pt or .pth extension."
        )

    if history_path.suffix.lower() != ".json":
        raise ValueError("history_path must use .json extension.")

    if checkpoint_path == history_path:
        raise ValueError(
            "checkpoint_path and history_path must be different."
        )


# =============================================================================
# Model·Loss·Optimizer
# =============================================================================


def create_training_components(
    *,
    device: torch.device,
    use_pretrained_weights: bool,
) -> TrainingComponents:
    """ResNet18 Feature Extractor 학습 구성요소를 생성한다.

    Optimizer 정책
    --------------
    Backbone은 requires_grad=False이고 새 FC Head만 학습한다.
    공유 create_optimizer()에는 Model 전체가 아니라 classification_head를
    전달해 Optimizer Param Group에 정확히 513개 Parameter만 등록한다.
    """
    if not isinstance(device, torch.device):
        raise TypeError("device must be torch.device.")

    if not isinstance(use_pretrained_weights, bool):
        raise TypeError("use_pretrained_weights must be bool.")

    model = create_resnet18_transfer_model(
        use_pretrained_weights=use_pretrained_weights,
        freeze_backbone=True,
        progress=True,
    ).to(device)

    loss_function = create_binary_classification_loss()

    optimizer = create_optimizer(
        model=model.classification_head,
        learning_rate=DEFAULT_LEARNING_RATE,
        weight_decay=DEFAULT_WEIGHT_DECAY,
    )

    validate_optimizer_targets_only_trainable_parameters(
        model=model,
        optimizer=optimizer,
    )

    return TrainingComponents(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
    )


def validate_optimizer_targets_only_trainable_parameters(
    *,
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Optimizer가 requires_grad=True Parameter만 정확히 포함하는지 검증한다."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be torch.nn.Module.")

    if not isinstance(optimizer, Optimizer):
        raise TypeError("optimizer must be torch.optim.Optimizer.")

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError("model has no trainable parameters.")

    optimizer_parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]

    trainable_ids = {id(parameter) for parameter in trainable_parameters}
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}

    if len(optimizer_parameters) != len(optimizer_ids):
        raise ValueError(
            "optimizer contains duplicate Parameter references."
        )

    if optimizer_ids != trainable_ids:
        missing_count = len(trainable_ids - optimizer_ids)
        unexpected_count = len(optimizer_ids - trainable_ids)

        raise ValueError(
            "optimizer parameters must exactly match model trainable "
            "parameters. "
            f"Missing: {missing_count}. "
            f"Unexpected: {unexpected_count}."
        )


# =============================================================================
# Training Result·Checkpoint
# =============================================================================


def validate_training_result(
    *,
    training_result: TrainingResult,
    epoch_count: int,
    checkpoint_path: Path,
) -> None:
    """공유 Training Pipeline 반환값과 Artifact를 검증한다."""
    if not isinstance(training_result, TrainingResult):
        raise TypeError("training_result must be TrainingResult.")

    if len(training_result.history) != epoch_count:
        raise ValueError(
            "training history length must match epoch_count."
        )

    if not 1 <= training_result.best_epoch_number <= epoch_count:
        raise ValueError("best_epoch_number is outside the epoch range.")

    if not math.isfinite(training_result.best_validation_loss):
        raise ValueError("best_validation_loss must be finite.")

    if not (
        0.0
        <= training_result.best_validation_accuracy
        <= 1.0
    ):
        raise ValueError(
            "best_validation_accuracy must be between 0 and 1."
        )

    if training_result.checkpoint_path != checkpoint_path:
        raise ValueError(
            "TrainingResult checkpoint path does not match "
            "the configured checkpoint path."
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Best Model Checkpoint was not created. "
            f"Expected path: {checkpoint_path}."
        )


def restore_best_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
) -> ResNet18Transfer:
    """새 ResNet18Transfer Instance에 Best Checkpoint를 복원한다.

    복원 Model은 weights=None으로 생성한다. Checkpoint가 Backbone과 Head의
    전체 state_dict를 저장하므로 복원 시 ImageNet Weight를 다시 다운로드할
    필요가 없다.
    """
    restored_model = create_resnet18_transfer_model(
        use_pretrained_weights=False,
        freeze_backbone=True,
        progress=False,
    ).to(device)

    _ = load_model_checkpoint(
        model=restored_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    restored_model.eval()
    return restored_model


# =============================================================================
# History JSON
# =============================================================================


def build_training_history_payload(
    *,
    components: TrainingComponents,
    data_loaders: object,
    device: torch.device,
    epoch_count: int,
    use_pretrained_weights: bool,
    training_duration_seconds: float,
    training_result: TrainingResult,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, object]:
    """TrainingResult를 Day 4 JSON Artifact 구조로 변환한다."""
    model = components.model
    optimizer = components.optimizer
    parameter_counts = model.parameter_counts()
    optimizer_parameter_count = sum(
        parameter.numel()
        for group in optimizer.param_groups
        for parameter in group["params"]
    )

    history = [
        {
            "epoch_number": item.epoch_number,
            "train_loss": item.train_result.average_loss,
            "train_accuracy": item.train_result.accuracy,
            "train_accuracy_percent": (
                item.train_result.accuracy * 100.0
            ),
            "validation_loss": item.validation_result.average_loss,
            "validation_accuracy": item.validation_result.accuracy,
            "validation_accuracy_percent": (
                item.validation_result.accuracy * 100.0
            ),
        }
        for item in training_result.history
    ]

    return {
        "project": {
            "name": PROJECT_NAME,
            "name_korean": PROJECT_NAME_KOREAN,
            "run_name": RUN_NAME,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "reproducibility": {
            "random_seed": DEFAULT_RANDOM_SEED,
            "deterministic_algorithms": False,
        },
        "data": {
            "image_size": [224, 224],
            "class_to_index": {
                "ok_front": 0,
                "def_front": 1,
            },
            "positive_class": "DEFECT",
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "drop_last": DROP_LAST,
            "persistent_workers": PERSISTENT_WORKERS,
            "train_sample_count": len(data_loaders.train_dataset),
            "validation_sample_count": len(
                data_loaders.validation_dataset
            ),
            "test_sample_count": len(data_loaders.test_dataset),
            "train_batch_count": len(data_loaders.train_loader),
            "validation_batch_count": len(
                data_loaders.validation_loader
            ),
            "test_batch_count": len(data_loaders.test_loader),
            "test_used_during_training": False,
        },
        "model": {
            "name": model.__class__.__name__,
            "module": model.__class__.__module__,
            "architecture": "torchvision_resnet18",
            "pretrained_weights": (
                "ResNet18_Weights.DEFAULT"
                if use_pretrained_weights
                else None
            ),
            "freeze_backbone": model.freeze_backbone,
            "batchnorm_policy": "frozen_backbone_eval",
            "classification_head": "Linear(512, 1)",
            "output": "single_binary_raw_logit",
            "total_parameter_count": parameter_counts.total,
            "trainable_parameter_count": parameter_counts.trainable,
            "frozen_parameter_count": parameter_counts.frozen,
            "optimizer_parameter_count": optimizer_parameter_count,
            "gradcam_target_layer": model.gradcam_target_layer_name,
        },
        "training_configuration": {
            "epoch_count": epoch_count,
            "loss_function": (
                components.loss_function.__class__.__name__
            ),
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "weight_decay": float(
                optimizer.param_groups[0]["weight_decay"]
            ),
            "classification_threshold": (
                DEFAULT_CLASSIFICATION_THRESHOLD
            ),
            "best_model_selection_metric": (
                BEST_MODEL_SELECTION_METRIC
            ),
        },
        "best_result": {
            "epoch_number": training_result.best_epoch_number,
            "validation_loss": training_result.best_validation_loss,
            "validation_accuracy": (
                training_result.best_validation_accuracy
            ),
            "validation_accuracy_percent": (
                training_result.best_validation_accuracy * 100.0
            ),
        },
        "runtime": {
            "training_duration_seconds": training_duration_seconds,
            "training_duration_minutes": (
                training_duration_seconds / 60.0
            ),
        },
        "artifacts": {
            "checkpoint_path": format_project_relative_path(
                checkpoint_path
            ),
            "training_history_path": format_project_relative_path(
                history_path
            ),
            "checkpoint_restored_after_training": True,
        },
        "history": history,
    }


def write_json_atomically(
    *,
    payload: dict[str, object],
    output_path: Path,
) -> None:
    """임시 파일 저장 후 교체하는 방식으로 JSON을 안전하게 기록한다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            json.dump(
                payload,
                output_file,
                ensure_ascii=False,
                indent=2,
            )
            output_file.write("\n")

        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# =============================================================================
# Console Output
# =============================================================================


def print_execution_configuration(
    *,
    components: TrainingComponents,
    data_loaders: object,
    device: torch.device,
    epoch_count: int,
    use_pretrained_weights: bool,
    checkpoint_path: Path,
    history_path: Path,
    validate_only: bool,
) -> None:
    """실제 학습 전에 공정 비교와 Parameter 설정을 출력한다."""
    counts = components.model.parameter_counts()
    optimizer_parameter_count = sum(
        parameter.numel()
        for group in components.optimizer.param_groups
        for parameter in group["params"]
    )

    print("=" * 100)
    print("DAY 4 - RESNET18 TRANSFER LEARNING TRAINING")
    print("=" * 100)
    print()
    print("[MODEL]")
    print(f"Model                      : {components.model.__class__.__name__}")
    print("Architecture               : torchvision ResNet18")
    print(f"Pretrained weights         : {use_pretrained_weights}")
    print(f"Freeze backbone            : {components.model.freeze_backbone}")
    print("BatchNorm policy           : frozen backbone eval")
    print("Classification head        : Linear(512, 1)")
    print(f"Total parameters           : {counts.total}")
    print(f"Trainable parameters       : {counts.trainable}")
    print(f"Frozen parameters          : {counts.frozen}")
    print(f"Optimizer parameters       : {optimizer_parameter_count}")
    print(
        "Grad-CAM target layer      : "
        f"{components.model.gradcam_target_layer_name}"
    )
    print()
    print("[DATA]")
    print(f"Train samples              : {len(data_loaders.train_dataset)}")
    print(
        "Validation samples         : "
        f"{len(data_loaders.validation_dataset)}"
    )
    print(f"Test samples               : {len(data_loaders.test_dataset)}")
    print("Image size                 : 224 x 224")
    print("Positive class             : DEFECT")
    print()
    print("[TRAINING]")
    print(f"Device                     : {device}")
    print(f"Epoch count                : {epoch_count}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print(
        "Loss function              : "
        f"{components.loss_function.__class__.__name__}"
    )
    print(
        "Optimizer                  : "
        f"{components.optimizer.__class__.__name__}"
    )
    print(f"Learning rate              : {DEFAULT_LEARNING_RATE}")
    print(f"Weight decay               : {DEFAULT_WEIGHT_DECAY}")
    print(
        "Classification threshold  : "
        f"{DEFAULT_CLASSIFICATION_THRESHOLD}"
    )
    print(
        "Best selection metric      : "
        f"{BEST_MODEL_SELECTION_METRIC}"
    )
    print(f"Validate only              : {validate_only}")
    print()
    print("[ARTIFACTS]")
    print(f"Checkpoint                 : {checkpoint_path}")
    print(f"Training history           : {history_path}")


def print_training_completion(
    *,
    training_result: TrainingResult,
    training_duration_seconds: float,
    checkpoint_path: Path,
    history_path: Path,
) -> None:
    """실제 Day 4 ResNet18 학습 완료 결과를 출력한다."""
    print()
    print("=" * 100)
    print("DAY 4 - RESNET18 TRANSFER LEARNING COMPLETED")
    print("=" * 100)
    print()
    print("[BEST RESULT]")
    print(f"Best epoch                 : {training_result.best_epoch_number}")
    print(
        "Best validation loss       : "
        f"{training_result.best_validation_loss:.6f}"
    )
    print(
        "Best validation accuracy   : "
        f"{training_result.best_validation_accuracy:.6f}"
    )
    print(
        "Best validation percent    : "
        f"{training_result.best_validation_accuracy * 100.0:.2f}%"
    )
    print()
    print("[RUNTIME]")
    print(f"Training seconds           : {training_duration_seconds:.2f}")
    print(
        "Training minutes           : "
        f"{training_duration_seconds / 60.0:.2f}"
    )
    print()
    print("[ARTIFACTS]")
    print(f"Best checkpoint            : {checkpoint_path}")
    print(f"Training history JSON      : {history_path}")
    print(f"Checkpoint exists          : {checkpoint_path.is_file()}")
    print(f"History JSON exists        : {history_path.is_file()}")
    print()
    print("[PASS] Day 4 ResNet18 transfer learning real training")


# =============================================================================
# Main
# =============================================================================


def main(arguments: Sequence[str] | None = None) -> None:
    """Day 4 ResNet18 전이학습 실행 Entry Point."""
    parsed = parse_arguments(arguments)

    checkpoint_path = resolve_project_path(parsed.checkpoint_path)
    history_path = resolve_project_path(parsed.history_path)

    validate_execution_arguments(
        epoch_count=parsed.epoch_count,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    settings = set_global_random_seed(
        seed=DEFAULT_RANDOM_SEED,
        deterministic_algorithms=False,
    )
    device = torch.device(settings.device)

    data_loaders = create_vision_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
        persistent_workers=PERSISTENT_WORKERS,
        random_seed=DEFAULT_RANDOM_SEED,
    )

    use_pretrained_weights = not parsed.no_pretrained_weights

    components = create_training_components(
        device=device,
        use_pretrained_weights=use_pretrained_weights,
    )

    print_execution_configuration(
        components=components,
        data_loaders=data_loaders,
        device=device,
        epoch_count=parsed.epoch_count,
        use_pretrained_weights=use_pretrained_weights,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        validate_only=parsed.validate_only,
    )

    if parsed.validate_only:
        print()
        print("=" * 100)
        print("[PASS] Day 4 ResNet18 training structure validation")
        print("=" * 100)
        return

    training_started_at = time.perf_counter()

    training_result = run_training(
        model=components.model,
        train_loader=data_loaders.train_loader,
        validation_loader=data_loaders.validation_loader,
        loss_function=components.loss_function,
        optimizer=components.optimizer,
        device=device,
        epoch_count=parsed.epoch_count,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
        checkpoint_path=checkpoint_path,
        verbose=not parsed.quiet,
    )

    training_duration_seconds = (
        time.perf_counter() - training_started_at
    )

    validate_training_result(
        training_result=training_result,
        epoch_count=parsed.epoch_count,
        checkpoint_path=checkpoint_path,
    )

    _ = restore_best_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    history_payload = build_training_history_payload(
        components=components,
        data_loaders=data_loaders,
        device=device,
        epoch_count=parsed.epoch_count,
        use_pretrained_weights=use_pretrained_weights,
        training_duration_seconds=training_duration_seconds,
        training_result=training_result,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    write_json_atomically(
        payload=history_payload,
        output_path=history_path,
    )

    if not history_path.is_file():
        raise FileNotFoundError(
            "Training History JSON was not created. "
            f"Expected path: {history_path}."
        )

    print_training_completion(
        training_result=training_result,
        training_duration_seconds=training_duration_seconds,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )


if __name__ == "__main__":
    main()
