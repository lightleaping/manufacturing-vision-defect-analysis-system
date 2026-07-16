"""
Day 3 CNN Baseline real training runner.

프로젝트
--------
Manufacturing Vision Defect Analysis System

한국어 프로젝트명
-----------------
제조 비전 결함 분석 시스템

이 Script의 역할
----------------
Day 1·Day 2·Day 3에서 구현한 기존 모듈을 연결하여
CNNBaseline의 실제 이미지 학습을 실행한다.

이 Script는 다음 기능을 새로 구현하지 않는다.

    Dataset Split

    Image Transform

    PyTorch Dataset

    DataLoader

    CNN Architecture

    Loss Function

    Optimizer

    Train Epoch

    Validation Epoch

    Multi-Epoch Training Pipeline

위 기능은 이미 src 모듈에 구현되어 있으며
현재 Script는 해당 기능을 순서대로 연결한다.

전체 실행 흐름
-------------
Global Random Seed

-> Vision DataLoaders

-> CNNBaseline

-> CPU Device

-> BCEWithLogitsLoss

-> Adam Optimizer

-> run_training()

-> Best Validation Loss Checkpoint

-> Training History JSON

현재 기본 학습 설정
------------------
Model:

    CNNBaseline

Device:

    cpu

Epoch:

    5

Batch Size:

    32

Learning Rate:

    0.001

Weight Decay:

    0.0

Classification Threshold:

    0.5

Random Seed:

    42

Best Model 기준:

    Lowest Validation Loss

Checkpoint:

    models/checkpoints/cnn_baseline_best.pt

History Artifact:

    reports/artifacts/day3_cnn_baseline_training_history.json

Dataset 사용 원칙
-----------------
Train:

    Model Parameter 학습

Validation:

    Best Epoch 선택

Test:

    현재 Script에서 사용하지 않음

Test Dataset은 Best Model 학습이 완료된 후
최종 성능 평가 단계에서만 사용한다.

실행 방법
---------
실제 학습 전 연결 구조 검증:

    python -m scripts.run_day3_cnn_baseline_training \
        --validate-only

기본 5 Epoch 실제 학습:

    python -m scripts.run_day3_cnn_baseline_training

사용자 지정 Epoch:

    python -m scripts.run_day3_cnn_baseline_training \
        --epoch-count 1

주의
----
--validate-only는 Dataset·DataLoader·Model·Loss·Optimizer를 생성하고
설정만 검증한다.

Train DataLoader를 반복하지 않으므로
Model Parameter는 변경되지 않으며 Checkpoint와 History JSON도 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer

from src.data.data_loader import (
    BATCH_SIZE,
    DROP_LAST,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED as DATA_LOADER_RANDOM_SEED,
    VisionDataLoaders,
    create_vision_data_loaders,
    validate_sampler_types,
)
from src.data.dataset_config import (
    CLASS_TO_INDEX,
    INDEX_TO_CLASS_NAME,
    PROJECT_ROOT,
)
from src.data.dataset_split import (
    RANDOM_SEED as DATASET_SPLIT_RANDOM_SEED,
    VALIDATION_RATIO,
)
from src.data.image_transforms import (
    IMAGE_SIZE,
)
from src.models.cnn_baseline import (
    CNNBaseline,
)
from src.reproducibility import (
    DEFAULT_RANDOM_SEED,
    ReproducibilitySettings,
    set_global_random_seed,
)
from src.training.epoch_runner import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
)
from src.training.loss_function import (
    create_binary_classification_loss,
)
from src.training.optimizer import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    create_optimizer,
)
from src.training.training_pipeline import (
    BEST_MODEL_SELECTION_METRIC,
    CHECKPOINT_VERSION,
    DEFAULT_CNN_CHECKPOINT_PATH,
    DEFAULT_EPOCH_COUNT,
    TrainingResult,
    run_training,
)


# =============================================================================
# Project Metadata
# =============================================================================

PROJECT_NAME = (
    "Manufacturing Vision Defect Analysis System"
)

PROJECT_NAME_KOREAN = (
    "제조 비전 결함 분석 시스템"
)

RUN_NAME = "day3_cnn_baseline_training"


# =============================================================================
# Default Artifact Paths
# =============================================================================

# training_pipeline.py의 기본 Checkpoint 경로는
# 프로젝트 Root 기준 상대 경로다.
#
# 실제 실행 Script에서는 현재 작업 Directory와 관계없이
# 항상 프로젝트 내부에 저장되도록 절대 경로로 변환한다.
DEFAULT_CHECKPOINT_OUTPUT_PATH = (
    PROJECT_ROOT
    / DEFAULT_CNN_CHECKPOINT_PATH
)


# Epoch별 Train·Validation 결과를 저장할 JSON 경로다.
DEFAULT_HISTORY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "artifacts"
    / "day3_cnn_baseline_training_history.json"
)


# =============================================================================
# Expected Current Dataset Configuration
# =============================================================================

# Day 2에서 실제 검증한 현재 Split 결과다.
#
# Dataset이나 Split 설정이 실수로 변경되면
# 실제 학습을 시작하기 전에 즉시 발견한다.
EXPECTED_TRAIN_SAMPLE_COUNT = 5_306

EXPECTED_VALIDATION_SAMPLE_COUNT = 1_327

EXPECTED_TEST_SAMPLE_COUNT = 715


# 현재 Batch Size 32와 drop_last=False 기준 Batch 수다.
EXPECTED_TRAIN_BATCH_COUNT = 166

EXPECTED_VALIDATION_BATCH_COUNT = 42

EXPECTED_TEST_BATCH_COUNT = 23


# 현재 CNNBaseline의 검증된 전체 Parameter 수다.
EXPECTED_CNN_PARAMETER_COUNT = 6_065


# =============================================================================
# Command Line Arguments
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """
    실제 학습 Script의 Command Line Argument를 생성하고 해석한다.

    지원 Argument
    ---------------
    --epoch-count:
        실행할 Epoch 수

        기본:

            5

    --checkpoint-path:
        Best Model Checkpoint 저장 경로

    --history-path:
        Training History JSON 저장 경로

    --validate-only:
        실제 Train·Validation을 실행하지 않고
        연결 구조와 설정만 검증

    --quiet:
        run_training()의 Epoch별 상세 Console 출력을 비활성화

    출력
    ----
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "Train the CNNBaseline model for the "
            "Manufacturing Vision Defect Analysis System."
        ),
    )

    parser.add_argument(
        "--epoch-count",
        type=int,
        default=DEFAULT_EPOCH_COUNT,
        help=(
            "Number of Train and Validation epochs. "
            f"Default: {DEFAULT_EPOCH_COUNT}."
        ),
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_OUTPUT_PATH,
        help=(
            "Best-model checkpoint output path. "
            "Supported extensions: .pt, .pth."
        ),
    )

    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_OUTPUT_PATH,
        help=(
            "Training history JSON output path."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the real data, model, loss, "
            "optimizer, device, and output configuration "
            "without running training."
        ),
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Disable the detailed epoch output from "
            "the reusable training pipeline."
        ),
    )

    return parser.parse_args()


# =============================================================================
# Path Utilities
# =============================================================================


def resolve_project_path(
    path: Path,
) -> Path:
    """
    상대 경로를 Project Root 기준 절대 경로로 변환한다.

    입력 예
    -------
    상대 경로:

        models/checkpoints/best.pt

    출력 예
    -------
    절대 경로:

        <project_root>/models/checkpoints/best.pt

    이미 절대 경로이면 그대로 반환한다.
    """
    if path.is_absolute():
        return path

    return (
        PROJECT_ROOT
        / path
    )


def format_project_relative_path(
    path: Path,
) -> str:
    """
    Project 내부 경로를 JSON·Console용 상대 경로로 변환한다.

    Project 외부 경로라면 절대 경로 문자열을 반환한다.

    Windows에서도 JSON에는 '/' 구분자를 사용한다.
    """
    resolved_project_root = (
        PROJECT_ROOT.resolve()
    )

    resolved_path = path.resolve()

    try:
        relative_path = (
            resolved_path.relative_to(
                resolved_project_root,
            )
        )

        return relative_path.as_posix()

    except ValueError:
        return str(
            resolved_path,
        )


# =============================================================================
# Execution Argument Validation
# =============================================================================


def validate_execution_arguments(
    epoch_count: int,
    checkpoint_path: Path,
    history_path: Path,
) -> None:
    """
    실제 학습 전에 Script Argument를 검증한다.

    검증 항목
    ---------
    Epoch Count:

        정수

        1 이상

    Checkpoint:

        .pt 또는 .pth

        Directory가 아님

    History:

        .json

        Directory가 아님
    """
    if (
        isinstance(
            epoch_count,
            bool,
        )
        or not isinstance(
            epoch_count,
            int,
        )
    ):
        raise TypeError(
            "epoch_count must be an integer. "
            f"Received type: "
            f"{type(epoch_count).__name__}."
        )

    if epoch_count <= 0:
        raise ValueError(
            "epoch_count must be greater than 0. "
            f"Received value: {epoch_count}."
        )

    if (
        checkpoint_path.suffix.lower()
        not in {
            ".pt",
            ".pth",
        }
    ):
        raise ValueError(
            "checkpoint_path must use a .pt or .pth extension. "
            f"Received path: {checkpoint_path}."
        )

    if (
        checkpoint_path.exists()
        and checkpoint_path.is_dir()
    ):
        raise ValueError(
            "checkpoint_path must point to a file, "
            "not a directory. "
            f"Received path: {checkpoint_path}."
        )

    if (
        history_path.suffix.lower()
        != ".json"
    ):
        raise ValueError(
            "history_path must use a .json extension. "
            f"Received path: {history_path}."
        )

    if (
        history_path.exists()
        and history_path.is_dir()
    ):
        raise ValueError(
            "history_path must point to a file, "
            "not a directory. "
            f"Received path: {history_path}."
        )


# =============================================================================
# Reproducibility Validation
# =============================================================================


def validate_random_seed_alignment() -> None:
    """
    Dataset Split·DataLoader·Global Seed가 모두 같은지 확인한다.

    현재 기대값
    -------------
    Dataset Split:

        42

    DataLoader:

        42

    Global Random Seed:

        42

    왜 필요한가
    -----------
    서로 다른 Seed를 사용하면 다음 재현성이 분리될 수 있다.

        Train·Validation Split

        Train Shuffle

        Model 초기 Weight

    현재 프로젝트에서는 모두 42로 통일한다.
    """
    random_seed_values = {
        "dataset_split": (
            DATASET_SPLIT_RANDOM_SEED
        ),
        "data_loader": (
            DATA_LOADER_RANDOM_SEED
        ),
        "global": (
            DEFAULT_RANDOM_SEED
        ),
    }

    if (
        len(
            set(
                random_seed_values.values()
            )
        )
        != 1
    ):
        raise ValueError(
            "dataset split, data loader, and global "
            "random seeds must match. "
            f"Received: {random_seed_values}."
        )


# =============================================================================
# Data Pipeline Validation
# =============================================================================


def validate_vision_data_loaders(
    data_loaders: VisionDataLoaders,
) -> None:
    """
    실제 Vision Dataset·DataLoader 구성을 검증한다.

    검증 항목
    ---------
    Train Sample:

        5,306

    Validation Sample:

        1,327

    Test Sample:

        715

    Train Batch:

        166

    Validation Batch:

        42

    Test Batch:

        23

    Sampler:

        Train:

            RandomSampler

        Validation:

            SequentialSampler

        Test:

            SequentialSampler

    주의
    ----
    실제 Batch를 미리 꺼내지 않는다.

    Train DataLoader의 Random Generator 상태를
    학습 전에 불필요하게 변경하지 않기 위해
    Dataset·Loader 길이와 Sampler만 검증한다.
    """
    if not isinstance(
        data_loaders,
        VisionDataLoaders,
    ):
        raise TypeError(
            "data_loaders must be a VisionDataLoaders instance. "
            f"Received type: "
            f"{type(data_loaders).__name__}."
        )

    actual_counts = {
        "train_samples": len(
            data_loaders.train_dataset
        ),
        "validation_samples": len(
            data_loaders.validation_dataset
        ),
        "test_samples": len(
            data_loaders.test_dataset
        ),
        "train_batches": len(
            data_loaders.train_loader
        ),
        "validation_batches": len(
            data_loaders.validation_loader
        ),
        "test_batches": len(
            data_loaders.test_loader
        ),
    }

    expected_counts = {
        "train_samples": (
            EXPECTED_TRAIN_SAMPLE_COUNT
        ),
        "validation_samples": (
            EXPECTED_VALIDATION_SAMPLE_COUNT
        ),
        "test_samples": (
            EXPECTED_TEST_SAMPLE_COUNT
        ),
        "train_batches": (
            EXPECTED_TRAIN_BATCH_COUNT
        ),
        "validation_batches": (
            EXPECTED_VALIDATION_BATCH_COUNT
        ),
        "test_batches": (
            EXPECTED_TEST_BATCH_COUNT
        ),
    }

    if actual_counts != expected_counts:
        raise ValueError(
            "real Vision Dataset or DataLoader counts "
            "do not match the verified Day 2 configuration. "
            f"Expected: {expected_counts}. "
            f"Received: {actual_counts}."
        )

    # Day 2에 구현한 기존 검증 함수를 그대로 재사용한다.
    validate_sampler_types(
        data_loaders=data_loaders,
    )


# =============================================================================
# Model·Loss·Optimizer Validation
# =============================================================================


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    requires_grad=True인 Model Parameter 수를 계산한다.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_optimizer_parameters(
    optimizer: Optimizer,
) -> int:
    """
    Optimizer에 등록된 전체 Scalar Parameter 수를 계산한다.
    """
    return sum(
        parameter.numel()
        for parameter_group in (
            optimizer.param_groups
        )
        for parameter in (
            parameter_group["params"]
        )
    )


def validate_training_components(
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> None:
    """
    실제 학습에 사용할 Model·Loss·Optimizer를 검증한다.

    검증 항목
    ---------
    Model:

        CNNBaseline

    Model Device:

        요청 Device와 일치

    Trainable Parameter:

        6,065

    Loss:

        BCEWithLogitsLoss

    Optimizer:

        Adam

    Optimizer Parameter:

        6,065

    Model Parameter와 Optimizer Parameter:

        같은 객체
    """
    if not isinstance(
        model,
        CNNBaseline,
    ):
        raise TypeError(
            "model must be a CNNBaseline for the Day 3 run. "
            f"Received type: "
            f"{type(model).__name__}."
        )

    model_parameters = list(
        model.parameters()
    )

    if not model_parameters:
        raise ValueError(
            "CNNBaseline must contain parameters."
        )

    for parameter in model_parameters:
        if parameter.device != device:
            raise ValueError(
                "all CNNBaseline parameters must be "
                "on the configured device. "
                f"Configured device: {device}. "
                f"Found: {parameter.device}."
            )

    trainable_parameter_count = (
        count_trainable_parameters(
            model=model,
        )
    )

    if (
        trainable_parameter_count
        != EXPECTED_CNN_PARAMETER_COUNT
    ):
        raise ValueError(
            "CNNBaseline trainable parameter count changed. "
            f"Expected: "
            f"{EXPECTED_CNN_PARAMETER_COUNT:,}. "
            f"Received: "
            f"{trainable_parameter_count:,}."
        )

    if (
        loss_function
        .__class__
        .__name__
        != "BCEWithLogitsLoss"
    ):
        raise TypeError(
            "Day 3 loss_function must be "
            "BCEWithLogitsLoss. "
            f"Received: "
            f"{loss_function.__class__.__name__}."
        )

    if (
        optimizer
        .__class__
        .__name__
        != "Adam"
    ):
        raise TypeError(
            "Day 3 optimizer must be Adam. "
            f"Received: "
            f"{optimizer.__class__.__name__}."
        )

    optimizer_parameter_count = (
        count_optimizer_parameters(
            optimizer=optimizer,
        )
    )

    if (
        optimizer_parameter_count
        != EXPECTED_CNN_PARAMETER_COUNT
    ):
        raise ValueError(
            "Adam Optimizer parameter count does not "
            "match CNNBaseline. "
            f"Expected: "
            f"{EXPECTED_CNN_PARAMETER_COUNT:,}. "
            f"Received: "
            f"{optimizer_parameter_count:,}."
        )

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    optimizer_parameter_ids = {
        id(parameter)
        for parameter_group in (
            optimizer.param_groups
        )
        for parameter in (
            parameter_group["params"]
        )
    }

    if (
        model_parameter_ids
        != optimizer_parameter_ids
    ):
        raise ValueError(
            "Adam Optimizer must reference the same "
            "trainable Parameter objects as CNNBaseline."
        )


# =============================================================================
# Pre-training Console Output
# =============================================================================


def print_execution_configuration(
    settings: ReproducibilitySettings,
    data_loaders: VisionDataLoaders,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch_count: int,
    checkpoint_path: Path,
    history_path: Path,
    validate_only: bool,
) -> None:
    """
    실제 학습 전 환경·데이터·모델·출력 설정을 출력한다.
    """
    trainable_parameter_count = (
        count_trainable_parameters(
            model=model,
        )
    )

    optimizer_parameter_count = (
        count_optimizer_parameters(
            optimizer=optimizer,
        )
    )

    parameter_group = (
        optimizer.param_groups[0]
    )

    print()
    print("=" * 100)
    print("DAY 3 - CNN BASELINE TRAINING")
    print("=" * 100)

    print()
    print("[PROJECT]")
    print(
        f"English name               : "
        f"{PROJECT_NAME}"
    )
    print(
        f"Korean name                : "
        f"{PROJECT_NAME_KOREAN}"
    )
    print(
        f"Project root               : "
        f"{PROJECT_ROOT}"
    )
    print(
        f"Run name                   : "
        f"{RUN_NAME}"
    )
    print(
        f"Validate only              : "
        f"{validate_only}"
    )

    print()
    print("[ENVIRONMENT]")
    print(
        f"Python                     : "
        f"{platform.python_version()}"
    )
    print(
        f"PyTorch                    : "
        f"{torch.__version__}"
    )
    print(
        f"Device                     : "
        f"{device}"
    )
    print(
        f"CUDA available             : "
        f"{torch.cuda.is_available()}"
    )

    print()
    print("[REPRODUCIBILITY]")
    print(
        f"Random seed                : "
        f"{settings.seed}"
    )
    print(
        f"Deterministic algorithms   : "
        f"{settings.deterministic_algorithms}"
    )
    print(
        f"CUDA seed applied          : "
        f"{settings.cuda_seed_applied}"
    )

    print()
    print("[DATA CONFIGURATION]")
    print(
        f"Image size                 : "
        f"{IMAGE_SIZE}"
    )
    print(
        f"Validation ratio           : "
        f"{VALIDATION_RATIO}"
    )
    print(
        f"Batch size                 : "
        f"{BATCH_SIZE}"
    )
    print(
        f"Num workers                : "
        f"{NUM_WORKERS}"
    )
    print(
        f"Pin memory                 : "
        f"{PIN_MEMORY}"
    )
    print(
        f"Drop last                  : "
        f"{DROP_LAST}"
    )
    print(
        f"Persistent workers         : "
        f"{PERSISTENT_WORKERS}"
    )
    print(
        f"Class mapping              : "
        f"{CLASS_TO_INDEX}"
    )
    print(
        f"Display classes            : "
        f"{INDEX_TO_CLASS_NAME}"
    )

    print()
    print("[REAL DATA]")
    print(
        f"Train samples              : "
        f"{len(data_loaders.train_dataset):,}"
    )
    print(
        f"Validation samples         : "
        f"{len(data_loaders.validation_dataset):,}"
    )
    print(
        f"Test samples               : "
        f"{len(data_loaders.test_dataset):,}"
    )
    print(
        f"Train batches              : "
        f"{len(data_loaders.train_loader):,}"
    )
    print(
        f"Validation batches         : "
        f"{len(data_loaders.validation_loader):,}"
    )
    print(
        f"Test batches               : "
        f"{len(data_loaders.test_loader):,}"
    )

    print()
    print("[MODEL]")
    print(
        f"Model                      : "
        f"{model.__class__.__name__}"
    )
    print(
        f"Trainable parameters       : "
        f"{trainable_parameter_count:,}"
    )
    print(
        f"Loss function              : "
        f"{loss_function.__class__.__name__}"
    )
    print(
        f"Optimizer                  : "
        f"{optimizer.__class__.__name__}"
    )
    print(
        f"Optimizer parameters       : "
        f"{optimizer_parameter_count:,}"
    )

    print()
    print("[TRAINING CONFIGURATION]")
    print(
        f"Epoch count                : "
        f"{epoch_count}"
    )
    print(
        f"Learning rate              : "
        f"{parameter_group['lr']}"
    )
    print(
        f"Weight decay               : "
        f"{parameter_group['weight_decay']}"
    )
    print(
        f"Classification threshold   : "
        f"{DEFAULT_CLASSIFICATION_THRESHOLD}"
    )
    print(
        f"Best selection metric      : "
        f"{BEST_MODEL_SELECTION_METRIC}"
    )

    print()
    print("[OUTPUT]")
    print(
        f"Checkpoint                 : "
        f"{checkpoint_path}"
    )
    print(
        f"Training history           : "
        f"{history_path}"
    )


# =============================================================================
# Training Result Validation
# =============================================================================


def validate_training_result(
    training_result: TrainingResult,
    epoch_count: int,
    checkpoint_path: Path,
) -> None:
    """
    실제 TrainingResult의 History·Best Result를 검증한다.

    검증 항목
    ---------
    History 길이:

        설정 Epoch 수와 일치

    Epoch 번호:

        1부터 순서대로 증가

    모든 Train Result:

        Sample 5,306

        Batch 166

    모든 Validation Result:

        Sample 1,327

        Batch 42

    Best Epoch:

        1~Epoch Count

    Best Validation Loss:

        History 최소값

    Best Validation Accuracy:

        0~1

    Checkpoint:

        요청 경로와 일치

        실제 파일 존재
    """
    if not isinstance(
        training_result,
        TrainingResult,
    ):
        raise TypeError(
            "training_result must be a TrainingResult. "
            f"Received type: "
            f"{type(training_result).__name__}."
        )

    if (
        len(
            training_result.history
        )
        != epoch_count
    ):
        raise ValueError(
            "Training History length must match epoch_count. "
            f"Expected: {epoch_count}. "
            f"Received: "
            f"{len(training_result.history)}."
        )

    expected_epoch_numbers = list(
        range(
            1,
            epoch_count + 1,
        )
    )

    actual_epoch_numbers = [
        history_item.epoch_number
        for history_item in (
            training_result.history
        )
    ]

    if (
        actual_epoch_numbers
        != expected_epoch_numbers
    ):
        raise ValueError(
            "Training History epoch numbers are invalid. "
            f"Expected: {expected_epoch_numbers}. "
            f"Received: {actual_epoch_numbers}."
        )

    for history_item in (
        training_result.history
    ):
        train_result = (
            history_item.train_result
        )

        validation_result = (
            history_item.validation_result
        )

        if (
            train_result.sample_count
            != EXPECTED_TRAIN_SAMPLE_COUNT
        ):
            raise ValueError(
                "Train sample count in Training History "
                "does not match the real Train Dataset. "
                f"Epoch: {history_item.epoch_number}. "
                f"Received: {train_result.sample_count}."
            )

        if (
            train_result.batch_count
            != EXPECTED_TRAIN_BATCH_COUNT
        ):
            raise ValueError(
                "Train batch count in Training History "
                "does not match the real Train DataLoader. "
                f"Epoch: {history_item.epoch_number}. "
                f"Received: {train_result.batch_count}."
            )

        if (
            validation_result.sample_count
            != EXPECTED_VALIDATION_SAMPLE_COUNT
        ):
            raise ValueError(
                "Validation sample count in Training History "
                "does not match the real Validation Dataset. "
                f"Epoch: {history_item.epoch_number}. "
                f"Received: "
                f"{validation_result.sample_count}."
            )

        if (
            validation_result.batch_count
            != EXPECTED_VALIDATION_BATCH_COUNT
        ):
            raise ValueError(
                "Validation batch count in Training History "
                "does not match the real Validation DataLoader. "
                f"Epoch: {history_item.epoch_number}. "
                f"Received: "
                f"{validation_result.batch_count}."
            )

    if not (
        1
        <= training_result.best_epoch_number
        <= epoch_count
    ):
        raise ValueError(
            "best_epoch_number must be contained in "
            "the configured Epoch range. "
            f"Received: "
            f"{training_result.best_epoch_number}."
        )

    validation_losses = [
        history_item
        .validation_result
        .average_loss
        for history_item in (
            training_result.history
        )
    ]

    expected_best_validation_loss = min(
        validation_losses
    )

    if not math.isclose(
        training_result.best_validation_loss,
        expected_best_validation_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "best_validation_loss must be the minimum "
            "Validation Loss in Training History. "
            f"Expected: "
            f"{expected_best_validation_loss}. "
            f"Received: "
            f"{training_result.best_validation_loss}."
        )

    if not (
        0.0
        <= training_result.best_validation_accuracy
        <= 1.0
    ):
        raise ValueError(
            "best_validation_accuracy must be "
            "between 0 and 1. "
            f"Received: "
            f"{training_result.best_validation_accuracy}."
        )

    if (
        training_result.checkpoint_path
        != checkpoint_path
    ):
        raise ValueError(
            "TrainingResult checkpoint path does not "
            "match the configured path. "
            f"Expected: {checkpoint_path}. "
            f"Received: "
            f"{training_result.checkpoint_path}."
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "Best Model Checkpoint was not created. "
            f"Expected path: {checkpoint_path}."
        )


# =============================================================================
# Checkpoint Validation
# =============================================================================


def load_and_validate_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    loss_function: nn.Module,
    training_result: TrainingResult,
    epoch_count: int,
) -> dict[str, object]:
    """
    저장된 Best Checkpoint를 다시 읽고 Metadata·State를 검증한다.

    출력
    ----
    dict[str, object]

        검증 완료된 Checkpoint
    """
    checkpoint_object = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint_object,
        dict,
    ):
        raise TypeError(
            "saved checkpoint must be a dictionary. "
            f"Received type: "
            f"{type(checkpoint_object).__name__}."
        )

    checkpoint: dict[str, object] = (
        checkpoint_object
    )

    required_keys = {
        "checkpoint_version",
        "model_name",
        "model_module",
        "loss_function_name",
        "optimizer_name",
        "epoch_number",
        "configured_epoch_count",
        "classification_threshold",
        "best_model_selection_metric",
        "model_state_dict",
        "optimizer_state_dict",
        "train_result",
        "validation_result",
    }

    missing_keys = (
        required_keys
        - set(
            checkpoint.keys()
        )
    )

    if missing_keys:
        raise KeyError(
            "saved checkpoint is missing required keys. "
            f"Missing: {sorted(missing_keys)}."
        )

    if (
        checkpoint["checkpoint_version"]
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "checkpoint_version does not match "
            "the current training pipeline. "
            f"Expected: {CHECKPOINT_VERSION}. "
            f"Received: "
            f"{checkpoint['checkpoint_version']}."
        )

    if (
        checkpoint["model_name"]
        != model.__class__.__name__
    ):
        raise ValueError(
            "checkpoint model_name does not match "
            "the actual Model. "
            f"Expected: "
            f"{model.__class__.__name__}. "
            f"Received: "
            f"{checkpoint['model_name']}."
        )

    if (
        checkpoint["loss_function_name"]
        != loss_function.__class__.__name__
    ):
        raise ValueError(
            "checkpoint loss_function_name does not "
            "match the actual Loss Function."
        )

    if (
        checkpoint["optimizer_name"]
        != optimizer.__class__.__name__
    ):
        raise ValueError(
            "checkpoint optimizer_name does not "
            "match the actual Optimizer."
        )

    if (
        checkpoint["configured_epoch_count"]
        != epoch_count
    ):
        raise ValueError(
            "checkpoint configured_epoch_count does not "
            "match the actual training configuration. "
            f"Expected: {epoch_count}. "
            f"Received: "
            f"{checkpoint['configured_epoch_count']}."
        )

    if (
        checkpoint[
            "best_model_selection_metric"
        ]
        != BEST_MODEL_SELECTION_METRIC
    ):
        raise ValueError(
            "checkpoint Best Model selection metric is invalid."
        )

    if (
        checkpoint["epoch_number"]
        != training_result.best_epoch_number
    ):
        raise ValueError(
            "checkpoint epoch_number does not match "
            "TrainingResult.best_epoch_number. "
            f"Expected: "
            f"{training_result.best_epoch_number}. "
            f"Received: "
            f"{checkpoint['epoch_number']}."
        )

    checkpoint_threshold = float(
        checkpoint[
            "classification_threshold"
        ]
    )

    if not math.isclose(
        checkpoint_threshold,
        DEFAULT_CLASSIFICATION_THRESHOLD,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "checkpoint Classification Threshold is invalid. "
            f"Expected: "
            f"{DEFAULT_CLASSIFICATION_THRESHOLD}. "
            f"Received: "
            f"{checkpoint_threshold}."
        )

    validation_result_object = checkpoint[
        "validation_result"
    ]

    if not isinstance(
        validation_result_object,
        dict,
    ):
        raise TypeError(
            "checkpoint validation_result must be a dictionary."
        )

    checkpoint_validation_loss = float(
        validation_result_object[
            "average_loss"
        ]
    )

    checkpoint_validation_accuracy = float(
        validation_result_object[
            "accuracy"
        ]
    )

    if not math.isclose(
        checkpoint_validation_loss,
        training_result.best_validation_loss,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "checkpoint Validation Loss does not match "
            "TrainingResult."
        )

    if not math.isclose(
        checkpoint_validation_accuracy,
        training_result.best_validation_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "checkpoint Validation Accuracy does not "
            "match TrainingResult."
        )

    model_state_object = checkpoint[
        "model_state_dict"
    ]

    if not isinstance(
        model_state_object,
        dict,
    ):
        raise TypeError(
            "checkpoint model_state_dict must be a dictionary."
        )

    if (
        set(
            model_state_object.keys()
        )
        != set(
            model.state_dict().keys()
        )
    ):
        raise ValueError(
            "checkpoint Model State keys do not match "
            "CNNBaseline State keys."
        )

    optimizer_state_object = checkpoint[
        "optimizer_state_dict"
    ]

    if not isinstance(
        optimizer_state_object,
        dict,
    ):
        raise TypeError(
            "checkpoint optimizer_state_dict "
            "must be a dictionary."
        )

    if not {
        "state",
        "param_groups",
    }.issubset(
        optimizer_state_object.keys()
    ):
        raise ValueError(
            "checkpoint Optimizer State is missing "
            "state or param_groups."
        )

    return checkpoint


# =============================================================================
# Training History JSON
# =============================================================================


def build_training_history_payload(
    settings: ReproducibilitySettings,
    data_loaders: VisionDataLoaders,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch_count: int,
    training_duration_seconds: float,
    training_result: TrainingResult,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, object]:
    """
    TrainingResult를 JSON 저장 가능한 Dictionary로 변환한다.

    저장 범위
    ---------
    Project Metadata

    Environment

    Reproducibility

    Dataset·DataLoader

    Model

    Training Configuration

    Best Result

    Epoch History

    Artifact Paths
    """
    parameter_group = (
        optimizer.param_groups[0]
    )

    history = []

    for history_item in (
        training_result.history
    ):
        history.append(
            {
                "epoch_number": (
                    history_item.epoch_number
                ),
                "train_loss": (
                    history_item
                    .train_result
                    .average_loss
                ),
                "train_accuracy": (
                    history_item
                    .train_result
                    .accuracy
                ),
                "train_accuracy_percent": (
                    history_item
                    .train_result
                    .accuracy
                    * 100.0
                ),
                "validation_loss": (
                    history_item
                    .validation_result
                    .average_loss
                ),
                "validation_accuracy": (
                    history_item
                    .validation_result
                    .accuracy
                ),
                "validation_accuracy_percent": (
                    history_item
                    .validation_result
                    .accuracy
                    * 100.0
                ),
            }
        )

    payload: dict[str, object] = {
        "project": {
            "name": PROJECT_NAME,
            "name_korean": (
                PROJECT_NAME_KOREAN
            ),
            "run_name": RUN_NAME,
        },
        "environment": {
            "python_version": (
                platform.python_version()
            ),
            "python_executable": (
                sys.executable
            ),
            "torch_version": (
                torch.__version__
            ),
            "device": str(
                device
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
        },
        "reproducibility": {
            "random_seed": (
                settings.seed
            ),
            "deterministic_algorithms": (
                settings
                .deterministic_algorithms
            ),
            "cuda_seed_applied": (
                settings
                .cuda_seed_applied
            ),
        },
        "data": {
            "image_size": list(
                IMAGE_SIZE
            ),
            "class_to_index": (
                CLASS_TO_INDEX
            ),
            "index_to_class_name": {
                str(
                    label
                ): class_name
                for (
                    label,
                    class_name,
                ) in (
                    INDEX_TO_CLASS_NAME.items()
                )
            },
            "validation_ratio": (
                VALIDATION_RATIO
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "num_workers": (
                NUM_WORKERS
            ),
            "pin_memory": (
                PIN_MEMORY
            ),
            "drop_last": (
                DROP_LAST
            ),
            "persistent_workers": (
                PERSISTENT_WORKERS
            ),
            "train_sample_count": len(
                data_loaders.train_dataset
            ),
            "validation_sample_count": len(
                data_loaders.validation_dataset
            ),
            "test_sample_count": len(
                data_loaders.test_dataset
            ),
            "train_batch_count": len(
                data_loaders.train_loader
            ),
            "validation_batch_count": len(
                data_loaders.validation_loader
            ),
            "test_batch_count": len(
                data_loaders.test_loader
            ),
            "test_used_during_training": (
                False
            ),
        },
        "model": {
            "name": (
                model.__class__.__name__
            ),
            "module": (
                model.__class__.__module__
            ),
            "trainable_parameter_count": (
                count_trainable_parameters(
                    model=model,
                )
            ),
            "output": (
                "single_binary_raw_logit"
            ),
            "positive_class": (
                "DEFECT"
            ),
        },
        "training_configuration": {
            "epoch_count": (
                epoch_count
            ),
            "loss_function": (
                loss_function
                .__class__
                .__name__
            ),
            "optimizer": (
                optimizer
                .__class__
                .__name__
            ),
            "learning_rate": float(
                parameter_group[
                    "lr"
                ]
            ),
            "weight_decay": float(
                parameter_group[
                    "weight_decay"
                ]
            ),
            "classification_threshold": (
                DEFAULT_CLASSIFICATION_THRESHOLD
            ),
            "best_model_selection_metric": (
                BEST_MODEL_SELECTION_METRIC
            ),
        },
        "best_result": {
            "epoch_number": (
                training_result
                .best_epoch_number
            ),
            "validation_loss": (
                training_result
                .best_validation_loss
            ),
            "validation_accuracy": (
                training_result
                .best_validation_accuracy
            ),
            "validation_accuracy_percent": (
                training_result
                .best_validation_accuracy
                * 100.0
            ),
        },
        "runtime": {
            "training_duration_seconds": (
                training_duration_seconds
            ),
        },
        "artifacts": {
            "checkpoint_path": (
                format_project_relative_path(
                    checkpoint_path
                )
            ),
            "training_history_path": (
                format_project_relative_path(
                    history_path
                )
            ),
        },
        "history": history,
    }

    return payload


def write_json_atomically(
    payload: dict[str, object],
    output_path: Path,
) -> None:
    """
    Training History JSON을 임시 파일에 먼저 저장한 뒤 교체한다.

    저장 흐름
    ---------
    day3_cnn_baseline_training_history.json.tmp

    ->

    JSON 저장 완료

    ->

    day3_cnn_baseline_training_history.json

    이유
    ----
    저장 도중 오류가 발생할 경우
    기존 정상 Artifact가 손상될 가능성을 줄인다.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_name(
            f"{output_path.name}.tmp"
        )
    )

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

            output_file.write(
                "\n"
            )

        temporary_path.replace(
            output_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# =============================================================================
# Completion Output
# =============================================================================


def print_training_completion(
    training_result: TrainingResult,
    training_duration_seconds: float,
    checkpoint_path: Path,
    history_path: Path,
) -> None:
    """
    실제 CNN Baseline 학습 완료 결과를 출력한다.
    """
    print()
    print("=" * 100)
    print("DAY 3 - CNN BASELINE TRAINING COMPLETED")
    print("=" * 100)

    print()
    print("[BEST RESULT]")
    print(
        f"Best epoch                 : "
        f"{training_result.best_epoch_number}"
    )
    print(
        f"Best validation loss       : "
        f"{training_result.best_validation_loss:.6f}"
    )
    print(
        f"Best validation accuracy   : "
        f"{training_result.best_validation_accuracy:.6f}"
    )
    print(
        f"Best validation percent    : "
        f"{training_result.best_validation_accuracy * 100:.2f}%"
    )

    print()
    print("[RUNTIME]")
    print(
        f"Training seconds           : "
        f"{training_duration_seconds:.2f}"
    )
    print(
        f"Training minutes           : "
        f"{training_duration_seconds / 60.0:.2f}"
    )

    print()
    print("[ARTIFACTS]")
    print(
        f"Best checkpoint            : "
        f"{checkpoint_path}"
    )
    print(
        f"Training history JSON      : "
        f"{history_path}"
    )
    print(
        f"Checkpoint exists          : "
        f"{checkpoint_path.is_file()}"
    )
    print(
        f"History JSON exists        : "
        f"{history_path.is_file()}"
    )

    print()
    print(
        "[PASS] Day 3 CNN Baseline real training"
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """
    Day 3 CNN Baseline 실제 학습 실행 Entry Point.

    실행 순서
    ---------
    1. Command Line Argument 해석
    2. 출력 경로 정규화
    3. 실행 설정 검증
    4. Seed 설정 일치 확인
    5. Global Random Seed 설정
    6. CPU Device 생성
    7. 실제 Vision DataLoader 생성
    8. Dataset·DataLoader 검증
    9. CNNBaseline 생성
    10. BCEWithLogitsLoss 생성
    11. Adam 생성
    12. Model·Loss·Optimizer 검증
    13. 전체 설정 출력
    14. --validate-only이면 종료
    15. 실제 여러 Epoch 학습
    16. TrainingResult 검증
    17. Best Checkpoint 검증
    18. Training History JSON 저장
    19. 최종 결과 출력
    """
    arguments = parse_arguments()

    checkpoint_path = (
        resolve_project_path(
            arguments.checkpoint_path
        )
    )

    history_path = (
        resolve_project_path(
            arguments.history_path
        )
    )

    validate_execution_arguments(
        epoch_count=(
            arguments.epoch_count
        ),
        checkpoint_path=(
            checkpoint_path
        ),
        history_path=(
            history_path
        ),
    )

    validate_random_seed_alignment()

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    #
    # DataLoader와 Model을 만들기 전에 Seed를 먼저 설정한다.
    #
    # 그래야:
    #
    #     Model 초기 Weight
    #
    #     Train DataLoader Shuffle
    #
    # 를 동일 환경에서 최대한 재현할 수 있다.
    settings = set_global_random_seed(
        seed=DEFAULT_RANDOM_SEED,
        deterministic_algorithms=False,
    )

    device = torch.device(
        settings.device
    )

    # ---------------------------------------------------------
    # Real Vision Data Pipeline
    # ---------------------------------------------------------
    data_loaders = (
        create_vision_data_loaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=DROP_LAST,
            persistent_workers=(
                PERSISTENT_WORKERS
            ),
            random_seed=(
                DEFAULT_RANDOM_SEED
            ),
        )
    )

    validate_vision_data_loaders(
        data_loaders=data_loaders,
    )

    # ---------------------------------------------------------
    # CNN Baseline
    # ---------------------------------------------------------
    model = CNNBaseline()

    model = model.to(
        device
    )

    # ---------------------------------------------------------
    # Binary Classification Loss
    # ---------------------------------------------------------
    loss_function = (
        create_binary_classification_loss()
    )

    # ---------------------------------------------------------
    # Adam Optimizer
    # ---------------------------------------------------------
    optimizer = create_optimizer(
        model=model,
        learning_rate=(
            DEFAULT_LEARNING_RATE
        ),
        weight_decay=(
            DEFAULT_WEIGHT_DECAY
        ),
    )

    validate_training_components(
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
    )

    print_execution_configuration(
        settings=settings,
        data_loaders=data_loaders,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epoch_count=(
            arguments.epoch_count
        ),
        checkpoint_path=(
            checkpoint_path
        ),
        history_path=(
            history_path
        ),
        validate_only=(
            arguments.validate_only
        ),
    )

    # ---------------------------------------------------------
    # Structure Validation Only
    # ---------------------------------------------------------
    if arguments.validate_only:
        print()
        print("=" * 100)
        print(
            "[PASS] Day 3 CNN Baseline "
            "training structure validation"
        )
        print("=" * 100)

        return

    # ---------------------------------------------------------
    # Real Training
    # ---------------------------------------------------------
    training_started_at = (
        time.perf_counter()
    )

    training_result = run_training(
        model=model,
        train_loader=(
            data_loaders.train_loader
        ),
        validation_loader=(
            data_loaders.validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device=device,
        epoch_count=(
            arguments.epoch_count
        ),
        classification_threshold=(
            DEFAULT_CLASSIFICATION_THRESHOLD
        ),
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=(
            not arguments.quiet
        ),
    )

    training_duration_seconds = (
        time.perf_counter()
        - training_started_at
    )

    # ---------------------------------------------------------
    # Training Result Validation
    # ---------------------------------------------------------
    validate_training_result(
        training_result=(
            training_result
        ),
        epoch_count=(
            arguments.epoch_count
        ),
        checkpoint_path=(
            checkpoint_path
        ),
    )

    # ---------------------------------------------------------
    # Best Checkpoint Validation
    # ---------------------------------------------------------
    _ = load_and_validate_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        training_result=(
            training_result
        ),
        epoch_count=(
            arguments.epoch_count
        ),
    )

    # ---------------------------------------------------------
    # Training History JSON
    # ---------------------------------------------------------
    history_payload = (
        build_training_history_payload(
            settings=settings,
            data_loaders=data_loaders,
            model=model,
            loss_function=(
                loss_function
            ),
            optimizer=optimizer,
            device=device,
            epoch_count=(
                arguments.epoch_count
            ),
            training_duration_seconds=(
                training_duration_seconds
            ),
            training_result=(
                training_result
            ),
            checkpoint_path=(
                checkpoint_path
            ),
            history_path=(
                history_path
            ),
        )
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

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print_training_completion(
        training_result=(
            training_result
        ),
        training_duration_seconds=(
            training_duration_seconds
        ),
        checkpoint_path=(
            checkpoint_path
        ),
        history_path=(
            history_path
        ),
    )


if __name__ == "__main__":
    main()