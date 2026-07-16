"""
Multi-epoch training pipeline and best-model checkpoint utilities.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 이미지 분류 모델에 대해
여러 Epoch의 Train·Validation을 반복하고, Epoch별 결과를 기록하며,
Validation Loss가 가장 낮은 Best Model을 Checkpoint로 저장한다.

현재 모델
---------
CNNBaseline

향후 재사용 대상
----------------
ResNet18 Transfer Learning Model

현재 클래스 정의
----------------
0 = NORMAL

1 = DEFECT

Positive Class:

    DEFECT

현재 기본 학습 설정
------------------
Epoch Count:

    5

Classification Threshold:

    0.5

Best Model Selection:

    Lowest Validation Loss

Default Checkpoint:

    models/checkpoints/cnn_baseline_best.pt

전체 학습 흐름
-------------
Model 생성

-> Device 이동

-> Loss Function 생성

-> Optimizer 생성

-> run_training()

-> Epoch 1

    -> train_one_epoch()

    -> validate_one_epoch()

    -> Epoch History 기록

    -> Best Validation Loss 확인

    -> Best이면 Checkpoint 저장

-> Epoch 2

-> ...

-> 마지막 Epoch

-> TrainingResult 반환

중요
----
Best Model은 마지막 Epoch Model과 다를 수 있다.

예:

    Epoch 1 Validation Loss:

        0.60

    Epoch 2 Validation Loss:

        0.48

    Epoch 3 Validation Loss:

        0.42

    Epoch 4 Validation Loss:

        0.46

    Epoch 5 Validation Loss:

        0.51

이 경우:

    마지막 Epoch:

        5

    Best Epoch:

        3

Checkpoint에는 Epoch 3 Model이 저장된다.

현재 run_training() 실행이 끝난 후 메모리에 남아 있는 Model은
마지막 Epoch의 Model이다.

Best Model Weight를 메모리 Model에 다시 불러오는 기능은
향후 Checkpoint Loading 단계에서 별도로 구현한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.epoch_runner import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    EpochResult,
    train_one_epoch,
    validate_one_epoch,
)


# =============================================================================
# Default Training Configuration
# =============================================================================

# 첫 CNN Baseline 실제 학습의 기본 Epoch 수다.
#
# 현재 환경:
#
#     CPU
#
#     Intel Core i5-1035G7
#
#     CUDA:
#
#         False
#
# 현재 Train Dataset:
#
#     5,306장
#
# 현재 Validation Dataset:
#
#     1,327장
#
# 실제 한 Epoch 학습 시간을 아직 측정하지 않았으므로
# 첫 Baseline은 5 Epoch로 시작한다.
DEFAULT_EPOCH_COUNT = 5


# CNN Baseline Best Model의 기본 저장 경로다.
#
# Validation Loss가 기존 Best Loss보다 낮아질 때마다
# 같은 경로에 새로운 Best Checkpoint를 저장한다.
DEFAULT_CNN_CHECKPOINT_PATH = Path(
    "models/checkpoints/cnn_baseline_best.pt"
)


# Checkpoint 구조 Version이다.
#
# 향후 Checkpoint Key 구조가 변경되면
# Version을 증가시켜 저장 형식을 구분할 수 있다.
CHECKPOINT_VERSION = 1


# Best Model 선택 기준의 이름이다.
#
# 현재:
#
#     Validation Loss 최소화
#
# 향후 다른 기준을 실험하더라도
# Checkpoint에 어떤 기준으로 Best Model을 선택했는지
# 명확하게 기록할 수 있다.
BEST_MODEL_SELECTION_METRIC = "validation_loss"


# =============================================================================
# Epoch History
# =============================================================================


@dataclass(frozen=True)
class EpochHistoryItem:
    """
    한 Epoch의 Train·Validation 결과 기록.

    왜 필요한가
    -----------
    여러 Epoch의 결과를 순서대로 저장해야 다음 작업이 가능하다.

        Loss Curve

        Accuracy Curve

        학습 결과 표

        README 결과 기록

        Day 3 보고서

        CNN·ResNet18 비교

    필드
    ----
    epoch_number:
        현재 Epoch 번호

        시작:

            1

    train_result:
        train_one_epoch()의 결과

    validation_result:
        validate_one_epoch()의 결과

    출력 예
    -------
    EpochHistoryItem(
        epoch_number=1,
        train_result=EpochResult(
            average_loss=0.6500,
            accuracy=0.6500,
            sample_count=5306,
            batch_count=166,
        ),
        validation_result=EpochResult(
            average_loss=0.6000,
            accuracy=0.7100,
            sample_count=1327,
            batch_count=42,
        ),
    )
    """

    epoch_number: int

    train_result: EpochResult

    validation_result: EpochResult

    def __post_init__(self) -> None:
        """
        Epoch History 항목을 검증한다.

        검증 항목
        ---------
        Epoch Number:

            정수

            bool 아님

            1 이상

        Train Result:

            EpochResult

        Validation Result:

            EpochResult
        """
        if (
            isinstance(
                self.epoch_number,
                bool,
            )
            or not isinstance(
                self.epoch_number,
                Integral,
            )
        ):
            raise TypeError(
                "epoch_number must be an integer. "
                f"Received type: "
                f"{type(self.epoch_number).__name__}."
            )

        if self.epoch_number <= 0:
            raise ValueError(
                "epoch_number must be greater than 0. "
                f"Received value: {self.epoch_number}."
            )

        if not isinstance(
            self.train_result,
            EpochResult,
        ):
            raise TypeError(
                "train_result must be an EpochResult. "
                f"Received type: "
                f"{type(self.train_result).__name__}."
            )

        if not isinstance(
            self.validation_result,
            EpochResult,
        ):
            raise TypeError(
                "validation_result must be an EpochResult. "
                f"Received type: "
                f"{type(self.validation_result).__name__}."
            )


# =============================================================================
# Final Training Result
# =============================================================================


@dataclass(frozen=True)
class TrainingResult:
    """
    여러 Epoch 학습이 완료된 후 반환하는 최종 결과.

    왜 필요한가
    -----------
    Training Pipeline의 결과를 하나의 명확한 객체로 반환하면
    학습 이력·Best Epoch·Checkpoint 정보를 일관되게 사용할 수 있다.

    필드
    ----
    history:
        모든 Epoch 결과

        Tuple을 사용한다.

        학습 완료 후 과거 결과가 실수로 변경되는 것을 방지한다.

    best_epoch_number:
        Validation Loss가 가장 낮았던 Epoch 번호

    best_validation_loss:
        Best Epoch의 Validation Average Loss

    best_validation_accuracy:
        Best Epoch의 Validation Accuracy

        Best Model 선택 기준은 Accuracy가 아니라 Validation Loss다.

        Accuracy는 Best Epoch의 참고 Metric으로 함께 저장한다.

    checkpoint_path:
        Best Model Checkpoint 경로

    출력 예
    -------
    TrainingResult(
        history=(
            EpochHistoryItem(...),
            EpochHistoryItem(...),
        ),
        best_epoch_number=2,
        best_validation_loss=0.4521,
        best_validation_accuracy=0.8214,
        checkpoint_path=Path(
            "models/checkpoints/cnn_baseline_best.pt"
        ),
    )
    """

    history: tuple[EpochHistoryItem, ...]

    best_epoch_number: int

    best_validation_loss: float

    best_validation_accuracy: float

    checkpoint_path: Path

    def __post_init__(self) -> None:
        """
        최종 Training Result를 검증한다.

        검증 항목
        ---------
        History:

            Tuple

            한 개 이상의 Epoch

            Epoch 번호가 1부터 순서대로 존재

        Best Epoch:

            History 안에 존재

        Best Validation Loss:

            유한한 값

            0 이상

            Best Epoch 결과와 일치

        Best Validation Accuracy:

            유한한 값

            0~1

            Best Epoch 결과와 일치

        Checkpoint Path:

            pathlib.Path
        """
        if not isinstance(
            self.history,
            tuple,
        ):
            raise TypeError(
                "history must be a tuple of EpochHistoryItem objects. "
                f"Received type: {type(self.history).__name__}."
            )

        if not self.history:
            raise ValueError(
                "history must contain at least one epoch result."
            )

        for history_item in self.history:
            if not isinstance(
                history_item,
                EpochHistoryItem,
            ):
                raise TypeError(
                    "every history item must be an "
                    "EpochHistoryItem. "
                    f"Received type: "
                    f"{type(history_item).__name__}."
                )

        expected_epoch_numbers = tuple(
            range(
                1,
                len(self.history) + 1,
            )
        )

        actual_epoch_numbers = tuple(
            history_item.epoch_number
            for history_item in self.history
        )

        if (
            actual_epoch_numbers
            != expected_epoch_numbers
        ):
            raise ValueError(
                "history epoch numbers must start at 1 "
                "and increase sequentially. "
                f"Expected: {expected_epoch_numbers}. "
                f"Received: {actual_epoch_numbers}."
            )

        if (
            isinstance(
                self.best_epoch_number,
                bool,
            )
            or not isinstance(
                self.best_epoch_number,
                Integral,
            )
        ):
            raise TypeError(
                "best_epoch_number must be an integer. "
                f"Received type: "
                f"{type(self.best_epoch_number).__name__}."
            )

        if not (
            1
            <= self.best_epoch_number
            <= len(self.history)
        ):
            raise ValueError(
                "best_epoch_number must refer to an epoch "
                "contained in history. "
                f"Received value: {self.best_epoch_number}. "
                f"History size: {len(self.history)}."
            )

        if not math.isfinite(
            self.best_validation_loss,
        ):
            raise ValueError(
                "best_validation_loss must be finite. "
                f"Received value: "
                f"{self.best_validation_loss}."
            )

        if self.best_validation_loss < 0.0:
            raise ValueError(
                "best_validation_loss must be greater than "
                "or equal to 0. "
                f"Received value: "
                f"{self.best_validation_loss}."
            )

        if not math.isfinite(
            self.best_validation_accuracy,
        ):
            raise ValueError(
                "best_validation_accuracy must be finite. "
                f"Received value: "
                f"{self.best_validation_accuracy}."
            )

        if not (
            0.0
            <= self.best_validation_accuracy
            <= 1.0
        ):
            raise ValueError(
                "best_validation_accuracy must be between "
                "0 and 1. "
                f"Received value: "
                f"{self.best_validation_accuracy}."
            )

        if not isinstance(
            self.checkpoint_path,
            Path,
        ):
            raise TypeError(
                "checkpoint_path must be a pathlib.Path. "
                f"Received type: "
                f"{type(self.checkpoint_path).__name__}."
            )

        best_history_item = self.history[
            self.best_epoch_number - 1
        ]

        expected_best_loss = (
            best_history_item
            .validation_result
            .average_loss
        )

        expected_best_accuracy = (
            best_history_item
            .validation_result
            .accuracy
        )

        if not math.isclose(
            self.best_validation_loss,
            expected_best_loss,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "best_validation_loss must match the "
                "Validation Loss of best_epoch_number. "
                f"Expected: {expected_best_loss}. "
                f"Received: {self.best_validation_loss}."
            )

        if not math.isclose(
            self.best_validation_accuracy,
            expected_best_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "best_validation_accuracy must match the "
                "Validation Accuracy of best_epoch_number. "
                f"Expected: {expected_best_accuracy}. "
                f"Received: "
                f"{self.best_validation_accuracy}."
            )


# =============================================================================
# Multi-Epoch Training Pipeline
# =============================================================================


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    epoch_count: int = DEFAULT_EPOCH_COUNT,
    classification_threshold: float = (
        DEFAULT_CLASSIFICATION_THRESHOLD
    ),
    checkpoint_path: Path | str = (
        DEFAULT_CNN_CHECKPOINT_PATH
    ),
    verbose: bool = True,
) -> TrainingResult:
    """
    여러 Epoch의 Train·Validation을 실행하고 Best Model을 저장한다.

    왜 필요한가
    -----------
    현재까지 구현한 기능은 각각 독립되어 있다.

        CNN Model

        Loss Function

        Optimizer

        Train Epoch

        Validation Epoch

    실제 모델 학습을 위해서는 이 기능들을 여러 Epoch 동안
    하나의 흐름으로 연결해야 한다.

    입력
    ----
    model:
        학습할 PyTorch Model

        현재:

            CNNBaseline

        향후:

            ResNet18

    train_loader:
        Train DataLoader

        현재 실제 Dataset:

            5,306장

        현재 Batch:

            166개

    validation_loader:
        Validation DataLoader

        현재 실제 Dataset:

            1,327장

        현재 Batch:

            42개

    loss_function:
        Binary Classification Loss

        현재:

            BCEWithLogitsLoss

    optimizer:
        Model Parameter를 갱신할 Optimizer

        현재:

            Adam

    device:
        Model과 Batch가 위치할 Device

        현재:

            cpu

    epoch_count:
        실행할 전체 Epoch 수

        기본:

            5

    classification_threshold:
        DEFECT Probability를 Class로 변환할 Threshold

        기본:

            0.5

    checkpoint_path:
        Best Model Checkpoint 저장 경로

        기본:

            models/checkpoints/cnn_baseline_best.pt

    verbose:
        Epoch 학습 결과를 Console에 출력할지 여부

        True:

            출력

        False:

            출력하지 않음

    처리 과정
    ---------
    1. 전체 입력을 검증한다.
    2. Model과 Device 상태를 확인한다.
    3. Model Trainable Parameter와 Optimizer 연결을 확인한다.
    4. Checkpoint 상위 Directory를 생성한다.
    5. Best Validation Loss를 infinity로 초기화한다.
    6. Epoch를 1부터 epoch_count까지 반복한다.
    7. train_one_epoch()를 실행한다.
    8. validate_one_epoch()를 실행한다.
    9. EpochHistoryItem을 생성한다.
    10. History에 Epoch 결과를 추가한다.
    11. Validation Loss가 기존 Best보다 낮은지 확인한다.
    12. 새로운 Best이면 Checkpoint를 저장한다.
    13. Epoch 결과를 Console에 출력한다.
    14. 모든 Epoch 완료 후 TrainingResult를 반환한다.

    출력
    ----
    TrainingResult

        history:

            모든 Epoch Train·Validation 결과

        best_epoch_number:

            Validation Loss가 가장 낮은 Epoch

        best_validation_loss:

            가장 낮은 Validation Loss

        best_validation_accuracy:

            Best Epoch의 Validation Accuracy

        checkpoint_path:

            Best Checkpoint 경로

    Best Model 선택
    ---------------
    기준:

        가장 낮은 Validation Loss

    비교:

        validation_loss

        <

        best_validation_loss

    Validation Loss가 이전 Best와 정확히 같은 경우:

        이전 Epoch를 유지한다.

    이유:

        같은 성능이면 더 먼저 도달한 Epoch를 유지한다.

    Checkpoint 저장 내용
    --------------------
    Checkpoint Version

    Model 이름

    Model Module

    Loss Function 이름

    Optimizer 이름

    Epoch 번호

    전체 Epoch 설정

    Classification Threshold

    Best Model 선택 기준

    Model State

    Optimizer State

    Train Result

    Validation Result

    Gradient
    --------
    Train Epoch:

        활성화

    Validation Epoch:

        비활성화

    예외 처리
    ---------
    잘못된 Model:

        TypeError

    잘못된 DataLoader:

        TypeError

    잘못된 Loss:

        TypeError

    잘못된 Optimizer:

        TypeError

    잘못된 Device:

        TypeError 또는 ValueError

    Model Device 불일치:

        ValueError

    Model과 Optimizer Parameter 불일치:

        ValueError

    잘못된 Epoch Count:

        TypeError 또는 ValueError

    잘못된 Threshold:

        TypeError 또는 ValueError

    잘못된 Checkpoint Path:

        TypeError 또는 ValueError

    잘못된 verbose:

        TypeError

    테스트 방법
    -----------
    작은 Dummy Dataset:

        6 Sample

    Epoch:

        3

    검증:

        History 길이:

            3

        Epoch Number:

            1, 2, 3

        Checkpoint:

            존재

        Best Epoch:

            1~3

        Best Validation Loss:

            History의 최소 Validation Loss

        Checkpoint Epoch:

            Best Epoch와 일치

    실무 확장 방향
    --------------
    향후 다음 기능을 추가할 수 있다.

        Early Stopping

        Learning Rate Scheduler

        Resume Training

        Automatic Mixed Precision

        Experiment Tracking

        TensorBoard

        Weights & Biases

    현재 CNN Baseline 범위에는 포함하지 않는다.
    """
    _validate_training_objects(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        optimizer=optimizer,
    )

    resolved_device = _resolve_device(
        device=device,
    )

    validated_epoch_count = _validate_epoch_count(
        epoch_count=epoch_count,
    )

    validated_threshold = (
        _validate_classification_threshold(
            classification_threshold=(
                classification_threshold
            ),
        )
    )

    resolved_checkpoint_path = (
        _resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
        )
    )

    _validate_verbose(
        verbose=verbose,
    )

    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    _validate_optimizer_parameter_connection(
        model=model,
        optimizer=optimizer,
    )

    _prepare_checkpoint_directory(
        checkpoint_path=(
            resolved_checkpoint_path
        ),
    )

    if verbose:
        _print_training_header(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=resolved_device,
            epoch_count=validated_epoch_count,
            classification_threshold=(
                validated_threshold
            ),
            checkpoint_path=(
                resolved_checkpoint_path
            ),
        )

    history_items: list[EpochHistoryItem] = []

    best_epoch_number = 0

    best_validation_loss = float(
        "inf"
    )

    best_validation_accuracy = 0.0

    for epoch_number in range(
        1,
        validated_epoch_count + 1,
    ):
        # --------------------------------------------------------------
        # Train
        # --------------------------------------------------------------
        train_result = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=resolved_device,
            classification_threshold=(
                validated_threshold
            ),
        )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------
        validation_result = (
            validate_one_epoch(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                device=resolved_device,
                classification_threshold=(
                    validated_threshold
                ),
            )
        )

        # --------------------------------------------------------------
        # History
        # --------------------------------------------------------------
        history_item = EpochHistoryItem(
            epoch_number=epoch_number,
            train_result=train_result,
            validation_result=(
                validation_result
            ),
        )

        history_items.append(
            history_item
        )

        # --------------------------------------------------------------
        # Best Model Selection
        # --------------------------------------------------------------
        #
        # '<'를 사용한다.
        #
        # Validation Loss가 정확히 같은 경우에는
        # 더 먼저 저장된 Epoch를 Best로 유지한다.
        best_model_updated = (
            validation_result.average_loss
            < best_validation_loss
        )

        if best_model_updated:
            best_epoch_number = (
                epoch_number
            )

            best_validation_loss = (
                validation_result
                .average_loss
            )

            best_validation_accuracy = (
                validation_result
                .accuracy
            )

            _save_best_checkpoint(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                epoch_number=epoch_number,
                epoch_count=(
                    validated_epoch_count
                ),
                classification_threshold=(
                    validated_threshold
                ),
                train_result=train_result,
                validation_result=(
                    validation_result
                ),
                checkpoint_path=(
                    resolved_checkpoint_path
                ),
            )

        if verbose:
            _print_epoch_summary(
                epoch_number=epoch_number,
                epoch_count=(
                    validated_epoch_count
                ),
                train_result=train_result,
                validation_result=(
                    validation_result
                ),
                best_model_updated=(
                    best_model_updated
                ),
                current_best_epoch=(
                    best_epoch_number
                ),
                current_best_validation_loss=(
                    best_validation_loss
                ),
            )

    # Epoch Count는 1 이상으로 검증했으므로
    # 정상 실행이라면 Best Epoch가 반드시 존재한다.
    if best_epoch_number <= 0:
        raise RuntimeError(
            "training completed without selecting a best epoch."
        )

    training_result = TrainingResult(
        history=tuple(
            history_items
        ),
        best_epoch_number=(
            best_epoch_number
        ),
        best_validation_loss=(
            best_validation_loss
        ),
        best_validation_accuracy=(
            best_validation_accuracy
        ),
        checkpoint_path=(
            resolved_checkpoint_path
        ),
    )

    if verbose:
        _print_training_completion(
            training_result=(
                training_result
            ),
        )

    return training_result


# =============================================================================
# Training Object Validation
# =============================================================================


def _validate_training_objects(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
) -> None:
    """
    Training Pipeline의 주요 PyTorch 객체 타입을 검증한다.

    왜 필요한가
    -----------
    여러 Epoch 학습을 시작한 후 늦게 오류를 발견하지 않도록
    Pipeline 진입 시점에 기본 객체를 먼저 확인한다.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be an instance of torch.nn.Module. "
            f"Received type: {type(model).__name__}."
        )

    if not isinstance(
        train_loader,
        DataLoader,
    ):
        raise TypeError(
            "train_loader must be an instance of "
            "torch.utils.data.DataLoader. "
            f"Received type: "
            f"{type(train_loader).__name__}."
        )

    if not isinstance(
        validation_loader,
        DataLoader,
    ):
        raise TypeError(
            "validation_loader must be an instance of "
            "torch.utils.data.DataLoader. "
            f"Received type: "
            f"{type(validation_loader).__name__}."
        )

    if not isinstance(
        loss_function,
        nn.Module,
    ):
        raise TypeError(
            "loss_function must be an instance of torch.nn.Module. "
            f"Received type: "
            f"{type(loss_function).__name__}."
        )

    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise TypeError(
            "optimizer must be an instance of "
            "torch.optim.Optimizer. "
            f"Received type: "
            f"{type(optimizer).__name__}."
        )


# =============================================================================
# Epoch Count Validation
# =============================================================================


def _validate_epoch_count(
    epoch_count: int,
) -> int:
    """
    Epoch 수를 검증한다.

    허용
    ----
    정수

    bool 아님

    1 이상

    출력
    ----
    int

        검증된 Epoch 수
    """
    if (
        isinstance(
            epoch_count,
            bool,
        )
        or not isinstance(
            epoch_count,
            Integral,
        )
    ):
        raise TypeError(
            "epoch_count must be an integer. "
            f"Received type: "
            f"{type(epoch_count).__name__}."
        )

    validated_epoch_count = int(
        epoch_count
    )

    if validated_epoch_count <= 0:
        raise ValueError(
            "epoch_count must be greater than 0. "
            f"Received value: "
            f"{validated_epoch_count}."
        )

    return validated_epoch_count


# =============================================================================
# Threshold Validation
# =============================================================================


def _validate_classification_threshold(
    classification_threshold: float,
) -> float:
    """
    Binary Classification Threshold를 검증한다.

    허용
    ----
    실수형 숫자

    유한한 값

    0 이상

    1 이하

    출력
    ----
    float

        검증된 Threshold
    """
    if (
        isinstance(
            classification_threshold,
            bool,
        )
        or not isinstance(
            classification_threshold,
            Real,
        )
    ):
        raise TypeError(
            "classification_threshold must be a real number. "
            f"Received type: "
            f"{type(classification_threshold).__name__}."
        )

    validated_threshold = float(
        classification_threshold
    )

    if not math.isfinite(
        validated_threshold
    ):
        raise ValueError(
            "classification_threshold must be finite. "
            f"Received value: "
            f"{validated_threshold}."
        )

    if not (
        0.0
        <= validated_threshold
        <= 1.0
    ):
        raise ValueError(
            "classification_threshold must be between 0 and 1. "
            f"Received value: "
            f"{validated_threshold}."
        )

    return validated_threshold


# =============================================================================
# Device Validation
# =============================================================================


def _resolve_device(
    device: torch.device | str,
) -> torch.device:
    """
    Device 입력을 torch.device로 정규화한다.

    현재
    ----
    cpu

    향후
    ----
    CUDA 사용 가능 환경에서는 cuda 지원
    """
    if not isinstance(
        device,
        (
            str,
            torch.device,
        ),
    ):
        raise TypeError(
            "device must be a string or torch.device. "
            f"Received type: {type(device).__name__}."
        )

    try:
        resolved_device = torch.device(
            device
        )
    except (
        RuntimeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid device: {device}."
        ) from error

    if (
        resolved_device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CUDA device was requested, "
            "but CUDA is not available."
        )

    return resolved_device


def _validate_model_device(
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    Model Parameter와 Buffer가 요청 Device에 있는지 확인한다.

    현재 정상 구조
    --------------
    Model:

        cpu

    Device:

        cpu
    """
    model_parameters = list(
        model.parameters()
    )

    if not model_parameters:
        raise ValueError(
            "model must contain at least one parameter."
        )

    for parameter in model_parameters:
        if parameter.device != device:
            raise ValueError(
                "all model parameters must be on the "
                "requested device. "
                f"Requested device: {device}. "
                f"Found parameter device: "
                f"{parameter.device}."
            )

    for buffer in model.buffers():
        if buffer.device != device:
            raise ValueError(
                "all model buffers must be on the "
                "requested device. "
                f"Requested device: {device}. "
                f"Found buffer device: "
                f"{buffer.device}."
            )


# =============================================================================
# Optimizer Connection Validation
# =============================================================================


def _validate_optimizer_parameter_connection(
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """
    Optimizer가 Model의 학습 가능한 Parameter를 정확히 참조하는지 확인한다.

    왜 필요한가
    -----------
    잘못된 Model로 Optimizer를 생성한 뒤 다른 Model을
    run_training()에 전달할 수 있다.

    예:

        model_a = CNNBaseline()

        optimizer = create_optimizer(
            model=model_a,
        )

        model_b = CNNBaseline()

        run_training(
            model=model_b,
            optimizer=optimizer,
        )

    이 경우 Optimizer는 model_a를 갱신하고,
    Forward는 model_b로 수행하는 심각한 오류가 발생한다.

    따라서 Parameter 객체 ID를 비교한다.

    현재 규칙
    ---------
    Model:

        requires_grad=True Parameter

    Optimizer:

        등록된 Parameter

    두 집합:

        정확히 같아야 함
    """
    trainable_model_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_model_parameters:
        raise ValueError(
            "model must contain at least one trainable "
            "parameter with requires_grad=True."
        )

    optimizer_parameters = [
        parameter
        for parameter_group in (
            optimizer.param_groups
        )
        for parameter in (
            parameter_group["params"]
        )
    ]

    if not optimizer_parameters:
        raise ValueError(
            "optimizer must contain at least one parameter."
        )

    optimizer_parameter_ids = [
        id(parameter)
        for parameter in (
            optimizer_parameters
        )
    ]

    if (
        len(
            optimizer_parameter_ids
        )
        != len(
            set(
                optimizer_parameter_ids
            )
        )
    ):
        raise ValueError(
            "optimizer must not contain duplicate parameters."
        )

    model_parameter_id_set = {
        id(parameter)
        for parameter in (
            trainable_model_parameters
        )
    }

    optimizer_parameter_id_set = set(
        optimizer_parameter_ids
    )

    if (
        optimizer_parameter_id_set
        != model_parameter_id_set
    ):
        raise ValueError(
            "optimizer parameters must exactly match "
            "the model parameters with requires_grad=True."
        )


# =============================================================================
# Checkpoint Path Validation
# =============================================================================


def _resolve_checkpoint_path(
    checkpoint_path: Path | str,
) -> Path:
    """
    Checkpoint 경로를 pathlib.Path로 변환하고 검증한다.

    허용
    ----
    pathlib.Path

    문자열 Path

    확장자:

        .pt

        .pth

    출력
    ----
    Path
    """
    if isinstance(
        checkpoint_path,
        str,
    ):
        if not checkpoint_path.strip():
            raise ValueError(
                "checkpoint_path string must not be empty."
            )

        resolved_path = Path(
            checkpoint_path
        )

    elif isinstance(
        checkpoint_path,
        Path,
    ):
        resolved_path = (
            checkpoint_path
        )

    else:
        raise TypeError(
            "checkpoint_path must be a string or pathlib.Path. "
            f"Received type: "
            f"{type(checkpoint_path).__name__}."
        )

    if not resolved_path.name:
        raise ValueError(
            "checkpoint_path must include a file name."
        )

    if (
        resolved_path.exists()
        and resolved_path.is_dir()
    ):
        raise ValueError(
            "checkpoint_path must point to a file, "
            "not a directory. "
            f"Received path: {resolved_path}."
        )

    supported_suffixes = {
        ".pt",
        ".pth",
    }

    if (
        resolved_path.suffix.lower()
        not in supported_suffixes
    ):
        raise ValueError(
            "checkpoint_path must use a .pt or .pth extension. "
            f"Received path: {resolved_path}."
        )

    return resolved_path


def _prepare_checkpoint_directory(
    checkpoint_path: Path,
) -> None:
    """
    Checkpoint 상위 Directory를 생성한다.

    예
    --
    입력:

        models/checkpoints/cnn_baseline_best.pt

    Directory가 없는 경우:

        models/

        models/checkpoints/

    자동 생성
    """
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# Verbose Validation
# =============================================================================


def _validate_verbose(
    verbose: bool,
) -> None:
    """
    Console 출력 여부 설정을 검증한다.
    """
    if not isinstance(
        verbose,
        bool,
    ):
        raise TypeError(
            "verbose must be a bool. "
            f"Received type: "
            f"{type(verbose).__name__}."
        )


# =============================================================================
# Best Checkpoint
# =============================================================================


def _save_best_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    loss_function: nn.Module,
    epoch_number: int,
    epoch_count: int,
    classification_threshold: float,
    train_result: EpochResult,
    validation_result: EpochResult,
    checkpoint_path: Path,
) -> None:
    """
    현재 Best Model의 Checkpoint를 저장한다.

    저장 내용
    ---------
    checkpoint_version

    model_name

    model_module

    loss_function_name

    optimizer_name

    epoch_number

    configured_epoch_count

    classification_threshold

    best_model_selection_metric

    model_state_dict

    optimizer_state_dict

    train_result

    validation_result

    저장 방식
    ---------
    임시 파일:

        <checkpoint>.tmp

    먼저 임시 파일에 저장한 뒤
    최종 Checkpoint 경로로 교체한다.

    이유
    ----
    저장 도중 프로세스가 중단되면
    기존 정상 Checkpoint가 손상될 가능성을 줄인다.
    """
    checkpoint = {
        "checkpoint_version": (
            CHECKPOINT_VERSION
        ),
        "model_name": (
            model.__class__.__name__
        ),
        "model_module": (
            model.__class__.__module__
        ),
        "loss_function_name": (
            loss_function
            .__class__
            .__name__
        ),
        "optimizer_name": (
            optimizer
            .__class__
            .__name__
        ),
        "epoch_number": (
            epoch_number
        ),
        "configured_epoch_count": (
            epoch_count
        ),
        "classification_threshold": (
            classification_threshold
        ),
        "best_model_selection_metric": (
            BEST_MODEL_SELECTION_METRIC
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "train_result": {
            "average_loss": (
                train_result.average_loss
            ),
            "accuracy": (
                train_result.accuracy
            ),
            "sample_count": (
                train_result.sample_count
            ),
            "batch_count": (
                train_result.batch_count
            ),
        },
        "validation_result": {
            "average_loss": (
                validation_result.average_loss
            ),
            "accuracy": (
                validation_result.accuracy
            ),
            "sample_count": (
                validation_result.sample_count
            ),
            "batch_count": (
                validation_result.batch_count
            ),
        },
    }

    temporary_checkpoint_path = (
        checkpoint_path.with_name(
            f"{checkpoint_path.name}.tmp"
        )
    )

    try:
        torch.save(
            checkpoint,
            temporary_checkpoint_path,
        )

        temporary_checkpoint_path.replace(
            checkpoint_path
        )

    finally:
        # replace()가 정상 완료되면 임시 파일은 이미 사라진다.
        #
        # 저장 중 오류가 발생해 임시 파일이 남은 경우에만 삭제한다.
        if temporary_checkpoint_path.exists():
            temporary_checkpoint_path.unlink()


# =============================================================================
# Console Output
# =============================================================================


def _get_data_loader_sample_count(
    data_loader: DataLoader,
) -> int | None:
    """
    DataLoader Dataset의 Sample 수를 가능한 경우 반환한다.

    IterableDataset처럼 길이를 알 수 없는 구조라면 None을 반환한다.
    """
    try:
        return len(
            data_loader.dataset
        )
    except TypeError:
        return None


def _get_data_loader_batch_count(
    data_loader: DataLoader,
) -> int | None:
    """
    DataLoader Batch 수를 가능한 경우 반환한다.
    """
    try:
        return len(
            data_loader
        )
    except TypeError:
        return None


def _format_optional_count(
    value: int | None,
) -> str:
    """
    알 수 없는 개수를 Console 출력용 문자열로 변환한다.
    """
    if value is None:
        return "UNKNOWN"

    return f"{value:,}"


def _print_training_header(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epoch_count: int,
    classification_threshold: float,
    checkpoint_path: Path,
) -> None:
    """
    전체 학습 시작 설정을 출력한다.
    """
    train_sample_count = (
        _get_data_loader_sample_count(
            data_loader=train_loader,
        )
    )

    train_batch_count = (
        _get_data_loader_batch_count(
            data_loader=train_loader,
        )
    )

    validation_sample_count = (
        _get_data_loader_sample_count(
            data_loader=validation_loader,
        )
    )

    validation_batch_count = (
        _get_data_loader_batch_count(
            data_loader=validation_loader,
        )
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print()
    print("=" * 88)
    print("VISION MODEL TRAINING")
    print("=" * 88)

    print()
    print("[CONFIGURATION]")
    print(
        f"Model                      : "
        f"{model.__class__.__name__}"
    )
    print(
        f"Device                     : "
        f"{device}"
    )
    print(
        f"Epoch count                : "
        f"{epoch_count}"
    )
    print(
        f"Classification threshold   : "
        f"{classification_threshold}"
    )
    print(
        f"Best selection metric      : "
        f"{BEST_MODEL_SELECTION_METRIC}"
    )
    print(
        f"Trainable parameters       : "
        f"{trainable_parameter_count:,}"
    )
    print(
        f"Checkpoint path            : "
        f"{checkpoint_path}"
    )

    print()
    print("[TRAIN DATA]")
    print(
        f"Samples                    : "
        f"{_format_optional_count(train_sample_count)}"
    )
    print(
        f"Batches                    : "
        f"{_format_optional_count(train_batch_count)}"
    )

    print()
    print("[VALIDATION DATA]")
    print(
        f"Samples                    : "
        f"{_format_optional_count(validation_sample_count)}"
    )
    print(
        f"Batches                    : "
        f"{_format_optional_count(validation_batch_count)}"
    )


def _print_epoch_summary(
    epoch_number: int,
    epoch_count: int,
    train_result: EpochResult,
    validation_result: EpochResult,
    best_model_updated: bool,
    current_best_epoch: int,
    current_best_validation_loss: float,
) -> None:
    """
    한 Epoch의 Train·Validation 결과를 출력한다.
    """
    print()
    print("-" * 88)
    print(
        f"EPOCH {epoch_number} / {epoch_count}"
    )
    print("-" * 88)

    print()
    print("[TRAIN]")
    print(
        f"Average loss               : "
        f"{train_result.average_loss:.6f}"
    )
    print(
        f"Accuracy                   : "
        f"{train_result.accuracy:.6f}"
    )
    print(
        f"Accuracy percent           : "
        f"{train_result.accuracy * 100:.2f}%"
    )
    print(
        f"Samples                    : "
        f"{train_result.sample_count:,}"
    )
    print(
        f"Batches                    : "
        f"{train_result.batch_count:,}"
    )

    print()
    print("[VALIDATION]")
    print(
        f"Average loss               : "
        f"{validation_result.average_loss:.6f}"
    )
    print(
        f"Accuracy                   : "
        f"{validation_result.accuracy:.6f}"
    )
    print(
        f"Accuracy percent           : "
        f"{validation_result.accuracy * 100:.2f}%"
    )
    print(
        f"Samples                    : "
        f"{validation_result.sample_count:,}"
    )
    print(
        f"Batches                    : "
        f"{validation_result.batch_count:,}"
    )

    print()
    print("[BEST MODEL]")
    print(
        f"Updated                    : "
        f"{best_model_updated}"
    )
    print(
        f"Current best epoch         : "
        f"{current_best_epoch}"
    )
    print(
        f"Current best val loss      : "
        f"{current_best_validation_loss:.6f}"
    )


def _print_training_completion(
    training_result: TrainingResult,
) -> None:
    """
    전체 학습 완료 결과를 출력한다.
    """
    print()
    print("=" * 88)
    print("TRAINING COMPLETED")
    print("=" * 88)

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
    print(
        f"Checkpoint path            : "
        f"{training_result.checkpoint_path}"
    )