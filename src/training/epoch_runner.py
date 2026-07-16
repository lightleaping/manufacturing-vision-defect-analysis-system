"""
Train and validation epoch runners for binary image classification.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 이미지 분류 모델에 대해
Train Epoch와 Validation Epoch를 각각 한 번 실행한다.

현재 클래스 정의
----------------
0 = NORMAL

1 = DEFECT

Positive Class:

    DEFECT

현재 모델 출력
---------------
CNNBaseline:

    Binary Raw Logit

    Shape:

        [batch_size]

확률 변환
---------
Raw Logit:

    -> torch.sigmoid()

    -> DEFECT Probability

초기 Classification Threshold
------------------------------
0.5

Probability < 0.5:

    0

    NORMAL

Probability >= 0.5:

    1

    DEFECT

Train Epoch
-----------
model.train()

-> Train DataLoader 전체 반복

-> Image·Label Device 이동

-> Integer Label을 float32 Target으로 변환

-> optimizer.zero_grad()

-> Model Forward

-> Loss 계산

-> Backward

-> Optimizer Step

-> Loss·Accuracy 집계

-> EpochResult

Validation Epoch
----------------
model.eval()

-> torch.inference_mode()

-> Validation DataLoader 전체 반복

-> Image·Label Device 이동

-> Integer Label을 float32 Target으로 변환

-> Model Forward

-> Loss·Accuracy 집계

-> EpochResult

중요
----
Train Epoch에서는 Gradient를 계산하고 Parameter를 갱신한다.

Validation Epoch에서는 Gradient를 계산하지 않으며
Parameter를 갱신하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.training.loss_function import prepare_binary_targets


# =============================================================================
# Classification Configuration
# =============================================================================

# 초기 Binary Classification Threshold다.
#
# Sigmoid Probability:
#
#     Probability < 0.5
#
#         -> 0
#
#         -> NORMAL
#
#     Probability >= 0.5
#
#         -> 1
#
#         -> DEFECT
#
# 향후 Validation Precision·Recall·F1 결과에 따라
# Threshold 최적화를 별도 실험할 수 있다.
#
# 현재 CNN Baseline에서는 기본값 0.5를 사용한다.
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5


# =============================================================================
# Epoch Result
# =============================================================================


@dataclass(frozen=True)
class EpochResult:
    """
    Train 또는 Validation Epoch의 집계 결과.

    왜 필요한가
    -----------
    Train과 Validation이 같은 결과 형식을 반환하면
    향후 Training Pipeline에서 일관된 방식으로 기록할 수 있다.

    Dictionary 대신 Dataclass를 사용하면 다음 장점이 있다.

        필드 이름 명확화

        타입 힌트

        자동 완성

        오타 감소

        테스트 용이

    필드
    ----
    average_loss:
        Epoch 전체 Sample 기준 평균 Loss

        범위:

            0 이상

    accuracy:
        Epoch 전체 Binary Classification Accuracy

        범위:

            0.0 이상

            1.0 이하

        예:

            0.85

            ->

            85%

    sample_count:
        Epoch에서 실제 처리한 전체 이미지 수

    batch_count:
        Epoch에서 실제 처리한 전체 Batch 수

    출력 예
    -------
    EpochResult(
        average_loss=0.6231,
        accuracy=0.8427,
        sample_count=5306,
        batch_count=166,
    )
    """

    average_loss: float

    accuracy: float

    sample_count: int

    batch_count: int

    def __post_init__(self) -> None:
        """
        생성된 Epoch 결과가 유효한지 검증한다.

        입력
        ----
        Dataclass에 전달된 네 개 필드

        처리 과정
        ---------
        1. 평균 Loss가 유한한지 확인한다.
        2. 평균 Loss가 음수가 아닌지 확인한다.
        3. Accuracy가 유한한지 확인한다.
        4. Accuracy가 0~1 범위인지 확인한다.
        5. Sample 수가 양수인지 확인한다.
        6. Batch 수가 양수인지 확인한다.

        출력
        ----
        정상:

            EpochResult 생성 완료

        잘못된 값:

            ValueError
        """
        if not math.isfinite(
            self.average_loss,
        ):
            raise ValueError(
                "average_loss must be finite. "
                f"Received value: {self.average_loss}."
            )

        if self.average_loss < 0.0:
            raise ValueError(
                "average_loss must be greater than or equal to 0. "
                f"Received value: {self.average_loss}."
            )

        if not math.isfinite(
            self.accuracy,
        ):
            raise ValueError(
                "accuracy must be finite. "
                f"Received value: {self.accuracy}."
            )

        if not (
            0.0
            <= self.accuracy
            <= 1.0
        ):
            raise ValueError(
                "accuracy must be between 0 and 1. "
                f"Received value: {self.accuracy}."
            )

        if self.sample_count <= 0:
            raise ValueError(
                "sample_count must be greater than 0. "
                f"Received value: {self.sample_count}."
            )

        if self.batch_count <= 0:
            raise ValueError(
                "batch_count must be greater than 0. "
                f"Received value: {self.batch_count}."
            )


# =============================================================================
# Train Epoch
# =============================================================================


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Optimizer,
    device: torch.device | str,
    classification_threshold: float = (
        DEFAULT_CLASSIFICATION_THRESHOLD
    ),
) -> EpochResult:
    """
    Train DataLoader 전체를 한 번 학습한다.

    왜 필요한가
    -----------
    Train Dataset 전체를 한 번 처리하고 Model Parameter를 갱신하는
    하나의 Epoch 실행 단위가 필요하다.

    입력
    ----
    model:
        학습할 PyTorch Model

        현재:

            CNNBaseline

    data_loader:
        Train DataLoader

        현재 실제 설정:

            Dataset:

                5,306장

            Batch Size:

                32

            Batch Count:

                166

            Sampler:

                RandomSampler

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

    classification_threshold:
        DEFECT Probability를 Binary Class로 변환할 Threshold

        기본:

            0.5

    처리 과정
    ---------
    1. 입력 객체와 설정값을 검증한다.
    2. Model과 Device가 일치하는지 확인한다.
    3. Model을 Training Mode로 변경한다.
    4. Train DataLoader 전체를 반복한다.
    5. Batch 형식을 검증한다.
    6. Image와 Label을 Device로 이동한다.
    7. Label을 BCE Loss용 float32 Target으로 변환한다.
    8. 이전 Gradient를 None으로 초기화한다.
    9. Model Forward로 Binary Raw Logit을 계산한다.
    10. Logit을 검증한다.
    11. Binary Classification Loss를 계산한다.
    12. Loss를 검증한다.
    13. Backward로 Gradient를 계산한다.
    14. Optimizer가 Parameter를 갱신한다.
    15. Sample 수 기준 Loss를 누적한다.
    16. Threshold 기반 Binary Accuracy를 누적한다.
    17. Epoch 전체 평균을 계산한다.

    출력
    ----
    EpochResult

        average_loss:

            전체 Train Sample 기준 평균 Loss

        accuracy:

            전체 Train Sample 기준 Accuracy

        sample_count:

            실제 처리한 Train 이미지 수

        batch_count:

            실제 처리한 Train Batch 수

    호출 관계
    ---------
    향후 Training Pipeline:

        for epoch in range(...):

            train_result = train_one_epoch(...)

            validation_result = validate_one_epoch(...)

    Gradient
    --------
    Train Epoch:

        Gradient 활성화

        loss.backward() 실행

        optimizer.step() 실행

    설계 이유
    ---------
    Train과 Validation 함수를 분리하여
    Gradient와 Parameter 갱신 여부를 명확하게 표현한다.

    예외 처리
    ---------
    잘못된 Model:

        TypeError

    잘못된 DataLoader:

        TypeError

    잘못된 Loss Function:

        TypeError

    잘못된 Optimizer:

        TypeError

    잘못된 Device:

        TypeError 또는 ValueError

    Model과 Device 불일치:

        ValueError

    잘못된 Threshold:

        TypeError 또는 ValueError

    잘못된 Batch:

        TypeError 또는 ValueError

    잘못된 Logit:

        TypeError 또는 ValueError

    잘못된 Loss:

        TypeError 또는 ValueError

    빈 DataLoader:

        ValueError

    테스트 방법
    -----------
    작은 TensorDataset과 CNNBaseline을 사용해 다음을 확인한다.

        Train Mode:

            True

        Parameter:

            Epoch 전후 변경

        EpochResult:

            유효한 Loss

            0~1 Accuracy

            정확한 Sample 수

            정확한 Batch 수

    실무 확장 방향
    --------------
    향후 다음 기능을 별도 추가할 수 있다.

        Gradient Clipping

        Automatic Mixed Precision

        Gradient Accumulation

        Batch Progress Logging

        Distributed Training

    현재 CNN Baseline 범위에는 포함하지 않는다.
    """
    resolved_device = _validate_common_epoch_arguments(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device=device,
        classification_threshold=classification_threshold,
    )

    _validate_optimizer(
        optimizer=optimizer,
    )

    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    # Train Epoch임을 명확히 한다.
    #
    # 현재 CNNBaseline에는 Dropout과 BatchNorm이 없지만,
    # 향후 ResNet18에는 BatchNorm이 포함되므로
    # 항상 명시적으로 Training Mode를 설정한다.
    model.train()

    total_loss = 0.0

    total_correct_count = 0

    total_sample_count = 0

    total_batch_count = 0

    for batch in data_loader:
        images, labels = _unpack_and_validate_batch(
            batch=batch,
        )

        # 현재 Transform 결과는 이미 float32다.
        #
        # Epoch Runner에서도 float32를 명시하여
        # CNN Weight dtype과 일관된 입력을 보장한다.
        images = images.to(
            device=resolved_device,
            dtype=torch.float32,
        )

        # Accuracy 계산에서는 원래 Class Label 의미를 유지한다.
        #
        # Dataset의 실제 dtype:
        #
        #     torch.int64
        labels = labels.to(
            device=resolved_device,
        )

        # BCEWithLogitsLoss용 Target:
        #
        # int64:
        #
        #     0
        #
        #     1
        #
        # ->
        #
        # float32:
        #
        #     0.0
        #
        #     1.0
        targets = prepare_binary_targets(
            labels=labels,
        )

        # PyTorch Gradient는 기본적으로 누적된다.
        #
        # Batch마다 새로운 Gradient를 계산하기 위해
        # 이전 Gradient를 None으로 초기화한다.
        optimizer.zero_grad(
            set_to_none=True,
        )

        # CNN Forward
        #
        # 입력:
        #
        #     [B, 3, H, W]
        #
        # 출력:
        #
        #     [B]
        #
        # 의미:
        #
        #     DEFECT Binary Raw Logit
        logits = model(
            images,
        )

        batch_size = images.shape[0]

        _validate_logits(
            logits=logits,
            expected_batch_size=batch_size,
        )

        # BCEWithLogitsLoss는 내부에서
        # Sigmoid와 Binary Cross Entropy를 함께 처리한다.
        loss = loss_function(
            logits,
            targets,
        )

        _validate_loss(
            loss=loss,
        )

        # Model Parameter Gradient 계산
        loss.backward()

        # 계산된 Gradient를 사용해
        # Weight와 Bias를 실제 갱신한다.
        optimizer.step()

        # BCEWithLogitsLoss(reduction="mean") 결과는
        # 현재 Batch의 평균 Loss다.
        #
        # Epoch 전체 Sample 평균을 정확히 계산하기 위해:
        #
        # Batch 평균 Loss
        #
        # ×
        #
        # Batch Sample 수
        #
        # 를 누적한다.
        total_loss += (
            loss.detach().item()
            * batch_size
        )

        # Metric 계산은 Gradient가 필요하지 않다.
        with torch.no_grad():
            batch_correct_count = (
                _count_correct_predictions(
                    logits=logits,
                    labels=labels,
                    classification_threshold=(
                        classification_threshold
                    ),
                )
            )

        total_correct_count += (
            batch_correct_count
        )

        total_sample_count += (
            batch_size
        )

        total_batch_count += 1

    return _build_epoch_result(
        total_loss=total_loss,
        total_correct_count=total_correct_count,
        total_sample_count=total_sample_count,
        total_batch_count=total_batch_count,
        epoch_name="train",
    )


# =============================================================================
# Validation Epoch
# =============================================================================


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device | str,
    classification_threshold: float = (
        DEFAULT_CLASSIFICATION_THRESHOLD
    ),
) -> EpochResult:
    """
    Validation DataLoader 전체를 한 번 평가한다.

    왜 필요한가
    -----------
    Train 과정에서 Model Parameter를 갱신한 후
    학습에 직접 사용하지 않은 Validation Dataset에서
    Loss와 Accuracy를 측정해야 한다.

    입력
    ----
    model:
        평가할 PyTorch Model

        현재:

            CNNBaseline

    data_loader:
        Validation DataLoader

        현재 실제 설정:

            Dataset:

                1,327장

            Batch Size:

                32

            Batch Count:

                42

            Sampler:

                SequentialSampler

    loss_function:
        Train과 동일한 Binary Classification Loss

        현재:

            BCEWithLogitsLoss

    device:
        Model과 Batch가 위치할 Device

        현재:

            cpu

    classification_threshold:
        DEFECT Probability를 Class로 변환할 Threshold

        기본:

            0.5

    처리 과정
    ---------
    1. 입력 객체와 설정값을 검증한다.
    2. Model과 Device가 일치하는지 확인한다.
    3. Model을 Evaluation Mode로 변경한다.
    4. torch.inference_mode()를 활성화한다.
    5. Validation DataLoader 전체를 반복한다.
    6. Image와 Label을 Device로 이동한다.
    7. Label을 BCE Loss용 float32 Target으로 변환한다.
    8. Model Forward를 실행한다.
    9. Validation Loss를 계산한다.
    10. Sample 수 기준 Loss를 누적한다.
    11. Threshold 기반 Accuracy를 누적한다.
    12. Epoch 전체 평균을 계산한다.

    출력
    ----
    EpochResult

        average_loss:

            전체 Validation Sample 기준 평균 Loss

        accuracy:

            전체 Validation Sample 기준 Accuracy

        sample_count:

            실제 처리한 Validation 이미지 수

        batch_count:

            실제 처리한 Validation Batch 수

    Validation에서 하지 않는 것
    ----------------------------
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    Parameter 갱신

    Gradient
    --------
    Validation Epoch:

        torch.inference_mode()

        Gradient 계산 없음

        계산 그래프 기록 없음

    설계 이유
    ---------
    Validation은 학습이 아니라 현재 Model 성능 측정이 목적이다.

    Gradient를 비활성화하면:

        메모리 사용 감소

        불필요한 계산 감소

        평가 목적 명확화

    테스트 방법
    -----------
    Validation 전후 Model Weight를 비교한다.

    예상:

        Weight 변경 없음

        model.training:

            False

        EpochResult:

            유효한 Loss

            0~1 Accuracy

            정확한 Sample 수

            정확한 Batch 수

    실무 확장 방향
    --------------
    향후 Evaluation 모듈에서 다음을 별도로 계산한다.

        Precision

        Recall

        F1 Score

        Confusion Matrix

        False Positive

        False Negative

        오분류 이미지
    """
    resolved_device = _validate_common_epoch_arguments(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device=device,
        classification_threshold=classification_threshold,
    )

    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    # Validation에서는 Evaluation Mode를 사용한다.
    #
    # 현재 CNNBaseline에는 BatchNorm이 없지만,
    # 향후 ResNet18에서는 반드시 필요하다.
    model.eval()

    total_loss = 0.0

    total_correct_count = 0

    total_sample_count = 0

    total_batch_count = 0

    # Validation에서는 Backward와 Parameter 갱신이 필요하지 않다.
    #
    # inference_mode()는 Gradient 계산과 계산 그래프 기록을
    # 비활성화한다.
    with torch.inference_mode():
        for batch in data_loader:
            images, labels = (
                _unpack_and_validate_batch(
                    batch=batch,
                )
            )

            images = images.to(
                device=resolved_device,
                dtype=torch.float32,
            )

            labels = labels.to(
                device=resolved_device,
            )

            targets = prepare_binary_targets(
                labels=labels,
            )

            logits = model(
                images,
            )

            batch_size = images.shape[0]

            _validate_logits(
                logits=logits,
                expected_batch_size=batch_size,
            )

            loss = loss_function(
                logits,
                targets,
            )

            _validate_loss(
                loss=loss,
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            batch_correct_count = (
                _count_correct_predictions(
                    logits=logits,
                    labels=labels,
                    classification_threshold=(
                        classification_threshold
                    ),
                )
            )

            total_correct_count += (
                batch_correct_count
            )

            total_sample_count += (
                batch_size
            )

            total_batch_count += 1

    return _build_epoch_result(
        total_loss=total_loss,
        total_correct_count=total_correct_count,
        total_sample_count=total_sample_count,
        total_batch_count=total_batch_count,
        epoch_name="validation",
    )


# =============================================================================
# Common Argument Validation
# =============================================================================


def _validate_common_epoch_arguments(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device | str,
    classification_threshold: float,
) -> torch.device:
    """
    Train·Validation 공통 입력을 검증한다.

    출력
    ----
    torch.device

        검증·정규화된 Device
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
        data_loader,
        DataLoader,
    ):
        raise TypeError(
            "data_loader must be an instance of "
            "torch.utils.data.DataLoader. "
            f"Received type: {type(data_loader).__name__}."
        )

    if not isinstance(
        loss_function,
        nn.Module,
    ):
        raise TypeError(
            "loss_function must be an instance of torch.nn.Module. "
            f"Received type: {type(loss_function).__name__}."
        )

    resolved_device = _resolve_device(
        device=device,
    )

    _validate_classification_threshold(
        classification_threshold=(
            classification_threshold
        ),
    )

    return resolved_device


def _validate_optimizer(
    optimizer: Optimizer,
) -> None:
    """
    Train Epoch에 전달된 Optimizer 타입을 검증한다.
    """
    if not isinstance(
        optimizer,
        Optimizer,
    ):
        raise TypeError(
            "optimizer must be an instance of "
            "torch.optim.Optimizer. "
            f"Received type: {type(optimizer).__name__}."
        )


def _resolve_device(
    device: torch.device | str,
) -> torch.device:
    """
    Device 입력을 torch.device 객체로 변환하고 검증한다.

    허용 예
    -------
    "cpu"

    torch.device("cpu")

    CUDA 사용 가능 환경:

        "cuda"

        torch.device("cuda")
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
            device,
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
            "CUDA device was requested, but CUDA is not available."
        )

    return resolved_device


def _validate_classification_threshold(
    classification_threshold: float,
) -> None:
    """
    Binary Classification Threshold를 검증한다.

    허용
    ----
    0.0 이상

    1.0 이하

    거부
    ----
    bool

    숫자가 아닌 값

    NaN

    inf

    0보다 작은 값

    1보다 큰 값
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
        classification_threshold,
    )

    if not math.isfinite(
        validated_threshold,
    ):
        raise ValueError(
            "classification_threshold must be finite. "
            f"Received value: {validated_threshold}."
        )

    if not (
        0.0
        <= validated_threshold
        <= 1.0
    ):
        raise ValueError(
            "classification_threshold must be between 0 and 1. "
            f"Received value: {validated_threshold}."
        )


# =============================================================================
# Model Device Validation
# =============================================================================


def _validate_model_device(
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    Model Parameter와 Buffer가 요청 Device에 있는지 확인한다.

    왜 필요한가
    -----------
    Image Batch는 Epoch Runner에서 지정 Device로 이동한다.

    Model이 다른 Device에 있으면 Forward에서
    Device 불일치 오류가 발생한다.

    현재 CPU 환경
    -------------
    Model:

        cpu

    Image:

        cpu

    Label:

        cpu

    향후 GPU 환경
    -------------
    Model은 Optimizer 생성 전에 다음처럼 이동해야 한다.

        model = model.to(
            device,
        )

    그 후 Optimizer를 생성한다.
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
                "all model parameters must be on the requested device. "
                f"Requested device: {device}. "
                f"Found parameter device: {parameter.device}."
            )

    for buffer in model.buffers():
        if buffer.device != device:
            raise ValueError(
                "all model buffers must be on the requested device. "
                f"Requested device: {device}. "
                f"Found buffer device: {buffer.device}."
            )


# =============================================================================
# Batch Validation
# =============================================================================


def _unpack_and_validate_batch(
    batch: object,
) -> tuple[Tensor, Tensor]:
    """
    DataLoader Batch를 Image Tensor와 Label Tensor로 분리하고 검증한다.

    예상 Batch
    ----------
    tuple 또는 list

    길이:

        2

    첫 번째:

        Image Tensor

    두 번째:

        Label Tensor

    Image Shape:

        [batch_size, channels, height, width]

    Label Shape:

        [batch_size]
    """
    if not isinstance(
        batch,
        (
            tuple,
            list,
        ),
    ):
        raise TypeError(
            "each data_loader batch must be a tuple or list "
            "containing images and labels. "
            f"Received type: {type(batch).__name__}."
        )

    if len(
        batch,
    ) != 2:
        raise ValueError(
            "each data_loader batch must contain exactly "
            "two items: images and labels. "
            f"Received item count: {len(batch)}."
        )

    images = batch[0]

    labels = batch[1]

    if not isinstance(
        images,
        Tensor,
    ):
        raise TypeError(
            "batch images must be a torch.Tensor. "
            f"Received type: {type(images).__name__}."
        )

    if not isinstance(
        labels,
        Tensor,
    ):
        raise TypeError(
            "batch labels must be a torch.Tensor. "
            f"Received type: {type(labels).__name__}."
        )

    if images.ndim != 4:
        raise ValueError(
            "batch images must have 4 dimensions in "
            "[batch_size, channels, height, width] format. "
            f"Received shape: {tuple(images.shape)}."
        )

    if labels.ndim != 1:
        raise ValueError(
            "batch labels must have 1 dimension in "
            "[batch_size] format. "
            f"Received shape: {tuple(labels.shape)}."
        )

    batch_size = images.shape[0]

    if batch_size <= 0:
        raise ValueError(
            "batch images must contain at least one image."
        )

    if labels.shape[0] != batch_size:
        raise ValueError(
            "image and label batch sizes must match. "
            f"Image batch size: {batch_size}. "
            f"Label batch size: {labels.shape[0]}."
        )

    if not images.is_floating_point():
        raise TypeError(
            "batch images must use a floating-point dtype. "
            f"Received dtype: {images.dtype}."
        )

    return images, labels


# =============================================================================
# Logit and Loss Validation
# =============================================================================


def _validate_logits(
    logits: Tensor,
    expected_batch_size: int,
) -> None:
    """
    Model의 Binary Raw Logit 출력을 검증한다.

    예상 Shape
    ----------
    [batch_size]

    예상 dtype
    ----------
    Floating Point

    예상 값
    -------
    모두 유한한 값
    """
    if not isinstance(
        logits,
        Tensor,
    ):
        raise TypeError(
            "model output logits must be a torch.Tensor. "
            f"Received type: {type(logits).__name__}."
        )

    if logits.ndim != 1:
        raise ValueError(
            "model output logits must have 1 dimension in "
            "[batch_size] format. "
            f"Received shape: {tuple(logits.shape)}."
        )

    if logits.shape[0] != expected_batch_size:
        raise ValueError(
            "logit batch size must match image batch size. "
            f"Expected: {expected_batch_size}. "
            f"Received: {logits.shape[0]}."
        )

    if not logits.is_floating_point():
        raise TypeError(
            "model output logits must use a floating-point dtype. "
            f"Received dtype: {logits.dtype}."
        )

    if not torch.isfinite(
        logits,
    ).all():
        raise ValueError(
            "model output logits must contain only finite values. "
            "NaN and infinity are not allowed."
        )


def _validate_loss(
    loss: Tensor,
) -> None:
    """
    Loss Function 출력을 검증한다.

    예상
    ----
    Tensor

    Scalar:

        Shape []

    Floating Point

    유한한 값

    0 이상
    """
    if not isinstance(
        loss,
        Tensor,
    ):
        raise TypeError(
            "loss_function output must be a torch.Tensor. "
            f"Received type: {type(loss).__name__}."
        )

    if loss.ndim != 0:
        raise ValueError(
            "loss_function output must be a scalar tensor. "
            f"Received shape: {tuple(loss.shape)}."
        )

    if not loss.is_floating_point():
        raise TypeError(
            "loss_function output must use a floating-point dtype. "
            f"Received dtype: {loss.dtype}."
        )

    if not torch.isfinite(
        loss,
    ):
        raise ValueError(
            "loss_function output must be finite. "
            "NaN and infinity are not allowed."
        )

    if loss.detach().item() < 0.0:
        raise ValueError(
            "loss_function output must be greater than "
            "or equal to 0. "
            f"Received value: {loss.detach().item()}."
        )


# =============================================================================
# Binary Accuracy
# =============================================================================


def _count_correct_predictions(
    logits: Tensor,
    labels: Tensor,
    classification_threshold: float,
) -> int:
    """
    Binary Logit과 Label을 이용해 정답 예측 수를 계산한다.

    입력
    ----
    logits:
        Binary Raw Logit

        Shape:

            [batch_size]

    labels:
        실제 Class Label

        Shape:

            [batch_size]

        값:

            0

            1

    classification_threshold:
        DEFECT Probability Threshold

    처리 과정
    ---------
    1. Raw Logit에 Sigmoid를 적용한다.
    2. DEFECT Probability를 계산한다.
    3. Threshold 이상이면 DEFECT 1로 분류한다.
    4. Threshold 미만이면 NORMAL 0으로 분류한다.
    5. 실제 Label과 같은 예측 수를 계산한다.

    출력
    ----
    int

        현재 Batch의 정확한 예측 수
    """
    probabilities = torch.sigmoid(
        logits.detach(),
    )

    predictions = (
        probabilities
        >= classification_threshold
    ).to(
        dtype=torch.int64,
    )

    expected_labels = labels.to(
        dtype=torch.int64,
    )

    correct_count = (
        predictions
        == expected_labels
    ).sum().item()

    return int(
        correct_count,
    )


# =============================================================================
# Epoch Result Builder
# =============================================================================


def _build_epoch_result(
    total_loss: float,
    total_correct_count: int,
    total_sample_count: int,
    total_batch_count: int,
    epoch_name: str,
) -> EpochResult:
    """
    Epoch 누적값을 평균 Loss·Accuracy로 변환한다.

    입력
    ----
    total_loss:
        Batch 평균 Loss에 Batch Size를 곱해 누적한
        전체 Sample Loss 합

    total_correct_count:
        전체 정답 예측 수

    total_sample_count:
        전체 처리 Sample 수

    total_batch_count:
        전체 처리 Batch 수

    epoch_name:
        오류 메시지에 사용할 Epoch 이름

        예:

            train

            validation

    출력
    ----
    EpochResult
    """
    if (
        total_sample_count <= 0
        or total_batch_count <= 0
    ):
        raise ValueError(
            f"{epoch_name} data_loader must contain "
            "at least one non-empty batch."
        )

    average_loss = (
        total_loss
        / total_sample_count
    )

    accuracy = (
        total_correct_count
        / total_sample_count
    )

    return EpochResult(
        average_loss=float(
            average_loss,
        ),
        accuracy=float(
            accuracy,
        ),
        sample_count=total_sample_count,
        batch_count=total_batch_count,
    )