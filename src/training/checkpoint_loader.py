"""
Validated PyTorch model checkpoint loader.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System에서 저장한
Best Model Checkpoint를 안전하게 읽고 현재 Model에 Weight를 복원한다.

현재 사용 대상
--------------
CNNBaseline Best Checkpoint

현재 실제 Checkpoint
--------------------
Path:

    models/checkpoints/cnn_baseline_best.pt

Best Epoch:

    2

Best Model Selection:

    Lowest Validation Loss

향후 재사용 대상
----------------
ResNet18 Transfer Learning Model

FastAPI Inference

Streamlit Dashboard

Batch Prediction

전체 복원 흐름
-------------
Model 생성

-> Device 이동

-> load_model_checkpoint()

-> Checkpoint Path 검증

-> torch.load()

-> Checkpoint Dictionary 검증

-> Required Key 검증

-> Checkpoint Version 검증

-> Model 이름·Module 검증

-> Epoch·Threshold 검증

-> Train·Validation Result 검증

-> Model State Key 검증

-> Model State Tensor 검증

-> model.load_state_dict(
       strict=True
   )

-> Parameter·Buffer Device 검증

-> Parameter·Buffer 유한성 검증

-> LoadedCheckpointInfo 반환

중요
----
이 모듈은 Model Weight를 복원하지만
Model Mode는 변경하지 않는다.

즉 다음을 자동 실행하지 않는다.

    model.train()

    model.eval()

Train·Evaluation Mode 설정은
실제 학습 또는 평가 함수의 책임이다.

Optimizer State
---------------
Checkpoint에는 optimizer_state_dict가 저장되어 있다.

하지만 현재 Loader는 Test·Inference용 Model 복원이 목적이므로
Optimizer State를 실제 Optimizer 객체에 복원하지 않는다.

향후 Resume Training 기능이 필요하면
Optimizer 복원 기능을 별도로 구현한다.

보안
----
현재 Checkpoint는 이 프로젝트에서 직접 학습하고 생성한
신뢰 가능한 로컬 Artifact다.

외부 출처의 신뢰할 수 없는 .pt·.pth 파일은
임의로 torch.load()하지 않는다.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path

import torch
from torch import Tensor, nn

from src.training.training_pipeline import (
    BEST_MODEL_SELECTION_METRIC,
    CHECKPOINT_VERSION,
)


# =============================================================================
# Supported Checkpoint Configuration
# =============================================================================

# 현재 Loader가 허용하는 PyTorch Checkpoint 확장자다.
SUPPORTED_CHECKPOINT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
    }
)


# training_pipeline.py가 저장하는 최상위 필수 Key다.
#
# 추가 Key는 허용한다.
#
# 하지만 아래 Key 중 하나라도 누락되면
# 현재 프로젝트의 정상 Checkpoint로 판단하지 않는다.
REQUIRED_CHECKPOINT_KEYS = frozenset(
    {
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
)


# Train·Validation Result Dictionary의 필수 Key다.
REQUIRED_EPOCH_RESULT_KEYS = frozenset(
    {
        "average_loss",
        "accuracy",
        "sample_count",
        "batch_count",
    }
)


# Optimizer State Dictionary의 기본 필수 Key다.
REQUIRED_OPTIMIZER_STATE_KEYS = frozenset(
    {
        "state",
        "param_groups",
    }
)


# =============================================================================
# Loaded Checkpoint Information
# =============================================================================


@dataclass(frozen=True)
class LoadedCheckpointInfo:
    """
    Model Weight 복원이 완료된 Checkpoint의 주요 Metadata.

    왜 필요한가
    -----------
    Weight 복원 후 Test·Inference 단계에서 다음 정보를 사용해야 한다.

        Best Epoch

        Validation Loss

        Validation Accuracy

        Classification Threshold

        Model 이름

        Checkpoint Version

    Dictionary를 그대로 반환하는 대신
    검증된 Metadata를 Dataclass로 반환한다.

    장점
    ----
    필드 이름 명확화

    타입 힌트

    자동 완성

    오타 감소

    불변 결과

    필드
    ----
    checkpoint_path:
        실제로 읽은 Checkpoint 경로

    checkpoint_version:
        Checkpoint 구조 Version

        현재:

            1

    model_name:
        저장된 Model Class 이름

        현재:

            CNNBaseline

    model_module:
        저장된 Model Module

        현재:

            src.models.cnn_baseline

    loss_function_name:
        학습에 사용한 Loss Function 이름

        현재:

            BCEWithLogitsLoss

    optimizer_name:
        학습에 사용한 Optimizer 이름

        현재:

            Adam

    epoch_number:
        저장된 Best Epoch

        현재 실제 결과:

            2

    configured_epoch_count:
        실제 학습에 설정한 전체 Epoch 수

        현재 실제 결과:

            5

    classification_threshold:
        Binary Classification Threshold

        현재:

            0.5

    best_model_selection_metric:
        Best Model 선택 기준

        현재:

            validation_loss

    validation_loss:
        Best Epoch의 Validation Average Loss

    validation_accuracy:
        Best Epoch의 Validation Accuracy

        내부 범위:

            0.0 ~ 1.0

    validation_sample_count:
        Best Epoch에서 평가한 Validation 이미지 수

        현재:

            1,327

    validation_batch_count:
        Best Epoch에서 평가한 Validation Batch 수

        현재:

            42
    """

    checkpoint_path: Path

    checkpoint_version: int

    model_name: str

    model_module: str

    loss_function_name: str

    optimizer_name: str

    epoch_number: int

    configured_epoch_count: int

    classification_threshold: float

    best_model_selection_metric: str

    validation_loss: float

    validation_accuracy: float

    validation_sample_count: int

    validation_batch_count: int

    def __post_init__(self) -> None:
        """
        LoadedCheckpointInfo의 모든 필드를 다시 검증한다.

        정상적인 Loader 실행에서는 이미 검증된 값만 전달된다.

        Dataclass를 외부에서 직접 생성하는 경우에도
        잘못된 Metadata가 생성되지 않도록 방어한다.
        """
        if not isinstance(
            self.checkpoint_path,
            Path,
        ):
            raise TypeError(
                "checkpoint_path must be a pathlib.Path. "
                f"Received type: "
                f"{type(self.checkpoint_path).__name__}."
            )

        if (
            self.checkpoint_path.suffix.lower()
            not in SUPPORTED_CHECKPOINT_SUFFIXES
        ):
            raise ValueError(
                "checkpoint_path must use a .pt or .pth extension. "
                f"Received path: {self.checkpoint_path}."
            )

        validated_checkpoint_version = (
            _validate_positive_integer(
                value=self.checkpoint_version,
                value_name="checkpoint_version",
            )
        )

        if (
            validated_checkpoint_version
            != CHECKPOINT_VERSION
        ):
            raise ValueError(
                "checkpoint_version is not supported. "
                f"Supported version: {CHECKPOINT_VERSION}. "
                f"Received version: "
                f"{validated_checkpoint_version}."
            )

        _ = _validate_non_empty_string(
            value=self.model_name,
            value_name="model_name",
        )

        _ = _validate_non_empty_string(
            value=self.model_module,
            value_name="model_module",
        )

        _ = _validate_non_empty_string(
            value=self.loss_function_name,
            value_name="loss_function_name",
        )

        _ = _validate_non_empty_string(
            value=self.optimizer_name,
            value_name="optimizer_name",
        )

        validated_epoch_number = (
            _validate_positive_integer(
                value=self.epoch_number,
                value_name="epoch_number",
            )
        )

        validated_epoch_count = (
            _validate_positive_integer(
                value=self.configured_epoch_count,
                value_name=(
                    "configured_epoch_count"
                ),
            )
        )

        if (
            validated_epoch_number
            > validated_epoch_count
        ):
            raise ValueError(
                "epoch_number must be less than or equal to "
                "configured_epoch_count. "
                f"Epoch: {validated_epoch_number}. "
                f"Configured epochs: "
                f"{validated_epoch_count}."
            )

        _ = _validate_probability(
            value=self.classification_threshold,
            value_name=(
                "classification_threshold"
            ),
        )

        validated_selection_metric = (
            _validate_non_empty_string(
                value=(
                    self
                    .best_model_selection_metric
                ),
                value_name=(
                    "best_model_selection_metric"
                ),
            )
        )

        if (
            validated_selection_metric
            != BEST_MODEL_SELECTION_METRIC
        ):
            raise ValueError(
                "best_model_selection_metric is not supported. "
                f"Expected: "
                f"{BEST_MODEL_SELECTION_METRIC}. "
                f"Received: "
                f"{validated_selection_metric}."
            )

        _ = _validate_non_negative_finite_real(
            value=self.validation_loss,
            value_name="validation_loss",
        )

        _ = _validate_probability(
            value=self.validation_accuracy,
            value_name="validation_accuracy",
        )

        _ = _validate_positive_integer(
            value=(
                self.validation_sample_count
            ),
            value_name=(
                "validation_sample_count"
            ),
        )

        _ = _validate_positive_integer(
            value=(
                self.validation_batch_count
            ),
            value_name=(
                "validation_batch_count"
            ),
        )


# =============================================================================
# Public Checkpoint Loading Function
# =============================================================================


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path | str,
    device: torch.device | str,
) -> LoadedCheckpointInfo:
    """
    검증된 PyTorch Checkpoint를 읽고 Model Weight를 복원한다.

    왜 필요한가
    -----------
    Training Pipeline이 끝난 뒤 메모리에 남는 Model은
    마지막 Epoch의 Model이다.

    현재 실제 학습:

        Last Epoch:

            5

        Best Epoch:

            2

    Test Dataset은 마지막 Epoch 5가 아니라
    Validation Loss가 가장 낮았던 Epoch 2 Model로 평가해야 한다.

    입력
    ----
    model:
        Weight를 복원할 PyTorch Model

        현재:

            새 CNNBaseline 객체

        권장 순서:

            model = CNNBaseline()

            model = model.to(
                device
            )

            load_model_checkpoint(
                model=model,
                ...
            )

    checkpoint_path:
        읽을 Checkpoint 파일

        허용:

            pathlib.Path

            문자열 Path

        확장자:

            .pt

            .pth

    device:
        Model과 Checkpoint Tensor를 배치할 Device

        현재:

            cpu

    처리 과정
    ---------
    1. Model 타입을 검증한다.
    2. Device를 torch.device로 변환한다.
    3. 현재 Model State가 존재하는지 확인한다.
    4. Model Parameter·Buffer Device를 확인한다.
    5. Checkpoint 경로를 검증한다.
    6. torch.load()로 Checkpoint를 읽는다.
    7. Checkpoint가 Mapping인지 확인한다.
    8. 최상위 필수 Key를 확인한다.
    9. Checkpoint Version을 확인한다.
    10. Model 이름을 현재 Model과 비교한다.
    11. Model Module을 현재 Model과 비교한다.
    12. Loss·Optimizer 이름을 검증한다.
    13. Epoch Metadata를 검증한다.
    14. Classification Threshold를 검증한다.
    15. Best Model 선택 기준을 검증한다.
    16. Train Result를 검증한다.
    17. Validation Result를 검증한다.
    18. Optimizer State 구조를 검증한다.
    19. Model State Dictionary를 검증한다.
    20. Checkpoint와 현재 Model State Key를 비교한다.
    21. Checkpoint State Tensor의 유한성을 확인한다.
    22. strict=True로 Weight를 복원한다.
    23. 복원 후 Parameter·Buffer Device를 확인한다.
    24. 복원 후 Parameter·Buffer 유한성을 확인한다.
    25. LoadedCheckpointInfo를 반환한다.

    출력
    ----
    LoadedCheckpointInfo

        Checkpoint 경로

        Version

        Model 이름

        Best Epoch

        Threshold

        Best Validation Loss

        Best Validation Accuracy

    Model 반환
    ----------
    Model 자체는 반환하지 않는다.

    입력으로 전달한 Model 객체에 Weight가 직접 복원된다.

    예:

        model = CNNBaseline()

        checkpoint_info = (
            load_model_checkpoint(
                model=model,
                checkpoint_path=...,
                device="cpu",
            )
        )

        # model은 이미 Best Weight를 가진다.

    Model Mode
    ----------
    이 함수는 다음을 호출하지 않는다.

        model.train()

        model.eval()

    즉 Loader 호출 전 Model Mode를 그대로 유지한다.

    Test 평가 함수에서 별도로:

        model.eval()

    을 호출한다.

    Strict Loading
    --------------
    다음 설정을 사용한다.

        strict=True

    따라서 다음 문제를 허용하지 않는다.

        Missing State Key

        Unexpected State Key

        Layer 이름 불일치

        Weight Shape 불일치

    Optimizer
    ---------
    Checkpoint의 optimizer_state_dict 구조는 확인한다.

    하지만 현재 Loader는 Test·Inference Model 복원이 목적이므로
    Optimizer 객체에는 State를 복원하지 않는다.

    예외 처리
    ---------
    잘못된 Model:

        TypeError

    Model State 없음:

        ValueError

    잘못된 Device:

        TypeError 또는 ValueError

    Model Device 불일치:

        ValueError

    Checkpoint 파일 없음:

        FileNotFoundError

    Checkpoint가 Directory:

        ValueError

    잘못된 확장자:

        ValueError

    읽을 수 없는 Checkpoint:

        RuntimeError

    필수 Key 누락:

        KeyError

    Version 불일치:

        ValueError

    Model 이름·Module 불일치:

        ValueError

    Epoch·Threshold 오류:

        TypeError 또는 ValueError

    State Key 불일치:

        ValueError

    State Weight Shape 불일치:

        ValueError

    NaN·inf Weight:

        ValueError

    테스트 방법
    -----------
    실제 Day 3 Checkpoint:

        models/checkpoints/cnn_baseline_best.pt

    새 CNNBaseline:

        Random Initial Weight

    복원 후:

        Best Epoch:

            2

        Configured Epoch:

            5

        Threshold:

            0.5

        Model State:

            Checkpoint State와 동일

        Parameter:

            CPU

            Finite
    """
    _validate_model(
        model=model,
    )

    resolved_device = _resolve_device(
        device=device,
    )

    _validate_model_state_exists(
        model=model,
    )

    # Checkpoint 복원 전에 Model이 이미 요청 Device에 있어야 한다.
    #
    # 현재 권장 순서:
    #
    #     model = CNNBaseline()
    #
    #     model = model.to(device)
    #
    #     load_model_checkpoint(...)
    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    resolved_checkpoint_path = (
        _resolve_checkpoint_path(
            checkpoint_path=checkpoint_path,
        )
    )

    checkpoint = _load_checkpoint_dictionary(
        checkpoint_path=(
            resolved_checkpoint_path
        ),
        device=resolved_device,
    )

    _validate_required_checkpoint_keys(
        checkpoint=checkpoint,
    )

    # ---------------------------------------------------------
    # Checkpoint Version
    # ---------------------------------------------------------
    checkpoint_version = (
        _validate_positive_integer(
            value=checkpoint[
                "checkpoint_version"
            ],
            value_name="checkpoint_version",
        )
    )

    if (
        checkpoint_version
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "checkpoint_version is not supported. "
            f"Supported version: {CHECKPOINT_VERSION}. "
            f"Received version: "
            f"{checkpoint_version}."
        )

    # ---------------------------------------------------------
    # Model Metadata
    # ---------------------------------------------------------
    model_name = (
        _validate_non_empty_string(
            value=checkpoint[
                "model_name"
            ],
            value_name="model_name",
        )
    )

    expected_model_name = (
        model.__class__.__name__
    )

    if model_name != expected_model_name:
        raise ValueError(
            "checkpoint model_name does not match "
            "the current Model. "
            f"Expected: {expected_model_name}. "
            f"Received: {model_name}."
        )

    model_module = (
        _validate_non_empty_string(
            value=checkpoint[
                "model_module"
            ],
            value_name="model_module",
        )
    )

    expected_model_module = (
        model.__class__.__module__
    )

    if (
        model_module
        != expected_model_module
    ):
        raise ValueError(
            "checkpoint model_module does not match "
            "the current Model. "
            f"Expected: {expected_model_module}. "
            f"Received: {model_module}."
        )

    loss_function_name = (
        _validate_non_empty_string(
            value=checkpoint[
                "loss_function_name"
            ],
            value_name=(
                "loss_function_name"
            ),
        )
    )

    optimizer_name = (
        _validate_non_empty_string(
            value=checkpoint[
                "optimizer_name"
            ],
            value_name="optimizer_name",
        )
    )

    # ---------------------------------------------------------
    # Epoch Metadata
    # ---------------------------------------------------------
    epoch_number = (
        _validate_positive_integer(
            value=checkpoint[
                "epoch_number"
            ],
            value_name="epoch_number",
        )
    )

    configured_epoch_count = (
        _validate_positive_integer(
            value=checkpoint[
                "configured_epoch_count"
            ],
            value_name=(
                "configured_epoch_count"
            ),
        )
    )

    if (
        epoch_number
        > configured_epoch_count
    ):
        raise ValueError(
            "checkpoint epoch_number must be less than "
            "or equal to configured_epoch_count. "
            f"Epoch: {epoch_number}. "
            f"Configured epochs: "
            f"{configured_epoch_count}."
        )

    # ---------------------------------------------------------
    # Classification Threshold
    # ---------------------------------------------------------
    classification_threshold = (
        _validate_probability(
            value=checkpoint[
                "classification_threshold"
            ],
            value_name=(
                "classification_threshold"
            ),
        )
    )

    # ---------------------------------------------------------
    # Best Model Selection Metric
    # ---------------------------------------------------------
    best_model_selection_metric = (
        _validate_non_empty_string(
            value=checkpoint[
                "best_model_selection_metric"
            ],
            value_name=(
                "best_model_selection_metric"
            ),
        )
    )

    if (
        best_model_selection_metric
        != BEST_MODEL_SELECTION_METRIC
    ):
        raise ValueError(
            "checkpoint best_model_selection_metric "
            "is not supported. "
            f"Expected: "
            f"{BEST_MODEL_SELECTION_METRIC}. "
            f"Received: "
            f"{best_model_selection_metric}."
        )

    # ---------------------------------------------------------
    # Train Result
    # ---------------------------------------------------------
    _ = _validate_epoch_result_mapping(
        result=checkpoint[
            "train_result"
        ],
        result_name="train_result",
    )

    # ---------------------------------------------------------
    # Validation Result
    # ---------------------------------------------------------
    (
        validation_loss,
        validation_accuracy,
        validation_sample_count,
        validation_batch_count,
    ) = _validate_epoch_result_mapping(
        result=checkpoint[
            "validation_result"
        ],
        result_name="validation_result",
    )

    # ---------------------------------------------------------
    # Optimizer State Structure
    # ---------------------------------------------------------
    _validate_optimizer_state_mapping(
        optimizer_state=checkpoint[
            "optimizer_state_dict"
        ],
    )

    # ---------------------------------------------------------
    # Model State
    # ---------------------------------------------------------
    model_state_dict = (
        _validate_model_state_mapping(
            model_state=checkpoint[
                "model_state_dict"
            ],
        )
    )

    _validate_model_state_keys(
        model=model,
        model_state_dict=(
            model_state_dict
        ),
    )

    _validate_checkpoint_state_tensors(
        model_state_dict=(
            model_state_dict
        ),
    )

    # ---------------------------------------------------------
    # Strict Weight Loading
    # ---------------------------------------------------------
    try:
        model.load_state_dict(
            state_dict=model_state_dict,
            strict=True,
        )

    except RuntimeError as error:
        raise ValueError(
            "checkpoint Model State could not be loaded "
            "into the current Model with strict=True. "
            "Check Layer names, State keys, and Tensor shapes."
        ) from error

    # ---------------------------------------------------------
    # Post-loading Validation
    # ---------------------------------------------------------
    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    _validate_loaded_model_state_finite(
        model=model,
    )

    return LoadedCheckpointInfo(
        checkpoint_path=(
            resolved_checkpoint_path
        ),
        checkpoint_version=(
            checkpoint_version
        ),
        model_name=model_name,
        model_module=model_module,
        loss_function_name=(
            loss_function_name
        ),
        optimizer_name=(
            optimizer_name
        ),
        epoch_number=epoch_number,
        configured_epoch_count=(
            configured_epoch_count
        ),
        classification_threshold=(
            classification_threshold
        ),
        best_model_selection_metric=(
            best_model_selection_metric
        ),
        validation_loss=(
            validation_loss
        ),
        validation_accuracy=(
            validation_accuracy
        ),
        validation_sample_count=(
            validation_sample_count
        ),
        validation_batch_count=(
            validation_batch_count
        ),
    )


# =============================================================================
# Basic Value Validation
# =============================================================================


def _validate_positive_integer(
    value: object,
    value_name: str,
) -> int:
    """
    bool이 아닌 1 이상의 정수를 검증한다.
    """
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Integral,
        )
    ):
        raise TypeError(
            f"{value_name} must be an integer. "
            f"Received type: "
            f"{type(value).__name__}."
        )

    validated_value = int(
        value
    )

    if validated_value <= 0:
        raise ValueError(
            f"{value_name} must be greater than 0. "
            f"Received value: "
            f"{validated_value}."
        )

    return validated_value


def _validate_non_empty_string(
    value: object,
    value_name: str,
) -> str:
    """
    비어 있지 않은 문자열을 검증한다.
    """
    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{value_name} must be a string. "
            f"Received type: "
            f"{type(value).__name__}."
        )

    validated_value = (
        value.strip()
    )

    if not validated_value:
        raise ValueError(
            f"{value_name} must not be empty."
        )

    return validated_value


def _validate_non_negative_finite_real(
    value: object,
    value_name: str,
) -> float:
    """
    bool이 아닌 유한한 0 이상의 실수를 검증한다.
    """
    if (
        isinstance(
            value,
            bool,
        )
        or not isinstance(
            value,
            Real,
        )
    ):
        raise TypeError(
            f"{value_name} must be a real number. "
            f"Received type: "
            f"{type(value).__name__}."
        )

    validated_value = float(
        value
    )

    if not math.isfinite(
        validated_value
    ):
        raise ValueError(
            f"{value_name} must be finite. "
            f"Received value: "
            f"{validated_value}."
        )

    if validated_value < 0.0:
        raise ValueError(
            f"{value_name} must be greater than "
            "or equal to 0. "
            f"Received value: "
            f"{validated_value}."
        )

    return validated_value


def _validate_probability(
    value: object,
    value_name: str,
) -> float:
    """
    유한한 0~1 범위의 실수를 검증한다.

    사용 대상
    ---------
    Classification Threshold

    Accuracy
    """
    validated_value = (
        _validate_non_negative_finite_real(
            value=value,
            value_name=value_name,
        )
    )

    if validated_value > 1.0:
        raise ValueError(
            f"{value_name} must be between 0 and 1. "
            f"Received value: "
            f"{validated_value}."
        )

    return validated_value


# =============================================================================
# Model Validation
# =============================================================================


def _validate_model(
    model: nn.Module,
) -> None:
    """
    Model이 torch.nn.Module인지 확인한다.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be an instance of torch.nn.Module. "
            f"Received type: "
            f"{type(model).__name__}."
        )


def _validate_model_state_exists(
    model: nn.Module,
) -> None:
    """
    Model에 복원할 State가 하나 이상 존재하는지 확인한다.

    Parameter 또는 Buffer가 모두 없는 Model은
    현재 Checkpoint Loader의 대상이 아니다.
    """
    if not model.state_dict():
        raise ValueError(
            "model must contain at least one Parameter "
            "or Buffer State."
        )


def _validate_model_device(
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    모든 Model Parameter·Buffer가 요청 Device에 있는지 확인한다.

    현재
    ----
    Model:

        cpu

    Device:

        cpu

    향후
    ----
    GPU 사용 시 Model을 먼저 GPU로 이동한 뒤
    Loader를 호출한다.
    """
    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():
        if parameter.device != device:
            raise ValueError(
                "all Model Parameters must be on "
                "the requested device. "
                f"Parameter: {parameter_name}. "
                f"Requested device: {device}. "
                f"Found device: "
                f"{parameter.device}."
            )

    for (
        buffer_name,
        buffer,
    ) in model.named_buffers():
        if buffer.device != device:
            raise ValueError(
                "all Model Buffers must be on "
                "the requested device. "
                f"Buffer: {buffer_name}. "
                f"Requested device: {device}. "
                f"Found device: "
                f"{buffer.device}."
            )


def _validate_loaded_model_state_finite(
    model: nn.Module,
) -> None:
    """
    Weight 복원 후 모든 Parameter·Buffer가 유한한지 확인한다.

    거부
    ----
    NaN

    positive infinity

    negative infinity
    """
    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():
        if not torch.isfinite(
            parameter.detach()
        ).all():
            raise ValueError(
                "loaded Model Parameter must contain "
                "only finite values. "
                f"Invalid Parameter: "
                f"{parameter_name}."
            )

    for (
        buffer_name,
        buffer,
    ) in model.named_buffers():
        if not torch.isfinite(
            buffer.detach()
        ).all():
            raise ValueError(
                "loaded Model Buffer must contain "
                "only finite values. "
                f"Invalid Buffer: "
                f"{buffer_name}."
            )


# =============================================================================
# Device Validation
# =============================================================================


def _resolve_device(
    device: torch.device | str,
) -> torch.device:
    """
    문자열 또는 torch.device를 검증하고 torch.device로 반환한다.
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
            f"Received type: "
            f"{type(device).__name__}."
        )

    try:
        resolved_device = (
            torch.device(
                device
            )
        )

    except (
        RuntimeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid device: {device}."
        ) from error

    if (
        resolved_device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CUDA device was requested, "
            "but CUDA is not available."
        )

    return resolved_device


# =============================================================================
# Checkpoint Path Validation
# =============================================================================


def _resolve_checkpoint_path(
    checkpoint_path: Path | str,
) -> Path:
    """
    Checkpoint Path를 pathlib.Path로 변환하고 검증한다.

    검증
    ----
    Path 또는 문자열

    비어 있지 않은 문자열

    .pt 또는 .pth

    실제 존재

    File

    Directory 아님
    """
    if isinstance(
        checkpoint_path,
        str,
    ):
        if not checkpoint_path.strip():
            raise ValueError(
                "checkpoint_path string "
                "must not be empty."
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
            "checkpoint_path must be a string "
            "or pathlib.Path. "
            f"Received type: "
            f"{type(checkpoint_path).__name__}."
        )

    if (
        resolved_path.suffix.lower()
        not in SUPPORTED_CHECKPOINT_SUFFIXES
    ):
        raise ValueError(
            "checkpoint_path must use a "
            ".pt or .pth extension. "
            f"Received path: "
            f"{resolved_path}."
        )

    if not resolved_path.exists():
        raise FileNotFoundError(
            "checkpoint file does not exist. "
            f"Received path: "
            f"{resolved_path}."
        )

    if resolved_path.is_dir():
        raise ValueError(
            "checkpoint_path must point to a file, "
            "not a directory. "
            f"Received path: "
            f"{resolved_path}."
        )

    if not resolved_path.is_file():
        raise ValueError(
            "checkpoint_path must point to "
            "a regular file. "
            f"Received path: "
            f"{resolved_path}."
        )

    return resolved_path


# =============================================================================
# Checkpoint File Loading
# =============================================================================


def _load_checkpoint_dictionary(
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    """
    Checkpoint 파일을 지정 Device로 읽고 Dictionary로 반환한다.

    현재 Checkpoint는 이 프로젝트가 직접 생성한
    신뢰 가능한 로컬 Artifact다.
    """
    try:
        checkpoint_object = (
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
        )

    except (
        RuntimeError,
        EOFError,
        OSError,
        ValueError,
        pickle.UnpicklingError,
    ) as error:
        raise RuntimeError(
            "failed to load the checkpoint file. "
            f"Checkpoint path: "
            f"{checkpoint_path}."
        ) from error

    if not isinstance(
        checkpoint_object,
        Mapping,
    ):
        raise TypeError(
            "loaded checkpoint must be a Mapping. "
            f"Received type: "
            f"{type(checkpoint_object).__name__}."
        )

    return dict(
        checkpoint_object
    )


# =============================================================================
# Checkpoint Structure Validation
# =============================================================================


def _validate_required_checkpoint_keys(
    checkpoint: Mapping[str, object],
) -> None:
    """
    최상위 Checkpoint 필수 Key가 모두 존재하는지 확인한다.

    추가 Key는 허용한다.
    """
    checkpoint_keys = set(
        checkpoint.keys()
    )

    missing_keys = (
        REQUIRED_CHECKPOINT_KEYS
        - checkpoint_keys
    )

    if missing_keys:
        raise KeyError(
            "checkpoint is missing required keys. "
            f"Missing keys: "
            f"{sorted(missing_keys)}."
        )


def _validate_epoch_result_mapping(
    result: object,
    result_name: str,
) -> tuple[
    float,
    float,
    int,
    int,
]:
    """
    Train·Validation Result Dictionary를 검증한다.

    출력
    ----
    average_loss

    accuracy

    sample_count

    batch_count
    """
    if not isinstance(
        result,
        Mapping,
    ):
        raise TypeError(
            f"{result_name} must be a Mapping. "
            f"Received type: "
            f"{type(result).__name__}."
        )

    result_keys = set(
        result.keys()
    )

    missing_keys = (
        REQUIRED_EPOCH_RESULT_KEYS
        - result_keys
    )

    if missing_keys:
        raise KeyError(
            f"{result_name} is missing required keys. "
            f"Missing keys: "
            f"{sorted(missing_keys)}."
        )

    average_loss = (
        _validate_non_negative_finite_real(
            value=result[
                "average_loss"
            ],
            value_name=(
                f"{result_name}.average_loss"
            ),
        )
    )

    accuracy = (
        _validate_probability(
            value=result[
                "accuracy"
            ],
            value_name=(
                f"{result_name}.accuracy"
            ),
        )
    )

    sample_count = (
        _validate_positive_integer(
            value=result[
                "sample_count"
            ],
            value_name=(
                f"{result_name}.sample_count"
            ),
        )
    )

    batch_count = (
        _validate_positive_integer(
            value=result[
                "batch_count"
            ],
            value_name=(
                f"{result_name}.batch_count"
            ),
        )
    )

    return (
        average_loss,
        accuracy,
        sample_count,
        batch_count,
    )


def _validate_optimizer_state_mapping(
    optimizer_state: object,
) -> None:
    """
    Checkpoint의 Optimizer State 기본 구조를 검증한다.

    현재 Test·Inference Loading에서는
    Optimizer 객체에 State를 복원하지 않는다.

    하지만 정상 Training Checkpoint인지 확인하기 위해
    기본 구조는 검증한다.
    """
    if not isinstance(
        optimizer_state,
        Mapping,
    ):
        raise TypeError(
            "optimizer_state_dict must be a Mapping. "
            f"Received type: "
            f"{type(optimizer_state).__name__}."
        )

    optimizer_state_keys = set(
        optimizer_state.keys()
    )

    missing_keys = (
        REQUIRED_OPTIMIZER_STATE_KEYS
        - optimizer_state_keys
    )

    if missing_keys:
        raise KeyError(
            "optimizer_state_dict is missing "
            "required keys. "
            f"Missing keys: "
            f"{sorted(missing_keys)}."
        )


# =============================================================================
# Model State Validation
# =============================================================================


def _validate_model_state_mapping(
    model_state: object,
) -> dict[str, Tensor]:
    """
    Checkpoint Model State를 검증하고 Dictionary로 반환한다.

    검증
    ----
    Mapping

    비어 있지 않음

    Key:

        비어 있지 않은 문자열

    Value:

        torch.Tensor
    """
    if not isinstance(
        model_state,
        Mapping,
    ):
        raise TypeError(
            "model_state_dict must be a Mapping. "
            f"Received type: "
            f"{type(model_state).__name__}."
        )

    if not model_state:
        raise ValueError(
            "model_state_dict must not be empty."
        )

    validated_state: dict[
        str,
        Tensor,
    ] = {}

    for (
        state_name,
        state_value,
    ) in model_state.items():
        validated_state_name = (
            _validate_non_empty_string(
                value=state_name,
                value_name=(
                    "model_state_dict key"
                ),
            )
        )

        if not isinstance(
            state_value,
            Tensor,
        ):
            raise TypeError(
                "every model_state_dict value "
                "must be a torch.Tensor. "
                f"State: "
                f"{validated_state_name}. "
                f"Received type: "
                f"{type(state_value).__name__}."
            )

        validated_state[
            validated_state_name
        ] = state_value

    return validated_state


def _validate_model_state_keys(
    model: nn.Module,
    model_state_dict: Mapping[
        str,
        Tensor,
    ],
) -> None:
    """
    Checkpoint State Key와 현재 Model State Key를 정확히 비교한다.

    오류
    ----
    Missing Key

    Unexpected Key
    """
    expected_keys = set(
        model.state_dict().keys()
    )

    checkpoint_keys = set(
        model_state_dict.keys()
    )

    missing_keys = (
        expected_keys
        - checkpoint_keys
    )

    unexpected_keys = (
        checkpoint_keys
        - expected_keys
    )

    if (
        missing_keys
        or unexpected_keys
    ):
        raise ValueError(
            "checkpoint Model State keys do not "
            "exactly match the current Model. "
            f"Missing keys: "
            f"{sorted(missing_keys)}. "
            f"Unexpected keys: "
            f"{sorted(unexpected_keys)}."
        )


def _validate_checkpoint_state_tensors(
    model_state_dict: Mapping[
        str,
        Tensor,
    ],
) -> None:
    """
    Checkpoint에 저장된 모든 Model State Tensor가 유한한지 확인한다.

    현재 CNNBaseline:
        Parameter Tensor

    향후 ResNet18:
        Parameter Tensor

        BatchNorm Buffer Tensor
    """
    for (
        state_name,
        state_tensor,
    ) in model_state_dict.items():
        if not torch.isfinite(
            state_tensor
        ).all():
            raise ValueError(
                "checkpoint Model State Tensor must "
                "contain only finite values. "
                f"Invalid State: "
                f"{state_name}."
            )