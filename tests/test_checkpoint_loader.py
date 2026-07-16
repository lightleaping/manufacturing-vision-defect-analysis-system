"""
Validated PyTorch checkpoint loader unit tests.

테스트 대상
----------
src/training/checkpoint_loader.py

테스트 목적
----------
Best Model Checkpoint가 현재 Model에 정확히 복원되는지 확인한다.

정상 Checkpoint뿐 아니라 다음 오류도 자동 검증한다.

    존재하지 않는 파일

    잘못된 확장자

    손상된 파일

    잘못된 Checkpoint 객체

    필수 Key 누락

    지원하지 않는 Version

    Model 이름 불일치

    Model Module 불일치

    잘못된 Epoch

    잘못된 Threshold

    잘못된 Validation Result

    잘못된 Optimizer State

    Model State Key 누락

    예상하지 않은 Model State Key

    Weight Shape 불일치

    NaN Weight

    Infinity Weight

현재 Model
----------
CNNBaseline

현재 Checkpoint Version
-----------------------
1

현재 Best Model 선택 기준
------------------------
validation_loss

현재 실제 학습 결과
--------------------
Best Epoch:

    2

Configured Epoch:

    5

Classification Threshold:

    0.5

이 테스트에서는 실제 프로젝트 Checkpoint를 수정하지 않는다.

pytest의 tmp_path에 독립적인 테스트 Checkpoint를 생성한다.
"""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
)
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from src.models.cnn_baseline import (
    CNNBaseline,
)
from src.training.checkpoint_loader import (
    REQUIRED_CHECKPOINT_KEYS,
    REQUIRED_EPOCH_RESULT_KEYS,
    REQUIRED_OPTIMIZER_STATE_KEYS,
    SUPPORTED_CHECKPOINT_SUFFIXES,
    LoadedCheckpointInfo,
    load_model_checkpoint,
)
from src.training.training_pipeline import (
    BEST_MODEL_SELECTION_METRIC,
    CHECKPOINT_VERSION,
)


# =============================================================================
# Test Models
# =============================================================================


class ParameterlessModel(nn.Module):
    """
    Parameter와 Buffer가 모두 없는 잘못된 Model.

    Checkpoint Loader는 복원할 State가 없는 Model을 거부해야 한다.
    """

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Batch 크기만큼 0 Logit을 반환한다.
        """
        return torch.zeros(
            images.shape[0],
            dtype=torch.float32,
            device=images.device,
        )


class BufferDeviceMismatchModel(
    nn.Module
):
    """
    Parameter와 Buffer의 Device가 다른 테스트 Model.

    Parameter:

        CPU

    Buffer:

        Meta

    Loader 호출 전 Device 검증에서 거부되어야 한다.
    """

    def __init__(self) -> None:
        """
        CPU Parameter와 Meta Buffer를 생성한다.
        """
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "meta_buffer",
            torch.empty(
                1,
                device="meta",
            ),
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        테스트에서는 Forward까지 실행하지 않는다.
        """
        return (
            images.mean(
                dim=(
                    1,
                    2,
                    3,
                ),
            )
            * self.weight
        )


# =============================================================================
# Test Data Creation
# =============================================================================


def clone_model_state(
    model: nn.Module,
) -> dict[str, Tensor]:
    """
    Model State를 독립적인 CPU Tensor Dictionary로 복사한다.

    왜 clone하는가
    --------------
    state_dict() Tensor는 Model Storage와 연결될 수 있다.

    테스트 중 Model이 변경되어도
    이전 State가 같이 변경되지 않도록 clone한다.
    """
    return {
        state_name: (
            state_tensor
            .detach()
            .cpu()
            .clone()
        )
        for (
            state_name,
            state_tensor,
        ) in model.state_dict().items()
    }


def state_dicts_are_equal(
    first_state: dict[
        str,
        Tensor,
    ],
    second_state: dict[
        str,
        Tensor,
    ],
) -> bool:
    """
    두 Model State의 Key·Tensor 값을 정확하게 비교한다.
    """
    if (
        set(
            first_state.keys()
        )
        != set(
            second_state.keys()
        )
    ):
        return False

    return all(
        torch.equal(
            first_state[
                state_name
            ],
            second_state[
                state_name
            ],
        )
        for state_name in (
            first_state
        )
    )


def create_source_model() -> CNNBaseline:
    """
    Checkpoint 원본 Weight를 가진 CNNBaseline을 생성한다.

    모든 Parameter를 서로 다른 일정한 값으로 채운다.

    목적
    ----
    새 Target Model의 Random Initial Weight와
    Checkpoint Weight가 확실히 다르게 만든다.
    """
    model = CNNBaseline()

    with torch.no_grad():
        for (
            parameter_index,
            parameter,
        ) in enumerate(
            model.parameters(),
            start=1,
        ):
            parameter.fill_(
                parameter_index
                * 0.01
            )

    return model


def create_epoch_result_payload(
    *,
    average_loss: float,
    accuracy: float,
    sample_count: int,
    batch_count: int,
) -> dict[str, object]:
    """
    Checkpoint 내부 Epoch Result Dictionary를 생성한다.
    """
    return {
        "average_loss": (
            average_loss
        ),
        "accuracy": (
            accuracy
        ),
        "sample_count": (
            sample_count
        ),
        "batch_count": (
            batch_count
        ),
    }


def create_valid_checkpoint(
    model: nn.Module,
) -> dict[str, object]:
    """
    현재 training_pipeline.py 형식과 일치하는
    정상 테스트 Checkpoint Dictionary를 생성한다.

    현재 테스트 Metadata
    --------------------
    Best Epoch:

        2

    Configured Epoch:

        5

    Threshold:

        0.5

    Validation Loss:

        0.465499

    Validation Accuracy:

        0.769405
    """
    return {
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
            "BCEWithLogitsLoss"
        ),
        "optimizer_name": (
            "Adam"
        ),
        "epoch_number": (
            2
        ),
        "configured_epoch_count": (
            5
        ),
        "classification_threshold": (
            0.5
        ),
        "best_model_selection_metric": (
            BEST_MODEL_SELECTION_METRIC
        ),
        "model_state_dict": (
            clone_model_state(
                model=model,
            )
        ),
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [],
        },
        "train_result": (
            create_epoch_result_payload(
                average_loss=0.469879,
                accuracy=0.791557,
                sample_count=5_306,
                batch_count=166,
            )
        ),
        "validation_result": (
            create_epoch_result_payload(
                average_loss=0.465499,
                accuracy=0.769405,
                sample_count=1_327,
                batch_count=42,
            )
        ),
    }


def save_checkpoint(
    checkpoint_path: Path,
    checkpoint: object,
) -> None:
    """
    테스트 Checkpoint 객체를 저장한다.
    """
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def create_valid_checkpoint_file(
    tmp_path: Path,
    *,
    suffix: str = ".pt",
) -> tuple[
    Path,
    CNNBaseline,
    dict[str, object],
]:
    """
    정상 Source Model과 Checkpoint 파일을 생성한다.

    출력
    ----
    Checkpoint Path

    Source CNNBaseline

    Checkpoint Dictionary
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint_path = (
        tmp_path
        / f"best{suffix}"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    return (
        checkpoint_path,
        source_model,
        checkpoint,
    )


def create_loaded_checkpoint_info(
    checkpoint_path: Path,
) -> LoadedCheckpointInfo:
    """
    정상 LoadedCheckpointInfo를 생성한다.
    """
    return LoadedCheckpointInfo(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint_version=(
            CHECKPOINT_VERSION
        ),
        model_name=(
            "CNNBaseline"
        ),
        model_module=(
            "src.models.cnn_baseline"
        ),
        loss_function_name=(
            "BCEWithLogitsLoss"
        ),
        optimizer_name=(
            "Adam"
        ),
        epoch_number=(
            2
        ),
        configured_epoch_count=(
            5
        ),
        classification_threshold=(
            0.5
        ),
        best_model_selection_metric=(
            BEST_MODEL_SELECTION_METRIC
        ),
        validation_loss=(
            0.465499
        ),
        validation_accuracy=(
            0.769405
        ),
        validation_sample_count=(
            1_327
        ),
        validation_batch_count=(
            42
        ),
    )


# =============================================================================
# Public Constants
# =============================================================================


def test_supported_checkpoint_suffixes() -> None:
    """
    .pt·.pth 확장자를 지원하는지 확인한다.
    """
    assert (
        SUPPORTED_CHECKPOINT_SUFFIXES
        == frozenset(
            {
                ".pt",
                ".pth",
            }
        )
    )


def test_required_checkpoint_keys_match_current_format() -> None:
    """
    현재 Training Pipeline Checkpoint 필수 Key를 확인한다.
    """
    expected_keys = {
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

    assert (
        REQUIRED_CHECKPOINT_KEYS
        == frozenset(
            expected_keys
        )
    )


def test_required_epoch_result_keys() -> None:
    """
    Train·Validation Result 필수 Key를 확인한다.
    """
    assert (
        REQUIRED_EPOCH_RESULT_KEYS
        == frozenset(
            {
                "average_loss",
                "accuracy",
                "sample_count",
                "batch_count",
            }
        )
    )


def test_required_optimizer_state_keys() -> None:
    """
    Optimizer State 필수 Key를 확인한다.
    """
    assert (
        REQUIRED_OPTIMIZER_STATE_KEYS
        == frozenset(
            {
                "state",
                "param_groups",
            }
        )
    )


# =============================================================================
# LoadedCheckpointInfo
# =============================================================================


def test_loaded_checkpoint_info_stores_expected_values(
    tmp_path: Path,
) -> None:
    """
    검증된 Checkpoint Metadata를 정상 저장하는지 확인한다.
    """
    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    info = (
        create_loaded_checkpoint_info(
            checkpoint_path=(
                checkpoint_path
            )
        )
    )

    assert (
        info.checkpoint_path
        == checkpoint_path
    )

    assert (
        info.checkpoint_version
        == 1
    )

    assert (
        info.model_name
        == "CNNBaseline"
    )

    assert (
        info.model_module
        == "src.models.cnn_baseline"
    )

    assert (
        info.loss_function_name
        == "BCEWithLogitsLoss"
    )

    assert (
        info.optimizer_name
        == "Adam"
    )

    assert info.epoch_number == 2

    assert (
        info.configured_epoch_count
        == 5
    )

    assert (
        info.classification_threshold
        == 0.5
    )

    assert (
        info.best_model_selection_metric
        == "validation_loss"
    )

    assert (
        info.validation_loss
        == 0.465499
    )

    assert (
        info.validation_accuracy
        == 0.769405
    )

    assert (
        info.validation_sample_count
        == 1_327
    )

    assert (
        info.validation_batch_count
        == 42
    )


def test_loaded_checkpoint_info_is_frozen(
    tmp_path: Path,
) -> None:
    """
    생성된 Checkpoint Metadata를 변경할 수 없는지 확인한다.
    """
    info = (
        create_loaded_checkpoint_info(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            )
        )
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        info.epoch_number = 3  # type: ignore[misc]


def test_loaded_checkpoint_info_rejects_non_path() -> None:
    """
    pathlib.Path가 아닌 Checkpoint Path를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "checkpoint_path must be "
            "a pathlib.Path"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path="best.pt",  # type: ignore[arg-type]
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


def test_loaded_checkpoint_info_rejects_unsupported_suffix(
    tmp_path: Path,
) -> None:
    """
    .pt·.pth 이외 Metadata Path를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "must use a .pt "
            "or .pth extension"
        ),
    ):
        create_loaded_checkpoint_info(
            checkpoint_path=(
                tmp_path
                / "best.json"
            )
        )


@pytest.mark.parametrize(
    "invalid_version",
    [
        True,
        False,
        1.5,
        "1",
        None,
    ],
)
def test_loaded_checkpoint_info_rejects_invalid_version_type(
    invalid_version: object,
    tmp_path: Path,
) -> None:
    """
    정수가 아닌 Version과 bool을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "checkpoint_version "
            "must be an integer"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=invalid_version,  # type: ignore[arg-type]
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    "invalid_version",
    [
        0,
        -1,
    ],
)
def test_loaded_checkpoint_info_rejects_non_positive_version(
    invalid_version: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Checkpoint Version을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_version "
            "must be greater than 0"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=(
                invalid_version
            ),
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


def test_loaded_checkpoint_info_rejects_unsupported_version(
    tmp_path: Path,
) -> None:
    """
    현재 Version 1 이외의 Version을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_version "
            "is not supported"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=2,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        (
            "model_name",
            "",
        ),
        (
            "model_module",
            "   ",
        ),
        (
            "loss_function_name",
            "",
        ),
        (
            "optimizer_name",
            " ",
        ),
    ],
)
def test_loaded_checkpoint_info_rejects_empty_string_fields(
    field_name: str,
    invalid_value: str,
    tmp_path: Path,
) -> None:
    """
    필수 문자열 Metadata의 빈 값을 거부하는지 확인한다.
    """
    values: dict[str, object] = {
        "checkpoint_path": (
            tmp_path
            / "best.pt"
        ),
        "checkpoint_version": 1,
        "model_name": "CNNBaseline",
        "model_module": (
            "src.models.cnn_baseline"
        ),
        "loss_function_name": (
            "BCEWithLogitsLoss"
        ),
        "optimizer_name": "Adam",
        "epoch_number": 2,
        "configured_epoch_count": 5,
        "classification_threshold": 0.5,
        "best_model_selection_metric": (
            "validation_loss"
        ),
        "validation_loss": 0.4,
        "validation_accuracy": 0.8,
        "validation_sample_count": 100,
        "validation_batch_count": 4,
    }

    values[
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} "
            "must not be empty"
        ),
    ):
        LoadedCheckpointInfo(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "epoch_number",
        "configured_epoch_count",
    ),
    [
        (
            6,
            5,
        ),
        (
            2,
            1,
        ),
    ],
)
def test_loaded_checkpoint_info_rejects_epoch_larger_than_configured_count(
    epoch_number: int,
    configured_epoch_count: int,
    tmp_path: Path,
) -> None:
    """
    Best Epoch가 설정 Epoch 수보다 크면 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "epoch_number must be "
            "less than or equal to"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=(
                epoch_number
            ),
            configured_epoch_count=(
                configured_epoch_count
            ),
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loaded_checkpoint_info_rejects_invalid_threshold(
    invalid_threshold: float,
    tmp_path: Path,
) -> None:
    """
    범위 밖·NaN·inf Threshold를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=(
                invalid_threshold
            ),
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


def test_loaded_checkpoint_info_rejects_unsupported_selection_metric(
    tmp_path: Path,
) -> None:
    """
    현재 지원하지 않는 Best Model 선택 기준을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "best_model_selection_metric "
            "is not supported"
        ),
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_accuracy"
            ),
            validation_loss=0.4,
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    "invalid_loss",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loaded_checkpoint_info_rejects_invalid_validation_loss(
    invalid_loss: float,
    tmp_path: Path,
) -> None:
    """
    음수·NaN·inf Validation Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=(
                invalid_loss
            ),
            validation_accuracy=0.8,
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loaded_checkpoint_info_rejects_invalid_validation_accuracy(
    invalid_accuracy: float,
    tmp_path: Path,
) -> None:
    """
    범위 밖·NaN·inf Validation Accuracy를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
    ):
        LoadedCheckpointInfo(
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            checkpoint_version=1,
            model_name="CNNBaseline",
            model_module=(
                "src.models.cnn_baseline"
            ),
            loss_function_name=(
                "BCEWithLogitsLoss"
            ),
            optimizer_name="Adam",
            epoch_number=2,
            configured_epoch_count=5,
            classification_threshold=0.5,
            best_model_selection_metric=(
                "validation_loss"
            ),
            validation_loss=0.4,
            validation_accuracy=(
                invalid_accuracy
            ),
            validation_sample_count=100,
            validation_batch_count=4,
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        (
            "validation_sample_count",
            0,
        ),
        (
            "validation_sample_count",
            -1,
        ),
        (
            "validation_batch_count",
            0,
        ),
        (
            "validation_batch_count",
            -1,
        ),
    ],
)
def test_loaded_checkpoint_info_rejects_non_positive_counts(
    field_name: str,
    invalid_value: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Validation Sample·Batch 수를 거부하는지 확인한다.
    """
    values: dict[str, object] = {
        "checkpoint_path": (
            tmp_path
            / "best.pt"
        ),
        "checkpoint_version": 1,
        "model_name": "CNNBaseline",
        "model_module": (
            "src.models.cnn_baseline"
        ),
        "loss_function_name": (
            "BCEWithLogitsLoss"
        ),
        "optimizer_name": "Adam",
        "epoch_number": 2,
        "configured_epoch_count": 5,
        "classification_threshold": 0.5,
        "best_model_selection_metric": (
            "validation_loss"
        ),
        "validation_loss": 0.4,
        "validation_accuracy": 0.8,
        "validation_sample_count": 100,
        "validation_batch_count": 4,
    }

    values[
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} "
            "must be greater than 0"
        ),
    ):
        LoadedCheckpointInfo(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# Successful Loading
# =============================================================================


def test_load_model_checkpoint_returns_loaded_checkpoint_info(
    tmp_path: Path,
) -> None:
    """
    정상 Checkpoint Loading 후 Metadata 객체를 반환하는지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    info = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    assert isinstance(
        info,
        LoadedCheckpointInfo,
    )


def test_load_model_checkpoint_returns_expected_metadata(
    tmp_path: Path,
) -> None:
    """
    정상 Checkpoint Metadata가 정확히 반환되는지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    info = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    assert (
        info.checkpoint_path
        == checkpoint_path
    )

    assert (
        info.checkpoint_version
        == CHECKPOINT_VERSION
    )

    assert (
        info.model_name
        == "CNNBaseline"
    )

    assert (
        info.model_module
        == "src.models.cnn_baseline"
    )

    assert (
        info.loss_function_name
        == "BCEWithLogitsLoss"
    )

    assert (
        info.optimizer_name
        == "Adam"
    )

    assert info.epoch_number == 2

    assert (
        info.configured_epoch_count
        == 5
    )

    assert (
        info.classification_threshold
        == 0.5
    )

    assert (
        info.best_model_selection_metric
        == BEST_MODEL_SELECTION_METRIC
    )

    assert (
        info.validation_loss
        == 0.465499
    )

    assert (
        info.validation_accuracy
        == 0.769405
    )

    assert (
        info.validation_sample_count
        == 1_327
    )

    assert (
        info.validation_batch_count
        == 42
    )


def test_load_model_checkpoint_replaces_target_model_state(
    tmp_path: Path,
) -> None:
    """
    새 Target Model의 초기 Weight가 Checkpoint Weight로 변경되는지 확인한다.
    """
    (
        checkpoint_path,
        source_model,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    target_state_before = (
        clone_model_state(
            model=target_model,
        )
    )

    source_state = (
        clone_model_state(
            model=source_model,
        )
    )

    assert (
        state_dicts_are_equal(
            first_state=(
                target_state_before
            ),
            second_state=(
                source_state
            ),
        )
        is False
    )

    _ = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    target_state_after = (
        clone_model_state(
            model=target_model,
        )
    )

    assert state_dicts_are_equal(
        first_state=(
            target_state_after
        ),
        second_state=(
            source_state
        ),
    )


def test_loaded_state_exactly_matches_checkpoint_state(
    tmp_path: Path,
) -> None:
    """
    복원된 Model State 값이 저장된 Checkpoint Tensor와 정확히 같은지 확인한다.
    """
    (
        checkpoint_path,
        _,
        checkpoint,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    _ = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    checkpoint_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        checkpoint_state,
        dict,
    )

    loaded_state = (
        clone_model_state(
            model=target_model,
        )
    )

    assert state_dicts_are_equal(
        first_state=(
            loaded_state
        ),
        second_state=(
            checkpoint_state
        ),
    )


@pytest.mark.parametrize(
    "training_mode",
    [
        True,
        False,
    ],
)
def test_loader_preserves_model_training_mode(
    training_mode: bool,
    tmp_path: Path,
) -> None:
    """
    Loader가 Train·Evaluation Mode를 변경하지 않는지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    target_model.train(
        mode=training_mode
    )

    mode_before = (
        target_model.training
    )

    _ = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    mode_after = (
        target_model.training
    )

    assert mode_before is (
        training_mode
    )

    assert mode_after is (
        training_mode
    )


def test_loader_accepts_string_checkpoint_path(
    tmp_path: Path,
) -> None:
    """
    문자열 Checkpoint Path를 허용하는지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    info = load_model_checkpoint(
        model=target_model,
        checkpoint_path=str(
            checkpoint_path
        ),
        device="cpu",
    )

    assert (
        info.checkpoint_path
        == checkpoint_path
    )


@pytest.mark.parametrize(
    "suffix",
    [
        ".pt",
        ".pth",
        ".PT",
        ".PTH",
    ],
)
def test_loader_accepts_supported_checkpoint_suffixes(
    suffix: str,
    tmp_path: Path,
) -> None:
    """
    .pt·.pth와 대문자 확장자를 허용하는지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
        suffix=suffix,
    )

    target_model = (
        CNNBaseline()
    )

    info = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device=torch.device(
            "cpu"
        ),
    )

    assert (
        info.checkpoint_path
        == checkpoint_path
    )


def test_loader_allows_additional_checkpoint_keys(
    tmp_path: Path,
) -> None:
    """
    필수 Key가 모두 존재하면 추가 Metadata Key를 허용하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "future_metadata"
    ] = {
        "note": (
            "additional key"
        ),
    }

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    target_model = (
        CNNBaseline()
    )

    info = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    assert (
        info.model_name
        == "CNNBaseline"
    )


def test_loaded_model_parameters_are_on_cpu_and_finite(
    tmp_path: Path,
) -> None:
    """
    복원 후 모든 Parameter가 CPU에 있고 유한한지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    _ = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    assert all(
        parameter.device
        == torch.device(
            "cpu"
        )
        for parameter in (
            target_model.parameters()
        )
    )

    assert all(
        torch.isfinite(
            parameter.detach()
        ).all().item()
        for parameter in (
            target_model.parameters()
        )
    )


def test_loaded_model_can_run_forward(
    tmp_path: Path,
) -> None:
    """
    복원된 CNNBaseline이 정상 Forward 가능한지 확인한다.
    """
    (
        checkpoint_path,
        _,
        _,
    ) = create_valid_checkpoint_file(
        tmp_path=tmp_path,
    )

    target_model = (
        CNNBaseline()
    )

    _ = load_model_checkpoint(
        model=target_model,
        checkpoint_path=(
            checkpoint_path
        ),
        device="cpu",
    )

    target_model.eval()

    images = torch.zeros(
        2,
        3,
        224,
        224,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        logits = target_model(
            images
        )

    assert logits.shape == (
        2,
    )

    assert (
        logits.dtype
        == torch.float32
    )

    assert torch.isfinite(
        logits
    ).all()


# =============================================================================
# Invalid Model
# =============================================================================


@pytest.mark.parametrize(
    "invalid_model",
    [
        None,
        "CNNBaseline",
        123,
        object(),
    ],
)
def test_loader_rejects_non_module_model(
    invalid_model: object,
    tmp_path: Path,
) -> None:
    """
    nn.Module이 아닌 Model을 거부하는지 확인한다.
    """
    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    with pytest.raises(
        TypeError,
        match=(
            "model must be an instance "
            "of torch.nn.Module"
        ),
    ):
        load_model_checkpoint(
            model=invalid_model,  # type: ignore[arg-type]
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_model_without_state(
    tmp_path: Path,
) -> None:
    """
    Parameter·Buffer State가 없는 Model을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "model must contain at least "
            "one Parameter or Buffer State"
        ),
    ):
        load_model_checkpoint(
            model=(
                ParameterlessModel()
            ),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device="cpu",
        )


def test_loader_rejects_model_parameter_device_mismatch(
    tmp_path: Path,
) -> None:
    """
    Model Parameter와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = CNNBaseline()

    model = model.to(
        device="meta"
    )

    with pytest.raises(
        ValueError,
        match=(
            "all Model Parameters "
            "must be on the requested device"
        ),
    ):
        load_model_checkpoint(
            model=model,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device="cpu",
        )


def test_loader_rejects_model_buffer_device_mismatch(
    tmp_path: Path,
) -> None:
    """
    Model Buffer와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = (
        BufferDeviceMismatchModel()
    )

    with pytest.raises(
        ValueError,
        match=(
            "all Model Buffers must be "
            "on the requested device"
        ),
    ):
        load_model_checkpoint(
            model=model,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Device
# =============================================================================


@pytest.mark.parametrize(
    "invalid_device",
    [
        None,
        0,
        1.5,
        object(),
    ],
)
def test_loader_rejects_invalid_device_type(
    invalid_device: object,
    tmp_path: Path,
) -> None:
    """
    문자열 또는 torch.device가 아닌 Device를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "device must be a string "
            "or torch.device"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device=invalid_device,  # type: ignore[arg-type]
        )


def test_loader_rejects_invalid_device_string(
    tmp_path: Path,
) -> None:
    """
    PyTorch가 해석할 수 없는 Device 문자열을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="Invalid device",
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device=(
                "not-a-device"
            ),
        )


def test_loader_rejects_unavailable_cuda(
    tmp_path: Path,
) -> None:
    """
    CUDA가 없는 환경에서 CUDA Device 요청을 거부하는지 확인한다.
    """
    if torch.cuda.is_available():
        pytest.skip(
            "CUDA is available."
        )

    with pytest.raises(
        ValueError,
        match=(
            "CUDA device was requested"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            device="cuda",
        )


# =============================================================================
# Invalid Checkpoint Path
# =============================================================================


@pytest.mark.parametrize(
    "invalid_path",
    [
        None,
        123,
        1.5,
        object(),
    ],
)
def test_loader_rejects_invalid_checkpoint_path_type(
    invalid_path: object,
) -> None:
    """
    문자열 또는 Path가 아닌 Checkpoint 경로를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "checkpoint_path must be a "
            "string or pathlib.Path"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=invalid_path,  # type: ignore[arg-type]
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "   ",
    ],
)
def test_loader_rejects_empty_checkpoint_string(
    invalid_path: str,
) -> None:
    """
    빈 문자열 Checkpoint Path를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_path string "
            "must not be empty"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                invalid_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "best",
        "best.json",
        "best.txt",
        "best.bin",
    ],
)
def test_loader_rejects_unsupported_checkpoint_suffix(
    invalid_path: str,
) -> None:
    """
    .pt·.pth 이외 Checkpoint 확장자를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_path must use "
            "a .pt or .pth extension"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                invalid_path
            ),
            device="cpu",
        )


def test_loader_rejects_missing_checkpoint_file(
    tmp_path: Path,
) -> None:
    """
    존재하지 않는 Checkpoint 파일을 거부하는지 확인한다.
    """
    missing_path = (
        tmp_path
        / "missing.pt"
    )

    with pytest.raises(
        FileNotFoundError,
        match=(
            "checkpoint file "
            "does not exist"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                missing_path
            ),
            device="cpu",
        )


def test_loader_rejects_directory_checkpoint_path(
    tmp_path: Path,
) -> None:
    """
    .pt 이름을 가진 Directory를 Checkpoint 파일로 거부하는지 확인한다.
    """
    directory_path = (
        tmp_path
        / "directory.pt"
    )

    directory_path.mkdir()

    with pytest.raises(
        ValueError,
        match=(
            "must point to a file, "
            "not a directory"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                directory_path
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Checkpoint File
# =============================================================================


def test_loader_rejects_corrupted_checkpoint_file(
    tmp_path: Path,
) -> None:
    """
    PyTorch Checkpoint가 아닌 손상된 Binary 파일을 거부하는지 확인한다.
    """
    checkpoint_path = (
        tmp_path
        / "corrupted.pt"
    )

    checkpoint_path.write_bytes(
        b"not-a-valid-pytorch-checkpoint"
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "failed to load "
            "the checkpoint file"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_checkpoint",
    [
        None,
        [
            1,
            2,
            3,
        ],
        "checkpoint",
        123,
    ],
)
def test_loader_rejects_non_mapping_checkpoint(
    invalid_checkpoint: object,
    tmp_path: Path,
) -> None:
    """
    Mapping이 아닌 Checkpoint 객체를 거부하는지 확인한다.
    """
    checkpoint_path = (
        tmp_path
        / "invalid.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=(
            invalid_checkpoint
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "loaded checkpoint "
            "must be a Mapping"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


# =============================================================================
# Missing Required Checkpoint Keys
# =============================================================================


@pytest.mark.parametrize(
    "missing_key",
    sorted(
        REQUIRED_CHECKPOINT_KEYS
    ),
)
def test_loader_rejects_each_missing_required_checkpoint_key(
    missing_key: str,
    tmp_path: Path,
) -> None:
    """
    최상위 필수 Key가 하나라도 누락되면 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    del checkpoint[
        missing_key
    ]

    checkpoint_path = (
        tmp_path
        / "missing_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        KeyError,
        match=(
            "checkpoint is missing "
            "required keys"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Checkpoint Metadata
# =============================================================================


@pytest.mark.parametrize(
    "invalid_version",
    [
        True,
        1.5,
        "1",
        None,
    ],
)
def test_loader_rejects_invalid_checkpoint_version_type(
    invalid_version: object,
    tmp_path: Path,
) -> None:
    """
    정수가 아닌 Checkpoint Version을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "checkpoint_version"
    ] = invalid_version

    checkpoint_path = (
        tmp_path
        / "invalid_version.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            "checkpoint_version "
            "must be an integer"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_version",
    [
        0,
        -1,
    ],
)
def test_loader_rejects_non_positive_checkpoint_version(
    invalid_version: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Checkpoint Version을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "checkpoint_version"
    ] = invalid_version

    checkpoint_path = (
        tmp_path
        / "invalid_version.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_version "
            "must be greater than 0"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_unsupported_checkpoint_version(
    tmp_path: Path,
) -> None:
    """
    현재 지원 Version 1과 다른 Version을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "checkpoint_version"
    ] = 2

    checkpoint_path = (
        tmp_path
        / "version_2.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint_version "
            "is not supported"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_model_name_mismatch(
    tmp_path: Path,
) -> None:
    """
    Checkpoint Model 이름이 현재 CNNBaseline과 다르면 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "model_name"
    ] = "ResNet18"

    checkpoint_path = (
        tmp_path
        / "wrong_model.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint model_name "
            "does not match"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_model_name",
    [
        "",
        "   ",
    ],
)
def test_loader_rejects_empty_model_name(
    invalid_model_name: str,
    tmp_path: Path,
) -> None:
    """
    빈 Checkpoint Model 이름을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "model_name"
    ] = invalid_model_name

    checkpoint_path = (
        tmp_path
        / "empty_model_name.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "model_name must not be empty"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_model_module_mismatch(
    tmp_path: Path,
) -> None:
    """
    Checkpoint Model Module이 현재 Model Module과 다르면 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "model_module"
    ] = (
        "different.module"
    )

    checkpoint_path = (
        tmp_path
        / "wrong_module.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint model_module "
            "does not match"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    (
        "metadata_key",
        "invalid_value",
    ),
    [
        (
            "loss_function_name",
            "",
        ),
        (
            "optimizer_name",
            "   ",
        ),
    ],
)
def test_loader_rejects_empty_training_metadata(
    metadata_key: str,
    invalid_value: str,
    tmp_path: Path,
) -> None:
    """
    빈 Loss·Optimizer 이름을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        metadata_key
    ] = invalid_value

    checkpoint_path = (
        tmp_path
        / "empty_metadata.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{metadata_key} "
            "must not be empty"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    (
        "epoch_key",
        "invalid_value",
    ),
    [
        (
            "epoch_number",
            0,
        ),
        (
            "epoch_number",
            -1,
        ),
        (
            "configured_epoch_count",
            0,
        ),
        (
            "configured_epoch_count",
            -1,
        ),
    ],
)
def test_loader_rejects_non_positive_epoch_metadata(
    epoch_key: str,
    invalid_value: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Best Epoch·Configured Epoch를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        epoch_key
    ] = invalid_value

    checkpoint_path = (
        tmp_path
        / "invalid_epoch.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{epoch_key} "
            "must be greater than 0"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_epoch_larger_than_configured_count(
    tmp_path: Path,
) -> None:
    """
    Best Epoch가 전체 설정 Epoch보다 크면 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "epoch_number"
    ] = 6

    checkpoint[
        "configured_epoch_count"
    ] = 5

    checkpoint_path = (
        tmp_path
        / "invalid_epoch_range.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "epoch_number must be "
            "less than or equal to"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loader_rejects_invalid_classification_threshold(
    invalid_threshold: float,
    tmp_path: Path,
) -> None:
    """
    범위 밖·NaN·inf Classification Threshold를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "classification_threshold"
    ] = invalid_threshold

    checkpoint_path = (
        tmp_path
        / "invalid_threshold.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_unsupported_best_model_selection_metric(
    tmp_path: Path,
) -> None:
    """
    Validation Loss 이외 Best Model 선택 기준을 현재 Loader가 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "best_model_selection_metric"
    ] = (
        "validation_accuracy"
    )

    checkpoint_path = (
        tmp_path
        / "invalid_metric.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "best_model_selection_metric "
            "is not supported"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Train·Validation Result
# =============================================================================


@pytest.mark.parametrize(
    "result_key",
    [
        "train_result",
        "validation_result",
    ],
)
def test_loader_rejects_non_mapping_epoch_result(
    result_key: str,
    tmp_path: Path,
) -> None:
    """
    Mapping이 아닌 Train·Validation Result를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        result_key
    ] = [
        1,
        2,
    ]

    checkpoint_path = (
        tmp_path
        / "invalid_result.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            f"{result_key} "
            "must be a Mapping"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    (
        "result_key",
        "missing_key",
    ),
    [
        (
            result_key,
            missing_key,
        )
        for result_key in (
            "train_result",
            "validation_result",
        )
        for missing_key in sorted(
            REQUIRED_EPOCH_RESULT_KEYS
        )
    ],
)
def test_loader_rejects_missing_epoch_result_key(
    result_key: str,
    missing_key: str,
    tmp_path: Path,
) -> None:
    """
    Train·Validation Result 필수 Key 누락을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    result = checkpoint[
        result_key
    ]

    assert isinstance(
        result,
        dict,
    )

    del result[
        missing_key
    ]

    checkpoint_path = (
        tmp_path
        / "missing_result_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        KeyError,
        match=(
            f"{result_key} "
            "is missing required keys"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_loss",
    [
        -0.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loader_rejects_invalid_validation_result_loss(
    invalid_loss: float,
    tmp_path: Path,
) -> None:
    """
    잘못된 Validation Average Loss를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    validation_result = checkpoint[
        "validation_result"
    ]

    assert isinstance(
        validation_result,
        dict,
    )

    validation_result[
        "average_loss"
    ] = invalid_loss

    checkpoint_path = (
        tmp_path
        / "invalid_validation_loss.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loader_rejects_invalid_validation_result_accuracy(
    invalid_accuracy: float,
    tmp_path: Path,
) -> None:
    """
    잘못된 Validation Accuracy를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    validation_result = checkpoint[
        "validation_result"
    ]

    assert isinstance(
        validation_result,
        dict,
    )

    validation_result[
        "accuracy"
    ] = invalid_accuracy

    checkpoint_path = (
        tmp_path
        / "invalid_validation_accuracy.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    (
        "count_key",
        "invalid_count",
    ),
    [
        (
            "sample_count",
            0,
        ),
        (
            "sample_count",
            -1,
        ),
        (
            "batch_count",
            0,
        ),
        (
            "batch_count",
            -1,
        ),
    ],
)
def test_loader_rejects_non_positive_validation_result_counts(
    count_key: str,
    invalid_count: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Validation Sample·Batch 수를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    validation_result = checkpoint[
        "validation_result"
    ]

    assert isinstance(
        validation_result,
        dict,
    )

    validation_result[
        count_key
    ] = invalid_count

    checkpoint_path = (
        tmp_path
        / "invalid_validation_count.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Optimizer State
# =============================================================================


@pytest.mark.parametrize(
    "invalid_optimizer_state",
    [
        None,
        [],
        "optimizer",
        123,
    ],
)
def test_loader_rejects_non_mapping_optimizer_state(
    invalid_optimizer_state: object,
    tmp_path: Path,
) -> None:
    """
    Mapping이 아닌 Optimizer State를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "optimizer_state_dict"
    ] = invalid_optimizer_state

    checkpoint_path = (
        tmp_path
        / "invalid_optimizer_state.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            "optimizer_state_dict "
            "must be a Mapping"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "missing_key",
    sorted(
        REQUIRED_OPTIMIZER_STATE_KEYS
    ),
)
def test_loader_rejects_missing_optimizer_state_key(
    missing_key: str,
    tmp_path: Path,
) -> None:
    """
    Optimizer State의 state·param_groups Key 누락을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    optimizer_state = checkpoint[
        "optimizer_state_dict"
    ]

    assert isinstance(
        optimizer_state,
        dict,
    )

    del optimizer_state[
        missing_key
    ]

    checkpoint_path = (
        tmp_path
        / "missing_optimizer_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        KeyError,
        match=(
            "optimizer_state_dict "
            "is missing required keys"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


# =============================================================================
# Invalid Model State
# =============================================================================


@pytest.mark.parametrize(
    "invalid_model_state",
    [
        None,
        [],
        "state",
        123,
    ],
)
def test_loader_rejects_non_mapping_model_state(
    invalid_model_state: object,
    tmp_path: Path,
) -> None:
    """
    Mapping이 아닌 Model State를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "model_state_dict"
    ] = invalid_model_state

    checkpoint_path = (
        tmp_path
        / "invalid_model_state.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            "model_state_dict "
            "must be a Mapping"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_empty_model_state(
    tmp_path: Path,
) -> None:
    """
    빈 Model State Dictionary를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    checkpoint[
        "model_state_dict"
    ] = {}

    checkpoint_path = (
        tmp_path
        / "empty_model_state.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "model_state_dict "
            "must not be empty"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_non_string_model_state_key(
    tmp_path: Path,
) -> None:
    """
    문자열이 아닌 Model State Key를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    first_tensor = next(
        iter(
            model_state.values()
        )
    )

    checkpoint[
        "model_state_dict"
    ] = {
        123: first_tensor,
    }

    checkpoint_path = (
        tmp_path
        / "invalid_state_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            "model_state_dict key "
            "must be a string"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_empty_model_state_key(
    tmp_path: Path,
) -> None:
    """
    빈 Model State Key를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    first_tensor = next(
        iter(
            model_state.values()
        )
    )

    checkpoint[
        "model_state_dict"
    ] = {
        "": first_tensor,
    }

    checkpoint_path = (
        tmp_path
        / "empty_state_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "model_state_dict key "
            "must not be empty"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_non_tensor_model_state_value(
    tmp_path: Path,
) -> None:
    """
    Tensor가 아닌 Model State 값을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    first_state_name = next(
        iter(
            model_state.keys()
        )
    )

    model_state[
        first_state_name
    ] = [
        1,
        2,
        3,
    ]

    checkpoint_path = (
        tmp_path
        / "invalid_state_value.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        TypeError,
        match=(
            "every model_state_dict "
            "value must be a torch.Tensor"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_missing_model_state_key(
    tmp_path: Path,
) -> None:
    """
    현재 Model이 요구하는 State Key가 누락되면 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    removed_key = next(
        iter(
            model_state.keys()
        )
    )

    del model_state[
        removed_key
    ]

    checkpoint_path = (
        tmp_path
        / "missing_state_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint Model State keys "
            "do not exactly match"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_unexpected_model_state_key(
    tmp_path: Path,
) -> None:
    """
    현재 Model에 존재하지 않는 추가 State Key를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    model_state[
        "unexpected.weight"
    ] = torch.zeros(
        1,
        dtype=torch.float32,
    )

    checkpoint_path = (
        tmp_path
        / "unexpected_state_key.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint Model State keys "
            "do not exactly match"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


def test_loader_rejects_model_state_shape_mismatch(
    tmp_path: Path,
) -> None:
    """
    State Key는 같지만 Weight Shape가 다르면 strict=True Loading을 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    model_state[
        "conv1.weight"
    ] = torch.zeros(
        1,
        dtype=torch.float32,
    )

    checkpoint_path = (
        tmp_path
        / "wrong_shape.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "could not be loaded "
            "into the current Model "
            "with strict=True"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_loader_rejects_non_finite_model_state_tensor(
    invalid_value: float,
    tmp_path: Path,
) -> None:
    """
    NaN·positive infinity·negative infinity Weight를 거부하는지 확인한다.
    """
    source_model = (
        create_source_model()
    )

    checkpoint = (
        create_valid_checkpoint(
            model=source_model,
        )
    )

    model_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        model_state,
        dict,
    )

    invalid_tensor = (
        model_state[
            "conv1.weight"
        ]
        .clone()
    )

    invalid_tensor.view(
        -1
    )[
        0
    ] = invalid_value

    model_state[
        "conv1.weight"
    ] = invalid_tensor

    checkpoint_path = (
        tmp_path
        / "non_finite_weight.pt"
    )

    save_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint=checkpoint,
    )

    with pytest.raises(
        ValueError,
        match=(
            "checkpoint Model State Tensor "
            "must contain only finite values"
        ),
    ):
        load_model_checkpoint(
            model=CNNBaseline(),
            checkpoint_path=(
                checkpoint_path
            ),
            device="cpu",
        )