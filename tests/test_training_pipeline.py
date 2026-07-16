"""
Multi-epoch training pipeline unit tests.

테스트 대상
----------
src/training/training_pipeline.py

테스트 목적
----------
여러 Epoch의 Train·Validation 실행 결과가 순서대로 기록되고,
Validation Loss가 가장 낮은 Epoch의 Model이 Checkpoint로 저장되는지
자동으로 검증한다.

현재 Best Model 선택 기준
------------------------
Lowest Validation Loss

현재 기본 설정
--------------
Epoch Count:

    5

Classification Threshold:

    0.5

Checkpoint:

    models/checkpoints/cnn_baseline_best.pt

Checkpoint 저장 내용
--------------------
Checkpoint Version

Model 이름

Model Module

Loss Function 이름

Optimizer 이름

Best Epoch

설정 Epoch 수

Classification Threshold

Best Model 선택 기준

Model State

Optimizer State

Train Result

Validation Result

이번 테스트 범위
----------------
Training Pipeline

Training History

Best Epoch

Best Validation Result

Checkpoint 저장

Checkpoint Metadata

입력 예외 처리

이번 테스트에서 아직 다루지 않는 기능
-----------------------------------
실제 Casting Dataset 5 Epoch 학습

Checkpoint Loading

Test Dataset 최종 평가

Precision

Recall

F1

Confusion Matrix
"""

from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

import src.training.training_pipeline as training_pipeline_module

from src.training.epoch_runner import (
    EpochResult,
)
from src.training.loss_function import (
    create_binary_classification_loss,
)
from src.training.optimizer import (
    create_optimizer,
)
from src.training.training_pipeline import (
    BEST_MODEL_SELECTION_METRIC,
    CHECKPOINT_VERSION,
    DEFAULT_CNN_CHECKPOINT_PATH,
    DEFAULT_EPOCH_COUNT,
    EpochHistoryItem,
    TrainingResult,
    run_training,
)


# =============================================================================
# Test Models
# =============================================================================


class MeanLogitModel(nn.Module):
    """
    이미지 평균을 Binary Raw Logit으로 변환하는 작은 테스트 Model.

    입력
    ----
    Shape:

        [batch_size, 3, height, width]

    처리
    ----
    이미지 전체 평균

    ->

    학습 가능한 Weight

    ->

    학습 가능한 Bias

    출력
    ----
    Shape:

        [batch_size]

    목적
    ----
    CNNBaseline보다 계산량이 작으므로
    Training Pipeline 자체를 빠르게 검증할 때 사용한다.
    """

    def __init__(self) -> None:
        """
        학습 가능한 Scalar Weight와 Bias를 생성한다.
        """
        super().__init__()

        self.weight = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self.bias = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        이미지 평균을 이용해 Binary Raw Logit을 계산한다.
        """
        image_means = images.mean(
            dim=(
                1,
                2,
                3,
            ),
        )

        return (
            image_means
            * self.weight
            + self.bias
        )


class ParameterlessModel(nn.Module):
    """
    Parameter가 없는 잘못된 Model.

    Training Pipeline의 Model Device 검증을 확인할 때 사용한다.
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


class BufferDeviceModel(nn.Module):
    """
    Parameter와 Buffer Device 불일치를 검증하기 위한 Model.

    Parameter:

        cpu

    Buffer:

        meta
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
        테스트에서는 Device 검증 이전에 Forward를 실행하지 않는다.
        """
        return (
            images.mean(
                dim=(
                    1,
                    2,
                    3,
                )
            )
            * self.weight
        )


# =============================================================================
# Helper Functions
# =============================================================================


def create_epoch_result(
    average_loss: float,
    accuracy: float = 0.5,
    sample_count: int = 6,
    batch_count: int = 3,
) -> EpochResult:
    """
    테스트용 EpochResult를 생성한다.

    입력
    ----
    average_loss:
        Epoch 평균 Loss

    accuracy:
        Epoch Accuracy

    sample_count:
        Sample 수

    batch_count:
        Batch 수

    출력
    ----
    EpochResult
    """
    return EpochResult(
        average_loss=average_loss,
        accuracy=accuracy,
        sample_count=sample_count,
        batch_count=batch_count,
    )


def create_history_item(
    epoch_number: int,
    train_loss: float,
    validation_loss: float,
    train_accuracy: float = 0.6,
    validation_accuracy: float = 0.7,
) -> EpochHistoryItem:
    """
    테스트용 EpochHistoryItem을 생성한다.
    """
    return EpochHistoryItem(
        epoch_number=epoch_number,
        train_result=create_epoch_result(
            average_loss=train_loss,
            accuracy=train_accuracy,
        ),
        validation_result=create_epoch_result(
            average_loss=validation_loss,
            accuracy=validation_accuracy,
        ),
    )


def create_standard_data_loader(
    sample_count: int = 6,
    batch_size: int = 2,
) -> DataLoader:
    """
    Training Pipeline 테스트용 Binary Image DataLoader를 생성한다.

    이미지 값
    ---------
    NORMAL:

        음수

    DEFECT:

        양수

    Label
    -----
    0 = NORMAL

    1 = DEFECT
    """
    image_values = []

    label_values = []

    for sample_index in range(
        sample_count
    ):
        label = sample_index % 2

        label_values.append(
            label
        )

        if label == 0:
            image_values.append(
                -1.0
            )
        else:
            image_values.append(
                1.0
            )

    images = torch.stack(
        [
            torch.full(
                (
                    3,
                    8,
                    8,
                ),
                fill_value=image_value,
                dtype=torch.float32,
            )
            for image_value in image_values
        ],
        dim=0,
    )

    labels = torch.tensor(
        label_values,
        dtype=torch.int64,
    )

    dataset = TensorDataset(
        images,
        labels,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def create_training_components() -> tuple[
    MeanLogitModel,
    DataLoader,
    DataLoader,
    nn.Module,
    Adam,
]:
    """
    정상 Training Pipeline 실행에 필요한 객체를 생성한다.

    출력
    ----
    Model

    Train DataLoader

    Validation DataLoader

    Loss Function

    Adam Optimizer
    """
    model = MeanLogitModel()

    train_loader = (
        create_standard_data_loader(
            sample_count=6,
            batch_size=2,
        )
    )

    validation_loader = (
        create_standard_data_loader(
            sample_count=6,
            batch_size=3,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
        learning_rate=0.01,
    )

    return (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    )


def clone_model_state(
    model: nn.Module,
) -> dict[str, Tensor]:
    """
    Model State를 CPU Tensor 복사본으로 저장한다.
    """
    return {
        key: value.detach().cpu().clone()
        for (
            key,
            value,
        ) in model.state_dict().items()
    }


def state_dicts_are_equal(
    first_state: dict[str, Tensor],
    second_state: dict[str, Tensor],
) -> bool:
    """
    두 Model State Dictionary의 Key와 Tensor 값을 비교한다.
    """
    if (
        set(
            first_state
        )
        != set(
            second_state
        )
    ):
        return False

    return all(
        torch.equal(
            first_state[key],
            second_state[key],
        )
        for key in first_state
    )


def load_checkpoint(
    checkpoint_path: Path,
) -> dict[str, object]:
    """
    테스트 Checkpoint를 CPU로 불러온다.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert isinstance(
        checkpoint,
        dict,
    )

    return checkpoint


# =============================================================================
# Default Configuration
# =============================================================================


def test_default_epoch_count_is_five() -> None:
    """
    첫 CNN Baseline 기본 Epoch 수가 5인지 확인한다.
    """
    assert DEFAULT_EPOCH_COUNT == 5


def test_default_checkpoint_path_is_expected_path() -> None:
    """
    CNN Baseline 기본 Checkpoint 경로를 확인한다.
    """
    assert DEFAULT_CNN_CHECKPOINT_PATH == Path(
        "models/checkpoints/cnn_baseline_best.pt"
    )


def test_checkpoint_version_is_one() -> None:
    """
    현재 Checkpoint 구조 Version이 1인지 확인한다.
    """
    assert CHECKPOINT_VERSION == 1


def test_best_model_selection_metric_is_validation_loss() -> None:
    """
    Best Model 선택 기준이 Validation Loss인지 확인한다.
    """
    assert (
        BEST_MODEL_SELECTION_METRIC
        == "validation_loss"
    )


# =============================================================================
# EpochHistoryItem
# =============================================================================


def test_epoch_history_item_stores_expected_values() -> None:
    """
    한 Epoch의 Train·Validation 결과를 저장하는지 확인한다.
    """
    train_result = create_epoch_result(
        average_loss=0.6,
        accuracy=0.7,
    )

    validation_result = (
        create_epoch_result(
            average_loss=0.5,
            accuracy=0.8,
        )
    )

    history_item = EpochHistoryItem(
        epoch_number=1,
        train_result=train_result,
        validation_result=(
            validation_result
        ),
    )

    assert history_item.epoch_number == 1

    assert (
        history_item.train_result
        is train_result
    )

    assert (
        history_item.validation_result
        is validation_result
    )


def test_epoch_history_item_is_frozen() -> None:
    """
    생성된 Epoch History를 수정할 수 없는지 확인한다.
    """
    history_item = create_history_item(
        epoch_number=1,
        train_loss=0.6,
        validation_loss=0.5,
    )

    with pytest.raises(
        AttributeError,
    ):
        history_item.epoch_number = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_epoch_number",
    [
        True,
        False,
        1.5,
        "1",
        None,
    ],
)
def test_epoch_history_item_rejects_invalid_epoch_number_type(
    invalid_epoch_number: object,
) -> None:
    """
    정수가 아닌 Epoch Number와 bool을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="epoch_number must be an integer",
    ):
        EpochHistoryItem(
            epoch_number=invalid_epoch_number,  # type: ignore[arg-type]
            train_result=create_epoch_result(
                average_loss=0.6,
            ),
            validation_result=(
                create_epoch_result(
                    average_loss=0.5,
                )
            ),
        )


@pytest.mark.parametrize(
    "invalid_epoch_number",
    [
        0,
        -1,
    ],
)
def test_epoch_history_item_rejects_non_positive_epoch_number(
    invalid_epoch_number: int,
) -> None:
    """
    0 이하 Epoch Number를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="epoch_number must be greater than 0",
    ):
        EpochHistoryItem(
            epoch_number=(
                invalid_epoch_number
            ),
            train_result=create_epoch_result(
                average_loss=0.6,
            ),
            validation_result=(
                create_epoch_result(
                    average_loss=0.5,
                )
            ),
        )


@pytest.mark.parametrize(
    "invalid_train_result",
    [
        None,
        {},
        "EpochResult",
        1,
    ],
)
def test_epoch_history_item_rejects_invalid_train_result(
    invalid_train_result: object,
) -> None:
    """
    EpochResult가 아닌 Train Result를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="train_result must be an EpochResult",
    ):
        EpochHistoryItem(
            epoch_number=1,
            train_result=invalid_train_result,  # type: ignore[arg-type]
            validation_result=(
                create_epoch_result(
                    average_loss=0.5,
                )
            ),
        )


@pytest.mark.parametrize(
    "invalid_validation_result",
    [
        None,
        {},
        "EpochResult",
        1,
    ],
)
def test_epoch_history_item_rejects_invalid_validation_result(
    invalid_validation_result: object,
) -> None:
    """
    EpochResult가 아닌 Validation Result를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="validation_result must be an EpochResult",
    ):
        EpochHistoryItem(
            epoch_number=1,
            train_result=create_epoch_result(
                average_loss=0.6,
            ),
            validation_result=invalid_validation_result,  # type: ignore[arg-type]
        )


# =============================================================================
# TrainingResult
# =============================================================================


def test_training_result_stores_expected_values() -> None:
    """
    전체 학습 결과를 정상 저장하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
            validation_accuracy=0.75,
        ),
        create_history_item(
            epoch_number=2,
            train_loss=0.5,
            validation_loss=0.4,
            validation_accuracy=0.8,
        ),
    )

    checkpoint_path = Path(
        "checkpoint.pt"
    )

    result = TrainingResult(
        history=history,
        best_epoch_number=2,
        best_validation_loss=0.4,
        best_validation_accuracy=0.8,
        checkpoint_path=checkpoint_path,
    )

    assert result.history == history

    assert result.best_epoch_number == 2

    assert result.best_validation_loss == 0.4

    assert (
        result.best_validation_accuracy
        == 0.8
    )

    assert (
        result.checkpoint_path
        == checkpoint_path
    )


def test_training_result_is_frozen() -> None:
    """
    학습 완료 후 TrainingResult를 변경할 수 없는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    result = TrainingResult(
        history=history,
        best_epoch_number=1,
        best_validation_loss=0.5,
        best_validation_accuracy=0.7,
        checkpoint_path=Path(
            "checkpoint.pt"
        ),
    )

    with pytest.raises(
        AttributeError,
    ):
        result.best_epoch_number = 2  # type: ignore[misc]


def test_training_result_rejects_non_tuple_history() -> None:
    """
    List History를 거부하는지 확인한다.
    """
    history_list = [
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    ]

    with pytest.raises(
        TypeError,
        match="history must be a tuple",
    ):
        TrainingResult(
            history=history_list,  # type: ignore[arg-type]
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


def test_training_result_rejects_empty_history() -> None:
    """
    Epoch가 하나도 없는 빈 History를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="at least one epoch result",
    ):
        TrainingResult(
            history=(),
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


def test_training_result_rejects_invalid_history_item() -> None:
    """
    EpochHistoryItem이 아닌 History 요소를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="every history item must be an EpochHistoryItem",
    ):
        TrainingResult(
            history=(
                "invalid",  # type: ignore[arg-type]
            ),
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "history",
    [
        (
            create_history_item(
                epoch_number=2,
                train_loss=0.6,
                validation_loss=0.5,
            ),
        ),
        (
            create_history_item(
                epoch_number=1,
                train_loss=0.6,
                validation_loss=0.5,
            ),
            create_history_item(
                epoch_number=3,
                train_loss=0.5,
                validation_loss=0.4,
            ),
        ),
        (
            create_history_item(
                epoch_number=1,
                train_loss=0.6,
                validation_loss=0.5,
            ),
            create_history_item(
                epoch_number=1,
                train_loss=0.5,
                validation_loss=0.4,
            ),
        ),
    ],
)
def test_training_result_rejects_non_sequential_epoch_numbers(
    history: tuple[
        EpochHistoryItem,
        ...,
    ],
) -> None:
    """
    Epoch 번호가 1부터 순서대로 증가하지 않으면 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="must start at 1 and increase sequentially",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=(
                history[
                    0
                ]
                .validation_result
                .average_loss
            ),
            best_validation_accuracy=(
                history[
                    0
                ]
                .validation_result
                .accuracy
            ),
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "invalid_best_epoch",
    [
        True,
        False,
        1.5,
        "1",
        None,
    ],
)
def test_training_result_rejects_invalid_best_epoch_type(
    invalid_best_epoch: object,
) -> None:
    """
    정수가 아닌 Best Epoch와 bool을 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        TypeError,
        match="best_epoch_number must be an integer",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=invalid_best_epoch,  # type: ignore[arg-type]
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "invalid_best_epoch",
    [
        0,
        -1,
        2,
    ],
)
def test_training_result_rejects_best_epoch_outside_history(
    invalid_best_epoch: int,
) -> None:
    """
    History에 존재하지 않는 Best Epoch를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must refer to an epoch contained in history",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=(
                invalid_best_epoch
            ),
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "invalid_loss",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_training_result_rejects_non_finite_best_validation_loss(
    invalid_loss: float,
) -> None:
    """
    NaN·inf Best Validation Loss를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="best_validation_loss must be finite",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=(
                invalid_loss
            ),
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


def test_training_result_rejects_negative_best_validation_loss() -> None:
    """
    음수 Best Validation Loss를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=-0.1,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_training_result_rejects_non_finite_best_accuracy(
    invalid_accuracy: float,
) -> None:
    """
    NaN·inf Best Validation Accuracy를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="best_validation_accuracy must be finite",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=(
                invalid_accuracy
            ),
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -0.1,
        1.1,
    ],
)
def test_training_result_rejects_accuracy_outside_zero_and_one(
    invalid_accuracy: float,
) -> None:
    """
    0~1 범위를 벗어난 Best Accuracy를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be between 0 and 1",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=(
                invalid_accuracy
            ),
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


def test_training_result_rejects_non_path_checkpoint() -> None:
    """
    pathlib.Path가 아닌 Checkpoint Path를 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        TypeError,
        match="checkpoint_path must be a pathlib.Path",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.7,
            checkpoint_path="checkpoint.pt",  # type: ignore[arg-type]
        )


def test_training_result_rejects_best_loss_mismatch() -> None:
    """
    Best Epoch의 Validation Loss와 다른 값을 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must match the Validation Loss",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=0.4,
            best_validation_accuracy=0.7,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


def test_training_result_rejects_best_accuracy_mismatch() -> None:
    """
    Best Epoch의 Validation Accuracy와 다른 값을 거부하는지 확인한다.
    """
    history = (
        create_history_item(
            epoch_number=1,
            train_loss=0.6,
            validation_loss=0.5,
            validation_accuracy=0.7,
        ),
    )

    with pytest.raises(
        ValueError,
        match="must match the Validation Accuracy",
    ):
        TrainingResult(
            history=history,
            best_epoch_number=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.8,
            checkpoint_path=Path(
                "checkpoint.pt"
            ),
        )


# =============================================================================
# Real Training Pipeline Integration
# =============================================================================


def test_run_training_returns_training_result(
    tmp_path: Path,
) -> None:
    """
    정상 학습 실행 후 TrainingResult를 반환하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=2,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert isinstance(
        result,
        TrainingResult,
    )


def test_run_training_records_all_epochs(
    tmp_path: Path,
) -> None:
    """
    설정한 Epoch 수만큼 History가 생성되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=3,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert len(
        result.history,
    ) == 3

    assert [
        item.epoch_number
        for item in result.history
    ] == [
        1,
        2,
        3,
    ]


def test_run_training_records_train_and_validation_counts(
    tmp_path: Path,
) -> None:
    """
    실제 처리한 Train·Validation Sample·Batch 수를 기록하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    history_item = result.history[0]

    assert (
        history_item
        .train_result
        .sample_count
        == 6
    )

    assert (
        history_item
        .train_result
        .batch_count
        == 3
    )

    assert (
        history_item
        .validation_result
        .sample_count
        == 6
    )

    assert (
        history_item
        .validation_result
        .batch_count
        == 2
    )


def test_run_training_changes_model_parameters(
    tmp_path: Path,
) -> None:
    """
    실제 Train Epoch에서 Model Parameter가 변경되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    state_before = clone_model_state(
        model=model,
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=2,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    state_after = clone_model_state(
        model=model,
    )

    assert not state_dicts_are_equal(
        first_state=state_before,
        second_state=state_after,
    )


def test_run_training_finishes_in_evaluation_mode(
    tmp_path: Path,
) -> None:
    """
    마지막 실행이 Validation이므로 Model이 Evaluation Mode인지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert model.training is False


def test_run_training_creates_checkpoint_file(
    tmp_path: Path,
) -> None:
    """
    Best Model Checkpoint가 실제 생성되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    assert checkpoint_path.is_file()

    assert (
        result.checkpoint_path
        == checkpoint_path
    )


def test_run_training_creates_nested_checkpoint_directory(
    tmp_path: Path,
) -> None:
    """
    Checkpoint 상위 Directory가 없어도 자동 생성하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "models"
        / "checkpoints"
        / "best.pt"
    )

    assert (
        checkpoint_path.parent.exists()
        is False
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    assert checkpoint_path.is_file()


def test_run_training_accepts_string_checkpoint_path(
    tmp_path: Path,
) -> None:
    """
    문자열 Checkpoint Path를 Path로 변환하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=str(
            checkpoint_path
        ),
        verbose=False,
    )

    assert isinstance(
        result.checkpoint_path,
        Path,
    )

    assert (
        result.checkpoint_path
        == checkpoint_path
    )


def test_run_training_accepts_pth_extension(
    tmp_path: Path,
) -> None:
    """
    .pth Checkpoint 확장자를 허용하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pth"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    assert checkpoint_path.is_file()

    assert (
        result.checkpoint_path
        == checkpoint_path
    )


def test_run_training_accepts_uppercase_checkpoint_extension(
    tmp_path: Path,
) -> None:
    """
    대문자 .PT 확장자도 허용하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.PT"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    assert checkpoint_path.is_file()


def test_temporary_checkpoint_file_is_removed_after_success(
    tmp_path: Path,
) -> None:
    """
    저장 완료 후 임시 .tmp 파일이 남지 않는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    temporary_path = (
        checkpoint_path.with_name(
            f"{checkpoint_path.name}.tmp"
        )
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    assert checkpoint_path.is_file()

    assert temporary_path.exists() is False


def test_existing_checkpoint_file_is_replaced(
    tmp_path: Path,
) -> None:
    """
    기존 파일이 있어도 새로운 Best Checkpoint로 교체하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    checkpoint_path.write_bytes(
        b"old-checkpoint"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint[
            "checkpoint_version"
        ]
        == CHECKPOINT_VERSION
    )


# =============================================================================
# Checkpoint Content
# =============================================================================


def test_checkpoint_contains_required_keys(
    tmp_path: Path,
) -> None:
    """
    Checkpoint에 필수 Metadata·State가 모두 저장되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
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

    assert required_keys.issubset(
        checkpoint.keys()
    )


def test_checkpoint_metadata_matches_training_objects(
    tmp_path: Path,
) -> None:
    """
    Model·Loss·Optimizer 이름이 실제 객체와 일치하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint["model_name"]
        == model.__class__.__name__
    )

    assert (
        checkpoint["model_module"]
        == model.__class__.__module__
    )

    assert (
        checkpoint[
            "loss_function_name"
        ]
        == loss_function
        .__class__
        .__name__
    )

    assert (
        checkpoint["optimizer_name"]
        == optimizer
        .__class__
        .__name__
    )


def test_checkpoint_stores_model_state_dict(
    tmp_path: Path,
) -> None:
    """
    Checkpoint에 Model State가 저장되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    checkpoint_state = checkpoint[
        "model_state_dict"
    ]

    assert isinstance(
        checkpoint_state,
        dict,
    )

    assert set(
        checkpoint_state
    ) == set(
        model.state_dict()
    )


def test_checkpoint_stores_optimizer_state_dict(
    tmp_path: Path,
) -> None:
    """
    Checkpoint에 Adam State가 저장되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    optimizer_state = checkpoint[
        "optimizer_state_dict"
    ]

    assert isinstance(
        optimizer_state,
        dict,
    )

    assert "state" in optimizer_state

    assert "param_groups" in optimizer_state

    assert len(
        optimizer_state["state"]
    ) > 0


def test_checkpoint_stores_configured_epoch_count(
    tmp_path: Path,
) -> None:
    """
    Checkpoint에 전체 설정 Epoch 수가 기록되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=3,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint[
            "configured_epoch_count"
        ]
        == 3
    )


def test_checkpoint_stores_custom_classification_threshold(
    tmp_path: Path,
) -> None:
    """
    사용자 지정 Threshold가 Checkpoint에 기록되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        classification_threshold=0.65,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint[
            "classification_threshold"
        ]
        == 0.65
    )


def test_checkpoint_selection_metric_is_validation_loss(
    tmp_path: Path,
) -> None:
    """
    Best Model 선택 기준이 Checkpoint에 기록되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint[
            "best_model_selection_metric"
        ]
        == BEST_MODEL_SELECTION_METRIC
    )


def test_checkpoint_result_values_match_training_result(
    tmp_path: Path,
) -> None:
    """
    Checkpoint의 Best Validation 결과가 TrainingResult와 일치하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=2,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert (
        checkpoint["epoch_number"]
        == result.best_epoch_number
    )

    assert (
        checkpoint[
            "validation_result"
        ][
            "average_loss"
        ]
        == result.best_validation_loss
    )

    assert (
        checkpoint[
            "validation_result"
        ][
            "accuracy"
        ]
        == result.best_validation_accuracy
    )


# =============================================================================
# Deterministic Best Model Selection
# =============================================================================


def test_run_training_selects_lowest_validation_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validation Loss가 가장 낮은 Epoch를 Best로 선택하는지 확인한다.

    Validation Loss:

        Epoch 1:

            0.60

        Epoch 2:

            0.40

        Epoch 3:

            0.50

    예상 Best:

        Epoch 2
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    train_results = iter(
        [
            create_epoch_result(
                average_loss=0.7,
                accuracy=0.6,
            ),
            create_epoch_result(
                average_loss=0.6,
                accuracy=0.7,
            ),
            create_epoch_result(
                average_loss=0.5,
                accuracy=0.8,
            ),
        ]
    )

    validation_results = iter(
        [
            create_epoch_result(
                average_loss=0.6,
                accuracy=0.70,
            ),
            create_epoch_result(
                average_loss=0.4,
                accuracy=0.80,
            ),
            create_epoch_result(
                average_loss=0.5,
                accuracy=0.90,
            ),
        ]
    )

    def fake_train_one_epoch(
        **_: object,
    ) -> EpochResult:
        return next(
            train_results
        )

    def fake_validate_one_epoch(
        **_: object,
    ) -> EpochResult:
        return next(
            validation_results
        )

    monkeypatch.setattr(
        training_pipeline_module,
        "train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "validate_one_epoch",
        fake_validate_one_epoch,
    )

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=3,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert result.best_epoch_number == 2

    assert (
        result.best_validation_loss
        == 0.4
    )

    assert (
        result.best_validation_accuracy
        == 0.8
    )

    assert (
        checkpoint["epoch_number"]
        == 2
    )


def test_best_accuracy_comes_from_lowest_loss_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Best Accuracy가 최고 Accuracy Epoch가 아니라
    Best Validation Loss Epoch에서 가져오는지 확인한다.

    Epoch 1:

        Loss 0.40

        Accuracy 0.70

    Epoch 2:

        Loss 0.50

        Accuracy 0.95

    Best 선택:

        Epoch 1

    저장 Accuracy:

        0.70
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    train_result = create_epoch_result(
        average_loss=0.6,
        accuracy=0.6,
    )

    validation_results = iter(
        [
            create_epoch_result(
                average_loss=0.4,
                accuracy=0.7,
            ),
            create_epoch_result(
                average_loss=0.5,
                accuracy=0.95,
            ),
        ]
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "train_one_epoch",
        lambda **_: train_result,
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "validate_one_epoch",
        lambda **_: next(
            validation_results
        ),
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=2,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert result.best_epoch_number == 1

    assert (
        result.best_validation_loss
        == 0.4
    )

    assert (
        result.best_validation_accuracy
        == 0.7
    )


def test_equal_validation_loss_keeps_earlier_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validation Loss가 같으면 먼저 저장된 Epoch를 유지하는지 확인한다.

    비교 연산:

        <

    사용:

        <=

    아님
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    train_result = create_epoch_result(
        average_loss=0.6,
        accuracy=0.6,
    )

    validation_results = iter(
        [
            create_epoch_result(
                average_loss=0.4,
                accuracy=0.7,
            ),
            create_epoch_result(
                average_loss=0.4,
                accuracy=0.9,
            ),
        ]
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "train_one_epoch",
        lambda **_: train_result,
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "validate_one_epoch",
        lambda **_: next(
            validation_results
        ),
    )

    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=2,
        checkpoint_path=(
            checkpoint_path
        ),
        verbose=False,
    )

    checkpoint = load_checkpoint(
        checkpoint_path=(
            checkpoint_path
        ),
    )

    assert result.best_epoch_number == 1

    assert (
        result.best_validation_accuracy
        == 0.7
    )

    assert (
        checkpoint["epoch_number"]
        == 1
    )


def test_decreasing_validation_loss_saves_last_epoch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Validation Loss가 계속 감소하면 마지막 Epoch가 Best인지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    train_result = create_epoch_result(
        average_loss=0.6,
        accuracy=0.6,
    )

    validation_results = iter(
        [
            create_epoch_result(
                average_loss=0.6,
                accuracy=0.6,
            ),
            create_epoch_result(
                average_loss=0.5,
                accuracy=0.7,
            ),
            create_epoch_result(
                average_loss=0.4,
                accuracy=0.8,
            ),
        ]
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "train_one_epoch",
        lambda **_: train_result,
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "validate_one_epoch",
        lambda **_: next(
            validation_results
        ),
    )

    result = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=3,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert result.best_epoch_number == 3

    assert (
        result.best_validation_loss
        == 0.4
    )


def test_pipeline_passes_threshold_and_device_to_epoch_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    검증된 Device와 Threshold가 Train·Validation 함수에 전달되는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    captured_train_arguments: list[
        dict[str, object]
    ] = []

    captured_validation_arguments: list[
        dict[str, object]
    ] = []

    epoch_result = create_epoch_result(
        average_loss=0.5,
        accuracy=0.5,
    )

    def fake_train_one_epoch(
        **arguments: object,
    ) -> EpochResult:
        captured_train_arguments.append(
            arguments
        )

        return epoch_result

    def fake_validate_one_epoch(
        **arguments: object,
    ) -> EpochResult:
        captured_validation_arguments.append(
            arguments
        )

        return epoch_result

    monkeypatch.setattr(
        training_pipeline_module,
        "train_one_epoch",
        fake_train_one_epoch,
    )

    monkeypatch.setattr(
        training_pipeline_module,
        "validate_one_epoch",
        fake_validate_one_epoch,
    )

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        classification_threshold=0.65,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert len(
        captured_train_arguments
    ) == 1

    assert len(
        captured_validation_arguments
    ) == 1

    assert (
        captured_train_arguments[
            0
        ][
            "device"
        ]
        == torch.device(
            "cpu"
        )
    )

    assert (
        captured_validation_arguments[
            0
        ][
            "device"
        ]
        == torch.device(
            "cpu"
        )
    )

    assert (
        captured_train_arguments[
            0
        ][
            "classification_threshold"
        ]
        == 0.65
    )

    assert (
        captured_validation_arguments[
            0
        ][
            "classification_threshold"
        ]
        == 0.65
    )


# =============================================================================
# Console Output
# =============================================================================


def test_verbose_false_produces_no_pipeline_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    verbose=False이면 Training Pipeline Console 출력이 없는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    captured = capsys.readouterr()

    assert captured.out == ""


def test_verbose_true_prints_training_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    verbose=True이면 학습 설정·Epoch·완료 결과를 출력하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    _ = run_training(
        model=model,
        train_loader=train_loader,
        validation_loader=(
            validation_loader
        ),
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=True,
    )

    captured = capsys.readouterr()

    assert "VISION MODEL TRAINING" in (
        captured.out
    )

    assert "EPOCH 1 / 1" in (
        captured.out
    )

    assert "TRAINING COMPLETED" in (
        captured.out
    )

    assert "Best validation loss" in (
        captured.out
    )


# =============================================================================
# Invalid Training Object Validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_model",
    [
        None,
        "model",
        123,
        object(),
    ],
)
def test_run_training_rejects_non_module_model(
    invalid_model: object,
    tmp_path: Path,
) -> None:
    """
    nn.Module이 아닌 Model을 거부하는지 확인한다.
    """
    (
        _,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        run_training(
            model=invalid_model,  # type: ignore[arg-type]
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_loader",
    [
        None,
        [],
        "loader",
        object(),
    ],
)
def test_run_training_rejects_non_dataloader_train_loader(
    invalid_loader: object,
    tmp_path: Path,
) -> None:
    """
    DataLoader가 아닌 Train Loader를 거부하는지 확인한다.
    """
    (
        model,
        _,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="train_loader must be an instance",
    ):
        run_training(
            model=model,
            train_loader=invalid_loader,  # type: ignore[arg-type]
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_loader",
    [
        None,
        [],
        "loader",
        object(),
    ],
)
def test_run_training_rejects_non_dataloader_validation_loader(
    invalid_loader: object,
    tmp_path: Path,
) -> None:
    """
    DataLoader가 아닌 Validation Loader를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        _,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="validation_loader must be an instance",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=invalid_loader,  # type: ignore[arg-type]
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_loss",
    [
        None,
        "loss",
        123,
        object(),
    ],
)
def test_run_training_rejects_non_module_loss(
    invalid_loss: object,
    tmp_path: Path,
) -> None:
    """
    nn.Module이 아닌 Loss Function을 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        _,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="loss_function must be an instance of torch.nn.Module",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=invalid_loss,  # type: ignore[arg-type]
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_optimizer",
    [
        None,
        "Adam",
        123,
        object(),
    ],
)
def test_run_training_rejects_non_optimizer(
    invalid_optimizer: object,
    tmp_path: Path,
) -> None:
    """
    torch.optim.Optimizer가 아닌 객체를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        _,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="optimizer must be an instance",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=invalid_optimizer,  # type: ignore[arg-type]
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


# =============================================================================
# Invalid Epoch Count
# =============================================================================


@pytest.mark.parametrize(
    "invalid_epoch_count",
    [
        True,
        False,
        1.5,
        "5",
        None,
    ],
)
def test_run_training_rejects_invalid_epoch_count_type(
    invalid_epoch_count: object,
    tmp_path: Path,
) -> None:
    """
    정수가 아닌 Epoch Count와 bool을 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="epoch_count must be an integer",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=invalid_epoch_count,  # type: ignore[arg-type]
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_epoch_count",
    [
        0,
        -1,
    ],
)
def test_run_training_rejects_non_positive_epoch_count(
    invalid_epoch_count: int,
    tmp_path: Path,
) -> None:
    """
    0 이하 Epoch Count를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="epoch_count must be greater than 0",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=(
                invalid_epoch_count
            ),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


# =============================================================================
# Invalid Threshold
# =============================================================================


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        False,
        None,
        "0.5",
        [
            0.5,
        ],
    ],
)
def test_run_training_rejects_invalid_threshold_type(
    invalid_threshold: object,
    tmp_path: Path,
) -> None:
    """
    Real Number가 아닌 Threshold와 bool을 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="classification_threshold must be a real number",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            classification_threshold=invalid_threshold,  # type: ignore[arg-type]
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_run_training_rejects_threshold_outside_zero_and_one(
    invalid_threshold: float,
    tmp_path: Path,
) -> None:
    """
    0~1 범위를 벗어난 Threshold를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="classification_threshold must be between 0 and 1",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            classification_threshold=(
                invalid_threshold
            ),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_run_training_rejects_non_finite_threshold(
    invalid_threshold: float,
    tmp_path: Path,
) -> None:
    """
    NaN·inf Threshold를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="classification_threshold must be finite",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            classification_threshold=(
                invalid_threshold
            ),
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


# =============================================================================
# Invalid Device
# =============================================================================


@pytest.mark.parametrize(
    "invalid_device",
    [
        None,
        0,
        1.0,
        object(),
    ],
)
def test_run_training_rejects_invalid_device_type(
    invalid_device: object,
    tmp_path: Path,
) -> None:
    """
    문자열 또는 torch.device가 아닌 Device를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="device must be a string or torch.device",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device=invalid_device,  # type: ignore[arg-type]
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_invalid_device_string(
    tmp_path: Path,
) -> None:
    """
    PyTorch가 해석할 수 없는 Device 문자열을 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="Invalid device",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="not-a-device",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_unavailable_cuda(
    tmp_path: Path,
) -> None:
    """
    CUDA가 없는 환경에서 CUDA 요청을 거부하는지 확인한다.
    """
    if torch.cuda.is_available():
        pytest.skip(
            "CUDA is available in this environment."
        )

    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="CUDA device was requested",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cuda",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_model_parameter_device_mismatch(
    tmp_path: Path,
) -> None:
    """
    Model Parameter Device와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = MeanLogitModel()

    model = model.to(
        device="meta"
    )

    train_loader = (
        create_standard_data_loader()
    )

    validation_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = Adam(
        model.parameters(),
        lr=0.001,
    )

    with pytest.raises(
        ValueError,
        match="all model parameters must be on the requested device",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_model_buffer_device_mismatch(
    tmp_path: Path,
) -> None:
    """
    Model Buffer Device와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = BufferDeviceModel()

    train_loader = (
        create_standard_data_loader()
    )

    validation_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="all model buffers must be on the requested device",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_parameterless_model(
    tmp_path: Path,
) -> None:
    """
    Parameter가 없는 Model을 거부하는지 확인한다.
    """
    model = ParameterlessModel()

    helper_model = MeanLogitModel()

    optimizer = create_optimizer(
        model=helper_model,
    )

    with pytest.raises(
        ValueError,
        match="model must contain at least one parameter",
    ):
        run_training(
            model=model,
            train_loader=(
                create_standard_data_loader()
            ),
            validation_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


# =============================================================================
# Optimizer Parameter Connection
# =============================================================================


def test_run_training_rejects_optimizer_from_different_model(
    tmp_path: Path,
) -> None:
    """
    다른 Model의 Parameter를 참조하는 Optimizer를 거부하는지 확인한다.
    """
    model = MeanLogitModel()

    different_model = (
        MeanLogitModel()
    )

    optimizer = create_optimizer(
        model=different_model,
    )

    with pytest.raises(
        ValueError,
        match="optimizer parameters must exactly match",
    ):
        run_training(
            model=model,
            train_loader=(
                create_standard_data_loader()
            ),
            validation_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_optimizer_missing_trainable_parameter(
    tmp_path: Path,
) -> None:
    """
    Model Trainable Parameter 일부만 포함한 Optimizer를 거부하는지 확인한다.
    """
    model = MeanLogitModel()

    optimizer = Adam(
        [
            model.weight,
        ],
        lr=0.001,
    )

    with pytest.raises(
        ValueError,
        match="optimizer parameters must exactly match",
    ):
        run_training(
            model=model,
            train_loader=(
                create_standard_data_loader()
            ),
            validation_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_rejects_duplicate_optimizer_parameter(
    tmp_path: Path,
) -> None:
    """
    동일 Parameter가 Optimizer에 중복 등록되면 거부하는지 확인한다.
    """
    model = MeanLogitModel()

    optimizer = create_optimizer(
        model=model,
    )

    optimizer.param_groups[
        0
    ][
        "params"
    ].append(
        model.weight
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicate parameters",
    ):
        run_training(
            model=model,
            train_loader=(
                create_standard_data_loader()
            ),
            validation_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


def test_run_training_supports_partial_model_freeze(
    tmp_path: Path,
) -> None:
    """
    Freeze Parameter를 제외하고 Optimizer를 생성한 정상 구조를 지원하는지 확인한다.
    """
    model = MeanLogitModel()

    model.bias.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    result = run_training(
        model=model,
        train_loader=(
            create_standard_data_loader()
        ),
        validation_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        optimizer=optimizer,
        device="cpu",
        epoch_count=1,
        checkpoint_path=(
            tmp_path
            / "best.pt"
        ),
        verbose=False,
    )

    assert len(
        result.history
    ) == 1


def test_run_training_rejects_model_with_all_parameters_frozen(
    tmp_path: Path,
) -> None:
    """
    모든 Parameter가 Freeze된 Model을 거부하는지 확인한다.
    """
    model = MeanLogitModel()

    optimizer = create_optimizer(
        model=model,
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(
        ValueError,
        match="at least one trainable parameter",
    ):
        run_training(
            model=model,
            train_loader=(
                create_standard_data_loader()
            ),
            validation_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=False,
        )


# =============================================================================
# Checkpoint Path Validation
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
def test_run_training_rejects_invalid_checkpoint_path_type(
    invalid_path: object,
) -> None:
    """
    문자열 또는 Path가 아닌 Checkpoint Path를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="checkpoint_path must be a string or pathlib.Path",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=invalid_path,  # type: ignore[arg-type]
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "",
        "   ",
    ],
)
def test_run_training_rejects_empty_checkpoint_string(
    invalid_path: str,
) -> None:
    """
    빈 문자열 Checkpoint Path를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                invalid_path
            ),
            verbose=False,
        )


@pytest.mark.parametrize(
    "invalid_path",
    [
        "checkpoint",
        "checkpoint.bin",
        "checkpoint.txt",
    ],
)
def test_run_training_rejects_unsupported_checkpoint_extension(
    invalid_path: str,
) -> None:
    """
    .pt·.pth 이외 확장자를 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="must use a .pt or .pth extension",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                invalid_path
            ),
            verbose=False,
        )


def test_run_training_rejects_directory_as_checkpoint_path(
    tmp_path: Path,
) -> None:
    """
    기존 Directory를 Checkpoint 파일 경로로 전달하면 거부하는지 확인한다.
    """
    checkpoint_directory = (
        tmp_path
        / "checkpoint_directory"
    )

    checkpoint_directory.mkdir()

    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        ValueError,
        match="must point to a file, not a directory",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                checkpoint_directory
            ),
            verbose=False,
        )


# =============================================================================
# Verbose Validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_verbose",
    [
        0,
        1,
        None,
        "True",
        [],
    ],
)
def test_run_training_rejects_non_boolean_verbose(
    invalid_verbose: object,
    tmp_path: Path,
) -> None:
    """
    bool이 아닌 verbose 값을 거부하는지 확인한다.
    """
    (
        model,
        train_loader,
        validation_loader,
        loss_function,
        optimizer,
    ) = create_training_components()

    with pytest.raises(
        TypeError,
        match="verbose must be a bool",
    ):
        run_training(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            loss_function=loss_function,
            optimizer=optimizer,
            device="cpu",
            epoch_count=1,
            checkpoint_path=(
                tmp_path
                / "best.pt"
            ),
            verbose=invalid_verbose,  # type: ignore[arg-type]
        )