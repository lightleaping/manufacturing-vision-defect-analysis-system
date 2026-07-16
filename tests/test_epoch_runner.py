"""
Train and validation epoch runner unit tests.

테스트 대상
----------
src/training/epoch_runner.py

테스트 목적
----------
Train Epoch와 Validation Epoch가 서로 다른 역할을 정확하게 수행하는지
자동으로 검증한다.

현재 클래스 정의
---------------
0 = NORMAL

1 = DEFECT

Positive Class:

    DEFECT

현재 모델 출력
---------------
Binary Raw Logit

Shape:

    [batch_size]

현재 Loss
---------
BCEWithLogitsLoss

현재 Optimizer
--------------
Adam

현재 Classification Threshold
------------------------------
0.5

Train Epoch
-----------
model.train()

-> Gradient 활성화

-> Forward

-> Loss

-> Backward

-> Optimizer Step

-> Parameter 변경

Validation Epoch
----------------
model.eval()

-> torch.inference_mode()

-> Forward

-> Loss·Accuracy 계산

-> Parameter 변경 없음

이번 테스트 범위
----------------
Epoch 한 번의 실행과 결과 집계만 검증한다.

여러 Epoch 반복, Training History, Best Model 저장,
Checkpoint는 아직 테스트하지 않는다.
"""

import math

import pytest
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import SGD
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.training.epoch_runner import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    EpochResult,
    train_one_epoch,
    validate_one_epoch,
)
from src.training.loss_function import (
    create_binary_classification_loss,
)
from src.training.optimizer import (
    create_optimizer,
)


# =============================================================================
# Helper Models
# =============================================================================


class SimpleMeanLogitModel(nn.Module):
    """
    이미지 전체 평균을 Binary Logit으로 변환하는 작은 테스트 Model.

    입력
    ----
    Shape:

        [batch_size, 3, height, width]

    출력
    ----
    Shape:

        [batch_size]

    구조
    ----
    이미지 평균

    ->

    Scalar Weight

    ->

    Bias

    ->

    Binary Raw Logit

    목적
    ----
    CNNBaseline보다 계산량이 작으므로
    Epoch Runner 자체의 동작을 빠르게 테스트할 때 사용한다.
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
        이미지 평균값을 이용해 Binary Logit을 계산한다.
        """
        image_means = images.mean(
            dim=(
                1,
                2,
                3,
            ),
        )

        logits = (
            image_means
            * self.weight
            + self.bias
        )

        return logits


class FirstPixelLogitModel(nn.Module):
    """
    이미지 첫 Pixel 값을 Binary Logit으로 사용하는 테스트 Model.

    목적
    ----
    테스트에서 원하는 Logit 값을 Image Tensor에 직접 기록해
    Loss와 Accuracy를 정확하게 계산할 때 사용한다.

    Image:

        images[
            batch_index,
            0,
            0,
            0,
        ]

    위 값을 Binary Logit으로 반환한다.
    """

    def __init__(self) -> None:
        """
        Model에 학습 가능한 Parameter가 존재하도록 Scale을 생성한다.
        """
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(
                1.0,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        각 이미지의 첫 번째 Channel·첫 Pixel 값을 Logit으로 반환한다.
        """
        return (
            images[
                :,
                0,
                0,
                0,
            ]
            * self.scale
        )


class GradientStateTrackingModel(nn.Module):
    """
    Forward 시 Gradient 활성화 상태를 기록하는 테스트 Model.

    Train Epoch 예상:

        torch.is_grad_enabled()

        -> True

    Validation Epoch 예상:

        torch.is_grad_enabled()

        -> False
    """

    def __init__(self) -> None:
        """
        학습 가능한 Bias와 Gradient 상태 기록 List를 생성한다.
        """
        super().__init__()

        self.bias = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self.gradient_enabled_states: list[bool] = []

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Forward 실행 시 Gradient 활성화 여부를 기록한다.
        """
        self.gradient_enabled_states.append(
            torch.is_grad_enabled()
        )

        batch_size = images.shape[0]

        return self.bias.expand(
            batch_size,
        )


class ForwardCountingModel(nn.Module):
    """
    Forward 호출 횟수를 기록하는 테스트 Model.

    목적
    ----
    DataLoader의 모든 Batch가 실제로 처리되는지 확인한다.
    """

    def __init__(self) -> None:
        """
        학습 가능한 Bias와 호출 횟수를 생성한다.
        """
        super().__init__()

        self.bias = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self.forward_call_count = 0

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Forward 호출 횟수를 증가시키고 Logit을 반환한다.
        """
        self.forward_call_count += 1

        return self.bias.expand(
            images.shape[0],
        )


class ParameterlessModel(nn.Module):
    """
    Parameter가 없는 Model.

    Epoch Runner가 Device 검증 단계에서
    Parameter 없는 Model을 거부하는지 확인할 때 사용한다.
    """

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        입력 Batch 크기만큼 0 Logit을 반환한다.
        """
        return torch.zeros(
            images.shape[0],
            dtype=torch.float32,
        )


# =============================================================================
# Invalid Output Models
# =============================================================================


class NonTensorOutputModel(nn.Module):
    """
    Tensor가 아닌 List를 반환하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Device 검증을 통과하기 위한 Dummy Parameter를 생성한다.
        """
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> list[float]:
        """
        잘못된 List 출력을 반환한다.
        """
        return [
            0.0
            for _ in range(
                images.shape[0]
            )
        ]


class TwoDimensionalLogitModel(nn.Module):
    """
    [B, 1] 형태의 잘못된 2차원 Logit을 반환한다.
    """

    def __init__(self) -> None:
        """
        Dummy Parameter를 생성한다.
        """
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        잘못된 [B, 1] Logit을 반환한다.
        """
        return (
            torch.zeros(
                images.shape[0],
                1,
                dtype=torch.float32,
                device=images.device,
            )
            + self.dummy
        )


class WrongBatchSizeLogitModel(nn.Module):
    """
    Image Batch보다 하나 많은 Logit을 반환하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Dummy Parameter를 생성한다.
        """
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        [B + 1] Logit을 반환한다.
        """
        return (
            torch.zeros(
                images.shape[0] + 1,
                dtype=torch.float32,
                device=images.device,
            )
            + self.dummy
        )


class IntegerLogitModel(nn.Module):
    """
    정수 dtype Logit을 반환하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Device 검증용 Dummy Parameter를 생성한다.
        """
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        torch.int64 Logit을 반환한다.
        """
        return torch.zeros(
            images.shape[0],
            dtype=torch.int64,
            device=images.device,
        )


class NonFiniteLogitModel(nn.Module):
    """
    NaN Logit을 반환하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Dummy Parameter를 생성한다.
        """
        super().__init__()

        self.dummy = nn.Parameter(
            torch.tensor(
                0.0,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        모든 값이 NaN인 Logit을 반환한다.
        """
        return (
            torch.full(
                (
                    images.shape[0],
                ),
                fill_value=float("nan"),
                dtype=torch.float32,
                device=images.device,
            )
            + self.dummy
        )


# =============================================================================
# Invalid Loss Functions
# =============================================================================


class NonTensorLoss(nn.Module):
    """
    Tensor가 아닌 float을 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> float:
        """
        Python float을 반환한다.
        """
        return 0.5


class VectorLoss(nn.Module):
    """
    Scalar가 아닌 [B] Tensor를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Sample별 Loss Vector를 반환한다.
        """
        return torch.ones_like(
            logits,
        )


class IntegerScalarLoss(nn.Module):
    """
    정수 dtype Scalar Tensor를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        torch.int64 Scalar를 반환한다.
        """
        return torch.tensor(
            1,
            dtype=torch.int64,
            device=logits.device,
        )


class NonFiniteLoss(nn.Module):
    """
    NaN Scalar Loss를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        계산 그래프에 연결된 NaN Scalar를 반환한다.
        """
        return (
            logits.sum()
            * float("nan")
        )


class NegativeLoss(nn.Module):
    """
    음수 Scalar Loss를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        -1.0 Loss를 반환한다.
        """
        return (
            logits.mean()
            * 0.0
            - 1.0
        )


# =============================================================================
# Counting Optimizer
# =============================================================================


class CountingSGD(SGD):
    """
    zero_grad()와 step() 호출 횟수를 기록하는 SGD Optimizer.

    목적
    ----
    Train Epoch에서 각 Batch마다 다음이 한 번씩 호출되는지 확인한다.

        optimizer.zero_grad()

        optimizer.step()
    """

    def __init__(
        self,
        parameters: object,
        learning_rate: float,
    ) -> None:
        """
        SGD를 생성하고 호출 횟수를 0으로 초기화한다.
        """
        super().__init__(
            params=parameters,
            lr=learning_rate,
        )

        self.zero_grad_call_count = 0

        self.step_call_count = 0

    def zero_grad(
        self,
        set_to_none: bool = True,
    ) -> None:
        """
        zero_grad() 호출 횟수를 기록한다.
        """
        self.zero_grad_call_count += 1

        super().zero_grad(
            set_to_none=set_to_none,
        )

    def step(
        self,
        closure: object = None,
    ) -> object:
        """
        step() 호출 횟수를 기록한다.
        """
        self.step_call_count += 1

        return super().step(
            closure=closure,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def create_image_batch(
    sample_count: int,
    image_value: float = 1.0,
) -> Tensor:
    """
    Epoch 테스트용 RGB 이미지 Batch를 생성한다.

    입력
    ----
    sample_count:
        이미지 수

    image_value:
        모든 Pixel에 채울 값

    출력
    ----
    Tensor

        Shape:

            [sample_count, 3, 8, 8]

        dtype:

            torch.float32
    """
    return torch.full(
        (
            sample_count,
            3,
            8,
            8,
        ),
        fill_value=image_value,
        dtype=torch.float32,
    )


def create_binary_labels(
    values: list[int],
) -> Tensor:
    """
    int64 Binary Label Tensor를 생성한다.
    """
    return torch.tensor(
        values,
        dtype=torch.int64,
    )


def create_standard_data_loader(
    sample_count: int = 5,
    batch_size: int = 2,
) -> DataLoader:
    """
    일반적인 Binary Image DataLoader를 생성한다.

    Label은 0·1을 번갈아 사용한다.
    """
    images = create_image_batch(
        sample_count=sample_count,
    )

    labels = torch.tensor(
        [
            index % 2
            for index in range(
                sample_count
            )
        ],
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


def create_single_batch_data_loader(
    batch: object,
) -> DataLoader:
    """
    하나의 사용자 지정 Batch를 그대로 반환하는 DataLoader를 생성한다.

    목적
    ----
    잘못된 Batch 형식에 대한 Epoch Runner 예외 처리를 테스트한다.

    batch_size=None을 사용하면 Dataset Item을 추가 Collate 없이
    하나의 Batch처럼 전달할 수 있다.
    """
    return DataLoader(
        dataset=[
            batch,
        ],
        batch_size=None,
        shuffle=False,
    )


def clone_model_parameters(
    model: nn.Module,
) -> list[Tensor]:
    """
    Model Parameter 값을 복사한다.
    """
    return [
        parameter
        .detach()
        .clone()
        for parameter in model.parameters()
    ]


def model_parameters_are_equal(
    first_parameters: list[Tensor],
    second_parameters: list[Tensor],
) -> bool:
    """
    두 Parameter Snapshot의 모든 값이 같은지 확인한다.
    """
    if len(
        first_parameters,
    ) != len(
        second_parameters,
    ):
        return False

    return all(
        torch.equal(
            first_parameter,
            second_parameter,
        )
        for (
            first_parameter,
            second_parameter,
        ) in zip(
            first_parameters,
            second_parameters,
            strict=True,
        )
    )


# =============================================================================
# Default Configuration
# =============================================================================


def test_default_classification_threshold_is_zero_point_five() -> None:
    """
    초기 Binary Classification Threshold가 0.5인지 확인한다.
    """
    assert (
        DEFAULT_CLASSIFICATION_THRESHOLD
        == 0.5
    )


# =============================================================================
# EpochResult
# =============================================================================


def test_epoch_result_stores_valid_values() -> None:
    """
    정상 Epoch 결과를 저장할 수 있는지 확인한다.
    """
    result = EpochResult(
        average_loss=0.625,
        accuracy=0.75,
        sample_count=8,
        batch_count=2,
    )

    assert result.average_loss == 0.625

    assert result.accuracy == 0.75

    assert result.sample_count == 8

    assert result.batch_count == 2


def test_epoch_result_is_frozen() -> None:
    """
    EpochResult 생성 후 필드를 변경할 수 없는지 확인한다.

    Training History에 저장된 과거 Epoch 결과가
    실수로 변경되는 것을 방지한다.
    """
    result = EpochResult(
        average_loss=0.5,
        accuracy=0.5,
        sample_count=4,
        batch_count=2,
    )

    with pytest.raises(
        AttributeError,
    ):
        result.accuracy = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid_average_loss",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_epoch_result_rejects_non_finite_average_loss(
    invalid_average_loss: float,
) -> None:
    """
    NaN·inf Average Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="average_loss must be finite",
    ):
        EpochResult(
            average_loss=invalid_average_loss,
            accuracy=0.5,
            sample_count=4,
            batch_count=2,
        )


def test_epoch_result_rejects_negative_average_loss() -> None:
    """
    음수 Average Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        EpochResult(
            average_loss=-0.1,
            accuracy=0.5,
            sample_count=4,
            batch_count=2,
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_epoch_result_rejects_non_finite_accuracy(
    invalid_accuracy: float,
) -> None:
    """
    NaN·inf Accuracy를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="accuracy must be finite",
    ):
        EpochResult(
            average_loss=0.5,
            accuracy=invalid_accuracy,
            sample_count=4,
            batch_count=2,
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        -0.1,
        1.1,
    ],
)
def test_epoch_result_rejects_accuracy_outside_zero_and_one(
    invalid_accuracy: float,
) -> None:
    """
    0~1 범위를 벗어난 Accuracy를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="accuracy must be between 0 and 1",
    ):
        EpochResult(
            average_loss=0.5,
            accuracy=invalid_accuracy,
            sample_count=4,
            batch_count=2,
        )


@pytest.mark.parametrize(
    "invalid_sample_count",
    [
        0,
        -1,
    ],
)
def test_epoch_result_rejects_non_positive_sample_count(
    invalid_sample_count: int,
) -> None:
    """
    0 이하 Sample 수를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="sample_count must be greater than 0",
    ):
        EpochResult(
            average_loss=0.5,
            accuracy=0.5,
            sample_count=invalid_sample_count,
            batch_count=2,
        )


@pytest.mark.parametrize(
    "invalid_batch_count",
    [
        0,
        -1,
    ],
)
def test_epoch_result_rejects_non_positive_batch_count(
    invalid_batch_count: int,
) -> None:
    """
    0 이하 Batch 수를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="batch_count must be greater than 0",
    ):
        EpochResult(
            average_loss=0.5,
            accuracy=0.5,
            sample_count=4,
            batch_count=invalid_batch_count,
        )


# =============================================================================
# Train Epoch Basic Behavior
# =============================================================================


def test_train_one_epoch_returns_epoch_result() -> None:
    """
    Train Epoch가 EpochResult를 반환하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert isinstance(
        result,
        EpochResult,
    )


def test_train_one_epoch_sets_model_to_training_mode() -> None:
    """
    Train Epoch가 Model을 Training Mode로 변경하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    model.eval()

    assert model.training is False

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    _ = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert model.training is True


def test_train_one_epoch_enables_gradient_during_forward() -> None:
    """
    Train Forward에서 Gradient가 활성화되는지 확인한다.
    """
    model = GradientStateTrackingModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=4,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    _ = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert (
        model.gradient_enabled_states
        == [
            True,
            True,
        ]
    )


def test_train_one_epoch_changes_trainable_parameters() -> None:
    """
    Train Epoch 후 Model Parameter가 실제로 변경되는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    images = create_image_batch(
        sample_count=4,
        image_value=1.0,
    )

    labels = create_binary_labels(
        [
            1,
            1,
            1,
            1,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    parameters_before = (
        clone_model_parameters(
            model=model,
        )
    )

    _ = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    parameters_after = (
        clone_model_parameters(
            model=model,
        )
    )

    assert not model_parameters_are_equal(
        first_parameters=parameters_before,
        second_parameters=parameters_after,
    )


def test_train_one_epoch_processes_every_batch() -> None:
    """
    Train DataLoader의 모든 Batch가 처리되는지 확인한다.
    """
    model = ForwardCountingModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=5,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert len(
        data_loader,
    ) == 3

    assert (
        model.forward_call_count
        == 3
    )

    assert result.batch_count == 3


def test_train_one_epoch_calls_zero_grad_and_step_for_each_batch() -> None:
    """
    각 Train Batch마다 zero_grad()와 step()이 한 번씩 호출되는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=5,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = CountingSGD(
        parameters=model.parameters(),
        learning_rate=0.01,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert result.batch_count == 3

    assert (
        optimizer.zero_grad_call_count
        == 3
    )

    assert (
        optimizer.step_call_count
        == 3
    )


def test_train_one_epoch_reports_exact_sample_and_batch_counts() -> None:
    """
    마지막 Batch 크기가 작아도 실제 Sample·Batch 수를 정확히 기록하는지 확인한다.

    Dataset:

        5장

    Batch Size:

        2

    Batch:

        2

        2

        1

    결과:

        Sample 5

        Batch 3
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=5,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert result.sample_count == 5

    assert result.batch_count == 3


def test_train_one_epoch_returns_finite_non_negative_loss() -> None:
    """
    Train Average Loss가 유한하고 0 이상인지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert math.isfinite(
        result.average_loss,
    )

    assert result.average_loss >= 0.0


def test_train_one_epoch_returns_accuracy_between_zero_and_one() -> None:
    """
    Train Accuracy가 0~1 범위인지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    result = train_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        device="cpu",
    )

    assert (
        0.0
        <= result.accuracy
        <= 1.0
    )


# =============================================================================
# Validation Epoch Basic Behavior
# =============================================================================


def test_validate_one_epoch_returns_epoch_result() -> None:
    """
    Validation Epoch가 EpochResult를 반환하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert isinstance(
        result,
        EpochResult,
    )


def test_validate_one_epoch_sets_model_to_evaluation_mode() -> None:
    """
    Validation Epoch가 Model을 Evaluation Mode로 변경하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    assert model.training is True

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    _ = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert model.training is False


def test_validate_one_epoch_disables_gradient_during_forward() -> None:
    """
    Validation Forward에서 Gradient가 비활성화되는지 확인한다.
    """
    model = GradientStateTrackingModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=4,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    _ = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert (
        model.gradient_enabled_states
        == [
            False,
            False,
        ]
    )


def test_validate_one_epoch_does_not_change_parameters() -> None:
    """
    Validation 전후 Model Parameter가 동일한지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    parameters_before = (
        clone_model_parameters(
            model=model,
        )
    )

    _ = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    parameters_after = (
        clone_model_parameters(
            model=model,
        )
    )

    assert model_parameters_are_equal(
        first_parameters=parameters_before,
        second_parameters=parameters_after,
    )


def test_validate_one_epoch_does_not_create_parameter_gradients() -> None:
    """
    Validation 후 Parameter Gradient가 생성되지 않는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    for parameter in model.parameters():
        parameter.grad = None

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    _ = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_validate_one_epoch_processes_every_batch() -> None:
    """
    Validation DataLoader의 모든 Batch가 처리되는지 확인한다.
    """
    model = ForwardCountingModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=5,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert (
        model.forward_call_count
        == 3
    )

    assert result.batch_count == 3


def test_validate_one_epoch_reports_exact_sample_and_batch_counts() -> None:
    """
    Validation Sample·Batch 수를 정확히 집계하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader(
            sample_count=5,
            batch_size=2,
        )
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert result.sample_count == 5

    assert result.batch_count == 3


def test_validate_one_epoch_returns_finite_non_negative_loss() -> None:
    """
    Validation Average Loss가 유한하고 0 이상인지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert math.isfinite(
        result.average_loss,
    )

    assert result.average_loss >= 0.0


def test_validate_one_epoch_returns_accuracy_between_zero_and_one() -> None:
    """
    Validation Accuracy가 0~1 범위인지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    assert (
        0.0
        <= result.accuracy
        <= 1.0
    )


# =============================================================================
# Sample-weighted Average Loss
# =============================================================================


def test_validation_average_loss_is_weighted_by_sample_count() -> None:
    """
    Epoch Average Loss가 Batch 단순 평균이 아니라
    전체 Sample 수 기준으로 계산되는지 확인한다.

    Dataset:

        3 Sample

    Batch Size:

        2

    Batch 구성:

        첫 Batch:

            2 Sample

        마지막 Batch:

            1 Sample

    각 Sample Loss를 모두 더한 뒤
    전체 Sample 수 3으로 나눈 값과 같아야 한다.
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        3,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    # 원하는 Logit:
    #
    # Sample 1:
    #
    #     0.0
    #
    # Sample 2:
    #
    #     0.0
    #
    # Sample 3:
    #
    #     4.0
    images[
        2,
        0,
        0,
        0,
    ] = 4.0

    labels = create_binary_labels(
        [
            0,
            1,
            0,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    loss_function = (
        create_binary_classification_loss()
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
        device="cpu",
    )

    expected_logits = torch.tensor(
        [
            0.0,
            0.0,
            4.0,
        ],
        dtype=torch.float32,
    )

    expected_targets = labels.to(
        dtype=torch.float32,
    )

    individual_losses = (
        functional
        .binary_cross_entropy_with_logits(
            input=expected_logits,
            target=expected_targets,
            reduction="none",
        )
    )

    expected_sample_average = (
        individual_losses
        .mean()
        .item()
    )

    first_batch_average = (
        individual_losses[
            :2
        ]
        .mean()
        .item()
    )

    second_batch_average = (
        individual_losses[
            2:
        ]
        .mean()
        .item()
    )

    incorrect_batch_average = (
        first_batch_average
        + second_batch_average
    ) / 2.0

    assert result.average_loss == (
        pytest.approx(
            expected_sample_average,
            rel=1e-6,
            abs=1e-6,
        )
    )

    assert result.average_loss != (
        pytest.approx(
            incorrect_batch_average,
            rel=1e-4,
            abs=1e-4,
        )
    )


# =============================================================================
# Binary Accuracy and Threshold
# =============================================================================


def test_validation_accuracy_matches_known_predictions() -> None:
    """
    미리 알고 있는 Logit과 Label에서 Accuracy가 1.0인지 확인한다.

    Logit:

        -2.0

        2.0

        0.0

        -0.1

    Threshold:

        0.5

    Prediction:

        0

        1

        1

        0
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        4,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    images[
        :,
        0,
        0,
        0,
    ] = torch.tensor(
        [
            -2.0,
            2.0,
            0.0,
            -0.1,
        ],
        dtype=torch.float32,
    )

    labels = create_binary_labels(
        [
            0,
            1,
            1,
            0,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert result.accuracy == 1.0


def test_probability_equal_to_threshold_is_predicted_as_defect() -> None:
    """
    Probability가 Threshold와 정확히 같으면 DEFECT 1로 분류하는지 확인한다.

    Logit:

        0.0

    Sigmoid:

        0.5

    Threshold:

        0.5

    코드:

        probability >= threshold

    Prediction:

        1

        DEFECT
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        2,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    labels = create_binary_labels(
        [
            1,
            1,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.5,
    )

    assert result.accuracy == 1.0


def test_custom_threshold_changes_binary_prediction() -> None:
    """
    사용자 지정 Threshold가 실제 Prediction에 적용되는지 확인한다.

    Logit:

        0

    Probability:

        0.5

    Threshold:

        0.6

    Prediction:

        0

        NORMAL
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        2,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    labels = create_binary_labels(
        [
            0,
            0,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.6,
    )

    assert result.accuracy == 1.0


def test_threshold_zero_predicts_every_sample_as_defect() -> None:
    """
    Threshold 0.0이면 모든 유효 확률이 DEFECT로 분류되는지 확인한다.
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        3,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    labels = create_binary_labels(
        [
            1,
            1,
            1,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.0,
    )

    assert result.accuracy == 1.0


def test_threshold_one_predicts_zero_for_probability_below_one() -> None:
    """
    Threshold 1.0에서 Probability 0.5는 NORMAL 0으로 분류되는지 확인한다.
    """
    model = FirstPixelLogitModel()

    images = torch.zeros(
        3,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    labels = create_binary_labels(
        [
            0,
            0,
            0,
        ]
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    result = validate_one_epoch(
        model=model,
        data_loader=data_loader,
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=1.0,
    )

    assert result.accuracy == 1.0


# =============================================================================
# Common Argument Validation
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
def test_validate_one_epoch_rejects_non_module_model(
    invalid_model: object,
) -> None:
    """
    nn.Module이 아닌 Model을 거부하는지 확인한다.
    """
    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        validate_one_epoch(
            model=invalid_model,  # type: ignore[arg-type]
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_data_loader",
    [
        None,
        [],
        "data_loader",
        object(),
    ],
)
def test_validate_one_epoch_rejects_non_dataloader(
    invalid_data_loader: object,
) -> None:
    """
    DataLoader가 아닌 객체를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    with pytest.raises(
        TypeError,
        match="data_loader must be an instance",
    ):
        validate_one_epoch(
            model=model,
            data_loader=invalid_data_loader,  # type: ignore[arg-type]
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_loss_function",
    [
        None,
        "BCEWithLogitsLoss",
        123,
        object(),
    ],
)
def test_validate_one_epoch_rejects_non_module_loss(
    invalid_loss_function: object,
) -> None:
    """
    nn.Module이 아닌 Loss Function을 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="loss_function must be an instance of torch.nn.Module",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=invalid_loss_function,  # type: ignore[arg-type]
            device="cpu",
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
def test_train_one_epoch_rejects_non_optimizer(
    invalid_optimizer: object,
) -> None:
    """
    torch.optim.Optimizer가 아닌 객체를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="optimizer must be an instance",
    ):
        train_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=invalid_optimizer,  # type: ignore[arg-type]
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_device",
    [
        None,
        0,
        1.0,
        object(),
    ],
)
def test_validate_one_epoch_rejects_invalid_device_type(
    invalid_device: object,
) -> None:
    """
    문자열 또는 torch.device가 아닌 Device를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="device must be a string or torch.device",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device=invalid_device,  # type: ignore[arg-type]
        )


def test_validate_one_epoch_rejects_invalid_device_string() -> None:
    """
    PyTorch가 해석할 수 없는 Device 문자열을 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="Invalid device",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="not-a-device",
        )


def test_validate_one_epoch_rejects_unavailable_cuda() -> None:
    """
    CUDA가 없는 환경에서 CUDA Device 요청을 거부하는지 확인한다.
    """
    if torch.cuda.is_available():
        pytest.skip(
            "CUDA is available in this environment."
        )

    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="CUDA device was requested",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cuda",
        )


def test_validate_one_epoch_rejects_model_without_parameters() -> None:
    """
    Parameter가 없는 Model을 거부하는지 확인한다.
    """
    model = ParameterlessModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="at least one parameter",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_model_on_different_device() -> None:
    """
    Model Parameter Device와 요청 Device가 다르면 거부하는지 확인한다.

    Meta Device를 사용해 CPU와 다른 Device 상태를 만든다.
    Forward 전에 Device 검증에서 오류가 발생해야 한다.
    """
    model = SimpleMeanLogitModel()

    model = model.to(
        device="meta",
    )

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="all model parameters must be on the requested device",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Classification Threshold Validation
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
def test_validate_one_epoch_rejects_invalid_threshold_type(
    invalid_threshold: object,
) -> None:
    """
    Real Number가 아닌 Threshold와 bool을 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="classification_threshold must be a real number",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
            classification_threshold=invalid_threshold,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_validate_one_epoch_rejects_threshold_outside_zero_and_one(
    invalid_threshold: float,
) -> None:
    """
    0~1 범위를 벗어난 Threshold를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="classification_threshold must be between 0 and 1",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
            classification_threshold=invalid_threshold,
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_one_epoch_rejects_non_finite_threshold(
    invalid_threshold: float,
) -> None:
    """
    NaN·inf Threshold를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="classification_threshold must be finite",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
            classification_threshold=invalid_threshold,
        )


# =============================================================================
# Batch Validation
# =============================================================================


def test_validate_one_epoch_rejects_non_sequence_batch() -> None:
    """
    tuple·list가 아닌 Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = torch.zeros(
        2,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        TypeError,
        match="each data_loader batch must be a tuple or list",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_batch",
    [
        (
            torch.zeros(
                2,
                3,
                8,
                8,
            ),
        ),
        (
            torch.zeros(
                2,
                3,
                8,
                8,
            ),
            torch.zeros(
                2,
                dtype=torch.int64,
            ),
            "extra",
        ),
    ],
)
def test_validate_one_epoch_rejects_wrong_batch_item_count(
    invalid_batch: tuple[object, ...],
) -> None:
    """
    Item 수가 2개가 아닌 Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="exactly two items",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_tensor_images() -> None:
    """
    Tensor가 아닌 Image Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        "images",
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        TypeError,
        match="batch images must be a torch.Tensor",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_tensor_labels() -> None:
    """
    Tensor가 아닌 Label Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        [
            0,
            1,
        ],
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        TypeError,
        match="batch labels must be a torch.Tensor",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_four_dimensional_images() -> None:
    """
    [B, C, H, W]가 아닌 Image Tensor를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                0,
                1,
                0,
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="batch images must have 4 dimensions",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_one_dimensional_labels() -> None:
    """
    [B]가 아닌 Label Tensor를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [
                    0,
                ],
                [
                    1,
                ],
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="batch labels must have 1 dimension",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_empty_image_batch() -> None:
    """
    이미지가 한 장도 없는 Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.empty(
            0,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.empty(
            0,
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="at least one image",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_mismatched_image_and_label_counts() -> None:
    """
    Image 수와 Label 수가 다른 Batch를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="image and label batch sizes must match",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_integer_image_tensor() -> None:
    """
    정수 dtype Image Tensor를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.int64,
        ),
        torch.tensor(
            [
                0,
                1,
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        TypeError,
        match="batch images must use a floating-point dtype",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_invalid_binary_label_value() -> None:
    """
    0·1 이외 Label이 Target 변환 단계에서 거부되는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    invalid_batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.int64,
        ),
    )

    data_loader = (
        create_single_batch_data_loader(
            batch=invalid_batch,
        )
    )

    with pytest.raises(
        ValueError,
        match="only binary values 0 and 1",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Model Logit Validation
# =============================================================================


def test_validate_one_epoch_rejects_non_tensor_model_output() -> None:
    """
    Tensor가 아닌 Model 출력을 거부하는지 확인한다.
    """
    model = NonTensorOutputModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="model output logits must be a torch.Tensor",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_two_dimensional_logits() -> None:
    """
    [B]가 아닌 [B, 1] Logit을 거부하는지 확인한다.
    """
    model = TwoDimensionalLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="model output logits must have 1 dimension",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_wrong_logit_batch_size() -> None:
    """
    Image Batch 수와 다른 Logit 수를 거부하는지 확인한다.
    """
    model = WrongBatchSizeLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="logit batch size must match image batch size",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_integer_logits() -> None:
    """
    정수 dtype Logit을 거부하는지 확인한다.
    """
    model = IntegerLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="model output logits must use a floating-point dtype",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_finite_logits() -> None:
    """
    NaN Logit을 거부하는지 확인한다.
    """
    model = NonFiniteLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="model output logits must contain only finite values",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Loss Output Validation
# =============================================================================


def test_validate_one_epoch_rejects_non_tensor_loss() -> None:
    """
    Tensor가 아닌 Loss 출력을 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="loss_function output must be a torch.Tensor",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=NonTensorLoss(),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_scalar_loss() -> None:
    """
    [B] 형태의 Loss Vector를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="loss_function output must be a scalar tensor",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=VectorLoss(),
            device="cpu",
        )


def test_validate_one_epoch_rejects_integer_loss() -> None:
    """
    정수 dtype Scalar Loss를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        TypeError,
        match="loss_function output must use a floating-point dtype",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=IntegerScalarLoss(),
            device="cpu",
        )


def test_validate_one_epoch_rejects_non_finite_loss() -> None:
    """
    NaN Scalar Loss를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="loss_function output must be finite",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=NonFiniteLoss(),
            device="cpu",
        )


def test_validate_one_epoch_rejects_negative_loss() -> None:
    """
    음수 Scalar Loss를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    data_loader = (
        create_standard_data_loader()
    )

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=NegativeLoss(),
            device="cpu",
        )


# =============================================================================
# Empty DataLoader
# =============================================================================


def test_train_one_epoch_rejects_empty_data_loader() -> None:
    """
    Batch가 하나도 없는 Train DataLoader를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    empty_dataset = TensorDataset(
        torch.empty(
            0,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.empty(
            0,
            dtype=torch.int64,
        ),
    )

    data_loader = DataLoader(
        dataset=empty_dataset,
        batch_size=2,
        shuffle=False,
    )

    optimizer = create_optimizer(
        model=model,
    )

    with pytest.raises(
        ValueError,
        match="train data_loader must contain",
    ):
        train_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            optimizer=optimizer,
            device="cpu",
        )


def test_validate_one_epoch_rejects_empty_data_loader() -> None:
    """
    Batch가 하나도 없는 Validation DataLoader를 거부하는지 확인한다.
    """
    model = SimpleMeanLogitModel()

    empty_dataset = TensorDataset(
        torch.empty(
            0,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.empty(
            0,
            dtype=torch.int64,
        ),
    )

    data_loader = DataLoader(
        dataset=empty_dataset,
        batch_size=2,
        shuffle=False,
    )

    with pytest.raises(
        ValueError,
        match="validation data_loader must contain",
    ):
        validate_one_epoch(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )