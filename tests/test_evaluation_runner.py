"""
Binary image classification evaluation runner unit tests.

테스트 대상
----------
src/evaluation/evaluation_runner.py

테스트 목적
----------
Binary Image Classifier의 Evaluation 결과가 정확하고,
Model·DataLoader·Batch·Logit·Loss 오류를 안전하게 방어하는지 검증한다.

현재 Class 정의
--------------
0:

    NORMAL

1:

    DEFECT

Positive Class:

    DEFECT

현재 Evaluation Result
---------------------
Average Loss

Accuracy

Sample Count

Batch Count

Classification Threshold

Ground Truth Labels

Raw Logits

DEFECT Probabilities

Binary Predictions

핵심 테스트
----------
Evaluation Mode

Gradient 비활성화

Model State 유지

모든 Batch 처리

Sample 가중 평균 Loss

Known Prediction

Custom Threshold

Threshold Equality

Tensor Shape

Tensor Dtype

Tensor Device

Result 내부 일관성

잘못된 Batch

잘못된 Label

잘못된 Logit

잘못된 Loss

빈 DataLoader
"""

from __future__ import annotations

from dataclasses import (
    FrozenInstanceError,
)
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from src.evaluation.evaluation_runner import (
    BinaryEvaluationResult,
    evaluate_binary_classifier,
)
from src.training.loss_function import (
    create_binary_classification_loss,
)


# =============================================================================
# Test Models
# =============================================================================


class ImageValueLogitModel(nn.Module):
    """
    이미지 첫 번째 Pixel 값을 Binary Logit으로 사용하는 테스트 Model.

    입력
    ----
    images:

        [B, 3, H, W]

    출력
    ----
    images[:, 0, 0, 0]

    ->

    Scale

    ->

    Bias

    ->

    [B]
    """

    def __init__(
        self,
        scale: float = 1.0,
        bias: float = 0.0,
    ) -> None:
        """
        학습 가능한 Scale·Bias를 생성한다.
        """
        super().__init__()

        self.scale = nn.Parameter(
            torch.tensor(
                scale,
                dtype=torch.float32,
            )
        )

        self.bias = nn.Parameter(
            torch.tensor(
                bias,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        이미지 첫 Pixel을 Logit으로 변환한다.
        """
        return (
            images[
                :,
                0,
                0,
                0,
            ]
            * self.scale
            + self.bias
        )


class RecordingEvaluationModel(
    ImageValueLogitModel
):
    """
    Forward 호출 횟수·Gradient 상태·입력 정보를 기록한다.
    """

    def __init__(self) -> None:
        """
        기록 필드를 초기화한다.
        """
        super().__init__()

        self.forward_call_count = 0

        self.gradient_enabled_values: list[
            bool
        ] = []

        self.input_dtypes: list[
            torch.dtype
        ] = []

        self.input_devices: list[
            torch.device
        ] = []

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        Evaluation 실행 상태를 기록한 뒤 Logit을 반환한다.
        """
        self.forward_call_count += 1

        self.gradient_enabled_values.append(
            torch.is_grad_enabled()
        )

        self.input_dtypes.append(
            images.dtype
        )

        self.input_devices.append(
            images.device
        )

        return super().forward(
            images=images,
        )


class ParameterlessModel(nn.Module):
    """
    Parameter·Buffer State가 없는 잘못된 Model.
    """

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        0 Logit을 반환한다.
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
    CPU Parameter와 Meta Buffer를 가진 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        서로 다른 Device의 Parameter·Buffer를 생성한다.
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
        Device 검증에서 먼저 중단되므로 실제 실행되지 않는다.
        """
        return (
            images[
                :,
                0,
                0,
                0,
            ]
            * self.weight
        )


class NonTensorOutputModel(nn.Module):
    """
    Tensor가 아닌 List를 출력하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Model State 검증 통과용 Parameter를 생성한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        images: Tensor,
    ) -> list[float]:
        """
        잘못된 List Output을 반환한다.
        """
        return [
            0.0
            for _ in range(
                images.shape[0]
            )
        ]


class TwoDimensionalOutputModel(
    nn.Module
):
    """
    [B, 1] Logit을 출력하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Parameter를 생성한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
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
        잘못된 2차원 Logit을 반환한다.
        """
        return torch.zeros(
            (
                images.shape[0],
                1,
            ),
            dtype=torch.float32,
            device=images.device,
        ) + (
            self.anchor
            * 0.0
        )


class ScalarOutputModel(nn.Module):
    """
    Scalar Logit을 출력하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Parameter를 생성한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
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
        0차원 Scalar를 반환한다.
        """
        return (
            self.anchor
            * 0.0
        )


class WrongBatchSizeOutputModel(
    nn.Module
):
    """
    입력보다 Sample이 한 개 많은 Logit을 출력한다.
    """

    def __init__(self) -> None:
        """
        Parameter를 생성한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
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
        [B + 1] Logit을 반환한다.
        """
        return torch.zeros(
            images.shape[0] + 1,
            dtype=torch.float32,
            device=images.device,
        ) + (
            self.anchor
            * 0.0
        )


class IntegerOutputModel(nn.Module):
    """
    int64 Logit을 출력하는 잘못된 Model.
    """

    def __init__(self) -> None:
        """
        Parameter를 생성한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
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
        int64 Logit을 반환한다.
        """
        return torch.zeros(
            images.shape[0],
            dtype=torch.int64,
            device=images.device,
        )


class NonFiniteOutputModel(nn.Module):
    """
    NaN·inf Logit을 출력하는 잘못된 Model.
    """

    def __init__(
        self,
        invalid_value: float,
    ) -> None:
        """
        출력할 비정상 값을 저장한다.
        """
        super().__init__()

        self.anchor = nn.Parameter(
            torch.tensor(
                0.0,
                dtype=torch.float32,
            )
        )

        self.invalid_value = (
            invalid_value
        )

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        """
        비정상 Logit을 반환한다.
        """
        return torch.full(
            (
                images.shape[0],
            ),
            fill_value=(
                self.invalid_value
            ),
            dtype=torch.float32,
            device=images.device,
        ) + (
            self.anchor
            * 0.0
        )


# =============================================================================
# Test Loss Functions
# =============================================================================


class TargetMeanLoss(nn.Module):
    """
    Target 평균을 Loss로 반환한다.

    Sample 가중 평균 Loss 테스트에 사용한다.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Target 평균 Scalar를 반환한다.
        """
        del logits

        return targets.mean()


class RecordingLoss(nn.Module):
    """
    Logit·Target Dtype과 Gradient 상태를 기록한다.
    """

    def __init__(self) -> None:
        """
        기록 필드를 초기화한다.
        """
        super().__init__()

        self.logit_dtypes: list[
            torch.dtype
        ] = []

        self.target_dtypes: list[
            torch.dtype
        ] = []

        self.gradient_enabled_values: list[
            bool
        ] = []

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        입력 정보를 기록하고 BCE Loss를 계산한다.
        """
        self.logit_dtypes.append(
            logits.dtype
        )

        self.target_dtypes.append(
            targets.dtype
        )

        self.gradient_enabled_values.append(
            torch.is_grad_enabled()
        )

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
        )


class NonTensorLoss(nn.Module):
    """
    Tensor가 아닌 float를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> float:
        """
        잘못된 Python float를 반환한다.
        """
        del logits

        del targets

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
        del targets

        return torch.ones_like(
            logits
        )


class IntegerScalarLoss(nn.Module):
    """
    int64 Scalar Tensor를 반환하는 잘못된 Loss.
    """

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        int64 Scalar를 반환한다.
        """
        del targets

        return torch.tensor(
            1,
            dtype=torch.int64,
            device=logits.device,
        )


class ConstantScalarLoss(nn.Module):
    """
    지정한 float 값을 Scalar Loss로 반환한다.
    """

    def __init__(
        self,
        value: float,
    ) -> None:
        """
        반환할 값을 저장한다.
        """
        super().__init__()

        self.value = value

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        지정한 Scalar Tensor를 반환한다.
        """
        del targets

        return torch.tensor(
            self.value,
            dtype=torch.float32,
            device=logits.device,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def create_image_tensor(
    image_values: list[float],
    *,
    dtype: torch.dtype = (
        torch.float32
    ),
    image_shape: tuple[
        int,
        int,
        int,
    ] = (
        3,
        8,
        8,
    ),
) -> Tensor:
    """
    각 Sample이 하나의 일정한 값을 갖는 이미지 Tensor를 생성한다.
    """
    return torch.stack(
        [
            torch.full(
                image_shape,
                fill_value=(
                    image_value
                ),
                dtype=dtype,
            )
            for image_value in (
                image_values
            )
        ],
        dim=0,
    )


def create_standard_data_loader(
    *,
    image_values: list[
        float
    ] | None = None,
    labels: list[
        int | float
    ] | None = None,
    batch_size: int = 2,
    image_dtype: torch.dtype = (
        torch.float32
    ),
    label_dtype: torch.dtype = (
        torch.int64
    ),
) -> DataLoader:
    """
    정상 Binary Image DataLoader를 생성한다.
    """
    if image_values is None:
        image_values = [
            -1.0,
            1.0,
            -0.5,
            0.5,
        ]

    if labels is None:
        labels = [
            0,
            1,
            0,
            1,
        ]

    images = create_image_tensor(
        image_values=image_values,
        dtype=image_dtype,
    )

    label_tensor = torch.tensor(
        labels,
        dtype=label_dtype,
    )

    dataset = TensorDataset(
        images,
        label_tensor,
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def create_single_batch_loader(
    batch: object,
) -> DataLoader:
    """
    지정한 객체를 그대로 한 번 반환하는 DataLoader를 생성한다.

    잘못된 Batch 구조 테스트에 사용한다.
    """
    dataset = [
        0,
    ]

    return DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=(
            lambda _: batch
        ),
    )


def clone_model_state(
    model: nn.Module,
) -> dict[str, Tensor]:
    """
    Model State를 독립적인 CPU Tensor로 복사한다.
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
    두 Model State의 Key·Tensor를 비교한다.
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


def create_valid_result_values() -> dict[
    str,
    object,
]:
    """
    정상 BinaryEvaluationResult 생성 인자를 반환한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    logits = torch.tensor(
        [
            -2.0,
            2.0,
            -0.2,
            0.2,
        ],
        dtype=torch.float32,
    )

    probabilities = torch.sigmoid(
        logits
    )

    threshold = 0.5

    predictions = (
        probabilities
        >= threshold
    ).to(
        dtype=torch.int64
    )

    accuracy = (
        predictions
        .eq(
            labels
        )
        .to(
            dtype=torch.float64
        )
        .mean()
        .item()
    )

    return {
        "average_loss": 0.4,
        "accuracy": accuracy,
        "sample_count": 4,
        "batch_count": 2,
        "classification_threshold": (
            threshold
        ),
        "labels": labels,
        "logits": logits,
        "probabilities": (
            probabilities
        ),
        "predictions": (
            predictions
        ),
    }


def create_valid_result() -> (
    BinaryEvaluationResult
):
    """
    정상 BinaryEvaluationResult를 생성한다.
    """
    return BinaryEvaluationResult(
        **create_valid_result_values(),  # type: ignore[arg-type]
    )


# =============================================================================
# BinaryEvaluationResult - Normal Behavior
# =============================================================================


def test_binary_evaluation_result_stores_expected_values() -> None:
    """
    정상 Scalar·Tensor 결과를 저장하는지 확인한다.
    """
    result = create_valid_result()

    assert result.average_loss == 0.4

    assert result.accuracy == 0.5

    assert result.sample_count == 4

    assert result.batch_count == 2

    assert (
        result.classification_threshold
        == 0.5
    )

    assert result.labels.shape == (
        4,
    )

    assert result.logits.shape == (
        4,
    )

    assert (
        result.probabilities.shape
        == (
            4,
        )
    )

    assert result.predictions.shape == (
        4,
    )


def test_binary_evaluation_result_is_frozen() -> None:
    """
    생성된 Result의 필드를 재할당할 수 없는지 확인한다.
    """
    result = create_valid_result()

    with pytest.raises(
        FrozenInstanceError,
    ):
        result.accuracy = 1.0  # type: ignore[misc]


def test_result_tensors_use_expected_dtypes() -> None:
    """
    결과 Tensor Dtype을 확인한다.
    """
    result = create_valid_result()

    assert (
        result.labels.dtype
        == torch.int64
    )

    assert (
        result.logits.dtype
        == torch.float32
    )

    assert (
        result.probabilities.dtype
        == torch.float32
    )

    assert (
        result.predictions.dtype
        == torch.int64
    )


def test_result_tensors_are_on_cpu() -> None:
    """
    모든 결과 Tensor가 CPU에 있는지 확인한다.
    """
    result = create_valid_result()

    assert (
        result.labels.device.type
        == "cpu"
    )

    assert (
        result.logits.device.type
        == "cpu"
    )

    assert (
        result.probabilities
        .device
        .type
        == "cpu"
    )

    assert (
        result.predictions
        .device
        .type
        == "cpu"
    )


def test_result_tensors_are_contiguous() -> None:
    """
    결과 Tensor가 Contiguous인지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            9,
            1,
            9,
            0,
            9,
            1,
            9,
        ],
        dtype=torch.int64,
    )[
        ::2
    ]

    logits = torch.tensor(
        [
            -1.0,
            9.0,
            1.0,
            9.0,
            -0.5,
            9.0,
            0.5,
            9.0,
        ],
        dtype=torch.float32,
    )[
        ::2
    ]

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= 0.5
    ).to(
        dtype=torch.int64
    )

    accuracy = (
        predictions
        .eq(
            labels
        )
        .to(
            dtype=torch.float64
        )
        .mean()
        .item()
    )

    result = BinaryEvaluationResult(
        average_loss=0.4,
        accuracy=accuracy,
        sample_count=4,
        batch_count=2,
        classification_threshold=0.5,
        labels=labels,
        logits=logits,
        probabilities=(
            probabilities
        ),
        predictions=(
            predictions
        ),
    )

    assert result.labels.is_contiguous()

    assert result.logits.is_contiguous()

    assert (
        result.probabilities
        .is_contiguous()
    )

    assert (
        result.predictions
        .is_contiguous()
    )


def test_result_tensors_are_independent_clones() -> None:
    """
    원본 Tensor를 변경해도 Result Tensor가 변경되지 않는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    labels = values[
        "labels"
    ]

    logits = values[
        "logits"
    ]

    probabilities = values[
        "probabilities"
    ]

    predictions = values[
        "predictions"
    ]

    assert isinstance(
        labels,
        Tensor,
    )

    assert isinstance(
        logits,
        Tensor,
    )

    assert isinstance(
        probabilities,
        Tensor,
    )

    assert isinstance(
        predictions,
        Tensor,
    )

    expected_labels = (
        labels.clone()
    )

    expected_logits = (
        logits.clone()
    )

    expected_probabilities = (
        probabilities.clone()
    )

    expected_predictions = (
        predictions.clone()
    )

    result = BinaryEvaluationResult(
        **values,  # type: ignore[arg-type]
    )

    labels.fill_(
        1
    )

    logits.fill_(
        0.0
    )

    probabilities.fill_(
        0.5
    )

    predictions.fill_(
        0
    )

    assert torch.equal(
        result.labels,
        expected_labels,
    )

    assert torch.equal(
        result.logits,
        expected_logits,
    )

    assert torch.equal(
        result.probabilities,
        expected_probabilities,
    )

    assert torch.equal(
        result.predictions,
        expected_predictions,
    )


def test_result_detaches_gradient_tensors() -> None:
    """
    Logit·Probability가 Gradient Graph에서 분리되는지 확인한다.
    """
    logits = torch.tensor(
        [
            -1.0,
            1.0,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    probabilities = torch.sigmoid(
        logits
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = (
        probabilities
        >= 0.5
    ).to(
        dtype=torch.int64
    )

    result = BinaryEvaluationResult(
        average_loss=0.3,
        accuracy=1.0,
        sample_count=2,
        batch_count=1,
        classification_threshold=0.5,
        labels=labels,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
    )

    assert (
        result.logits.requires_grad
        is False
    )

    assert (
        result.probabilities
        .requires_grad
        is False
    )


@pytest.mark.parametrize(
    "threshold",
    [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ],
)
def test_result_accepts_valid_thresholds(
    threshold: float,
) -> None:
    """
    0~1 범위 Threshold를 허용하는지 확인한다.
    """
    logits = torch.tensor(
        [
            -2.0,
            0.0,
            2.0,
        ],
        dtype=torch.float32,
    )

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities
        >= threshold
    ).to(
        dtype=torch.int64
    )

    labels = predictions.clone()

    result = BinaryEvaluationResult(
        average_loss=0.2,
        accuracy=1.0,
        sample_count=3,
        batch_count=1,
        classification_threshold=(
            threshold
        ),
        labels=labels,
        logits=logits,
        probabilities=(
            probabilities
        ),
        predictions=(
            predictions
        ),
    )

    assert (
        result.classification_threshold
        == threshold
    )


# =============================================================================
# BinaryEvaluationResult - Invalid Scalar Values
# =============================================================================


@pytest.mark.parametrize(
    "invalid_loss",
    [
        True,
        False,
        None,
        "0.4",
        object(),
    ],
)
def test_result_rejects_invalid_average_loss_type(
    invalid_loss: object,
) -> None:
    """
    Real Number가 아닌 Loss와 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "average_loss"
    ] = invalid_loss

    with pytest.raises(
        TypeError,
        match=(
            "average_loss must be "
            "a real number"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
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
def test_result_rejects_invalid_average_loss_value(
    invalid_loss: float,
) -> None:
    """
    음수·NaN·inf Loss를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "average_loss"
    ] = invalid_loss

    with pytest.raises(
        ValueError,
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_accuracy",
    [
        True,
        False,
        None,
        "0.5",
        object(),
    ],
)
def test_result_rejects_invalid_accuracy_type(
    invalid_accuracy: object,
) -> None:
    """
    Real Number가 아닌 Accuracy와 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "accuracy"
    ] = invalid_accuracy

    with pytest.raises(
        TypeError,
        match=(
            "accuracy must be "
            "a real number"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
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
def test_result_rejects_invalid_accuracy_value(
    invalid_accuracy: float,
) -> None:
    """
    범위 밖·NaN·inf Accuracy를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "accuracy"
    ] = invalid_accuracy

    with pytest.raises(
        ValueError,
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "sample_count",
        "batch_count",
    ],
)
@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        False,
        1.5,
        "2",
        None,
    ],
)
def test_result_rejects_invalid_count_type(
    field_name: str,
    invalid_count: object,
) -> None:
    """
    정수가 아닌 Sample·Batch Count와 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        field_name
    ] = invalid_count

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} "
            "must be an integer"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "sample_count",
        "batch_count",
    ],
)
@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
    ],
)
def test_result_rejects_non_positive_count(
    field_name: str,
    invalid_count: int,
) -> None:
    """
    0 이하 Sample·Batch Count를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        field_name
    ] = invalid_count

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} "
            "must be greater than 0"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_batch_count_larger_than_sample_count() -> None:
    """
    Batch 수가 Sample 수보다 크면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "batch_count"
    ] = 5

    with pytest.raises(
        ValueError,
        match=(
            "batch_count must be "
            "less than or equal to "
            "sample_count"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        False,
        None,
        "0.5",
        object(),
    ],
)
def test_result_rejects_invalid_threshold_type(
    invalid_threshold: object,
) -> None:
    """
    Real Number가 아닌 Threshold와 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "classification_threshold"
    ] = invalid_threshold

    with pytest.raises(
        TypeError,
        match=(
            "classification_threshold "
            "must be a real number"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
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
def test_result_rejects_invalid_threshold_value(
    invalid_threshold: float,
) -> None:
    """
    범위 밖·NaN·inf Threshold를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "classification_threshold"
    ] = invalid_threshold

    with pytest.raises(
        ValueError,
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# BinaryEvaluationResult - Invalid Tensor Structure
# =============================================================================


@pytest.mark.parametrize(
    "field_name",
    [
        "labels",
        "logits",
        "probabilities",
        "predictions",
    ],
)
@pytest.mark.parametrize(
    "invalid_tensor",
    [
        None,
        [],
        "tensor",
        123,
    ],
)
def test_result_rejects_non_tensor_fields(
    field_name: str,
    invalid_tensor: object,
) -> None:
    """
    Tensor가 아닌 결과 필드를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        field_name
    ] = invalid_tensor

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must be "
            "a torch.Tensor"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "labels",
        "logits",
        "probabilities",
        "predictions",
    ],
)
def test_result_rejects_two_dimensional_tensors(
    field_name: str,
) -> None:
    """
    [N, 1] 형태의 2차원 결과 Tensor를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    tensor = values[
        field_name
    ]

    assert isinstance(
        tensor,
        Tensor,
    )

    values[
        field_name
    ] = tensor.unsqueeze(
        dim=1
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must be "
            "one-dimensional"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "dtype",
    ),
    [
        (
            "labels",
            torch.float32,
        ),
        (
            "logits",
            torch.float64,
        ),
        (
            "probabilities",
            torch.float64,
        ),
        (
            "predictions",
            torch.float32,
        ),
    ],
)
def test_result_rejects_unexpected_tensor_dtype(
    field_name: str,
    dtype: torch.dtype,
) -> None:
    """
    결과 Tensor의 고정 Dtype과 다른 Dtype을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    tensor = values[
        field_name
    ]

    assert isinstance(
        tensor,
        Tensor,
    )

    values[
        field_name
    ] = tensor.to(
        dtype=dtype
    )

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must use dtype"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "dtype",
    ),
    [
        (
            "labels",
            torch.int64,
        ),
        (
            "logits",
            torch.float32,
        ),
        (
            "probabilities",
            torch.float32,
        ),
        (
            "predictions",
            torch.int64,
        ),
    ],
)
def test_result_rejects_empty_tensor(
    field_name: str,
    dtype: torch.dtype,
) -> None:
    """
    비어 있는 결과 Tensor를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        field_name
    ] = torch.empty(
        0,
        dtype=dtype,
    )

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must not "
            "be empty"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_different_tensor_lengths() -> None:
    """
    Label·Logit·Probability·Prediction 길이가 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    predictions = values[
        "predictions"
    ]

    assert isinstance(
        predictions,
        Tensor,
    )

    values[
        "predictions"
    ] = predictions[
        :-1
    ]

    with pytest.raises(
        ValueError,
        match=(
            "must contain the same "
            "number of elements"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_tensor_length_and_sample_count_mismatch() -> None:
    """
    Tensor 길이와 sample_count가 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "sample_count"
    ] = 5

    with pytest.raises(
        ValueError,
        match=(
            "result Tensor length "
            "must match sample_count"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_label",
    [
        -1,
        2,
        3,
    ],
)
def test_result_rejects_non_binary_label(
    invalid_label: int,
) -> None:
    """
    0·1 이외 Ground Truth Label을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    labels = values[
        "labels"
    ]

    assert isinstance(
        labels,
        Tensor,
    )

    invalid_labels = (
        labels.clone()
    )

    invalid_labels[
        0
    ] = invalid_label

    values[
        "labels"
    ] = invalid_labels

    with pytest.raises(
        ValueError,
        match=(
            "labels must contain only "
            "binary values"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_prediction",
    [
        -1,
        2,
        3,
    ],
)
def test_result_rejects_non_binary_prediction(
    invalid_prediction: int,
) -> None:
    """
    0·1 이외 Prediction을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    predictions = values[
        "predictions"
    ]

    assert isinstance(
        predictions,
        Tensor,
    )

    invalid_predictions = (
        predictions.clone()
    )

    invalid_predictions[
        0
    ] = invalid_prediction

    values[
        "predictions"
    ] = invalid_predictions

    with pytest.raises(
        ValueError,
        match=(
            "predictions must contain "
            "only binary values"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_result_rejects_non_finite_logit(
    invalid_value: float,
) -> None:
    """
    NaN·inf Logit을 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    logits = values[
        "logits"
    ]

    assert isinstance(
        logits,
        Tensor,
    )

    invalid_logits = (
        logits.clone()
    )

    invalid_logits[
        0
    ] = invalid_value

    values[
        "logits"
    ] = invalid_logits

    with pytest.raises(
        ValueError,
        match=(
            "logits must contain "
            "only finite values"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_result_rejects_non_finite_probability(
    invalid_value: float,
) -> None:
    """
    NaN·inf Probability를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    probabilities = values[
        "probabilities"
    ]

    assert isinstance(
        probabilities,
        Tensor,
    )

    invalid_probabilities = (
        probabilities.clone()
    )

    invalid_probabilities[
        0
    ] = invalid_value

    values[
        "probabilities"
    ] = (
        invalid_probabilities
    )

    with pytest.raises(
        ValueError,
        match=(
            "probabilities must contain "
            "only finite values"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_probability",
    [
        -0.1,
        1.1,
    ],
)
def test_result_rejects_probability_outside_zero_and_one(
    invalid_probability: float,
) -> None:
    """
    0~1 범위 밖 Probability를 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    probabilities = values[
        "probabilities"
    ]

    assert isinstance(
        probabilities,
        Tensor,
    )

    invalid_probabilities = (
        probabilities.clone()
    )

    invalid_probabilities[
        0
    ] = invalid_probability

    values[
        "probabilities"
    ] = (
        invalid_probabilities
    )

    with pytest.raises(
        ValueError,
        match=(
            "probabilities must contain "
            "values between 0 and 1"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_probability_not_matching_sigmoid_logit() -> None:
    """
    Probability가 sigmoid(Logit)과 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    probabilities = values[
        "probabilities"
    ]

    assert isinstance(
        probabilities,
        Tensor,
    )

    invalid_probabilities = (
        probabilities.clone()
    )

    invalid_probabilities[
        0
    ] = 0.5

    values[
        "probabilities"
    ] = (
        invalid_probabilities
    )

    with pytest.raises(
        ValueError,
        match=(
            "probabilities must match "
            "sigmoid"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_prediction_not_matching_probability() -> None:
    """
    Prediction이 Probability·Threshold와 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    predictions = values[
        "predictions"
    ]

    assert isinstance(
        predictions,
        Tensor,
    )

    invalid_predictions = (
        predictions.clone()
    )

    invalid_predictions[
        0
    ] = (
        1
        - int(
            invalid_predictions[
                0
            ].item()
        )
    )

    values[
        "predictions"
    ] = (
        invalid_predictions
    )

    with pytest.raises(
        ValueError,
        match=(
            "predictions must match "
            "probabilities"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


def test_result_rejects_accuracy_not_matching_predictions() -> None:
    """
    Accuracy가 Label·Prediction 결과와 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_result_values()
    )

    values[
        "accuracy"
    ] = 1.0

    with pytest.raises(
        ValueError,
        match=(
            "accuracy must match "
            "labels and predictions"
        ),
    ):
        BinaryEvaluationResult(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# Evaluation - Normal Behavior
# =============================================================================


def test_evaluate_returns_binary_evaluation_result() -> None:
    """
    정상 평가 후 BinaryEvaluationResult를 반환하는지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert isinstance(
        result,
        BinaryEvaluationResult,
    )


def test_evaluate_sets_model_to_evaluation_mode() -> None:
    """
    평가 시작 시 Model을 Evaluation Mode로 변경하는지 확인한다.
    """
    model = (
        ImageValueLogitModel()
    )

    model.train()

    assert model.training is True

    _ = evaluate_binary_classifier(
        model=model,
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert model.training is False


def test_evaluate_disables_gradient_during_forward() -> None:
    """
    모든 Forward가 Gradient 비활성화 상태인지 확인한다.
    """
    model = (
        RecordingEvaluationModel()
    )

    _ = evaluate_binary_classifier(
        model=model,
        data_loader=(
            create_standard_data_loader(
                batch_size=2,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert (
        model.forward_call_count
        == 2
    )

    assert (
        model.gradient_enabled_values
        == [
            False,
            False,
        ]
    )


def test_evaluate_does_not_change_model_state() -> None:
    """
    평가 전후 Model Parameter 값이 같은지 확인한다.
    """
    model = (
        ImageValueLogitModel(
            scale=2.0,
            bias=0.3,
        )
    )

    state_before = (
        clone_model_state(
            model=model,
        )
    )

    _ = evaluate_binary_classifier(
        model=model,
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    state_after = (
        clone_model_state(
            model=model,
        )
    )

    assert state_dicts_are_equal(
        first_state=state_before,
        second_state=state_after,
    )


def test_evaluate_does_not_create_parameter_gradients() -> None:
    """
    평가 후 Parameter Gradient가 생성되지 않는지 확인한다.
    """
    model = (
        ImageValueLogitModel()
    )

    assert all(
        parameter.grad is None
        for parameter in (
            model.parameters()
        )
    )

    _ = evaluate_binary_classifier(
        model=model,
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert all(
        parameter.grad is None
        for parameter in (
            model.parameters()
        )
    )


def test_evaluate_processes_every_batch() -> None:
    """
    DataLoader의 모든 Batch를 처리하는지 확인한다.
    """
    model = (
        RecordingEvaluationModel()
    )

    data_loader = (
        create_standard_data_loader(
            image_values=[
                -1.0,
                1.0,
                -0.5,
                0.5,
                -0.2,
            ],
            labels=[
                0,
                1,
                0,
                1,
                0,
            ],
            batch_size=2,
        )
    )

    result = (
        evaluate_binary_classifier(
            model=model,
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )
    )

    assert (
        model.forward_call_count
        == 3
    )

    assert result.batch_count == 3

    assert result.sample_count == 5


def test_evaluate_returns_exact_sample_and_batch_counts() -> None:
    """
    실제 Sample·Batch 수를 정확히 반환하는지 확인한다.
    """
    data_loader = (
        create_standard_data_loader(
            image_values=[
                -1.0,
                1.0,
                -0.5,
                0.5,
                -0.2,
                0.2,
            ],
            labels=[
                0,
                1,
                0,
                1,
                0,
                1,
            ],
            batch_size=4,
        )
    )

    result = (
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )
    )

    assert result.sample_count == 6

    assert result.batch_count == 2


def test_evaluate_returns_expected_tensor_shapes() -> None:
    """
    전체 Sample 수와 같은 1차원 Tensor를 반환하는지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert result.labels.shape == (
        4,
    )

    assert result.logits.shape == (
        4,
    )

    assert (
        result.probabilities.shape
        == (
            4,
        )
    )

    assert result.predictions.shape == (
        4,
    )


def test_evaluate_returns_expected_tensor_dtypes() -> None:
    """
    평가 결과 Tensor Dtype을 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert (
        result.labels.dtype
        == torch.int64
    )

    assert (
        result.logits.dtype
        == torch.float32
    )

    assert (
        result.probabilities.dtype
        == torch.float32
    )

    assert (
        result.predictions.dtype
        == torch.int64
    )


def test_evaluate_returns_cpu_tensors() -> None:
    """
    평가 결과 Tensor가 모두 CPU인지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert (
        result.labels.device.type
        == "cpu"
    )

    assert (
        result.logits.device.type
        == "cpu"
    )

    assert (
        result.probabilities
        .device
        .type
        == "cpu"
    )

    assert (
        result.predictions
        .device
        .type
        == "cpu"
    )


def test_evaluate_preserves_dataset_order() -> None:
    """
    shuffle=False DataLoader 순서대로 Label·Logit을 수집하는지 확인한다.
    """
    image_values = [
        -2.0,
        1.5,
        -0.3,
        0.7,
    ]

    labels = [
        0,
        1,
        0,
        1,
    ]

    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=(
                    image_values
                ),
                labels=labels,
                batch_size=3,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert torch.equal(
        result.labels,
        torch.tensor(
            labels,
            dtype=torch.int64,
        ),
    )

    assert torch.allclose(
        result.logits,
        torch.tensor(
            image_values,
            dtype=torch.float32,
        ),
    )


def test_evaluate_known_predictions_and_accuracy() -> None:
    """
    알려진 Logit·Label에서 Prediction·Accuracy를 확인한다.

    Prediction:

        [0, 1, 0, 1]

    Label:

        [0, 1, 1, 0]

    Accuracy:

        2 / 4

        0.5
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=[
                    -1.0,
                    1.0,
                    -0.2,
                    0.2,
                ],
                labels=[
                    0,
                    1,
                    1,
                    0,
                ],
                batch_size=2,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.5,
    )

    assert torch.equal(
        result.predictions,
        torch.tensor(
            [
                0,
                1,
                0,
                1,
            ],
            dtype=torch.int64,
        ),
    )

    assert result.accuracy == 0.5


def test_evaluate_probabilities_match_sigmoid_logits() -> None:
    """
    Probability가 sigmoid(Logit)과 일치하는지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert torch.allclose(
        result.probabilities,
        torch.sigmoid(
            result.logits
        ),
    )


def test_evaluate_predictions_match_threshold() -> None:
    """
    Prediction이 Probability·Threshold와 일치하는지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.7,
    )

    expected_predictions = (
        result.probabilities
        >= 0.7
    ).to(
        dtype=torch.int64
    )

    assert torch.equal(
        result.predictions,
        expected_predictions,
    )


def test_evaluate_uses_greater_than_or_equal_at_threshold() -> None:
    """
    Probability가 Threshold와 같으면 Positive Class로 분류하는지 확인한다.

    Logit:

        0

    Sigmoid:

        0.5

    Threshold:

        0.5

    Prediction:

        1
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=[
                    0.0,
                ],
                labels=[
                    1,
                ],
                batch_size=1,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.5,
    )

    assert (
        result.probabilities[
            0
        ].item()
        == 0.5
    )

    assert (
        result.predictions[
            0
        ].item()
        == 1
    )

    assert result.accuracy == 1.0


def test_evaluate_supports_zero_threshold() -> None:
    """
    Threshold 0에서는 모든 유한 Probability가 Positive인지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=[
                    -5.0,
                    0.0,
                    5.0,
                ],
                labels=[
                    1,
                    1,
                    1,
                ],
                batch_size=2,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=0.0,
    )

    assert torch.equal(
        result.predictions,
        torch.ones(
            3,
            dtype=torch.int64,
        ),
    )


def test_evaluate_supports_one_threshold() -> None:
    """
    Threshold 1에서는 유한 Logit의 Sigmoid가 1 미만이므로 모두 Negative인지 확인한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=[
                    -5.0,
                    0.0,
                    5.0,
                ],
                labels=[
                    0,
                    0,
                    0,
                ],
                batch_size=2,
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
        classification_threshold=1.0,
    )

    assert torch.equal(
        result.predictions,
        torch.zeros(
            3,
            dtype=torch.int64,
        ),
    )


def test_evaluate_uses_sample_weighted_average_loss() -> None:
    """
    Batch 평균의 단순 평균이 아니라 Sample 가중 평균을 사용하는지 확인한다.

    Batch 1:

        4 Samples

        Target Mean:

            0.0

    Batch 2:

        2 Samples

        Target Mean:

            1.0

    Sample 가중 평균:

        (
            0.0 * 4
            +
            1.0 * 2
        )

        /

        6

        =

        1 / 3

    Batch 평균의 단순 평균:

        (
            0.0
            +
            1.0
        )

        /

        2

        =

        0.5

    두 값은 달라야 한다.
    """
    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                image_values=[
                    -1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                ],
                labels=[
                    0,
                    0,
                    0,
                    0,
                    1,
                    1,
                ],
                batch_size=4,
            )
        ),
        loss_function=(
            TargetMeanLoss()
        ),
        device="cpu",
    )

    expected_weighted_loss = (
        2.0
        / 6.0
    )

    assert result.average_loss == (
        pytest.approx(
            expected_weighted_loss,
        )
    )

    assert (
        result.average_loss
        != pytest.approx(
            0.5,
        )
    )


def test_evaluate_converts_float64_images_to_float32() -> None:
    """
    float64 Image를 Model 입력 직전에 float32로 변환하는지 확인한다.
    """
    model = (
        RecordingEvaluationModel()
    )

    _ = evaluate_binary_classifier(
        model=model,
        data_loader=(
            create_standard_data_loader(
                image_dtype=(
                    torch.float64
                ),
            )
        ),
        loss_function=(
            create_binary_classification_loss()
        ),
        device="cpu",
    )

    assert (
        model.input_dtypes
        == [
            torch.float32,
            torch.float32,
        ]
    )


def test_evaluate_converts_float_labels_to_float32_targets() -> None:
    """
    0.0·1.0 Label을 Loss 입력용 float32 Target으로 변환하는지 확인한다.
    """
    loss_function = (
        RecordingLoss()
    )

    result = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader(
                labels=[
                    0.0,
                    1.0,
                    0.0,
                    1.0,
                ],
                label_dtype=(
                    torch.float64
                ),
            )
        ),
        loss_function=(
            loss_function
        ),
        device="cpu",
    )

    assert (
        loss_function.target_dtypes
        == [
            torch.float32,
            torch.float32,
        ]
    )

    assert (
        result.labels.dtype
        == torch.int64
    )


def test_evaluate_loss_runs_without_gradient() -> None:
    """
    Loss 계산도 Gradient 비활성화 상태인지 확인한다.
    """
    loss_function = (
        RecordingLoss()
    )

    _ = evaluate_binary_classifier(
        model=(
            ImageValueLogitModel()
        ),
        data_loader=(
            create_standard_data_loader()
        ),
        loss_function=(
            loss_function
        ),
        device="cpu",
    )

    assert (
        loss_function
        .gradient_enabled_values
        == [
            False,
            False,
        ]
    )


# =============================================================================
# Evaluation - Invalid Objects
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
def test_evaluate_rejects_non_module_model(
    invalid_model: object,
) -> None:
    """
    nn.Module이 아닌 Model을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "model must be an instance "
            "of torch.nn.Module"
        ),
    ):
        evaluate_binary_classifier(
            model=invalid_model,  # type: ignore[arg-type]
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
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
def test_evaluate_rejects_non_dataloader(
    invalid_loader: object,
) -> None:
    """
    DataLoader가 아닌 객체를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "data_loader must be "
            "an instance"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=invalid_loader,  # type: ignore[arg-type]
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
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
def test_evaluate_rejects_non_module_loss(
    invalid_loss: object,
) -> None:
    """
    nn.Module이 아닌 Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "loss_function must be "
            "an instance"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=invalid_loss,  # type: ignore[arg-type]
            device="cpu",
        )


def test_evaluate_rejects_model_without_state() -> None:
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
        evaluate_binary_classifier(
            model=(
                ParameterlessModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Evaluation - Invalid Device
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
def test_evaluate_rejects_invalid_device_type(
    invalid_device: object,
) -> None:
    """
    문자열·torch.device가 아닌 Device를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "device must be a string "
            "or torch.device"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device=invalid_device,  # type: ignore[arg-type]
        )


def test_evaluate_rejects_invalid_device_string() -> None:
    """
    PyTorch가 해석할 수 없는 Device 문자열을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match="Invalid device",
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="not-a-device",
        )


def test_evaluate_rejects_unavailable_cuda() -> None:
    """
    CUDA가 없는 환경에서 CUDA 요청을 거부하는지 확인한다.
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
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cuda",
        )


def test_evaluate_rejects_model_parameter_device_mismatch() -> None:
    """
    Model Parameter Device와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = (
        ImageValueLogitModel()
    )

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
        evaluate_binary_classifier(
            model=model,
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_model_buffer_device_mismatch() -> None:
    """
    Model Buffer Device와 요청 Device가 다르면 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "all Model Buffers "
            "must be on the requested device"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                BufferDeviceMismatchModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Evaluation - Invalid Threshold
# =============================================================================


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        True,
        False,
        None,
        "0.5",
        object(),
    ],
)
def test_evaluate_rejects_invalid_threshold_type(
    invalid_threshold: object,
) -> None:
    """
    Real Number가 아닌 Threshold와 bool을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "classification_threshold "
            "must be a real number"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
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
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evaluate_rejects_invalid_threshold_value(
    invalid_threshold: float,
) -> None:
    """
    범위 밖·NaN·inf Threshold를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
            classification_threshold=(
                invalid_threshold
            ),
        )


# =============================================================================
# Evaluation - Invalid Batch
# =============================================================================


def test_evaluate_rejects_non_tuple_or_list_batch() -> None:
    """
    Tuple·List가 아닌 Batch를 거부하는지 확인한다.
    """
    data_loader = (
        create_single_batch_loader(
            batch=torch.tensor(
                [
                    1,
                ]
            ),
        )
    )

    with pytest.raises(
        TypeError,
        match=(
            "each data_loader batch "
            "must be a tuple or list"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "batch",
    [
        (),
        (
            torch.zeros(
                1,
                3,
                8,
                8,
            ),
        ),
        (
            torch.zeros(
                1,
                3,
                8,
                8,
            ),
            torch.zeros(
                1,
                dtype=torch.int64,
            ),
            "extra",
        ),
    ],
)
def test_evaluate_rejects_batch_with_wrong_item_count(
    batch: tuple[Any, ...],
) -> None:
    """
    Item 수가 2가 아닌 Batch를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly "
            "two items"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_non_tensor_images() -> None:
    """
    Tensor가 아닌 Image를 거부하는지 확인한다.
    """
    batch = (
        "images",
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "batch images must be "
            "a torch.Tensor"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_non_tensor_labels() -> None:
    """
    Tensor가 아닌 Label을 거부하는지 확인한다.
    """
    batch = (
        torch.zeros(
            1,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        [
            0,
        ],
    )

    with pytest.raises(
        TypeError,
        match=(
            "batch labels must be "
            "a torch.Tensor"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_images",
    [
        torch.zeros(
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        torch.zeros(
            1,
            3,
            8,
            8,
            1,
            dtype=torch.float32,
        ),
    ],
)
def test_evaluate_rejects_non_four_dimensional_images(
    invalid_images: Tensor,
) -> None:
    """
    4차원이 아닌 Image Batch를 거부하는지 확인한다.
    """
    batch = (
        invalid_images,
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "batch images must be "
            "four-dimensional"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_empty_image_batch() -> None:
    """
    Batch Size가 0인 Image를 거부하는지 확인한다.
    """
    batch = (
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

    with pytest.raises(
        ValueError,
        match=(
            "batch images must contain "
            "at least one sample"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "channel_count",
    [
        1,
        2,
        4,
    ],
)
def test_evaluate_rejects_non_rgb_images(
    channel_count: int,
) -> None:
    """
    Channel 수가 3이 아닌 Image를 거부하는지 확인한다.
    """
    batch = (
        torch.zeros(
            1,
            channel_count,
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

    with pytest.raises(
        ValueError,
        match=(
            "must contain exactly "
            "three RGB channels"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "image_shape",
    [
        (
            1,
            3,
            0,
            8,
        ),
        (
            1,
            3,
            8,
            0,
        ),
    ],
)
def test_evaluate_rejects_non_positive_image_size(
    image_shape: tuple[
        int,
        int,
        int,
        int,
    ],
) -> None:
    """
    Height·Width가 0인 Image를 거부하는지 확인한다.
    """
    batch = (
        torch.empty(
            image_shape,
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "height and width "
            "must be greater than 0"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "integer_dtype",
    [
        torch.int32,
        torch.int64,
        torch.uint8,
    ],
)
def test_evaluate_rejects_integer_images(
    integer_dtype: torch.dtype,
) -> None:
    """
    Floating Point가 아닌 Image Dtype을 거부하는지 확인한다.
    """
    batch = (
        torch.zeros(
            1,
            3,
            8,
            8,
            dtype=integer_dtype,
        ),
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        TypeError,
        match=(
            "batch images must use "
            "a floating-point dtype"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
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
def test_evaluate_rejects_non_finite_images(
    invalid_value: float,
) -> None:
    """
    NaN·inf Image 값을 거부하는지 확인한다.
    """
    images = torch.zeros(
        1,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    images[
        0,
        0,
        0,
        0,
    ] = invalid_value

    batch = (
        images,
        torch.tensor(
            [
                0,
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "batch images must contain "
            "only finite values"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_non_one_dimensional_labels() -> None:
    """
    [B, 1] Label을 거부하는지 확인한다.
    """
    batch = (
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

    with pytest.raises(
        ValueError,
        match=(
            "batch labels must be "
            "one-dimensional"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_empty_labels() -> None:
    """
    비어 있는 Label Tensor를 거부하는지 확인한다.
    """
    batch = (
        torch.zeros(
            1,
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

    with pytest.raises(
        ValueError,
        match=(
            "batch labels must contain "
            "at least one sample"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_image_label_count_mismatch() -> None:
    """
    Image 수와 Label 수가 다르면 거부하는지 확인한다.
    """
    batch = (
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

    with pytest.raises(
        ValueError,
        match=(
            "images and labels must "
            "contain the same number"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_labels",
    [
        torch.tensor(
            [
                0,
                2,
            ],
            dtype=torch.int64,
        ),
        torch.tensor(
            [
                -1,
                1,
            ],
            dtype=torch.int64,
        ),
        torch.tensor(
            [
                0.0,
                0.5,
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                True,
                False,
            ],
            dtype=torch.bool,
        ),
    ],
)
def test_evaluate_rejects_invalid_binary_labels(
    invalid_labels: Tensor,
) -> None:
    """
    Binary 0·1 규칙을 위반한 Label을 거부하는지 확인한다.
    """
    batch = (
        torch.zeros(
            2,
            3,
            8,
            8,
            dtype=torch.float32,
        ),
        invalid_labels,
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_single_batch_loader(
                    batch=batch,
                )
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Evaluation - Invalid Model Output
# =============================================================================


def test_evaluate_rejects_non_tensor_logits() -> None:
    """
    Tensor가 아닌 Model Output을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "model output logits must "
            "be a torch.Tensor"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                NonTensorOutputModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "model",
    [
        TwoDimensionalOutputModel(),
        ScalarOutputModel(),
    ],
)
def test_evaluate_rejects_non_one_dimensional_logits(
    model: nn.Module,
) -> None:
    """
    [B]가 아닌 Logit Shape를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "model output logits must "
            "be one-dimensional"
        ),
    ):
        evaluate_binary_classifier(
            model=model,
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_logit_batch_size_mismatch() -> None:
    """
    Logit 수와 Image Batch Size가 다르면 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "model output batch size "
            "must match"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                WrongBatchSizeOutputModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_integer_logits() -> None:
    """
    Floating Point가 아닌 Logit을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "model output logits must "
            "use a floating-point dtype"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                IntegerOutputModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
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
def test_evaluate_rejects_non_finite_logits(
    invalid_value: float,
) -> None:
    """
    NaN·inf Logit을 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "model output logits must "
            "contain only finite values"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                NonFiniteOutputModel(
                    invalid_value=(
                        invalid_value
                    )
                )
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )


# =============================================================================
# Evaluation - Invalid Loss Output
# =============================================================================


def test_evaluate_rejects_non_tensor_loss() -> None:
    """
    Tensor가 아닌 Loss Output을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "loss_function output must "
            "be a torch.Tensor"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                NonTensorLoss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_non_scalar_loss() -> None:
    """
    [B] Loss Vector를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "loss_function output must "
            "be a scalar Tensor"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                VectorLoss()
            ),
            device="cpu",
        )


def test_evaluate_rejects_integer_loss() -> None:
    """
    Floating Point가 아닌 Scalar Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "loss_function output must "
            "use a floating-point dtype"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                IntegerScalarLoss()
            ),
            device="cpu",
        )


@pytest.mark.parametrize(
    "invalid_loss",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_evaluate_rejects_non_finite_loss(
    invalid_loss: float,
) -> None:
    """
    NaN·inf Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "loss_function output "
            "must be finite"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                ConstantScalarLoss(
                    value=invalid_loss,
                )
            ),
            device="cpu",
        )


def test_evaluate_rejects_negative_loss() -> None:
    """
    음수 Loss를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "loss_function output must "
            "be greater than or equal to 0"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=(
                create_standard_data_loader()
            ),
            loss_function=(
                ConstantScalarLoss(
                    value=-0.1,
                )
            ),
            device="cpu",
        )


# =============================================================================
# Evaluation - Empty DataLoader
# =============================================================================


def test_evaluate_rejects_empty_data_loader() -> None:
    """
    Sample이 하나도 없는 DataLoader를 거부하는지 확인한다.
    """
    images = torch.empty(
        0,
        3,
        8,
        8,
        dtype=torch.float32,
    )

    labels = torch.empty(
        0,
        dtype=torch.int64,
    )

    data_loader = DataLoader(
        dataset=TensorDataset(
            images,
            labels,
        ),
        batch_size=2,
        shuffle=False,
    )

    with pytest.raises(
        ValueError,
        match=(
            "data_loader must contain "
            "at least one sample"
        ),
    ):
        evaluate_binary_classifier(
            model=(
                ImageValueLogitModel()
            ),
            data_loader=data_loader,
            loss_function=(
                create_binary_classification_loss()
            ),
            device="cpu",
        )