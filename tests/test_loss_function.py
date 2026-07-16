"""
Binary classification loss utility unit tests.

테스트 대상
----------
src/training/loss_function.py

테스트 목적
----------
CNNBaseline의 Binary Raw Logit과 Dataset의 Binary Label을 사용하여
Loss를 안전하고 일관된 방식으로 계산할 수 있는지 검증한다.

현재 클래스 정의
---------------
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

현재 Dataset Label
------------------
CastingDefectDataset:

    Integer Class Label

    Shape:

        [batch_size]

    dtype:

        torch.int64

Loss 입력 Target
----------------
prepare_binary_targets():

    Floating Point Binary Target

    Shape:

        [batch_size]

    dtype:

        torch.float32

Loss Function
-------------
BCEWithLogitsLoss

현재 설정:

    reduction="mean"

    pos_weight=None

주의
----
이번 테스트에서는 Optimizer, Train Epoch, Validation Epoch,
전체 Training Pipeline을 검증하지 않는다.
"""

import math

import pytest
import torch
from torch import Tensor, nn

from src.training.loss_function import (
    DEFAULT_LOSS_REDUCTION,
    USE_POSITIVE_CLASS_WEIGHT,
    create_binary_classification_loss,
    prepare_binary_targets,
)


# =============================================================================
# Loss Configuration
# =============================================================================


def test_default_loss_reduction_is_mean() -> None:
    """
    기본 Loss Reduction 설정이 mean인지 확인한다.

    mean은 Batch 내 Sample Loss의 평균값 하나를 반환한다.
    """
    assert DEFAULT_LOSS_REDUCTION == "mean"


def test_positive_class_weight_is_disabled_for_first_baseline() -> None:
    """
    첫 CNN Baseline에서 Positive Class Weight를 사용하지 않는지 확인한다.

    현재 Train Split:

        NORMAL:
            2,300

        DEFECT:
            3,006

    현재 Class 불균형은 극심하지 않으므로
    첫 Baseline에서는 Unweighted Loss를 사용한다.
    """
    assert USE_POSITIVE_CLASS_WEIGHT is False


# =============================================================================
# Loss Function Factory
# =============================================================================


def test_create_binary_classification_loss_returns_expected_type() -> None:
    """
    Loss Factory가 BCEWithLogitsLoss 객체를 반환하는지 확인한다.
    """
    loss_function = create_binary_classification_loss()

    assert isinstance(
        loss_function,
        nn.BCEWithLogitsLoss,
    )


def test_created_loss_uses_mean_reduction() -> None:
    """
    생성된 Loss가 Batch 평균을 반환하도록 설정됐는지 확인한다.
    """
    loss_function = create_binary_classification_loss()

    assert loss_function.reduction == "mean"


def test_created_loss_does_not_use_positive_class_weight() -> None:
    """
    생성된 Loss에 pos_weight가 적용되지 않았는지 확인한다.
    """
    loss_function = create_binary_classification_loss()

    assert loss_function.pos_weight is None


def test_loss_factory_returns_independent_instances() -> None:
    """
    Loss Factory를 여러 번 호출하면 서로 다른 객체가 생성되는지 확인한다.

    각 Training Pipeline이 자신의 Loss 객체를 독립적으로 사용할 수 있다.
    """
    first_loss_function = (
        create_binary_classification_loss()
    )

    second_loss_function = (
        create_binary_classification_loss()
    )

    assert first_loss_function is not second_loss_function


# =============================================================================
# Binary Target Preparation
# =============================================================================


def test_prepare_binary_targets_converts_int64_to_float32() -> None:
    """
    Dataset의 int64 Label을 BCE Loss용 float32 Target으로 변환하는지 확인한다.
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

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert targets.dtype == torch.float32


def test_prepare_binary_targets_preserves_shape() -> None:
    """
    Target 변환 전후의 Batch Shape가 같은지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert labels.shape == (
        5,
    )

    assert targets.shape == (
        5,
    )


def test_prepare_binary_targets_preserves_binary_values() -> None:
    """
    dtype을 변환해도 NORMAL·DEFECT Label 의미가 유지되는지 확인한다.

    변환:

        0

        ->

        0.0

        1

        ->

        1.0
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

    targets = prepare_binary_targets(
        labels=labels,
    )

    expected_targets = torch.tensor(
        [
            0.0,
            1.0,
            1.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    assert torch.equal(
        targets,
        expected_targets,
    )


def test_prepare_binary_targets_supports_single_label_batch() -> None:
    """
    Batch Size가 1인 Label도 정상 처리하는지 확인한다.

    향후 FastAPI 단일 이미지 추론 및 단일 Sample 검증에서 중요하다.
    """
    labels = torch.tensor(
        [
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert targets.shape == (
        1,
    )

    assert targets.dtype == torch.float32

    assert targets.item() == 1.0


def test_prepare_binary_targets_supports_explicit_cpu_string() -> None:
    """
    문자열 'cpu'를 Device 인자로 전달할 수 있는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
        device="cpu",
    )

    assert targets.device.type == "cpu"


def test_prepare_binary_targets_supports_torch_device() -> None:
    """
    torch.device 객체를 Device 인자로 전달할 수 있는지 확인한다.
    """
    device = torch.device(
        "cpu",
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
        device=device,
    )

    assert targets.device == device


def test_prepare_binary_targets_keeps_existing_device_when_device_is_none() -> None:
    """
    device=None이면 원래 Label Tensor의 Device를 유지하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
        device="cpu",
    )

    targets = prepare_binary_targets(
        labels=labels,
        device=None,
    )

    assert targets.device == labels.device


@pytest.mark.parametrize(
    "integer_dtype",
    [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_prepare_binary_targets_supports_integer_dtypes(
    integer_dtype: torch.dtype,
) -> None:
    """
    지원하는 여러 정수 dtype을 float32 Binary Target으로 변환하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=integer_dtype,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert targets.dtype == torch.float32

    assert torch.equal(
        targets,
        torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize(
    "floating_dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ],
)
def test_prepare_binary_targets_supports_valid_floating_dtypes(
    floating_dtype: torch.dtype,
) -> None:
    """
    이미 Floating Point인 올바른 Binary Label도 float32로 정규화하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0.0,
            1.0,
        ],
        dtype=floating_dtype,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert targets.dtype == torch.float32

    assert torch.equal(
        targets,
        torch.tensor(
            [
                0.0,
                1.0,
            ],
            dtype=torch.float32,
        ),
    )


def test_prepare_binary_targets_does_not_modify_original_integer_labels() -> None:
    """
    Target 변환 과정이 원본 Dataset Label Tensor를 변경하지 않는지 확인한다.
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

    original_labels = labels.clone()

    _ = prepare_binary_targets(
        labels=labels,
    )

    assert torch.equal(
        labels,
        original_labels,
    )

    assert labels.dtype == torch.int64


def test_prepare_binary_targets_returns_finite_values() -> None:
    """
    정상 Binary Label을 변환한 결과가 모두 유한한 값인지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    assert torch.isfinite(
        targets,
    ).all()


# =============================================================================
# Invalid Binary Target Input
# =============================================================================


@pytest.mark.parametrize(
    "invalid_labels",
    [
        [
            0,
            1,
        ],
        (
            0,
            1,
        ),
        None,
        "0,1",
    ],
)
def test_prepare_binary_targets_rejects_non_tensor_input(
    invalid_labels: object,
) -> None:
    """
    Tensor가 아닌 Label 입력을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="labels must be a torch.Tensor",
    ):
        prepare_binary_targets(
            labels=invalid_labels,  # type: ignore[arg-type]
        )


def test_prepare_binary_targets_rejects_scalar_tensor() -> None:
    """
    Batch 차원이 없는 Scalar Label Tensor를 거부하는지 확인한다.
    """
    scalar_label = torch.tensor(
        1,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="labels must have 1 dimension",
    ):
        prepare_binary_targets(
            labels=scalar_label,
        )


def test_prepare_binary_targets_rejects_two_dimensional_tensor() -> None:
    """
    [B, 1] 형태의 2차원 Label Tensor를 거부하는지 확인한다.

    현재 공식 Label Shape는 [B]다.
    """
    labels = torch.tensor(
        [
            [
                0,
            ],
            [
                1,
            ],
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="labels must have 1 dimension",
    ):
        prepare_binary_targets(
            labels=labels,
        )


def test_prepare_binary_targets_rejects_empty_label_batch() -> None:
    """
    Label이 하나도 없는 빈 Batch를 거부하는지 확인한다.
    """
    empty_labels = torch.empty(
        0,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="at least one binary class label",
    ):
        prepare_binary_targets(
            labels=empty_labels,
        )


def test_prepare_binary_targets_rejects_boolean_dtype() -> None:
    """
    bool Label을 거부하는지 확인한다.

    bool은 내부적으로 0·1처럼 동작할 수 있지만,
    현재 Dataset Label은 명시적인 Class Index여야 한다.
    """
    boolean_labels = torch.tensor(
        [
            False,
            True,
        ],
        dtype=torch.bool,
    )

    with pytest.raises(
        TypeError,
        match="must not use torch.bool",
    ):
        prepare_binary_targets(
            labels=boolean_labels,
        )


def test_prepare_binary_targets_rejects_complex_dtype() -> None:
    """
    Complex Number dtype을 Binary Label로 사용하지 못하도록 거부하는지 확인한다.
    """
    complex_labels = torch.tensor(
        [
            0.0 + 0.0j,
            1.0 + 0.0j,
        ],
        dtype=torch.complex64,
    )

    with pytest.raises(
        TypeError,
        match="integer or floating-point dtype",
    ):
        prepare_binary_targets(
            labels=complex_labels,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -1,
        2,
        3,
        100,
    ],
)
def test_prepare_binary_targets_rejects_invalid_integer_binary_value(
    invalid_value: int,
) -> None:
    """
    0·1 이외의 정수 Label을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            invalid_value,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match="only binary values 0 and 1",
    ):
        prepare_binary_targets(
            labels=labels,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        0.1,
        0.5,
        0.9,
        1.1,
    ],
)
def test_prepare_binary_targets_rejects_non_binary_floating_value(
    invalid_value: float,
) -> None:
    """
    0.0·1.0 이외의 실수 Label을 거부하는지 확인한다.

    확률값은 Label이 아니다.

    Target은 실제 Class를 의미하는 0 또는 1이어야 한다.
    """
    labels = torch.tensor(
        [
            0.0,
            invalid_value,
            1.0,
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="only binary values 0 and 1",
    ):
        prepare_binary_targets(
            labels=labels,
        )


@pytest.mark.parametrize(
    "non_finite_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_prepare_binary_targets_rejects_non_finite_value(
    non_finite_value: float,
) -> None:
    """
    NaN·양의 무한대·음의 무한대 Label을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0.0,
            non_finite_value,
            1.0,
        ],
        dtype=torch.float32,
    )

    with pytest.raises(
        ValueError,
        match="finite values",
    ):
        prepare_binary_targets(
            labels=labels,
        )


# =============================================================================
# BCEWithLogitsLoss Calculation
# =============================================================================


def test_zero_logits_produce_expected_binary_cross_entropy() -> None:
    """
    Logit이 모두 0일 때 BCE Loss가 ln(2)인지 확인한다.

    Sigmoid(0):

        0.5

    정답이 0이든 1이든 예측 확률이 0.5이면
    Binary Cross Entropy는 다음 값이다.

        -log(0.5)

        =

        ln(2)

        ≈

        0.693147
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert loss.item() == pytest.approx(
        math.log(2.0),
        rel=1e-6,
        abs=1e-6,
    )


def test_correct_confident_logits_have_lower_loss_than_wrong_logits() -> None:
    """
    정답 방향의 자신 있는 Logit이 반대 방향 Logit보다 낮은 Loss를 갖는지 확인한다.

    올바른 예측:

        NORMAL Label 0

        ->

        음수 Logit

        DEFECT Label 1

        ->

        양수 Logit

    잘못된 예측:

        NORMAL Label 0

        ->

        양수 Logit

        DEFECT Label 1

        ->

        음수 Logit
    """
    loss_function = (
        create_binary_classification_loss()
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    correct_logits = torch.tensor(
        [
            -5.0,
            5.0,
        ],
        dtype=torch.float32,
    )

    wrong_logits = torch.tensor(
        [
            5.0,
            -5.0,
        ],
        dtype=torch.float32,
    )

    correct_loss = loss_function(
        correct_logits,
        targets,
    )

    wrong_loss = loss_function(
        wrong_logits,
        targets,
    )

    assert correct_loss < wrong_loss


def test_loss_output_is_scalar_tensor() -> None:
    """
    reduction='mean'인 Loss 출력이 Scalar Tensor인지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -1.0,
            1.0,
            0.5,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert isinstance(
        loss,
        Tensor,
    )

    assert loss.ndim == 0

    assert loss.shape == torch.Size([])


def test_loss_output_is_float32_for_float32_logits() -> None:
    """
    float32 Logit과 float32 Target의 Loss가 float32인지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -1.0,
            1.0,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert loss.dtype == torch.float32


def test_loss_output_is_finite() -> None:
    """
    정상 Logit과 Target으로 계산한 Loss에 NaN·inf가 없는지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -2.0,
            1.5,
            0.2,
            -0.7,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert torch.isfinite(
        loss,
    )


def test_bce_with_logits_loss_remains_finite_for_extreme_correct_logits() -> None:
    """
    매우 큰 절댓값의 올바른 Logit에서도 Loss가 유한한지 확인한다.

    BCEWithLogitsLoss를 사용하는 중요한 이유 중 하나는
    Sigmoid와 BCE를 따로 계산하는 것보다 수치적으로 안정적이기 때문이다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -100.0,
            100.0,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert torch.isfinite(
        loss,
    )

    assert loss.item() >= 0.0


def test_bce_with_logits_loss_remains_finite_for_extreme_wrong_logits() -> None:
    """
    매우 큰 절댓값의 잘못된 Logit에서도 Loss가 유한한지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            100.0,
            -100.0,
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    assert torch.isfinite(
        loss,
    )

    assert loss.item() > 0.0


# =============================================================================
# Gradient Validation
# =============================================================================


def test_loss_backward_creates_logit_gradients() -> None:
    """
    Loss에서 Backward를 수행하면 Logit Gradient가 생성되는지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -2.0,
            1.5,
            0.2,
            -0.7,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    labels = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None

    assert logits.grad.shape == (
        4,
    )


def test_loss_backward_creates_only_finite_logit_gradients() -> None:
    """
    Backward 결과의 Logit Gradient에 NaN·inf가 없는지 확인한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            -2.0,
            1.5,
            0.2,
            -0.7,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    labels = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None

    assert torch.isfinite(
        logits.grad,
    ).all()


def test_gradient_direction_moves_logits_toward_correct_binary_class() -> None:
    """
    BCE Loss Gradient 방향이 정답 Class 방향과 일치하는지 확인한다.

    Gradient Descent는 다음 계산을 수행한다.

        new_value

        =

        old_value

        -

        learning_rate × gradient

    NORMAL Target 0:
        Logit을 낮춰야 하므로 Gradient는 양수여야 한다.

    DEFECT Target 1:
        Logit을 높여야 하므로 Gradient는 음수여야 한다.
    """
    loss_function = (
        create_binary_classification_loss()
    )

    logits = torch.tensor(
        [
            0.0,
            0.0,
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    targets = prepare_binary_targets(
        labels=labels,
    )

    loss = loss_function(
        logits,
        targets,
    )

    loss.backward()

    assert logits.grad is not None

    # NORMAL Target 0:
    #
    # Gradient Descent에서 Logit을 더 낮추기 위한 양수 Gradient
    assert logits.grad[0].item() > 0.0

    # DEFECT Target 1:
    #
    # Gradient Descent에서 Logit을 더 높이기 위한 음수 Gradient
    assert logits.grad[1].item() < 0.0