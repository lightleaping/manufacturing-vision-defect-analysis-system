"""
Adam optimizer utility unit tests.

테스트 대상
----------
src/training/optimizer.py

테스트 목적
----------
CNNBaseline의 학습 가능한 Parameter가 Adam Optimizer에 정확하게
연결되는지 검증한다.

현재 Optimizer 설정
-------------------
Optimizer:

    Adam

Learning Rate:

    0.001

Weight Decay:

    0.0

Betas:

    (0.9, 0.999)

Epsilon:

    1e-8

현재 CNNBaseline
---------------
전체 Parameter:

    6,065개

현재 모든 Parameter:

    requires_grad=True

전체 학습 흐름
-------------
Model

-> create_optimizer()

-> optimizer.zero_grad()

-> Model Forward

-> BCEWithLogitsLoss

-> loss.backward()

-> Gradient 생성

-> optimizer.step()

-> Parameter 갱신

주의
----
이번 테스트에서는 Train Epoch, Validation Epoch,
전체 Training Pipeline을 검증하지 않는다.

Optimizer 생성과 한 번의 Parameter 갱신까지만 검증한다.
"""

import pytest
import torch
from torch import nn
from torch.optim import Adam

from src.models.cnn_baseline import CNNBaseline
from src.training.loss_function import (
    create_binary_classification_loss,
    prepare_binary_targets,
)
from src.training.optimizer import (
    DEFAULT_ADAM_BETAS,
    DEFAULT_ADAM_EPSILON,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    create_optimizer,
)


# =============================================================================
# Test Helper Models
# =============================================================================


class EmptyModel(nn.Module):
    """
    Parameter가 없는 Model.

    create_optimizer()가 Parameter 없는 Model을
    명확한 오류로 거부하는지 검증할 때 사용한다.
    """

    def __init__(self) -> None:
        """
        Parameter가 없는 nn.Module을 생성한다.
        """
        super().__init__()


class SmallLinearModel(nn.Module):
    """
    Optimizer 동작을 가볍게 검증하기 위한 작은 Linear Model.

    입력
    ----
    Shape:

        [batch_size, 2]

    출력
    ----
    Shape:

        [batch_size, 1]
    """

    def __init__(self) -> None:
        """
        2개 입력 Feature를 1개 출력으로 변환하는 Layer를 생성한다.
        """
        super().__init__()

        self.linear = nn.Linear(
            in_features=2,
            out_features=1,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        입력 Tensor를 Linear Layer에 전달한다.
        """
        return self.linear(
            inputs,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def count_model_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Model의 학습 가능한 Parameter 개수를 계산한다.

    입력
    ----
    model:
        확인할 PyTorch Model

    출력
    ----
    int:
        requires_grad=True인 전체 Scalar Parameter 수
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_optimizer_parameters(
    optimizer: Adam,
) -> int:
    """
    Optimizer에 등록된 전체 Scalar Parameter 수를 계산한다.

    입력
    ----
    optimizer:
        확인할 Adam Optimizer

    출력
    ----
    int:
        모든 Parameter Group에 등록된 Parameter 수
    """
    return sum(
        parameter.numel()
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    )


def get_model_trainable_parameter_ids(
    model: nn.Module,
) -> set[int]:
    """
    Model의 학습 가능한 Parameter 객체 ID를 반환한다.

    Tensor는 값 비교가 아니라 객체 Identity로 비교한다.
    """
    return {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }


def get_optimizer_parameter_ids(
    optimizer: Adam,
) -> set[int]:
    """
    Optimizer에 등록된 Parameter 객체 ID를 반환한다.
    """
    return {
        id(parameter)
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    }


# =============================================================================
# Default Optimizer Configuration
# =============================================================================


def test_default_learning_rate_is_expected_value() -> None:
    """
    CNN Baseline 기본 Learning Rate가 0.001인지 확인한다.
    """
    assert DEFAULT_LEARNING_RATE == 1e-3

    assert DEFAULT_LEARNING_RATE == 0.001


def test_default_weight_decay_is_zero() -> None:
    """
    첫 CNN Baseline에서 Weight Decay를 사용하지 않는지 확인한다.
    """
    assert DEFAULT_WEIGHT_DECAY == 0.0


def test_default_adam_betas_are_expected_values() -> None:
    """
    Adam 기본 Betas 설정을 확인한다.
    """
    assert DEFAULT_ADAM_BETAS == (
        0.9,
        0.999,
    )


def test_default_adam_epsilon_is_expected_value() -> None:
    """
    Adam Epsilon 설정이 1e-8인지 확인한다.
    """
    assert DEFAULT_ADAM_EPSILON == 1e-8


# =============================================================================
# Optimizer Creation
# =============================================================================


def test_create_optimizer_returns_adam() -> None:
    """
    Optimizer Factory가 Adam 객체를 반환하는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    assert isinstance(
        optimizer,
        Adam,
    )


def test_created_optimizer_has_one_parameter_group() -> None:
    """
    현재 CNN Baseline Optimizer가 하나의 Parameter Group을 사용하는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    assert len(
        optimizer.param_groups,
    ) == 1


def test_created_optimizer_uses_default_learning_rate() -> None:
    """
    생성된 Optimizer의 Learning Rate를 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    parameter_group = optimizer.param_groups[0]

    assert parameter_group["lr"] == (
        DEFAULT_LEARNING_RATE
    )


def test_created_optimizer_uses_default_weight_decay() -> None:
    """
    생성된 Optimizer의 Weight Decay를 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    parameter_group = optimizer.param_groups[0]

    assert parameter_group["weight_decay"] == (
        DEFAULT_WEIGHT_DECAY
    )


def test_created_optimizer_uses_default_adam_betas() -> None:
    """
    생성된 Optimizer의 Adam Betas를 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    parameter_group = optimizer.param_groups[0]

    assert parameter_group["betas"] == (
        DEFAULT_ADAM_BETAS
    )


def test_created_optimizer_uses_default_adam_epsilon() -> None:
    """
    생성된 Optimizer의 Epsilon을 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    parameter_group = optimizer.param_groups[0]

    assert parameter_group["eps"] == (
        DEFAULT_ADAM_EPSILON
    )


def test_optimizer_contains_all_cnn_trainable_parameters() -> None:
    """
    CNNBaseline의 학습 가능한 6,065개 Parameter가 모두
    Optimizer에 등록되는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    model_parameter_count = (
        count_model_trainable_parameters(
            model=model,
        )
    )

    optimizer_parameter_count = (
        count_optimizer_parameters(
            optimizer=optimizer,
        )
    )

    assert model_parameter_count == 6_065

    assert optimizer_parameter_count == 6_065

    assert (
        optimizer_parameter_count
        == model_parameter_count
    )


def test_optimizer_uses_same_parameter_objects_as_model() -> None:
    """
    Optimizer가 Model Parameter의 복사본이 아니라
    실제 같은 Parameter 객체를 참조하는지 확인한다.

    같은 객체를 참조해야 optimizer.step()이
    Model Weight를 실제로 변경할 수 있다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    model_parameter_ids = (
        get_model_trainable_parameter_ids(
            model=model,
        )
    )

    optimizer_parameter_ids = (
        get_optimizer_parameter_ids(
            optimizer=optimizer,
        )
    )

    assert (
        optimizer_parameter_ids
        == model_parameter_ids
    )


def test_optimizer_does_not_contain_duplicate_parameters() -> None:
    """
    동일한 Parameter가 Optimizer에 중복 등록되지 않았는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    optimizer_parameters = [
        parameter
        for parameter_group in optimizer.param_groups
        for parameter in parameter_group["params"]
    ]

    optimizer_parameter_ids = [
        id(parameter)
        for parameter in optimizer_parameters
    ]

    assert len(
        optimizer_parameter_ids,
    ) == len(
        set(
            optimizer_parameter_ids,
        )
    )


def test_optimizer_factory_returns_independent_instances() -> None:
    """
    Factory를 여러 번 호출하면 서로 다른 Optimizer 객체가 생성되는지 확인한다.
    """
    model = CNNBaseline()

    first_optimizer = create_optimizer(
        model=model,
    )

    second_optimizer = create_optimizer(
        model=model,
    )

    assert first_optimizer is not second_optimizer


# =============================================================================
# Custom Hyperparameter Configuration
# =============================================================================


def test_create_optimizer_accepts_custom_learning_rate() -> None:
    """
    사용자 지정 Learning Rate를 적용할 수 있는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
        learning_rate=5e-4,
    )

    assert (
        optimizer.param_groups[0]["lr"]
        == 5e-4
    )


def test_create_optimizer_accepts_custom_weight_decay() -> None:
    """
    사용자 지정 Weight Decay를 적용할 수 있는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
        weight_decay=1e-4,
    )

    assert (
        optimizer.param_groups[0]["weight_decay"]
        == 1e-4
    )


def test_create_optimizer_accepts_integer_learning_rate() -> None:
    """
    Real Number인 정수 Learning Rate도 float으로 변환되는지 확인한다.

    실제 학습에서는 0.001을 사용하지만
    입력 검증 동작을 확인하기 위한 테스트다.
    """
    model = SmallLinearModel()

    optimizer = create_optimizer(
        model=model,
        learning_rate=1,
    )

    assert (
        optimizer.param_groups[0]["lr"]
        == 1.0
    )


def test_create_optimizer_accepts_integer_weight_decay() -> None:
    """
    0 이상의 정수 Weight Decay도 float으로 처리되는지 확인한다.
    """
    model = SmallLinearModel()

    optimizer = create_optimizer(
        model=model,
        weight_decay=1,
    )

    assert (
        optimizer.param_groups[0]["weight_decay"]
        == 1.0
    )


def test_zero_weight_decay_is_valid() -> None:
    """
    현재 기본 설정인 Weight Decay 0.0이 정상 허용되는지 확인한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
        weight_decay=0.0,
    )

    assert (
        optimizer.param_groups[0]["weight_decay"]
        == 0.0
    )


# =============================================================================
# Trainable and Frozen Parameter Handling
# =============================================================================


def test_frozen_parameters_are_excluded_from_optimizer() -> None:
    """
    requires_grad=False인 Parameter가 Optimizer에서 제외되는지 확인한다.

    향후 ResNet18 Backbone Freeze에 사용할 수 있는 구조인지 검증한다.
    """
    model = CNNBaseline()

    for parameter in model.conv1.parameters():
        parameter.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    optimizer_parameter_ids = (
        get_optimizer_parameter_ids(
            optimizer=optimizer,
        )
    )

    frozen_parameter_ids = {
        id(parameter)
        for parameter in model.conv1.parameters()
    }

    assert optimizer_parameter_ids.isdisjoint(
        frozen_parameter_ids,
    )


def test_remaining_trainable_parameters_are_included_after_partial_freeze() -> None:
    """
    일부 Layer를 Freeze해도 나머지 Trainable Parameter가
    모두 Optimizer에 포함되는지 확인한다.
    """
    model = CNNBaseline()

    for parameter in model.conv1.parameters():
        parameter.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    model_trainable_parameter_ids = (
        get_model_trainable_parameter_ids(
            model=model,
        )
    )

    optimizer_parameter_ids = (
        get_optimizer_parameter_ids(
            optimizer=optimizer,
        )
    )

    assert (
        optimizer_parameter_ids
        == model_trainable_parameter_ids
    )


def test_partial_freeze_reduces_optimizer_parameter_count() -> None:
    """
    Conv1을 Freeze하면 해당 Layer의 Parameter가
    Optimizer Parameter 수에서 제외되는지 확인한다.

    Conv1 Parameter:

        Weight:

            8 × 3 × 3 × 3

            = 216

        Bias:

            8

        Total:

            224

    전체:

        6,065

    Conv1 Freeze 후:

        6,065 - 224

        = 5,841
    """
    model = CNNBaseline()

    for parameter in model.conv1.parameters():
        parameter.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    optimizer_parameter_count = (
        count_optimizer_parameters(
            optimizer=optimizer,
        )
    )

    assert optimizer_parameter_count == 5_841


# =============================================================================
# Invalid Model Validation
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
def test_create_optimizer_rejects_non_module_model(
    invalid_model: object,
) -> None:
    """
    nn.Module이 아닌 객체를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match="model must be an instance of torch.nn.Module",
    ):
        create_optimizer(
            model=invalid_model,  # type: ignore[arg-type]
        )


def test_create_optimizer_rejects_model_without_parameters() -> None:
    """
    Parameter가 하나도 없는 Model을 거부하는지 확인한다.
    """
    model = EmptyModel()

    with pytest.raises(
        ValueError,
        match="at least one parameter",
    ):
        create_optimizer(
            model=model,
        )


def test_create_optimizer_rejects_model_with_all_parameters_frozen() -> None:
    """
    모든 Parameter가 Freeze된 Model을 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(
        ValueError,
        match="at least one trainable parameter",
    ):
        create_optimizer(
            model=model,
        )


# =============================================================================
# Invalid Learning Rate Validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_learning_rate",
    [
        True,
        False,
        None,
        "0.001",
        [
            0.001,
        ],
    ],
)
def test_create_optimizer_rejects_invalid_learning_rate_type(
    invalid_learning_rate: object,
) -> None:
    """
    Real Number가 아닌 Learning Rate와 bool을 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        TypeError,
        match="learning_rate must be a real number",
    ):
        create_optimizer(
            model=model,
            learning_rate=invalid_learning_rate,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_learning_rate",
    [
        0.0,
        -0.001,
    ],
)
def test_create_optimizer_rejects_non_positive_learning_rate(
    invalid_learning_rate: float,
) -> None:
    """
    0 이하 Learning Rate를 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        ValueError,
        match="greater than 0",
    ):
        create_optimizer(
            model=model,
            learning_rate=invalid_learning_rate,
        )


@pytest.mark.parametrize(
    "invalid_learning_rate",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_create_optimizer_rejects_non_finite_learning_rate(
    invalid_learning_rate: float,
) -> None:
    """
    NaN·양의 무한대·음의 무한대 Learning Rate를 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        ValueError,
        match="learning_rate must be finite",
    ):
        create_optimizer(
            model=model,
            learning_rate=invalid_learning_rate,
        )


# =============================================================================
# Invalid Weight Decay Validation
# =============================================================================


@pytest.mark.parametrize(
    "invalid_weight_decay",
    [
        True,
        False,
        None,
        "0.0",
        [
            0.0,
        ],
    ],
)
def test_create_optimizer_rejects_invalid_weight_decay_type(
    invalid_weight_decay: object,
) -> None:
    """
    Real Number가 아닌 Weight Decay와 bool을 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        TypeError,
        match="weight_decay must be a real number",
    ):
        create_optimizer(
            model=model,
            weight_decay=invalid_weight_decay,  # type: ignore[arg-type]
        )


def test_create_optimizer_rejects_negative_weight_decay() -> None:
    """
    음수 Weight Decay를 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        ValueError,
        match="greater than or equal to 0",
    ):
        create_optimizer(
            model=model,
            weight_decay=-1e-4,
        )


@pytest.mark.parametrize(
    "invalid_weight_decay",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_create_optimizer_rejects_non_finite_weight_decay(
    invalid_weight_decay: float,
) -> None:
    """
    NaN·양의 무한대·음의 무한대 Weight Decay를 거부하는지 확인한다.
    """
    model = SmallLinearModel()

    with pytest.raises(
        ValueError,
        match="weight_decay must be finite",
    ):
        create_optimizer(
            model=model,
            weight_decay=invalid_weight_decay,
        )


# =============================================================================
# Gradient and Optimizer State
# =============================================================================


def test_optimizer_state_is_empty_before_first_step() -> None:
    """
    Adam State가 첫 optimizer.step() 전에는 비어 있는지 확인한다.

    Adam은 첫 Parameter 갱신 시점에
    Gradient 평균·제곱 평균 상태를 생성한다.
    """
    model = CNNBaseline()

    optimizer = create_optimizer(
        model=model,
    )

    assert len(
        optimizer.state,
    ) == 0


def test_zero_grad_with_set_to_none_clears_gradients() -> None:
    """
    set_to_none=True가 기존 Gradient를 None으로 초기화하는지 확인한다.
    """
    model = SmallLinearModel()

    optimizer = create_optimizer(
        model=model,
    )

    inputs = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
        ],
        dtype=torch.float32,
    )

    outputs = model(
        inputs,
    )

    temporary_loss = outputs.sum()

    temporary_loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in model.parameters()
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    assert all(
        parameter.grad is None
        for parameter in model.parameters()
    )


def test_backward_creates_finite_gradients_before_optimizer_step() -> None:
    """
    CNN Forward와 BCE Loss 이후 모든 Trainable Parameter에
    유한한 Gradient가 생성되는지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = CNNBaseline()

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    images = torch.randn(
        2,
        3,
        32,
        32,
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

    optimizer.zero_grad(
        set_to_none=True,
    )

    logits = model(
        images,
    )

    loss = loss_function(
        logits,
        targets,
    )

    loss.backward()

    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue

        assert parameter.grad is not None

        assert torch.isfinite(
            parameter.grad,
        ).all()


# =============================================================================
# Parameter Update
# =============================================================================


def test_optimizer_step_changes_cnn_classifier_weight() -> None:
    """
    Adam Step 이후 CNN Classifier Weight가 실제로 변경되는지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = CNNBaseline()

    loss_function = (
        create_binary_classification_loss()
    )

    optimizer = create_optimizer(
        model=model,
    )

    images = torch.randn(
        2,
        3,
        32,
        32,
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

    classifier_weight_before = (
        model
        .classifier
        .weight
        .detach()
        .clone()
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    logits = model(
        images,
    )

    loss = loss_function(
        logits,
        targets,
    )

    loss.backward()

    optimizer.step()

    classifier_weight_after = (
        model
        .classifier
        .weight
        .detach()
        .clone()
    )

    assert not torch.equal(
        classifier_weight_before,
        classifier_weight_after,
    )


def test_optimizer_state_is_created_after_first_step() -> None:
    """
    첫 Adam Step 이후 Optimizer State가 생성되는지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = SmallLinearModel()

    optimizer = create_optimizer(
        model=model,
    )

    inputs = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
            [
                -1.0,
                0.5,
            ],
        ],
        dtype=torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    outputs = model(
        inputs,
    )

    loss = (
        outputs
        .pow(2)
        .mean()
    )

    loss.backward()

    optimizer.step()

    assert len(
        optimizer.state,
    ) > 0


def test_trainable_parameter_changes_after_step() -> None:
    """
    일부 Freeze 상태에서 Trainable Parameter는 갱신되는지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = SmallLinearModel()

    model.linear.bias.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    weight_before = (
        model
        .linear
        .weight
        .detach()
        .clone()
    )

    inputs = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
        ],
        dtype=torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    outputs = model(
        inputs,
    )

    loss = outputs.sum()

    loss.backward()

    optimizer.step()

    weight_after = (
        model
        .linear
        .weight
        .detach()
        .clone()
    )

    assert not torch.equal(
        weight_before,
        weight_after,
    )


def test_frozen_parameter_does_not_change_after_step() -> None:
    """
    Optimizer에서 제외된 Frozen Parameter가 갱신되지 않는지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = SmallLinearModel()

    model.linear.bias.requires_grad = False

    optimizer = create_optimizer(
        model=model,
    )

    bias_before = (
        model
        .linear
        .bias
        .detach()
        .clone()
    )

    inputs = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
        ],
        dtype=torch.float32,
    )

    optimizer.zero_grad(
        set_to_none=True,
    )

    outputs = model(
        inputs,
    )

    loss = outputs.sum()

    loss.backward()

    optimizer.step()

    bias_after = (
        model
        .linear
        .bias
        .detach()
        .clone()
    )

    assert torch.equal(
        bias_before,
        bias_after,
    )


def test_multiple_optimizer_steps_keep_parameters_finite() -> None:
    """
    여러 번의 작은 학습 Step 후에도 Parameter가
    NaN·inf 없이 유한한 상태인지 확인한다.
    """
    torch.manual_seed(
        42,
    )

    model = SmallLinearModel()

    optimizer = create_optimizer(
        model=model,
        learning_rate=1e-3,
    )

    inputs = torch.tensor(
        [
            [
                1.0,
                2.0,
            ],
            [
                -1.0,
                0.5,
            ],
        ],
        dtype=torch.float32,
    )

    for _ in range(3):
        optimizer.zero_grad(
            set_to_none=True,
        )

        outputs = model(
            inputs,
        )

        loss = (
            outputs
            .pow(2)
            .mean()
        )

        loss.backward()

        optimizer.step()

    for parameter in model.parameters():
        assert torch.isfinite(
            parameter,
        ).all()