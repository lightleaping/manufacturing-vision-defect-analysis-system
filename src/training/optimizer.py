"""
Optimizer utilities for vision-model training.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 이미지 분류 모델을
학습하기 위한 Adam Optimizer를 생성한다.

현재 CNN Baseline 설정
----------------------
Optimizer:

    Adam

Learning Rate:

    0.001

Weight Decay:

    0.0

Adam Betas:

    (0.9, 0.999)

Adam Epsilon:

    1e-8

현재 모델
---------
CNNBaseline

전체 Parameter:

    6,065개

현재 CNNBaseline의 모든 Parameter는 학습 가능하다.

전체 학습 흐름
-------------
Model

-> Trainable Parameter 수집

-> create_optimizer()

-> Adam Optimizer

-> optimizer.zero_grad()

-> Model Forward

-> Loss 계산

-> loss.backward()

-> optimizer.step()

중요
----
Optimizer는 Loss를 직접 계산하지 않는다.

loss.backward()가 각 Parameter의 Gradient를 계산하면,
optimizer.step()이 그 Gradient를 이용해 Weight와 Bias를 갱신한다.
"""

import math
from numbers import Real

from torch import nn
from torch.optim import Adam, Optimizer


# =============================================================================
# Default Optimizer Configuration
# =============================================================================

# Adam Optimizer의 초기 Learning Rate다.
#
# 1e-3과 0.001은 같은 값이다.
#
# 첫 CNN Baseline에서는 복잡한 Learning Rate 탐색 없이
# Adam의 대표적인 초기값인 0.001을 사용한다.
DEFAULT_LEARNING_RATE = 1e-3


# 첫 CNN Baseline에서는 Weight Decay를 적용하지 않는다.
#
# 현재 모델은 약 6천 개 Parameter를 가진 작은 모델이며,
# 아직 실제 Train·Validation 결과를 확인하지 않았다.
#
# 과적합이 확인되기 전에 정규화 요소를 추가하지 않고
# 기본 학습 특성을 먼저 확인한다.
DEFAULT_WEIGHT_DECAY = 0.0


# Adam의 첫 번째·두 번째 Momentum 관련 계수다.
#
# 첫 번째 값:
#
#     Gradient 평균 정보를 추적한다.
#
# 두 번째 값:
#
#     Gradient 제곱 평균 정보를 추적한다.
#
# 현재 Baseline에서는 PyTorch Adam 기본값을 그대로 사용한다.
DEFAULT_ADAM_BETAS = (
    0.9,
    0.999,
)


# Adam 계산 과정에서 0으로 나누는 문제를 방지하고
# 수치 안정성을 높이기 위해 사용하는 작은 값이다.
#
# 현재 Baseline에서는 PyTorch Adam 기본값을 그대로 사용한다.
DEFAULT_ADAM_EPSILON = 1e-8


# =============================================================================
# Optimizer Factory
# =============================================================================


def create_optimizer(
    model: nn.Module,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> Optimizer:
    """
    학습 가능한 Model Parameter를 사용하는 Adam Optimizer를 생성한다.

    왜 필요한가
    -----------
    loss.backward()는 Model Parameter의 Gradient를 계산하지만,
    Gradient 계산만으로 Weight와 Bias가 변경되지는 않는다.

    Optimizer는 계산된 Gradient를 사용해
    Model Parameter를 실제로 갱신한다.

    입력
    ----
    model:
        학습할 PyTorch Model

        예상 타입:

            nn.Module

        현재 사용 모델:

            CNNBaseline

    learning_rate:
        Parameter 갱신 크기를 제어하는 Learning Rate

        기본값:

            0.001

        허용 조건:

            숫자

            유한한 값

            0보다 큰 값

    weight_decay:
        Weight 크기를 제한하는 정규화 계수

        기본값:

            0.0

        허용 조건:

            숫자

            유한한 값

            0 이상인 값

    처리 과정
    ---------
    1. model이 nn.Module인지 확인한다.
    2. Learning Rate가 유효한지 확인한다.
    3. Weight Decay가 유효한지 확인한다.
    4. Model에 Parameter가 존재하는지 확인한다.
    5. requires_grad=True인 Parameter만 수집한다.
    6. 학습 가능한 Parameter가 존재하는지 확인한다.
    7. Adam Optimizer를 생성한다.

    출력
    ----
    Optimizer

        실제 객체 타입:

            torch.optim.Adam

        설정:

            learning_rate:

                기본 0.001

            weight_decay:

                기본 0.0

            betas:

                (0.9, 0.999)

            eps:

                1e-8

    호출 관계
    ---------
    향후 Training Pipeline:

        model = CNNBaseline()

        optimizer = create_optimizer(
            model=model,
        )

        Train Batch:

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

    설계 이유
    ---------
    Adam:
        Parameter별 Gradient 통계를 사용해
        비교적 안정적인 초기 학습을 제공한다.

    Learning Rate 0.001:
        작은 CNN Baseline의 첫 학습에 사용할
        단순하고 일반적인 초기값이다.

    Weight Decay 0.0:
        아직 과적합 여부를 확인하지 않았으므로
        첫 Baseline에서는 추가 정규화를 적용하지 않는다.

    Trainable Parameter만 연결:
        향후 ResNet18 전이학습에서 Freeze한 Parameter를
        Optimizer에서 자동으로 제외할 수 있다.

    예외 처리
    ---------
    model이 nn.Module이 아닌 경우:

        TypeError

    Learning Rate가 숫자가 아닌 경우:

        TypeError

    Learning Rate가 NaN 또는 inf인 경우:

        ValueError

    Learning Rate가 0 이하인 경우:

        ValueError

    Weight Decay가 숫자가 아닌 경우:

        TypeError

    Weight Decay가 NaN 또는 inf인 경우:

        ValueError

    Weight Decay가 음수인 경우:

        ValueError

    Model에 Parameter가 없는 경우:

        ValueError

    모든 Parameter가 Freeze된 경우:

        ValueError

    테스트 방법
    -----------
    CNNBaseline 생성:

        model = CNNBaseline()

    Optimizer 생성:

        optimizer = create_optimizer(
            model=model,
        )

    확인:

        Optimizer 타입:

            Adam

        Learning Rate:

            0.001

        Weight Decay:

            0.0

        Optimizer Parameter 수:

            6,065

    실무 확장 방향
    --------------
    향후 다음 기능을 별도 실험할 수 있다.

        AdamW

        SGD + Momentum

        Weight Decay

        Learning Rate Scheduler

        Layer별 Learning Rate

        ResNet18 Backbone Freeze

    현재 CNN Baseline 범위에는 포함하지 않는다.
    """
    _validate_model(
        model=model,
    )

    validated_learning_rate = _validate_learning_rate(
        learning_rate=learning_rate,
    )

    validated_weight_decay = _validate_weight_decay(
        weight_decay=weight_decay,
    )

    trainable_parameters = _collect_trainable_parameters(
        model=model,
    )

    optimizer = Adam(
        params=trainable_parameters,
        lr=validated_learning_rate,
        betas=DEFAULT_ADAM_BETAS,
        eps=DEFAULT_ADAM_EPSILON,
        weight_decay=validated_weight_decay,
    )

    return optimizer


# =============================================================================
# Model Validation
# =============================================================================


def _validate_model(
    model: nn.Module,
) -> None:
    """
    Optimizer에 전달할 Model 타입을 검증한다.

    입력
    ----
    model:
        검증할 객체

    처리 과정
    ---------
    nn.Module 인스턴스인지 확인한다.

    출력
    ----
    정상:

        반환값 없음

    잘못된 타입:

        TypeError

    설계 이유
    ---------
    Optimizer는 PyTorch Parameter를 필요로 한다.

    잘못된 객체가 전달되면 model.parameters() 호출 시
    이해하기 어려운 AttributeError가 발생할 수 있다.

    함수 진입 시점에 타입을 확인하면
    오류 원인을 더 명확하게 전달할 수 있다.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be an instance of torch.nn.Module. "
            f"Received type: {type(model).__name__}."
        )


# =============================================================================
# Hyperparameter Validation
# =============================================================================


def _validate_learning_rate(
    learning_rate: float,
) -> float:
    """
    Learning Rate를 검증하고 float 값으로 반환한다.

    입력
    ----
    learning_rate:
        검증할 Learning Rate

    허용 조건
    ---------
    숫자

    유한한 값

    0보다 큰 값

    출력
    ----
    float

        검증된 Learning Rate

    예외 처리
    ---------
    bool:

        TypeError

    숫자가 아닌 값:

        TypeError

    NaN·inf:

        ValueError

    0 이하:

        ValueError
    """
    if (
        isinstance(
            learning_rate,
            bool,
        )
        or not isinstance(
            learning_rate,
            Real,
        )
    ):
        raise TypeError(
            "learning_rate must be a real number. "
            f"Received type: {type(learning_rate).__name__}."
        )

    validated_learning_rate = float(
        learning_rate,
    )

    if not math.isfinite(
        validated_learning_rate,
    ):
        raise ValueError(
            "learning_rate must be finite. "
            f"Received value: {validated_learning_rate}."
        )

    if validated_learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be greater than 0. "
            f"Received value: {validated_learning_rate}."
        )

    return validated_learning_rate


def _validate_weight_decay(
    weight_decay: float,
) -> float:
    """
    Weight Decay를 검증하고 float 값으로 반환한다.

    입력
    ----
    weight_decay:
        검증할 Weight Decay

    허용 조건
    ---------
    숫자

    유한한 값

    0 이상인 값

    출력
    ----
    float

        검증된 Weight Decay

    예외 처리
    ---------
    bool:

        TypeError

    숫자가 아닌 값:

        TypeError

    NaN·inf:

        ValueError

    음수:

        ValueError
    """
    if (
        isinstance(
            weight_decay,
            bool,
        )
        or not isinstance(
            weight_decay,
            Real,
        )
    ):
        raise TypeError(
            "weight_decay must be a real number. "
            f"Received type: {type(weight_decay).__name__}."
        )

    validated_weight_decay = float(
        weight_decay,
    )

    if not math.isfinite(
        validated_weight_decay,
    ):
        raise ValueError(
            "weight_decay must be finite. "
            f"Received value: {validated_weight_decay}."
        )

    if validated_weight_decay < 0.0:
        raise ValueError(
            "weight_decay must be greater than or equal to 0. "
            f"Received value: {validated_weight_decay}."
        )

    return validated_weight_decay


# =============================================================================
# Trainable Parameter Collection
# =============================================================================


def _collect_trainable_parameters(
    model: nn.Module,
) -> list[nn.Parameter]:
    """
    Model에서 학습 가능한 Parameter만 수집한다.

    왜 필요한가
    -----------
    PyTorch Model에는 다음 두 종류의 Parameter가 있을 수 있다.

    학습 가능:

        requires_grad=True

    Freeze:

        requires_grad=False

    Optimizer는 실제로 학습할 Parameter만 관리해야 한다.

    입력
    ----
    model:
        검증이 완료된 PyTorch Model

    처리 과정
    ---------
    1. 모든 Parameter를 List로 수집한다.
    2. Parameter 존재 여부를 확인한다.
    3. requires_grad=True인 Parameter만 선택한다.
    4. 학습 가능한 Parameter 존재 여부를 확인한다.

    출력
    ----
    list[nn.Parameter]

        학습 가능한 Parameter 목록

    예외 처리
    ---------
    Model에 Parameter가 없는 경우:

        ValueError

    모든 Parameter가 Freeze된 경우:

        ValueError

    설계 이유
    ---------
    현재 CNNBaseline은 모든 Parameter를 학습한다.

    향후 ResNet18 전이학습에서는 Backbone 일부를 Freeze할 수 있다.

    requires_grad=True인 Parameter만 Optimizer에 전달하면
    현재 CNN과 향후 전이학습 구조에 모두 사용할 수 있다.
    """
    all_parameters = list(
        model.parameters()
    )

    if not all_parameters:
        raise ValueError(
            "model must contain at least one parameter."
        )

    trainable_parameters = [
        parameter
        for parameter in all_parameters
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "model must contain at least one trainable parameter "
            "with requires_grad=True."
        )

    return trainable_parameters