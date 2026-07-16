"""
Binary classification loss utilities for the CNN Baseline model.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 정상·불량 이진 이미지
분류 학습에서 사용할 Loss Function을 생성한다.

현재 클래스 정의
----------------
0 = NORMAL
1 = DEFECT

Positive Class:
    DEFECT

모델 출력
---------
CNNBaseline은 이미지 한 장마다 Binary Raw Logit 하나를 반환한다.

Batch 입력:

    [batch_size, 3, 224, 224]

모델 출력:

    [batch_size]

Dataset Label
-------------
현재 Dataset과 DataLoader는 Class Label을 다음 형식으로 반환한다.

    dtype:
        torch.int64

    값:
        0 또는 1

BCEWithLogitsLoss Target
------------------------
BCEWithLogitsLoss 계산 전에는 Label을 다음 형식으로 변환한다.

    dtype:
        torch.float32

    값:
        0.0 또는 1.0

전체 호출 흐름
-------------
Dataset

-> Integer Label

-> prepare_binary_targets()

-> Floating Point Binary Target

-> CNN Raw Logit

-> BCEWithLogitsLoss

-> Scalar Loss

중요
----
모델 내부에서는 Sigmoid를 적용하지 않는다.

BCEWithLogitsLoss가 내부에서 Sigmoid와 Binary Cross Entropy 계산을
수치적으로 안정적인 방식으로 함께 처리한다.
"""

import torch
from torch import Tensor, nn


# =============================================================================
# Loss Configuration
# =============================================================================

# BCEWithLogitsLoss는 기본적으로 Batch의 개별 Loss를 계산한 뒤
# 평균값 하나를 반환한다.
#
# 예:
#
# 개별 Sample Loss:
#
#     [0.52, 0.71, 0.63, 0.44]
#
# reduction="mean":
#
#     0.575
#
# 최종 출력 Shape:
#
#     []
#
# 즉, 역전파에 사용할 Scalar Tensor 하나가 반환된다.
DEFAULT_LOSS_REDUCTION = "mean"


# 현재 CNN Baseline에서는 Class Weight를 사용하지 않는다.
#
# Train Split:
#
# NORMAL:
#     2,300장
#     43.35%
#
# DEFECT:
#     3,006장
#     56.65%
#
# 불균형이 매우 심하지 않으므로 첫 Baseline에서는
# 모든 Sample에 같은 중요도를 적용한다.
#
# 향후 Validation 결과에서 특정 Class Recall이 지나치게 낮거나
# 한 Class로 예측이 편향될 경우 별도 실험으로 검토한다.
USE_POSITIVE_CLASS_WEIGHT = False


# =============================================================================
# Loss Function Factory
# =============================================================================


def create_binary_classification_loss() -> nn.BCEWithLogitsLoss:
    """
    CNN Binary Classification용 BCEWithLogitsLoss를 생성한다.

    왜 필요한가
    -----------
    CNNBaseline은 확률이 아닌 Binary Raw Logit을 출력한다.

    Raw Logit과 Binary Target을 이용해 모델을 학습하려면
    Binary Classification Loss Function이 필요하다.

    현재는 다음 Loss를 사용한다.

        BCEWithLogitsLoss

    입력
    ----
    없음

    현재 CNN Baseline은 Class Weight를 사용하지 않으므로
    외부 설정값을 입력받지 않는다.

    처리 과정
    ---------
    1. BCEWithLogitsLoss 객체를 생성한다.
    2. Batch 내 Sample Loss의 평균값을 반환하도록 설정한다.
    3. Positive Class Weight는 적용하지 않는다.

    출력
    ----
    nn.BCEWithLogitsLoss

        Raw Logit과 Binary Target을 입력받아
        Scalar Loss를 계산하는 PyTorch Loss Module

    호출 관계
    ---------
    향후 Training Pipeline

    -> create_binary_classification_loss()

    -> BCEWithLogitsLoss 생성

    -> Train Epoch

    -> CNN Raw Logit 계산

    -> Loss 계산

    -> loss.backward()

    -> Optimizer Step

    설계 이유
    ---------
    BCEWithLogitsLoss는 다음 계산을 내부에서 함께 처리한다.

        Sigmoid

        +

        Binary Cross Entropy

    Sigmoid와 BCELoss를 따로 사용하는 것보다
    수치적으로 더 안정적이다.

    현재 Reduction
    --------------
    mean

    Batch 내 모든 Sample Loss의 평균을 반환한다.

    예외 처리
    ---------
    이 함수 자체에는 사용자 입력값이 없으므로
    별도의 입력 예외가 없다.

    Logit과 Target의 Shape·dtype 문제는
    Loss 호출 시 PyTorch가 검증한다.

    Binary Target의 값과 dtype은
    prepare_binary_targets()에서 사전에 검증한다.

    테스트 방법
    -----------
    Dummy Logit:

        [4]

    Integer Label:

        [4]

    Target 변환:

        int64

        ->

        float32

    Loss 결과:

        Scalar Tensor

    실무 확장 방향
    --------------
    Validation 결과에서 Class 편향이 확인되면 다음을 검토할 수 있다.

        pos_weight

        Focal Loss

        Class-balanced Loss

    현재 Baseline에서는 구현하지 않는다.
    """
    loss_function = nn.BCEWithLogitsLoss(
        reduction=DEFAULT_LOSS_REDUCTION,
    )

    return loss_function


# =============================================================================
# Binary Target Preparation
# =============================================================================


def prepare_binary_targets(
    labels: Tensor,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Dataset Label을 BCEWithLogitsLoss용 Binary Target으로 변환한다.

    왜 필요한가
    -----------
    현재 Dataset은 Class Label을 정수형으로 반환한다.

    예:

        tensor([0, 1, 1, 0])

        dtype:
            torch.int64

    BCEWithLogitsLoss 계산에는 Floating Point Target이 필요하다.

    따라서 Loss 계산 전에 다음 형식으로 변환한다.

        tensor([0.0, 1.0, 1.0, 0.0])

        dtype:
            torch.float32

    입력
    ----
    labels:
        Dataset 또는 DataLoader에서 전달된 Label Tensor

        예상 Shape:

            [batch_size]

        허용 값:

            0

            1

        현재 실제 dtype:

            torch.int64

    device:
        Target Tensor를 이동할 Device

        예:

            "cpu"

            torch.device("cpu")

            "cuda"

        None이면 기존 Device를 유지한다.

    처리 과정
    ---------
    1. Tensor인지 확인한다.
    2. 1차원 Label Batch인지 확인한다.
    3. 빈 Label Batch인지 확인한다.
    4. bool dtype인지 확인한다.
    5. 지원하는 숫자 dtype인지 확인한다.
    6. float32로 변환한다.
    7. 필요한 경우 지정 Device로 이동한다.
    8. NaN 또는 inf가 없는지 확인한다.
    9. 모든 값이 0 또는 1인지 확인한다.

    출력
    ----
    Tensor

        Shape:

            [batch_size]

        dtype:

            torch.float32

        값:

            0.0 또는 1.0

        Device:

            device가 지정된 경우 해당 Device

            device가 None이면 기존 Device

    호출 관계
    ---------
    향후 Train Epoch:

        Integer Labels

        -> prepare_binary_targets()

        -> Float Targets

        -> BCEWithLogitsLoss

    향후 Validation Epoch:

        Integer Labels

        -> prepare_binary_targets()

        -> Float Targets

        -> BCEWithLogitsLoss

    설계 이유
    ---------
    Dataset은 Class Index 역할을 유지한다.

        int64

    Loss 계산 단계에서만 Binary Floating Target으로 변환한다.

        float32

    Dataset의 책임과 학습 Loss의 책임을 분리할 수 있다.

    예외 처리
    ---------
    Tensor가 아닌 입력:

        TypeError

    1차원이 아닌 Label:

        ValueError

    빈 Label:

        ValueError

    bool Label:

        TypeError

    지원하지 않는 dtype:

        TypeError

    NaN 또는 inf:

        ValueError

    0·1 이외의 Label:

        ValueError

    테스트 방법
    -----------
    입력:

        tensor(
            [0, 1, 1, 0],
            dtype=torch.int64,
        )

    예상 출력:

        tensor(
            [0.0, 1.0, 1.0, 0.0],
            dtype=torch.float32,
        )

    실무 확장 방향
    --------------
    Multi-class Classification으로 확장할 경우
    이 함수는 사용하지 않는다.

    Multi-class 문제에서는 일반적으로 다음 구조를 사용한다.

        Integer Class Index

        +

        CrossEntropyLoss
    """
    if not isinstance(
        labels,
        Tensor,
    ):
        raise TypeError(
            "labels must be a torch.Tensor. "
            f"Received type: {type(labels).__name__}."
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must have 1 dimension in "
            "[batch_size] format. "
            f"Received shape: {tuple(labels.shape)}."
        )

    if labels.numel() == 0:
        raise ValueError(
            "labels must contain at least one binary class label."
        )

    # Python과 PyTorch에서 bool은 0 또는 1처럼 동작할 수 있다.
    #
    # 그러나 현재 Label은 명시적인 Class Index여야 하므로
    # bool dtype은 허용하지 않는다.
    if labels.dtype == torch.bool:
        raise TypeError(
            "labels must not use torch.bool. "
            "Use integer class labels 0 and 1 instead."
        )

    supported_integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }

    is_supported_integer = (
        labels.dtype in supported_integer_dtypes
    )

    is_floating_point = labels.is_floating_point()

    if (
        not is_supported_integer
        and not is_floating_point
    ):
        raise TypeError(
            "labels must use an integer or floating-point dtype. "
            f"Received dtype: {labels.dtype}."
        )

    # Dataset Label은 보통 int64이지만,
    # BCEWithLogitsLoss Target은 float32로 변환한다.
    #
    # device가 지정되지 않으면 기존 Device를 유지한다.
    if device is None:
        targets = labels.to(
            dtype=torch.float32,
        )
    else:
        targets = labels.to(
            device=device,
            dtype=torch.float32,
        )

    # float Label을 직접 전달하는 경우
    # NaN 또는 inf가 포함될 가능성을 방어한다.
    if not torch.isfinite(
        targets,
    ).all():
        raise ValueError(
            "labels must contain only finite values. "
            "NaN and infinity are not allowed."
        )

    is_zero = targets == 0.0
    is_one = targets == 1.0

    valid_binary_values = (
        is_zero
        | is_one
    )

    if not valid_binary_values.all():
        invalid_values = torch.unique(
            targets[
                ~valid_binary_values
            ]
        ).detach().cpu().tolist()

        raise ValueError(
            "labels must contain only binary values 0 and 1. "
            f"Received invalid values: {invalid_values}."
        )

    return targets