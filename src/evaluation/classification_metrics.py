"""
Binary classification metrics for manufacturing image inspection.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 Binary Classification 결과에서
Accuracy·Precision·Recall·F1 Score·Confusion Matrix를 계산한다.

현재 Class 정의
--------------
0:

    NORMAL

1:

    DEFECT

Negative Class:

    NORMAL

Positive Class:

    DEFECT

현재 입력
---------
Ground Truth Labels:

    evaluation_result.labels

Binary Predictions:

    evaluation_result.predictions

현재 출력
---------
Accuracy

Precision

Recall

F1 Score

True Negative

False Positive

False Negative

True Positive

Sample Count

2 x 2 Confusion Matrix

Confusion Matrix 순서
---------------------
Row:

    Actual Class

Column:

    Predicted Class

Class 순서:

    0 = NORMAL

    1 = DEFECT

Matrix:

    [
        [TN, FP],
        [FN, TP],
    ]

즉:

                    Predicted

                    NORMAL    DEFECT

Actual NORMAL          TN        FP

Actual DEFECT          FN        TP

Metric 정의
-----------
Accuracy:

    (TN + TP)

    /

    Total Samples

Precision:

    TP

    /

    (TP + FP)

Recall:

    TP

    /

    (TP + FN)

F1 Score:

    2 * Precision * Recall

    /

    (Precision + Recall)

Zero Division 정책
-------------------
Precision 분모가 0:

    Precision = 0.0

Recall 분모가 0:

    Recall = 0.0

F1 분모가 0:

    F1 Score = 0.0

왜 오류 대신 0.0인가
---------------------
예를 들어 Model이 DEFECT를 한 번도 예측하지 않으면:

    TP + FP = 0

Precision은 수학적으로 정의되지 않는다.

그러나 평가 Pipeline을 중단하지 않고
해당 Metric을 0.0으로 기록하는 정책을 사용한다.

현재 구현 방식
--------------
sklearn 함수를 호출하지 않고
TN·FP·FN·TP를 직접 계산한다.

이유
----
Metric 정의 학습

Binary Class 의미 명확화

Zero Division 정책 명시

Confusion Matrix 순서 명시

Torch Tensor 기반 처리

향후 sklearn 결과와 교차 검증 가능
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor


# =============================================================================
# Binary Class Configuration
# =============================================================================

# 정상 Class Label
NEGATIVE_CLASS_LABEL = 0


# 불량 Class Label
#
# Precision·Recall·F1은 이 Class를 Positive로 계산한다.
POSITIVE_CLASS_LABEL = 1


# Confusion Matrix Class 순서
#
# Row:
#
#     Actual
#
# Column:
#
#     Predicted
#
# Matrix:
#
#     [
#         [TN, FP],
#         [FN, TP],
#     ]
CONFUSION_MATRIX_CLASS_ORDER = (
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
)


# Metric 입력으로 허용할 Integer Tensor Dtype이다.
#
# 최종 내부 표현은 모두 torch.int64로 통일한다.
SUPPORTED_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


# Metric 비교 허용 오차다.
#
# 직접 계산한 Metric과 Dataclass 입력 Metric이
# 같은지 확인할 때 사용한다.
METRIC_ABSOLUTE_TOLERANCE = 1e-12


# =============================================================================
# Binary Classification Metrics
# =============================================================================


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    """
    Binary Classification Metric과 Confusion Matrix 결과.

    왜 필요한가
    -----------
    개별 float·int 값을 Dictionary로 반환할 수도 있다.

    하지만 Dataclass를 사용하면:

        필드 이름 명확화

        타입 힌트

        자동 완성

        오타 감소

        불변 결과

        내부 일관성 검증

    이 가능하다.

    필드
    ----
    accuracy:
        전체 Sample 중 정답 비율

        계산:

            (TN + TP)

            /

            sample_count

        범위:

            0.0 ~ 1.0

    precision:
        DEFECT라고 예측한 Sample 중
        실제 DEFECT 비율

        계산:

            TP

            /

            (TP + FP)

        범위:

            0.0 ~ 1.0

    recall:
        실제 DEFECT 중
        Model이 DEFECT로 찾은 비율

        계산:

            TP

            /

            (TP + FN)

        범위:

            0.0 ~ 1.0

    f1_score:
        Precision·Recall의 조화 평균

        계산:

            2 * Precision * Recall

            /

            (Precision + Recall)

        범위:

            0.0 ~ 1.0

    sample_count:
        전체 평가 Sample 수

    true_negative:
        실제 NORMAL이고
        NORMAL로 예측한 수

    false_positive:
        실제 NORMAL인데
        DEFECT로 잘못 예측한 수

        제조 관점:

            정상 제품을 불량으로 오판

    false_negative:
        실제 DEFECT인데
        NORMAL로 잘못 예측한 수

        제조 관점:

            실제 불량을 놓침

    true_positive:
        실제 DEFECT이고
        DEFECT로 예측한 수

    confusion_matrix:
        Shape:

            [2, 2]

        Dtype:

            torch.int64

        Device:

            cpu

        구조:

            [
                [TN, FP],
                [FN, TP],
            ]

    내부 일관성
    -----------
    다음 관계를 모두 검증한다.

        TN + FP + FN + TP

        ==

        sample_count

    그리고:

        confusion_matrix

        ==

        [
            [TN, FP],
            [FN, TP],
        ]

    그리고:

        accuracy

        ==

        (TN + TP) / sample_count

    그리고:

        precision

        ==

        TP / (TP + FP)

    그리고:

        recall

        ==

        TP / (TP + FN)

    그리고:

        f1_score

        ==

        2PR / (P + R)
    """

    accuracy: float

    precision: float

    recall: float

    f1_score: float

    sample_count: int

    true_negative: int

    false_positive: int

    false_negative: int

    true_positive: int

    confusion_matrix: Tensor

    def __post_init__(self) -> None:
        """
        Metric·Count·Confusion Matrix의 유효성과 일관성을 검증한다.

        Confusion Matrix는 다음 상태로 정규화한다.

            Gradient 분리

            CPU 이동

            int64

            Contiguous

            Clone

        이유
        ----
        외부 Tensor Storage를 수정해도
        생성된 Metric 객체가 변경되지 않도록 한다.
        """
        validated_accuracy = (
            _validate_probability(
                value=self.accuracy,
                value_name="accuracy",
            )
        )

        validated_precision = (
            _validate_probability(
                value=self.precision,
                value_name="precision",
            )
        )

        validated_recall = (
            _validate_probability(
                value=self.recall,
                value_name="recall",
            )
        )

        validated_f1_score = (
            _validate_probability(
                value=self.f1_score,
                value_name="f1_score",
            )
        )

        validated_sample_count = (
            _validate_positive_integer(
                value=self.sample_count,
                value_name="sample_count",
            )
        )

        validated_true_negative = (
            _validate_non_negative_integer(
                value=self.true_negative,
                value_name="true_negative",
            )
        )

        validated_false_positive = (
            _validate_non_negative_integer(
                value=self.false_positive,
                value_name="false_positive",
            )
        )

        validated_false_negative = (
            _validate_non_negative_integer(
                value=self.false_negative,
                value_name="false_negative",
            )
        )

        validated_true_positive = (
            _validate_non_negative_integer(
                value=self.true_positive,
                value_name="true_positive",
            )
        )

        normalized_confusion_matrix = (
            _normalize_confusion_matrix(
                confusion_matrix=(
                    self.confusion_matrix
                )
            )
        )

        # Frozen Dataclass의 검증 단계에서
        # 정규화된 Scalar·Tensor 값을 저장한다.
        object.__setattr__(
            self,
            "accuracy",
            validated_accuracy,
        )

        object.__setattr__(
            self,
            "precision",
            validated_precision,
        )

        object.__setattr__(
            self,
            "recall",
            validated_recall,
        )

        object.__setattr__(
            self,
            "f1_score",
            validated_f1_score,
        )

        object.__setattr__(
            self,
            "sample_count",
            validated_sample_count,
        )

        object.__setattr__(
            self,
            "true_negative",
            validated_true_negative,
        )

        object.__setattr__(
            self,
            "false_positive",
            validated_false_positive,
        )

        object.__setattr__(
            self,
            "false_negative",
            validated_false_negative,
        )

        object.__setattr__(
            self,
            "true_positive",
            validated_true_positive,
        )

        object.__setattr__(
            self,
            "confusion_matrix",
            normalized_confusion_matrix,
        )

        # ------------------------------------------------------
        # Count Total
        # ------------------------------------------------------
        confusion_count_total = (
            validated_true_negative
            + validated_false_positive
            + validated_false_negative
            + validated_true_positive
        )

        if (
            confusion_count_total
            != validated_sample_count
        ):
            raise ValueError(
                "true_negative, false_positive, "
                "false_negative, and true_positive "
                "must sum to sample_count. "
                f"Confusion count total: "
                f"{confusion_count_total}. "
                f"Sample count: "
                f"{validated_sample_count}."
            )

        # ------------------------------------------------------
        # Confusion Matrix
        # ------------------------------------------------------
        expected_confusion_matrix = (
            torch.tensor(
                [
                    [
                        validated_true_negative,
                        validated_false_positive,
                    ],
                    [
                        validated_false_negative,
                        validated_true_positive,
                    ],
                ],
                dtype=torch.int64,
                device="cpu",
            )
        )

        if not torch.equal(
            normalized_confusion_matrix,
            expected_confusion_matrix,
        ):
            raise ValueError(
                "confusion_matrix must match "
                "true_negative, false_positive, "
                "false_negative, and true_positive."
            )

        # ------------------------------------------------------
        # Expected Metrics
        # ------------------------------------------------------
        expected_accuracy = (
            (
                validated_true_negative
                + validated_true_positive
            )
            / validated_sample_count
        )

        expected_precision = (
            _safe_divide(
                numerator=(
                    validated_true_positive
                ),
                denominator=(
                    validated_true_positive
                    + validated_false_positive
                ),
            )
        )

        expected_recall = (
            _safe_divide(
                numerator=(
                    validated_true_positive
                ),
                denominator=(
                    validated_true_positive
                    + validated_false_negative
                ),
            )
        )

        expected_f1_score = (
            _calculate_f1_score(
                precision=(
                    expected_precision
                ),
                recall=(
                    expected_recall
                ),
            )
        )

        _validate_metric_matches_expected(
            metric_value=(
                validated_accuracy
            ),
            expected_value=(
                expected_accuracy
            ),
            metric_name="accuracy",
        )

        _validate_metric_matches_expected(
            metric_value=(
                validated_precision
            ),
            expected_value=(
                expected_precision
            ),
            metric_name="precision",
        )

        _validate_metric_matches_expected(
            metric_value=(
                validated_recall
            ),
            expected_value=(
                expected_recall
            ),
            metric_name="recall",
        )

        _validate_metric_matches_expected(
            metric_value=(
                validated_f1_score
            ),
            expected_value=(
                expected_f1_score
            ),
            metric_name="f1_score",
        )


# =============================================================================
# Public Metric Function
# =============================================================================


def calculate_binary_classification_metrics(
    labels: Tensor,
    predictions: Tensor,
) -> BinaryClassificationMetrics:
    """
    Ground Truth Label과 Binary Prediction에서 분류 Metric을 계산한다.

    왜 필요한가
    -----------
    Evaluation Runner는 다음 전체 예측 결과를 수집한다.

        Labels

        Logits

        Probabilities

        Predictions

    현재 함수는 그중:

        Labels

        Predictions

    를 사용하여 최종 Binary Classification 성능을 계산한다.

    입력
    ----
    labels:
        Ground Truth Label Tensor

        정상 Shape:

            [sample_count]

        허용 Integer Dtype:

            uint8

            int8

            int16

            int32

            int64

        값:

            0 또는 1

        현재 의미:

            0 = NORMAL

            1 = DEFECT

    predictions:
        Model Binary Prediction Tensor

        정상 Shape:

            [sample_count]

        허용 Integer Dtype:

            uint8

            int8

            int16

            int32

            int64

        값:

            0 또는 1

    처리 과정
    ---------
    1. Label Tensor를 검증한다.
    2. Prediction Tensor를 검증한다.
    3. 두 Tensor를 CPU int64로 정규화한다.
    4. Tensor 길이가 같은지 확인한다.
    5. True Negative를 계산한다.
    6. False Positive를 계산한다.
    7. False Negative를 계산한다.
    8. True Positive를 계산한다.
    9. Accuracy를 계산한다.
    10. Precision을 계산한다.
    11. Recall을 계산한다.
    12. F1 Score를 계산한다.
    13. 2 x 2 Confusion Matrix를 생성한다.
    14. BinaryClassificationMetrics를 반환한다.

    출력
    ----
    BinaryClassificationMetrics

    Confusion Matrix
    ----------------
    반환 순서:

        [
            [TN, FP],
            [FN, TP],
        ]

    Positive Class
    --------------
    현재 Positive Class:

        DEFECT

        Label 1

    Accuracy
    --------
    계산:

        (
            true_negative

            +

            true_positive
        )

        /

        sample_count

    Precision
    ---------
    계산:

        true_positive

        /

        (
            true_positive

            +

            false_positive
        )

    의미:

        DEFECT라고 예측한 제품 중

        실제 DEFECT 비율

    Recall
    ------
    계산:

        true_positive

        /

        (
            true_positive

            +

            false_negative
        )

    의미:

        실제 DEFECT 제품 중

        Model이 찾아낸 비율

    F1 Score
    --------
    계산:

        2

        *

        precision

        *

        recall

        /

        (
            precision

            +

            recall
        )

    Zero Division
    -------------
    분모가 0이면:

        0.0

    을 반환한다.

    예:

        DEFECT Prediction이 하나도 없음

        ->

        TP + FP = 0

        ->

        Precision = 0.0

    Tensor Device
    -------------
    입력이 GPU Tensor여도
    Metric 계산 전 CPU int64 Tensor로 정규화한다.

    입력 Tensor 변경
    ----------------
    입력 Tensor를 수정하지 않는다.

    내부에서:

        detach()

        cpu()

        int64

        contiguous()

        clone()

    을 사용한다.

    예외 처리
    ---------
    Tensor가 아님:

        TypeError

    1차원이 아님:

        ValueError

    빈 Tensor:

        ValueError

    지원하지 않는 Dtype:

        TypeError

    0·1 이외 값:

        ValueError

    Label·Prediction 길이 불일치:

        ValueError

    테스트 예
    ---------
    Labels:

        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ]

    Predictions:

        [
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
        ]

    결과:

        TN = 2

        FP = 1

        FN = 2

        TP = 3

        Accuracy = 5 / 8

        Precision = 3 / 4

        Recall = 3 / 5

        F1 = 2 / 3
    """
    normalized_labels = (
        _normalize_binary_tensor(
            tensor=labels,
            tensor_name="labels",
        )
    )

    normalized_predictions = (
        _normalize_binary_tensor(
            tensor=predictions,
            tensor_name="predictions",
        )
    )

    if (
        normalized_labels.shape[0]
        != normalized_predictions.shape[0]
    ):
        raise ValueError(
            "labels and predictions must contain "
            "the same number of samples. "
            f"Label count: "
            f"{normalized_labels.shape[0]}. "
            f"Prediction count: "
            f"{normalized_predictions.shape[0]}."
        )

    sample_count = int(
        normalized_labels.shape[0]
    )

    # ---------------------------------------------------------
    # True Negative
    # ---------------------------------------------------------
    #
    # Actual:
    #
    #     NORMAL
    #
    # Prediction:
    #
    #     NORMAL
    true_negative = int(
        (
            (
                normalized_labels
                == NEGATIVE_CLASS_LABEL
            )
            &
            (
                normalized_predictions
                == NEGATIVE_CLASS_LABEL
            )
        )
        .sum()
        .item()
    )

    # ---------------------------------------------------------
    # False Positive
    # ---------------------------------------------------------
    #
    # Actual:
    #
    #     NORMAL
    #
    # Prediction:
    #
    #     DEFECT
    false_positive = int(
        (
            (
                normalized_labels
                == NEGATIVE_CLASS_LABEL
            )
            &
            (
                normalized_predictions
                == POSITIVE_CLASS_LABEL
            )
        )
        .sum()
        .item()
    )

    # ---------------------------------------------------------
    # False Negative
    # ---------------------------------------------------------
    #
    # Actual:
    #
    #     DEFECT
    #
    # Prediction:
    #
    #     NORMAL
    false_negative = int(
        (
            (
                normalized_labels
                == POSITIVE_CLASS_LABEL
            )
            &
            (
                normalized_predictions
                == NEGATIVE_CLASS_LABEL
            )
        )
        .sum()
        .item()
    )

    # ---------------------------------------------------------
    # True Positive
    # ---------------------------------------------------------
    #
    # Actual:
    #
    #     DEFECT
    #
    # Prediction:
    #
    #     DEFECT
    true_positive = int(
        (
            (
                normalized_labels
                == POSITIVE_CLASS_LABEL
            )
            &
            (
                normalized_predictions
                == POSITIVE_CLASS_LABEL
            )
        )
        .sum()
        .item()
    )

    # ---------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------
    accuracy = (
        (
            true_negative
            + true_positive
        )
        / sample_count
    )

    # ---------------------------------------------------------
    # Precision
    # ---------------------------------------------------------
    precision = _safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_positive
        ),
    )

    # ---------------------------------------------------------
    # Recall
    # ---------------------------------------------------------
    recall = _safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_negative
        ),
    )

    # ---------------------------------------------------------
    # F1 Score
    # ---------------------------------------------------------
    f1_score = (
        _calculate_f1_score(
            precision=precision,
            recall=recall,
        )
    )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------
    confusion_matrix = torch.tensor(
        [
            [
                true_negative,
                false_positive,
            ],
            [
                false_negative,
                true_positive,
            ],
        ],
        dtype=torch.int64,
        device="cpu",
    )

    return BinaryClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        sample_count=sample_count,
        true_negative=(
            true_negative
        ),
        false_positive=(
            false_positive
        ),
        false_negative=(
            false_negative
        ),
        true_positive=(
            true_positive
        ),
        confusion_matrix=(
            confusion_matrix
        ),
    )


# =============================================================================
# Scalar Validation
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


def _validate_non_negative_integer(
    value: object,
    value_name: str,
) -> int:
    """
    bool이 아닌 0 이상의 정수를 검증한다.

    사용
    ----
    TN

    FP

    FN

    TP
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

    if validated_value < 0:
        raise ValueError(
            f"{value_name} must be greater than "
            "or equal to 0. "
            f"Received value: "
            f"{validated_value}."
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
    유한한 0~1 범위 실수를 검증한다.

    사용
    ----
    Accuracy

    Precision

    Recall

    F1 Score
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
# Binary Tensor Validation
# =============================================================================


def _normalize_binary_tensor(
    tensor: Tensor,
    tensor_name: str,
) -> Tensor:
    """
    Label·Prediction Tensor를 검증하고 CPU int64로 정규화한다.

    검증
    ----
    torch.Tensor

    1차원

    비어 있지 않음

    bool 아님

    지원 Integer Dtype

    값:

        0 또는 1

    출력
    ----
    Detached

    CPU

    int64

    Contiguous

    Clone
    """
    if not isinstance(
        tensor,
        Tensor,
    ):
        raise TypeError(
            f"{tensor_name} must be a torch.Tensor. "
            f"Received type: "
            f"{type(tensor).__name__}."
        )

    if tensor.ndim != 1:
        raise ValueError(
            f"{tensor_name} must be one-dimensional. "
            f"Expected shape: [sample_count]. "
            f"Received shape: "
            f"{tuple(tensor.shape)}."
        )

    if tensor.numel() <= 0:
        raise ValueError(
            f"{tensor_name} must not be empty."
        )

    if tensor.dtype == torch.bool:
        raise TypeError(
            f"{tensor_name} must use an integer dtype, "
            "not torch.bool."
        )

    if (
        tensor.dtype
        not in SUPPORTED_INTEGER_DTYPES
    ):
        raise TypeError(
            f"{tensor_name} must use a supported "
            "integer dtype. "
            f"Supported dtypes: "
            f"{sorted(str(dtype) for dtype in SUPPORTED_INTEGER_DTYPES)}. "
            f"Received dtype: "
            f"{tensor.dtype}."
        )

    normalized_tensor = (
        tensor
        .detach()
        .to(
            device="cpu",
            dtype=torch.int64,
        )
        .contiguous()
        .clone()
    )

    unique_values = torch.unique(
        normalized_tensor
    )

    actual_values = {
        int(
            value.item()
        )
        for value in (
            unique_values
        )
    }

    valid_values = {
        NEGATIVE_CLASS_LABEL,
        POSITIVE_CLASS_LABEL,
    }

    invalid_values = (
        actual_values
        - valid_values
    )

    if invalid_values:
        raise ValueError(
            f"{tensor_name} must contain only "
            f"binary values "
            f"{NEGATIVE_CLASS_LABEL} and "
            f"{POSITIVE_CLASS_LABEL}. "
            f"Invalid values: "
            f"{sorted(invalid_values)}."
        )

    return normalized_tensor


# =============================================================================
# Confusion Matrix Validation
# =============================================================================


def _normalize_confusion_matrix(
    confusion_matrix: Tensor,
) -> Tensor:
    """
    Confusion Matrix를 검증하고 독립적인 CPU int64 Tensor로 반환한다.

    정상
    ----
    Tensor

    Shape:

        [2, 2]

    Dtype:

        torch.int64

    값:

        0 이상의 정수

    출력
    ----
    Detached

    CPU

    int64

    Contiguous

    Clone
    """
    if not isinstance(
        confusion_matrix,
        Tensor,
    ):
        raise TypeError(
            "confusion_matrix must be "
            "a torch.Tensor. "
            f"Received type: "
            f"{type(confusion_matrix).__name__}."
        )

    if (
        tuple(
            confusion_matrix.shape
        )
        != (
            2,
            2,
        )
    ):
        raise ValueError(
            "confusion_matrix must have shape [2, 2]. "
            f"Received shape: "
            f"{tuple(confusion_matrix.shape)}."
        )

    if (
        confusion_matrix.dtype
        != torch.int64
    ):
        raise TypeError(
            "confusion_matrix must use "
            "dtype torch.int64. "
            f"Received dtype: "
            f"{confusion_matrix.dtype}."
        )

    normalized_confusion_matrix = (
        confusion_matrix
        .detach()
        .to(
            device="cpu",
            dtype=torch.int64,
        )
        .contiguous()
        .clone()
    )

    if (
        normalized_confusion_matrix
        .min()
        .item()
        < 0
    ):
        raise ValueError(
            "confusion_matrix must contain "
            "only non-negative counts."
        )

    return normalized_confusion_matrix


# =============================================================================
# Metric Calculation Helpers
# =============================================================================


def _safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    분모가 0이면 0.0을 반환하고
    아니면 일반 나눗셈을 수행한다.

    사용
    ----
    Precision

    Recall

    Zero Division 정책
    -------------------
    denominator == 0:

        0.0
    """
    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def _calculate_f1_score(
    precision: float,
    recall: float,
) -> float:
    """
    Precision·Recall에서 F1 Score를 계산한다.

    계산
    ----
    2PR

    /

    (P + R)

    Zero Division
    -------------
    Precision + Recall == 0:

        0.0
    """
    return _safe_divide(
        numerator=(
            2.0
            * precision
            * recall
        ),
        denominator=(
            precision
            + recall
        ),
    )


def _validate_metric_matches_expected(
    metric_value: float,
    expected_value: float,
    metric_name: str,
) -> None:
    """
    전달된 Metric이 Count에서 다시 계산한 값과 같은지 확인한다.
    """
    if not math.isclose(
        metric_value,
        expected_value,
        rel_tol=0.0,
        abs_tol=(
            METRIC_ABSOLUTE_TOLERANCE
        ),
    ):
        raise ValueError(
            f"{metric_name} must match "
            "the confusion counts. "
            f"Expected: {expected_value}. "
            f"Received: {metric_value}."
        )