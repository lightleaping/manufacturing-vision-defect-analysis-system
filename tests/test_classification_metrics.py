"""
Binary classification metrics unit tests.

테스트 대상
----------
src/evaluation/classification_metrics.py

테스트 목적
----------
Manufacturing Vision Defect Analysis System의 Binary Classification Metric이
현재 Class 정의와 일치하는지 검증한다.

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

Confusion Matrix 순서
---------------------
Row:

    Actual

Column:

    Predicted

Matrix:

    [
        [TN, FP],
        [FN, TP],
    ]

핵심 Metric
-----------
Accuracy:

    (TN + TP)

    /

    Total

Precision:

    TP

    /

    (TP + FP)

Recall:

    TP

    /

    (TP + FN)

F1:

    2 * Precision * Recall

    /

    (Precision + Recall)

Zero Division 정책
-------------------
Precision 분모가 0:

    0.0

Recall 분모가 0:

    0.0

F1 분모가 0:

    0.0

주요 테스트 범위
---------------
Public Constant

정상 Metric

Known Confusion Matrix

모든 예측 정답

모든 예측 오답

NORMAL 단일 Class

DEFECT 단일 Class

Positive Prediction 없음

Positive Label 없음

Zero Division

지원 Integer Dtype

CPU int64 Confusion Matrix

Frozen Dataclass

Metric·Count 내부 일관성

잘못된 Tensor

잘못된 Shape

잘못된 Dtype

잘못된 Binary 값

Label·Prediction 길이 불일치
"""

from __future__ import annotations

import math
from dataclasses import (
    FrozenInstanceError,
)

import pytest
import torch
from torch import Tensor

from src.evaluation.classification_metrics import (
    CONFUSION_MATRIX_CLASS_ORDER,
    METRIC_ABSOLUTE_TOLERANCE,
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    SUPPORTED_INTEGER_DTYPES,
    BinaryClassificationMetrics,
    calculate_binary_classification_metrics,
)


# =============================================================================
# Test Helpers
# =============================================================================


def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    """
    테스트 기대값 계산용 Zero Division 함수.

    Production 내부 함수에 의존하지 않고
    테스트에서 기대 Metric을 독립적으로 계산한다.
    """
    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def calculate_expected_f1(
    precision: float,
    recall: float,
) -> float:
    """
    테스트 기대 F1 Score를 계산한다.
    """
    return safe_divide(
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


def create_metrics_from_counts(
    *,
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> BinaryClassificationMetrics:
    """
    Confusion Count에서 정상 BinaryClassificationMetrics를 생성한다.

    Dataclass 직접 검증 테스트에 사용한다.
    """
    sample_count = (
        true_negative
        + false_positive
        + false_negative
        + true_positive
    )

    accuracy = (
        (
            true_negative
            + true_positive
        )
        / sample_count
    )

    precision = safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_positive
        ),
    )

    recall = safe_divide(
        numerator=true_positive,
        denominator=(
            true_positive
            + false_negative
        ),
    )

    f1_score = calculate_expected_f1(
        precision=precision,
        recall=recall,
    )

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


def create_valid_metric_values() -> dict[
    str,
    object,
]:
    """
    정상 Dataclass 생성 인자를 반환한다.

    Confusion Count
    ---------------
    TN:

        2

    FP:

        1

    FN:

        2

    TP:

        3

    Sample:

        8

    Accuracy:

        5 / 8

    Precision:

        3 / 4

    Recall:

        3 / 5

    F1:

        2 / 3
    """
    true_negative = 2

    false_positive = 1

    false_negative = 2

    true_positive = 3

    sample_count = 8

    accuracy = (
        5
        / 8
    )

    precision = (
        3
        / 4
    )

    recall = (
        3
        / 5
    )

    f1_score = (
        2
        / 3
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "sample_count": sample_count,
        "true_negative": (
            true_negative
        ),
        "false_positive": (
            false_positive
        ),
        "false_negative": (
            false_negative
        ),
        "true_positive": (
            true_positive
        ),
        "confusion_matrix": (
            torch.tensor(
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
            )
        ),
    }


def create_valid_metrics() -> (
    BinaryClassificationMetrics
):
    """
    정상 BinaryClassificationMetrics를 생성한다.
    """
    return BinaryClassificationMetrics(
        **create_valid_metric_values(),  # type: ignore[arg-type]
    )


# =============================================================================
# Public Constants
# =============================================================================


def test_negative_class_label_is_normal_zero() -> None:
    """
    NORMAL Class가 Label 0인지 확인한다.
    """
    assert (
        NEGATIVE_CLASS_LABEL
        == 0
    )


def test_positive_class_label_is_defect_one() -> None:
    """
    DEFECT Positive Class가 Label 1인지 확인한다.
    """
    assert (
        POSITIVE_CLASS_LABEL
        == 1
    )


def test_confusion_matrix_class_order_is_normal_then_defect() -> None:
    """
    Confusion Matrix Class 순서가 0·1인지 확인한다.
    """
    assert (
        CONFUSION_MATRIX_CLASS_ORDER
        == (
            0,
            1,
        )
    )


def test_supported_integer_dtypes() -> None:
    """
    Metric 입력에서 허용하는 Integer Dtype을 확인한다.
    """
    assert (
        SUPPORTED_INTEGER_DTYPES
        == frozenset(
            {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
        )
    )


def test_metric_absolute_tolerance() -> None:
    """
    Metric 내부 일관성 비교 허용 오차를 확인한다.
    """
    assert (
        METRIC_ABSOLUTE_TOLERANCE
        == 1e-12
    )


# =============================================================================
# BinaryClassificationMetrics - Normal Behavior
# =============================================================================


def test_metrics_stores_expected_values() -> None:
    """
    정상 Metric·Count를 저장하는지 확인한다.
    """
    metrics = (
        create_valid_metrics()
    )

    assert math.isclose(
        metrics.accuracy,
        5 / 8,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        metrics.precision,
        3 / 4,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        metrics.recall,
        3 / 5,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        metrics.f1_score,
        2 / 3,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert (
        metrics.sample_count
        == 8
    )

    assert (
        metrics.true_negative
        == 2
    )

    assert (
        metrics.false_positive
        == 1
    )

    assert (
        metrics.false_negative
        == 2
    )

    assert (
        metrics.true_positive
        == 3
    )


def test_metrics_confusion_matrix_has_expected_values() -> None:
    """
    Confusion Matrix가 [[TN, FP], [FN, TP]]인지 확인한다.
    """
    metrics = (
        create_valid_metrics()
    )

    expected_matrix = torch.tensor(
        [
            [
                2,
                1,
            ],
            [
                2,
                3,
            ],
        ],
        dtype=torch.int64,
    )

    assert torch.equal(
        metrics.confusion_matrix,
        expected_matrix,
    )


def test_metrics_is_frozen() -> None:
    """
    생성된 Metric 결과를 재할당할 수 없는지 확인한다.
    """
    metrics = (
        create_valid_metrics()
    )

    with pytest.raises(
        FrozenInstanceError,
    ):
        metrics.recall = 1.0  # type: ignore[misc]


def test_confusion_matrix_is_cpu_int64() -> None:
    """
    Confusion Matrix가 CPU int64 Tensor인지 확인한다.
    """
    metrics = (
        create_valid_metrics()
    )

    assert (
        metrics.confusion_matrix
        .device
        .type
        == "cpu"
    )

    assert (
        metrics.confusion_matrix
        .dtype
        == torch.int64
    )


def test_confusion_matrix_is_contiguous() -> None:
    """
    비연속 입력 Confusion Matrix가 Contiguous로 정규화되는지 확인한다.
    """
    matrix_storage = torch.tensor(
        [
            [
                2,
                99,
                1,
                99,
            ],
            [
                2,
                99,
                3,
                99,
            ],
        ],
        dtype=torch.int64,
    )

    non_contiguous_matrix = (
        matrix_storage[
            :,
            ::2
        ]
    )

    assert (
        non_contiguous_matrix
        .is_contiguous()
        is False
    )

    metrics = (
        BinaryClassificationMetrics(
            accuracy=5 / 8,
            precision=3 / 4,
            recall=3 / 5,
            f1_score=2 / 3,
            sample_count=8,
            true_negative=2,
            false_positive=1,
            false_negative=2,
            true_positive=3,
            confusion_matrix=(
                non_contiguous_matrix
            ),
        )
    )

    assert (
        metrics.confusion_matrix
        .is_contiguous()
    )


def test_confusion_matrix_is_independent_clone() -> None:
    """
    원본 Matrix를 변경해도 Metric Matrix가 변경되지 않는지 확인한다.
    """
    original_matrix = torch.tensor(
        [
            [
                2,
                1,
            ],
            [
                2,
                3,
            ],
        ],
        dtype=torch.int64,
    )

    expected_matrix = (
        original_matrix.clone()
    )

    metrics = (
        BinaryClassificationMetrics(
            accuracy=5 / 8,
            precision=3 / 4,
            recall=3 / 5,
            f1_score=2 / 3,
            sample_count=8,
            true_negative=2,
            false_positive=1,
            false_negative=2,
            true_positive=3,
            confusion_matrix=(
                original_matrix
            ),
        )
    )

    original_matrix.fill_(
        0
    )

    assert torch.equal(
        metrics.confusion_matrix,
        expected_matrix,
    )


def test_metrics_accepts_zero_confusion_counts() -> None:
    """
    일부 Confusion Count가 0이어도 정상 처리하는지 확인한다.
    """
    metrics = (
        create_metrics_from_counts(
            true_negative=10,
            false_positive=0,
            false_negative=0,
            true_positive=5,
        )
    )

    assert (
        metrics.false_positive
        == 0
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.accuracy
        == 1.0
    )

    assert (
        metrics.precision
        == 1.0
    )

    assert (
        metrics.recall
        == 1.0
    )

    assert (
        metrics.f1_score
        == 1.0
    )


def test_metrics_supports_precision_zero_division_policy() -> None:
    """
    Positive Prediction이 없을 때 Precision 0.0을 허용하는지 확인한다.
    """
    metrics = (
        create_metrics_from_counts(
            true_negative=4,
            false_positive=0,
            false_negative=3,
            true_positive=0,
        )
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


def test_metrics_supports_no_positive_label_policy() -> None:
    """
    실제 DEFECT가 없을 때 Recall·F1 0.0 정책을 허용하는지 확인한다.
    """
    metrics = (
        create_metrics_from_counts(
            true_negative=5,
            false_positive=0,
            false_negative=0,
            true_positive=0,
        )
    )

    assert (
        metrics.accuracy
        == 1.0
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


# =============================================================================
# BinaryClassificationMetrics - Invalid Metric Values
# =============================================================================


@pytest.mark.parametrize(
    "field_name",
    [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        True,
        False,
        None,
        "0.5",
        object(),
    ],
)
def test_metrics_rejects_invalid_metric_type(
    field_name: str,
    invalid_value: object,
) -> None:
    """
    Real Number가 아닌 Metric과 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        field_name
    ] = invalid_value

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} "
            "must be a real number"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ],
)
@pytest.mark.parametrize(
    "invalid_value",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_metrics_rejects_invalid_metric_value(
    field_name: str,
    invalid_value: float,
) -> None:
    """
    범위 밖·NaN·inf Metric을 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    (
        "field_name",
        "invalid_value",
    ),
    [
        (
            "accuracy",
            0.7,
        ),
        (
            "precision",
            0.7,
        ),
        (
            "recall",
            0.7,
        ),
        (
            "f1_score",
            0.7,
        ),
    ],
)
def test_metrics_rejects_metric_not_matching_confusion_counts(
    field_name: str,
    invalid_value: float,
) -> None:
    """
    Confusion Count에서 계산한 값과 다른 Metric을 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        field_name
    ] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must match "
            "the confusion counts"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# BinaryClassificationMetrics - Invalid Counts
# =============================================================================


def test_metrics_rejects_invalid_sample_count_type() -> None:
    """
    정수가 아닌 sample_count를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "sample_count"
    ] = 8.0

    with pytest.raises(
        TypeError,
        match=(
            "sample_count must be "
            "an integer"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        False,
        None,
        "8",
        1.5,
    ],
)
def test_metrics_rejects_invalid_sample_count_types(
    invalid_count: object,
) -> None:
    """
    bool·문자열·실수 Sample Count를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "sample_count"
    ] = invalid_count

    with pytest.raises(
        TypeError,
        match=(
            "sample_count must be "
            "an integer"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        0,
        -1,
    ],
)
def test_metrics_rejects_non_positive_sample_count(
    invalid_count: int,
) -> None:
    """
    0 이하 Sample Count를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "sample_count"
    ] = invalid_count

    with pytest.raises(
        ValueError,
        match=(
            "sample_count must be "
            "greater than 0"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ],
)
@pytest.mark.parametrize(
    "invalid_count",
    [
        True,
        False,
        None,
        "1",
        1.5,
    ],
)
def test_metrics_rejects_invalid_confusion_count_type(
    field_name: str,
    invalid_count: object,
) -> None:
    """
    정수가 아닌 Confusion Count와 bool을 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        field_name
    ] = invalid_count

    with pytest.raises(
        TypeError,
        match=(
            f"{field_name} must be "
            "an integer"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ],
)
def test_metrics_rejects_negative_confusion_count(
    field_name: str,
) -> None:
    """
    음수 Confusion Count를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        field_name
    ] = -1

    with pytest.raises(
        ValueError,
        match=(
            f"{field_name} must be "
            "greater than or equal to 0"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


def test_metrics_rejects_confusion_count_total_mismatch() -> None:
    """
    TN·FP·FN·TP 합계가 Sample Count와 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "sample_count"
    ] = 9

    with pytest.raises(
        ValueError,
        match=(
            "must sum to "
            "sample_count"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# BinaryClassificationMetrics - Invalid Confusion Matrix
# =============================================================================


@pytest.mark.parametrize(
    "invalid_matrix",
    [
        None,
        [],
        "matrix",
        123,
    ],
)
def test_metrics_rejects_non_tensor_confusion_matrix(
    invalid_matrix: object,
) -> None:
    """
    Tensor가 아닌 Confusion Matrix를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "confusion_matrix"
    ] = invalid_matrix

    with pytest.raises(
        TypeError,
        match=(
            "confusion_matrix must be "
            "a torch.Tensor"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_shape",
    [
        (
            4,
        ),
        (
            1,
            4,
        ),
        (
            2,
            3,
        ),
        (
            3,
            2,
        ),
        (
            1,
            2,
            2,
        ),
    ],
)
def test_metrics_rejects_wrong_confusion_matrix_shape(
    invalid_shape: tuple[int, ...],
) -> None:
    """
    [2, 2]가 아닌 Confusion Matrix를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "confusion_matrix"
    ] = torch.zeros(
        invalid_shape,
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "confusion_matrix must "
            "have shape"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        torch.int32,
        torch.float32,
        torch.float64,
        torch.bool,
    ],
)
def test_metrics_rejects_wrong_confusion_matrix_dtype(
    invalid_dtype: torch.dtype,
) -> None:
    """
    int64가 아닌 Confusion Matrix를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "confusion_matrix"
    ] = torch.tensor(
        [
            [
                2,
                1,
            ],
            [
                2,
                3,
            ],
        ],
        dtype=invalid_dtype,
    )

    with pytest.raises(
        TypeError,
        match=(
            "confusion_matrix must use "
            "dtype torch.int64"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


def test_metrics_rejects_negative_confusion_matrix_value() -> None:
    """
    음수 Matrix Count를 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "confusion_matrix"
    ] = torch.tensor(
        [
            [
                2,
                -1,
            ],
            [
                2,
                3,
            ],
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "must contain only "
            "non-negative counts"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


def test_metrics_rejects_confusion_matrix_not_matching_counts() -> None:
    """
    Matrix 값이 개별 TN·FP·FN·TP와 다르면 거부하는지 확인한다.
    """
    values = (
        create_valid_metric_values()
    )

    values[
        "confusion_matrix"
    ] = torch.tensor(
        [
            [
                1,
                2,
            ],
            [
                2,
                3,
            ],
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "confusion_matrix must match"
        ),
    ):
        BinaryClassificationMetrics(
            **values,  # type: ignore[arg-type]
        )


# =============================================================================
# Metric Calculation - Known Example
# =============================================================================


def test_calculate_known_binary_metrics() -> None:
    """
    알려진 Label·Prediction에서 모든 Metric을 확인한다.

    Labels:

        [0, 0, 0, 1, 1, 1, 1, 1]

    Predictions:

        [0, 1, 0, 1, 0, 1, 1, 0]

    TN:

        2

    FP:

        1

    FN:

        2

    TP:

        3
    """
    labels = torch.tensor(
        [
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            1,
            0,
            1,
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.sample_count
        == 8
    )

    assert (
        metrics.true_negative
        == 2
    )

    assert (
        metrics.false_positive
        == 1
    )

    assert (
        metrics.false_negative
        == 2
    )

    assert (
        metrics.true_positive
        == 3
    )

    assert metrics.accuracy == (
        pytest.approx(
            5 / 8
        )
    )

    assert metrics.precision == (
        pytest.approx(
            3 / 4
        )
    )

    assert metrics.recall == (
        pytest.approx(
            3 / 5
        )
    )

    assert metrics.f1_score == (
        pytest.approx(
            2 / 3
        )
    )

    assert torch.equal(
        metrics.confusion_matrix,
        torch.tensor(
            [
                [
                    2,
                    1,
                ],
                [
                    2,
                    3,
                ],
            ],
            dtype=torch.int64,
        ),
    )


# =============================================================================
# Metric Calculation - Common Scenarios
# =============================================================================


def test_calculate_all_predictions_correct() -> None:
    """
    모든 Prediction이 정답이면 모든 Metric이 1인지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    predictions = labels.clone()

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 3
    )

    assert (
        metrics.false_positive
        == 0
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 3
    )

    assert (
        metrics.accuracy
        == 1.0
    )

    assert (
        metrics.precision
        == 1.0
    )

    assert (
        metrics.recall
        == 1.0
    )

    assert (
        metrics.f1_score
        == 1.0
    )


def test_calculate_all_predictions_wrong() -> None:
    """
    NORMAL·DEFECT가 모두 반대로 예측되면 Accuracy·F1이 0인지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            0,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            1,
            1,
            0,
            0,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 0
    )

    assert (
        metrics.false_positive
        == 2
    )

    assert (
        metrics.false_negative
        == 2
    )

    assert (
        metrics.true_positive
        == 0
    )

    assert (
        metrics.accuracy
        == 0.0
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


def test_calculate_only_normal_samples_all_correct() -> None:
    """
    실제 NORMAL만 있고 모두 정상 예측한 경우를 확인한다.

    Positive Class가 존재하지 않으므로:

        Precision = 0

        Recall = 0

        F1 = 0

    Accuracy는:

        1
    """
    labels = torch.tensor(
        [
            0,
            0,
            0,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            0,
            0,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 3
    )

    assert (
        metrics.false_positive
        == 0
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 0
    )

    assert (
        metrics.accuracy
        == 1.0
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


def test_calculate_only_defect_samples_all_correct() -> None:
    """
    실제 DEFECT만 있고 모두 DEFECT로 예측한 경우를 확인한다.
    """
    labels = torch.tensor(
        [
            1,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            1,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 0
    )

    assert (
        metrics.false_positive
        == 0
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 3
    )

    assert (
        metrics.accuracy
        == 1.0
    )

    assert (
        metrics.precision
        == 1.0
    )

    assert (
        metrics.recall
        == 1.0
    )

    assert (
        metrics.f1_score
        == 1.0
    )


def test_calculate_no_positive_predictions() -> None:
    """
    Model이 DEFECT를 한 번도 예측하지 않은 경우를 확인한다.

    Precision 분모:

        TP + FP

        0

    정책:

        Precision = 0.0
    """
    labels = torch.tensor(
        [
            0,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            0,
            0,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 1
    )

    assert (
        metrics.false_positive
        == 0
    )

    assert (
        metrics.false_negative
        == 2
    )

    assert (
        metrics.true_positive
        == 0
    )

    assert metrics.accuracy == (
        pytest.approx(
            1 / 3
        )
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


def test_calculate_no_positive_labels_with_positive_predictions() -> None:
    """
    실제 DEFECT는 없지만 모두 DEFECT로 오판한 경우를 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            0,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            1,
            1,
        ],
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 0
    )

    assert (
        metrics.false_positive
        == 2
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 0
    )

    assert (
        metrics.accuracy
        == 0.0
    )

    assert (
        metrics.precision
        == 0.0
    )

    assert (
        metrics.recall
        == 0.0
    )

    assert (
        metrics.f1_score
        == 0.0
    )


def test_calculate_all_predictions_positive() -> None:
    """
    모든 Sample을 DEFECT로 예측한 경우를 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            0,
            1,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.ones(
        5,
        dtype=torch.int64,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.true_negative
        == 0
    )

    assert (
        metrics.false_positive
        == 2
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 3
    )

    assert metrics.accuracy == (
        pytest.approx(
            3 / 5
        )
    )

    assert metrics.precision == (
        pytest.approx(
            3 / 5
        )
    )

    assert (
        metrics.recall
        == 1.0
    )

    assert metrics.f1_score == (
        pytest.approx(
            0.75
        )
    )


@pytest.mark.parametrize(
    (
        "label",
        "prediction",
        "expected_counts",
        "expected_accuracy",
        "expected_precision",
        "expected_recall",
        "expected_f1",
    ),
    [
        (
            0,
            0,
            (
                1,
                0,
                0,
                0,
            ),
            1.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            0,
            1,
            (
                0,
                1,
                0,
                0,
            ),
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1,
            0,
            (
                0,
                0,
                1,
                0,
            ),
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        (
            1,
            1,
            (
                0,
                0,
                0,
                1,
            ),
            1.0,
            1.0,
            1.0,
            1.0,
        ),
    ],
)
def test_calculate_each_single_sample_case(
    label: int,
    prediction: int,
    expected_counts: tuple[
        int,
        int,
        int,
        int,
    ],
    expected_accuracy: float,
    expected_precision: float,
    expected_recall: float,
    expected_f1: float,
) -> None:
    """
    가능한 단일 Sample 분류 결과 네 가지를 모두 검증한다.
    """
    metrics = (
        calculate_binary_classification_metrics(
            labels=torch.tensor(
                [
                    label,
                ],
                dtype=torch.int64,
            ),
            predictions=torch.tensor(
                [
                    prediction,
                ],
                dtype=torch.int64,
            ),
        )
    )

    (
        expected_tn,
        expected_fp,
        expected_fn,
        expected_tp,
    ) = expected_counts

    assert (
        metrics.true_negative
        == expected_tn
    )

    assert (
        metrics.false_positive
        == expected_fp
    )

    assert (
        metrics.false_negative
        == expected_fn
    )

    assert (
        metrics.true_positive
        == expected_tp
    )

    assert (
        metrics.accuracy
        == expected_accuracy
    )

    assert (
        metrics.precision
        == expected_precision
    )

    assert (
        metrics.recall
        == expected_recall
    )

    assert (
        metrics.f1_score
        == expected_f1
    )


# =============================================================================
# Metric Calculation - Tensor Normalization
# =============================================================================


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
def test_calculate_accepts_supported_integer_dtypes(
    integer_dtype: torch.dtype,
) -> None:
    """
    지원 Integer Dtype 입력을 모두 허용하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=integer_dtype,
    )

    predictions = torch.tensor(
        [
            0,
            1,
            1,
            1,
        ],
        dtype=integer_dtype,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.sample_count
        == 4
    )

    assert (
        metrics.true_negative
        == 1
    )

    assert (
        metrics.false_positive
        == 1
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 2
    )


def test_calculate_accepts_different_supported_integer_dtypes() -> None:
    """
    Label·Prediction Dtype이 서로 달라도 지원 Integer이면 허용하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            1,
        ],
        dtype=torch.int16,
    )

    predictions = torch.tensor(
        [
            0,
            0,
            1,
        ],
        dtype=torch.uint8,
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.sample_count
        == 3
    )

    assert (
        metrics.true_negative
        == 1
    )

    assert (
        metrics.false_negative
        == 1
    )

    assert (
        metrics.true_positive
        == 1
    )


def test_calculate_accepts_non_contiguous_inputs() -> None:
    """
    비연속 Label·Prediction Tensor를 허용하는지 확인한다.
    """
    label_storage = torch.tensor(
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
    )

    prediction_storage = torch.tensor(
        [
            0,
            9,
            1,
            9,
            1,
            9,
            1,
            9,
        ],
        dtype=torch.int64,
    )

    labels = label_storage[
        ::2
    ]

    predictions = (
        prediction_storage[
            ::2
        ]
    )

    assert (
        labels.is_contiguous()
        is False
    )

    assert (
        predictions.is_contiguous()
        is False
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert (
        metrics.sample_count
        == 4
    )

    assert (
        metrics.true_negative
        == 1
    )

    assert (
        metrics.false_positive
        == 1
    )

    assert (
        metrics.false_negative
        == 0
    )

    assert (
        metrics.true_positive
        == 2
    )


def test_calculate_does_not_modify_input_tensors() -> None:
    """
    Metric 계산이 입력 Label·Prediction Tensor를 변경하지 않는지 확인한다.
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

    predictions = torch.tensor(
        [
            0,
            1,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    labels_before = (
        labels.clone()
    )

    predictions_before = (
        predictions.clone()
    )

    _ = (
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )
    )

    assert torch.equal(
        labels,
        labels_before,
    )

    assert torch.equal(
        predictions,
        predictions_before,
    )


def test_calculate_returns_cpu_int64_confusion_matrix() -> None:
    """
    계산 결과 Confusion Matrix가 CPU int64인지 확인한다.
    """
    metrics = (
        calculate_binary_classification_metrics(
            labels=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.int32,
            ),
            predictions=torch.tensor(
                [
                    0,
                    1,
                ],
                dtype=torch.int16,
            ),
        )
    )

    assert (
        metrics.confusion_matrix
        .device
        .type
        == "cpu"
    )

    assert (
        metrics.confusion_matrix
        .dtype
        == torch.int64
    )

    assert (
        metrics.confusion_matrix
        .shape
        == (
            2,
            2,
        )
    )


# =============================================================================
# Metric Calculation - Invalid Tensor Type
# =============================================================================


@pytest.mark.parametrize(
    "invalid_labels",
    [
        None,
        [],
        "labels",
        123,
        object(),
    ],
)
def test_calculate_rejects_non_tensor_labels(
    invalid_labels: object,
) -> None:
    """
    Tensor가 아닌 Label을 거부하는지 확인한다.
    """
    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match=(
            "labels must be "
            "a torch.Tensor"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=invalid_labels,  # type: ignore[arg-type]
            predictions=predictions,
        )


@pytest.mark.parametrize(
    "invalid_predictions",
    [
        None,
        [],
        "predictions",
        123,
        object(),
    ],
)
def test_calculate_rejects_non_tensor_predictions(
    invalid_predictions: object,
) -> None:
    """
    Tensor가 아닌 Prediction을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match=(
            "predictions must be "
            "a torch.Tensor"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=invalid_predictions,  # type: ignore[arg-type]
        )


# =============================================================================
# Metric Calculation - Invalid Tensor Shape
# =============================================================================


@pytest.mark.parametrize(
    "invalid_labels",
    [
        torch.tensor(
            1,
            dtype=torch.int64,
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
        torch.zeros(
            1,
            1,
            2,
            dtype=torch.int64,
        ),
    ],
)
def test_calculate_rejects_non_one_dimensional_labels(
    invalid_labels: Tensor,
) -> None:
    """
    1차원이 아닌 Label Tensor를 거부하는지 확인한다.
    """
    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "labels must be "
            "one-dimensional"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=invalid_labels,
            predictions=predictions,
        )


@pytest.mark.parametrize(
    "invalid_predictions",
    [
        torch.tensor(
            1,
            dtype=torch.int64,
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
        torch.zeros(
            1,
            1,
            2,
            dtype=torch.int64,
        ),
    ],
)
def test_calculate_rejects_non_one_dimensional_predictions(
    invalid_predictions: Tensor,
) -> None:
    """
    1차원이 아닌 Prediction Tensor를 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "predictions must be "
            "one-dimensional"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=invalid_predictions,
        )


def test_calculate_rejects_empty_labels() -> None:
    """
    빈 Label Tensor를 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "labels must not be empty"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=torch.empty(
                0,
                dtype=torch.int64,
            ),
            predictions=torch.empty(
                0,
                dtype=torch.int64,
            ),
        )


def test_calculate_rejects_empty_predictions() -> None:
    """
    Label은 존재하지만 Prediction이 비어 있으면 거부하는지 확인한다.
    """
    with pytest.raises(
        ValueError,
        match=(
            "predictions must not "
            "be empty"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=torch.tensor(
                [
                    0,
                ],
                dtype=torch.int64,
            ),
            predictions=torch.empty(
                0,
                dtype=torch.int64,
            ),
        )


def test_calculate_rejects_label_prediction_length_mismatch() -> None:
    """
    Label·Prediction Sample 수가 다르면 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            0,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "labels and predictions "
            "must contain the same "
            "number of samples"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


# =============================================================================
# Metric Calculation - Invalid Tensor Dtype
# =============================================================================


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_calculate_rejects_non_integer_label_dtype(
    invalid_dtype: torch.dtype,
) -> None:
    """
    지원 Integer가 아닌 Label Dtype을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=invalid_dtype,
    )

    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match=(
            "labels must use "
            "a supported integer dtype"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_calculate_rejects_non_integer_prediction_dtype(
    invalid_dtype: torch.dtype,
) -> None:
    """
    지원 Integer가 아닌 Prediction Dtype을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=invalid_dtype,
    )

    with pytest.raises(
        TypeError,
        match=(
            "predictions must use "
            "a supported integer dtype"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


def test_calculate_rejects_bool_labels() -> None:
    """
    bool Label Tensor를 Integer Label로 허용하지 않는지 확인한다.
    """
    labels = torch.tensor(
        [
            False,
            True,
        ],
        dtype=torch.bool,
    )

    predictions = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        TypeError,
        match=(
            "labels must use an "
            "integer dtype, not torch.bool"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


def test_calculate_rejects_bool_predictions() -> None:
    """
    bool Prediction Tensor를 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            False,
            True,
        ],
        dtype=torch.bool,
    )

    with pytest.raises(
        TypeError,
        match=(
            "predictions must use an "
            "integer dtype, not torch.bool"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


# =============================================================================
# Metric Calculation - Invalid Binary Values
# =============================================================================


@pytest.mark.parametrize(
    "invalid_label",
    [
        -3,
        -1,
        2,
        3,
        99,
    ],
)
def test_calculate_rejects_non_binary_label_values(
    invalid_label: int,
) -> None:
    """
    0·1 이외 Label 값을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            invalid_label,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "labels must contain only "
            "binary values"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )


@pytest.mark.parametrize(
    "invalid_prediction",
    [
        -3,
        -1,
        2,
        3,
        99,
    ],
)
def test_calculate_rejects_non_binary_prediction_values(
    invalid_prediction: int,
) -> None:
    """
    0·1 이외 Prediction 값을 거부하는지 확인한다.
    """
    labels = torch.tensor(
        [
            0,
            1,
            1,
        ],
        dtype=torch.int64,
    )

    predictions = torch.tensor(
        [
            0,
            invalid_prediction,
            1,
        ],
        dtype=torch.int64,
    )

    with pytest.raises(
        ValueError,
        match=(
            "predictions must contain only "
            "binary values"
        ),
    ):
        calculate_binary_classification_metrics(
            labels=labels,
            predictions=predictions,
        )