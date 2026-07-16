"""
Binary image classification evaluation runner.

이 모듈의 역할
---------------
Manufacturing Vision Defect Analysis System의 Binary Image Classifier를
Evaluation Mode로 실행하고 전체 Dataset의 예측 결과를 수집한다.

현재 Model
----------
CNNBaseline

향후 재사용 대상
----------------
ResNet18 Transfer Learning Model

현재 Class 정의
--------------
0:

    NORMAL

1:

    DEFECT

Positive Class:

    DEFECT

현재 평가 입력
--------------
Best CNN Checkpoint가 복원된 Model

Test DataLoader

BCEWithLogitsLoss

CPU Device

Classification Threshold:

    0.5

전체 평가 흐름
-------------
Best Model

-> model.eval()

-> torch.inference_mode()

-> Test Batch

-> Images Device 이동

-> Labels Binary Target 변환

-> Model Forward

-> Raw Logits

-> BCEWithLogitsLoss

-> Sigmoid

-> DEFECT Probabilities

-> Classification Threshold

-> Binary Predictions

-> 전체 결과 CPU 수집

-> BinaryEvaluationResult

현재 반환 결과
--------------
Average Loss

Accuracy

Sample Count

Batch Count

Classification Threshold

Ground Truth Labels

Raw Logits

DEFECT Probabilities

Binary Predictions

아직 이 모듈에서 계산하지 않는 Metric
------------------------------------
Precision

Recall

F1 Score

Confusion Matrix

위 Metric은 별도의 classification_metrics.py에서 계산한다.

중요
----
이 함수는 Evaluation 시작 시 model.eval()을 호출한다.

따라서 함수 실행 후 Model은 Evaluation Mode를 유지한다.

Model Parameter는 변경하지 않는다.

Gradient도 생성하지 않는다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.training.epoch_runner import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
)
from src.training.loss_function import (
    prepare_binary_targets,
)


# =============================================================================
# Binary Evaluation Result
# =============================================================================


@dataclass(frozen=True)
class BinaryEvaluationResult:
    """
    Binary Image Classification 전체 평가 결과.

    왜 필요한가
    -----------
    Test Dataset 평가 후 다음 작업을 수행해야 한다.

        Accuracy 확인

        Precision 계산

        Recall 계산

        F1 계산

        Confusion Matrix 생성

        오분류 Sample 확인

        Confidence 분석

    따라서 집계 Metric뿐 아니라 Sample별 예측 결과도 저장한다.

    필드
    ----
    average_loss:
        전체 Dataset의 Sample 가중 평균 Loss

    accuracy:
        전체 Dataset Accuracy

        범위:

            0.0 ~ 1.0

    sample_count:
        실제 평가한 전체 Sample 수

        현재 실제 Test:

            715

    batch_count:
        실제 처리한 Batch 수

        현재 실제 Test:

            23

    classification_threshold:
        Probability를 Binary Class로 변환한 Threshold

        현재:

            0.5

    labels:
        Ground Truth Label

        Shape:

            [sample_count]

        Dtype:

            torch.int64

        Device:

            cpu

        값:

            0 또는 1

    logits:
        Model Raw Output

        Shape:

            [sample_count]

        Dtype:

            torch.float32

        Device:

            cpu

    probabilities:
        DEFECT Class Probability

        계산:

            sigmoid(logits)

        Shape:

            [sample_count]

        Dtype:

            torch.float32

        범위:

            0.0 ~ 1.0

    predictions:
        Threshold 기반 Binary Prediction

        계산:

            probabilities
            >=
            classification_threshold

        Shape:

            [sample_count]

        Dtype:

            torch.int64

        값:

            0 또는 1

    내부 일관성
    -----------
    다음 관계를 검증한다.

        len(labels)

        ==

        len(logits)

        ==

        len(probabilities)

        ==

        len(predictions)

        ==

        sample_count

    또한:

        sigmoid(logits)

        ≈

        probabilities

    그리고:

        probabilities >= threshold

        ==

        predictions

    마지막으로:

        mean(predictions == labels)

        ≈

        accuracy
    """

    average_loss: float

    accuracy: float

    sample_count: int

    batch_count: int

    classification_threshold: float

    labels: Tensor

    logits: Tensor

    probabilities: Tensor

    predictions: Tensor

    def __post_init__(self) -> None:
        """
        Evaluation Result의 Scalar·Tensor·내부 관계를 검증한다.

        Tensor는 다음 형태로 정규화한다.

            Gradient 분리

            CPU 이동

            Contiguous

            Clone

        이유
        ----
        외부 Tensor Storage와 결과 객체가 연결되지 않도록 한다.

        또한 Metric·JSON·시각화 단계에서
        CPU Tensor를 일관되게 사용한다.
        """
        validated_average_loss = (
            _validate_non_negative_finite_real(
                value=self.average_loss,
                value_name="average_loss",
            )
        )

        validated_accuracy = (
            _validate_probability(
                value=self.accuracy,
                value_name="accuracy",
            )
        )

        validated_sample_count = (
            _validate_positive_integer(
                value=self.sample_count,
                value_name="sample_count",
            )
        )

        validated_batch_count = (
            _validate_positive_integer(
                value=self.batch_count,
                value_name="batch_count",
            )
        )

        if (
            validated_batch_count
            > validated_sample_count
        ):
            raise ValueError(
                "batch_count must be less than or equal to "
                "sample_count. "
                f"Batch count: {validated_batch_count}. "
                f"Sample count: {validated_sample_count}."
            )

        validated_threshold = (
            _validate_probability(
                value=(
                    self
                    .classification_threshold
                ),
                value_name=(
                    "classification_threshold"
                ),
            )
        )

        normalized_labels = (
            _normalize_result_tensor(
                tensor=self.labels,
                tensor_name="labels",
                expected_dtype=(
                    torch.int64
                ),
            )
        )

        normalized_logits = (
            _normalize_result_tensor(
                tensor=self.logits,
                tensor_name="logits",
                expected_dtype=(
                    torch.float32
                ),
            )
        )

        normalized_probabilities = (
            _normalize_result_tensor(
                tensor=self.probabilities,
                tensor_name=(
                    "probabilities"
                ),
                expected_dtype=(
                    torch.float32
                ),
            )
        )

        normalized_predictions = (
            _normalize_result_tensor(
                tensor=self.predictions,
                tensor_name="predictions",
                expected_dtype=(
                    torch.int64
                ),
            )
        )

        # Frozen Dataclass지만 __post_init__ 내부에서는
        # object.__setattr__()로 정규화된 값을 저장할 수 있다.
        object.__setattr__(
            self,
            "average_loss",
            validated_average_loss,
        )

        object.__setattr__(
            self,
            "accuracy",
            validated_accuracy,
        )

        object.__setattr__(
            self,
            "sample_count",
            validated_sample_count,
        )

        object.__setattr__(
            self,
            "batch_count",
            validated_batch_count,
        )

        object.__setattr__(
            self,
            "classification_threshold",
            validated_threshold,
        )

        object.__setattr__(
            self,
            "labels",
            normalized_labels,
        )

        object.__setattr__(
            self,
            "logits",
            normalized_logits,
        )

        object.__setattr__(
            self,
            "probabilities",
            normalized_probabilities,
        )

        object.__setattr__(
            self,
            "predictions",
            normalized_predictions,
        )

        # ------------------------------------------------------
        # Tensor Length
        # ------------------------------------------------------
        result_lengths = {
            "labels": (
                normalized_labels.numel()
            ),
            "logits": (
                normalized_logits.numel()
            ),
            "probabilities": (
                normalized_probabilities
                .numel()
            ),
            "predictions": (
                normalized_predictions
                .numel()
            ),
        }

        if (
            len(
                set(
                    result_lengths.values()
                )
            )
            != 1
        ):
            raise ValueError(
                "labels, logits, probabilities, and predictions "
                "must contain the same number of elements. "
                f"Received lengths: {result_lengths}."
            )

        if (
            normalized_labels.numel()
            != validated_sample_count
        ):
            raise ValueError(
                "result Tensor length must match sample_count. "
                f"Tensor length: "
                f"{normalized_labels.numel()}. "
                f"Sample count: "
                f"{validated_sample_count}."
            )

        # ------------------------------------------------------
        # Binary Values
        # ------------------------------------------------------
        _validate_binary_integer_tensor(
            tensor=normalized_labels,
            tensor_name="labels",
        )

        _validate_binary_integer_tensor(
            tensor=normalized_predictions,
            tensor_name="predictions",
        )

        # ------------------------------------------------------
        # Finite Values
        # ------------------------------------------------------
        _validate_finite_tensor(
            tensor=normalized_logits,
            tensor_name="logits",
        )

        _validate_finite_tensor(
            tensor=(
                normalized_probabilities
            ),
            tensor_name="probabilities",
        )

        # ------------------------------------------------------
        # Probability Range
        # ------------------------------------------------------
        if (
            normalized_probabilities.min().item()
            < 0.0
            or
            normalized_probabilities.max().item()
            > 1.0
        ):
            raise ValueError(
                "probabilities must contain values "
                "between 0 and 1."
            )

        # ------------------------------------------------------
        # Logit -> Probability
        # ------------------------------------------------------
        expected_probabilities = (
            torch.sigmoid(
                normalized_logits
            )
        )

        if not torch.allclose(
            normalized_probabilities,
            expected_probabilities,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError(
                "probabilities must match sigmoid(logits)."
            )

        # ------------------------------------------------------
        # Probability -> Prediction
        # ------------------------------------------------------
        expected_predictions = (
            normalized_probabilities
            >= validated_threshold
        ).to(
            dtype=torch.int64
        )

        if not torch.equal(
            normalized_predictions,
            expected_predictions,
        ):
            raise ValueError(
                "predictions must match probabilities "
                "and classification_threshold."
            )

        # ------------------------------------------------------
        # Prediction -> Accuracy
        # ------------------------------------------------------
        expected_accuracy = (
            normalized_predictions
            .eq(
                normalized_labels
            )
            .to(
                dtype=torch.float64
            )
            .mean()
            .item()
        )

        if not math.isclose(
            validated_accuracy,
            expected_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "accuracy must match labels and predictions. "
                f"Expected: {expected_accuracy}. "
                f"Received: {validated_accuracy}."
            )


# =============================================================================
# Public Evaluation Function
# =============================================================================


def evaluate_binary_classifier(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device | str,
    classification_threshold: float = (
        DEFAULT_CLASSIFICATION_THRESHOLD
    ),
) -> BinaryEvaluationResult:
    """
    Binary Image Classifier를 전체 Dataset에서 평가한다.

    왜 필요한가
    -----------
    Validation Epoch 함수는 학습 중 Best Model을 선택하기 위한
    Loss·Accuracy 집계에 집중한다.

    최종 Test 평가에서는 다음 정보도 필요하다.

        모든 Ground Truth Label

        모든 Raw Logit

        모든 DEFECT Probability

        모든 Binary Prediction

    이 결과를 사용해 다음 단계에서 계산한다.

        Precision

        Recall

        F1 Score

        Confusion Matrix

        오분류 분석

    입력
    ----
    model:
        평가할 PyTorch Model

        현재:

            Best Epoch Weight가 복원된 CNNBaseline

    data_loader:
        평가 Dataset DataLoader

        현재:

            Test DataLoader

            715 Samples

            23 Batches

            shuffle=False

    loss_function:
        Binary Classification Loss

        현재:

            BCEWithLogitsLoss

    device:
        Model·Batch Device

        현재:

            cpu

    classification_threshold:
        DEFECT Probability를 Class로 변환할 Threshold

        현재:

            0.5

        Checkpoint Metadata의 값을 전달한다.

    처리 과정
    ---------
    1. 입력 객체를 검증한다.
    2. Device를 torch.device로 정규화한다.
    3. Threshold를 검증한다.
    4. Model State와 Device를 검증한다.
    5. model.eval()을 호출한다.
    6. torch.inference_mode()에 진입한다.
    7. 모든 Batch를 반복한다.
    8. Images를 float32·Device로 이동한다.
    9. Labels를 Binary Target으로 변환한다.
    10. Model Forward로 Raw Logit을 계산한다.
    11. BCE Loss를 계산한다.
    12. Sigmoid로 DEFECT Probability를 계산한다.
    13. Threshold로 Prediction을 계산한다.
    14. Sample 가중 Loss를 누적한다.
    15. 정답 수를 누적한다.
    16. Label·Logit·Probability·Prediction을 CPU에 수집한다.
    17. 전체 Tensor를 연결한다.
    18. BinaryEvaluationResult를 반환한다.

    출력
    ----
    BinaryEvaluationResult

    Loss 계산
    ---------
    BCEWithLogitsLoss의 Batch 평균을 단순 평균하지 않는다.

    현재 마지막 Batch는 다른 Batch보다 작을 수 있다.

    따라서:

        total_loss

        +=

        batch_loss

        *

        batch_size

    마지막:

        average_loss

        =

        total_loss

        /

        total_sample_count

    Sample 가중 평균을 사용한다.

    Accuracy 계산
    -------------
    Probability:

        sigmoid(logits)

    Prediction:

        probability >= threshold

    Accuracy:

        correct_prediction_count

        /

        total_sample_count

    Gradient
    --------
    다음을 사용한다.

        torch.inference_mode()

    따라서:

        Gradient 계산 안 함

        Parameter Gradient 생성 안 함

        Model Parameter 변경 안 함

    Model Mode
    ----------
    평가 시작 시:

        model.eval()

    함수 종료 후에도:

        model.training == False

    를 유지한다.

    Result Device
    -------------
    모든 결과 Tensor는 CPU로 반환한다.

        labels:

            cpu

        logits:

            cpu

        probabilities:

            cpu

        predictions:

            cpu

    Dataset 순서
    ------------
    DataLoader가 shuffle=False이면
    반환 Tensor 순서는 Dataset 순서와 같다.

    향후:

        result.predictions[0]

    은:

        test_dataset.samples[0]

    과 연결할 수 있다.

    예외 처리
    ---------
    잘못된 Model:

        TypeError

    잘못된 DataLoader:

        TypeError

    잘못된 Loss:

        TypeError

    잘못된 Device:

        TypeError 또는 ValueError

    Model State 없음:

        ValueError

    Model Device 불일치:

        ValueError

    잘못된 Threshold:

        TypeError 또는 ValueError

    잘못된 Batch:

        TypeError 또는 ValueError

    잘못된 Image:

        TypeError 또는 ValueError

    잘못된 Label:

        TypeError 또는 ValueError

    잘못된 Logit:

        TypeError 또는 ValueError

    잘못된 Loss:

        TypeError 또는 ValueError

    빈 DataLoader:

        ValueError

    테스트 방법
    -----------
    작은 Dummy Dataset:

        6 Samples

        2 Batches

    확인:

        Model Evaluation Mode

        Gradient 비활성화

        Parameter 변경 없음

        Sample Count

        Batch Count

        Loss

        Accuracy

        Label Shape

        Logit Shape

        Probability Shape

        Prediction Shape

        Tensor Device

        Tensor Dtype

        내부 값 일관성
    """
    _validate_evaluation_objects(
        model=model,
        data_loader=data_loader,
        loss_function=loss_function,
    )

    resolved_device = _resolve_device(
        device=device,
    )

    validated_threshold = (
        _validate_probability(
            value=(
                classification_threshold
            ),
            value_name=(
                "classification_threshold"
            ),
        )
    )

    _validate_model_state_exists(
        model=model,
    )

    _validate_model_device(
        model=model,
        device=resolved_device,
    )

    # Evaluation Mode:
    #
    # Dropout:
    #
    #     비활성화
    #
    # BatchNorm:
    #
    #     저장된 Running Statistics 사용
    model.eval()

    total_loss = 0.0

    total_correct_count = 0

    total_sample_count = 0

    total_batch_count = 0

    collected_labels: list[
        Tensor
    ] = []

    collected_logits: list[
        Tensor
    ] = []

    collected_probabilities: list[
        Tensor
    ] = []

    collected_predictions: list[
        Tensor
    ] = []

    # Gradient 추적을 완전히 비활성화한다.
    with torch.inference_mode():
        for batch in data_loader:
            (
                images,
                labels,
            ) = _unpack_and_validate_batch(
                batch=batch,
            )

            images = images.to(
                device=resolved_device,
                dtype=torch.float32,
                non_blocking=True,
            )

            labels = labels.to(
                device=resolved_device,
                non_blocking=True,
            )

            # 기존 Loss 모듈의 Binary Label 검증을 재사용한다.
            targets = prepare_binary_targets(
                labels=labels,
                device=resolved_device,
            )

            integer_labels = targets.to(
                dtype=torch.int64
            )

            logits = model(
                images
            )

            _validate_logits(
                logits=logits,
                expected_batch_size=(
                    images.shape[0]
                ),
                device=resolved_device,
            )

            loss = loss_function(
                logits,
                targets,
            )

            _validate_loss(
                loss=loss,
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities
                >= validated_threshold
            ).to(
                dtype=torch.int64
            )

            batch_size = int(
                images.shape[0]
            )

            batch_correct_count = int(
                predictions
                .eq(
                    integer_labels
                )
                .sum()
                .item()
            )

            # BCEWithLogitsLoss(reduction="mean")의
            # Batch 평균 Loss를 Sample 수만큼 가중한다.
            total_loss += (
                float(
                    loss.item()
                )
                * batch_size
            )

            total_correct_count += (
                batch_correct_count
            )

            total_sample_count += (
                batch_size
            )

            total_batch_count += 1

            # Metric·시각화·오분류 분석을 위해
            # 모든 결과를 CPU Tensor로 수집한다.
            collected_labels.append(
                integer_labels
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.int64,
                )
                .contiguous()
                .clone()
            )

            collected_logits.append(
                logits
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
                .clone()
            )

            collected_probabilities.append(
                probabilities
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.float32,
                )
                .contiguous()
                .clone()
            )

            collected_predictions.append(
                predictions
                .detach()
                .to(
                    device="cpu",
                    dtype=torch.int64,
                )
                .contiguous()
                .clone()
            )

    if (
        total_sample_count
        <= 0
    ):
        raise ValueError(
            "data_loader must contain at least one sample."
        )

    if (
        total_batch_count
        <= 0
    ):
        raise ValueError(
            "data_loader must contain at least one batch."
        )

    average_loss = (
        total_loss
        / total_sample_count
    )

    accuracy = (
        total_correct_count
        / total_sample_count
    )

    all_labels = torch.cat(
        collected_labels,
        dim=0,
    )

    all_logits = torch.cat(
        collected_logits,
        dim=0,
    )

    all_probabilities = torch.cat(
        collected_probabilities,
        dim=0,
    )

    all_predictions = torch.cat(
        collected_predictions,
        dim=0,
    )

    return BinaryEvaluationResult(
        average_loss=(
            average_loss
        ),
        accuracy=(
            accuracy
        ),
        sample_count=(
            total_sample_count
        ),
        batch_count=(
            total_batch_count
        ),
        classification_threshold=(
            validated_threshold
        ),
        labels=all_labels,
        logits=all_logits,
        probabilities=(
            all_probabilities
        ),
        predictions=(
            all_predictions
        ),
    )


# =============================================================================
# Basic Scalar Validation
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

    Classification Threshold
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
# Result Tensor Validation
# =============================================================================


def _normalize_result_tensor(
    tensor: Tensor,
    tensor_name: str,
    expected_dtype: torch.dtype,
) -> Tensor:
    """
    Evaluation Result Tensor를 검증하고 독립적인 CPU Tensor로 변환한다.

    검증
    ----
    torch.Tensor

    1차원

    비어 있지 않음

    기대 Dtype

    Gradient 없음

    출력
    ----
    Detached

    CPU

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

    if (
        tensor.dtype
        != expected_dtype
    ):
        raise TypeError(
            f"{tensor_name} must use dtype "
            f"{expected_dtype}. "
            f"Received dtype: "
            f"{tensor.dtype}."
        )

    return (
        tensor
        .detach()
        .to(
            device="cpu",
            dtype=expected_dtype,
        )
        .contiguous()
        .clone()
    )


def _validate_binary_integer_tensor(
    tensor: Tensor,
    tensor_name: str,
) -> None:
    """
    int64 Tensor 값이 0 또는 1인지 확인한다.
    """
    unique_values = torch.unique(
        tensor
    )

    valid_values = {
        0,
        1,
    }

    actual_values = {
        int(
            value.item()
        )
        for value in (
            unique_values
        )
    }

    invalid_values = (
        actual_values
        - valid_values
    )

    if invalid_values:
        raise ValueError(
            f"{tensor_name} must contain only "
            "binary values 0 and 1. "
            f"Invalid values: "
            f"{sorted(invalid_values)}."
        )


def _validate_finite_tensor(
    tensor: Tensor,
    tensor_name: str,
) -> None:
    """
    Tensor에 NaN·positive infinity·negative infinity가 없는지 확인한다.
    """
    if not torch.isfinite(
        tensor
    ).all():
        raise ValueError(
            f"{tensor_name} must contain "
            "only finite values."
        )


# =============================================================================
# Evaluation Object Validation
# =============================================================================


def _validate_evaluation_objects(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
) -> None:
    """
    Evaluation에 필요한 PyTorch 객체 타입을 검증한다.
    """
    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be an instance of torch.nn.Module. "
            f"Received type: "
            f"{type(model).__name__}."
        )

    if not isinstance(
        data_loader,
        DataLoader,
    ):
        raise TypeError(
            "data_loader must be an instance of "
            "torch.utils.data.DataLoader. "
            f"Received type: "
            f"{type(data_loader).__name__}."
        )

    if not isinstance(
        loss_function,
        nn.Module,
    ):
        raise TypeError(
            "loss_function must be an instance of "
            "torch.nn.Module. "
            f"Received type: "
            f"{type(loss_function).__name__}."
        )


# =============================================================================
# Model Validation
# =============================================================================


def _validate_model_state_exists(
    model: nn.Module,
) -> None:
    """
    Model에 Parameter 또는 Buffer State가 존재하는지 확인한다.
    """
    if not model.state_dict():
        raise ValueError(
            "model must contain at least one "
            "Parameter or Buffer State."
        )


def _validate_model_device(
    model: nn.Module,
    device: torch.device,
) -> None:
    """
    모든 Model Parameter·Buffer가 요청 Device에 있는지 확인한다.
    """
    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():
        if parameter.device != device:
            raise ValueError(
                "all Model Parameters must be on "
                "the requested device. "
                f"Parameter: {parameter_name}. "
                f"Requested: {device}. "
                f"Found: {parameter.device}."
            )

    for (
        buffer_name,
        buffer,
    ) in model.named_buffers():
        if buffer.device != device:
            raise ValueError(
                "all Model Buffers must be on "
                "the requested device. "
                f"Buffer: {buffer_name}. "
                f"Requested: {device}. "
                f"Found: {buffer.device}."
            )


# =============================================================================
# Device Validation
# =============================================================================


def _resolve_device(
    device: torch.device | str,
) -> torch.device:
    """
    문자열 또는 torch.device를 torch.device로 정규화한다.
    """
    if not isinstance(
        device,
        (
            str,
            torch.device,
        ),
    ):
        raise TypeError(
            "device must be a string or torch.device. "
            f"Received type: "
            f"{type(device).__name__}."
        )

    try:
        resolved_device = torch.device(
            device
        )

    except (
        RuntimeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Invalid device: {device}."
        ) from error

    if (
        resolved_device.type
        == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError(
            "CUDA device was requested, "
            "but CUDA is not available."
        )

    return resolved_device


# =============================================================================
# Batch Validation
# =============================================================================


def _unpack_and_validate_batch(
    batch: object,
) -> tuple[
    Tensor,
    Tensor,
]:
    """
    DataLoader Batch를 Images·Labels로 분리하고 검증한다.

    정상 Batch
    ----------
    Tuple 또는 List

    길이:

        2

    첫 번째:

        Images Tensor

    두 번째:

        Labels Tensor

    Image
    -----
    Shape:

        [batch_size, 3, height, width]

    Dtype:

        Floating Point

    값:

        Finite

    Label
    -----
    Shape:

        [batch_size]

    Binary 값 검증은
    prepare_binary_targets()가 수행한다.
    """
    if not isinstance(
        batch,
        (
            tuple,
            list,
        ),
    ):
        raise TypeError(
            "each data_loader batch must be "
            "a tuple or list containing "
            "(images, labels). "
            f"Received type: "
            f"{type(batch).__name__}."
        )

    if len(batch) != 2:
        raise ValueError(
            "each data_loader batch must contain "
            "exactly two items: images and labels. "
            f"Received item count: "
            f"{len(batch)}."
        )

    images = batch[0]

    labels = batch[1]

    if not isinstance(
        images,
        Tensor,
    ):
        raise TypeError(
            "batch images must be a torch.Tensor. "
            f"Received type: "
            f"{type(images).__name__}."
        )

    if not isinstance(
        labels,
        Tensor,
    ):
        raise TypeError(
            "batch labels must be a torch.Tensor. "
            f"Received type: "
            f"{type(labels).__name__}."
        )

    if images.ndim != 4:
        raise ValueError(
            "batch images must be four-dimensional. "
            "Expected shape: "
            "[batch_size, 3, height, width]. "
            f"Received shape: "
            f"{tuple(images.shape)}."
        )

    if images.shape[0] <= 0:
        raise ValueError(
            "batch images must contain "
            "at least one sample."
        )

    if images.shape[1] != 3:
        raise ValueError(
            "batch images must contain exactly "
            "three RGB channels. "
            f"Received channel count: "
            f"{images.shape[1]}."
        )

    if (
        images.shape[2] <= 0
        or images.shape[3] <= 0
    ):
        raise ValueError(
            "batch image height and width "
            "must be greater than 0. "
            f"Received shape: "
            f"{tuple(images.shape)}."
        )

    if not images.is_floating_point():
        raise TypeError(
            "batch images must use "
            "a floating-point dtype. "
            f"Received dtype: "
            f"{images.dtype}."
        )

    if not torch.isfinite(
        images
    ).all():
        raise ValueError(
            "batch images must contain "
            "only finite values."
        )

    if labels.ndim != 1:
        raise ValueError(
            "batch labels must be one-dimensional. "
            "Expected shape: [batch_size]. "
            f"Received shape: "
            f"{tuple(labels.shape)}."
        )

    if labels.shape[0] <= 0:
        raise ValueError(
            "batch labels must contain "
            "at least one sample."
        )

    if (
        images.shape[0]
        != labels.shape[0]
    ):
        raise ValueError(
            "batch images and labels must contain "
            "the same number of samples. "
            f"Image count: {images.shape[0]}. "
            f"Label count: {labels.shape[0]}."
        )

    return (
        images,
        labels,
    )


# =============================================================================
# Logit Validation
# =============================================================================


def _validate_logits(
    logits: Tensor,
    expected_batch_size: int,
    device: torch.device,
) -> None:
    """
    Binary Classifier Raw Logit을 검증한다.

    정상
    ----
    Tensor

    Shape:

        [batch_size]

    Floating Point

    요청 Device

    모든 값 유한
    """
    if not isinstance(
        logits,
        Tensor,
    ):
        raise TypeError(
            "model output logits must be "
            "a torch.Tensor. "
            f"Received type: "
            f"{type(logits).__name__}."
        )

    if logits.ndim != 1:
        raise ValueError(
            "model output logits must be "
            "one-dimensional. "
            "Expected shape: [batch_size]. "
            f"Received shape: "
            f"{tuple(logits.shape)}."
        )

    if (
        logits.shape[0]
        != expected_batch_size
    ):
        raise ValueError(
            "model output batch size must match "
            "the input image batch size. "
            f"Expected: {expected_batch_size}. "
            f"Received: {logits.shape[0]}."
        )

    if not logits.is_floating_point():
        raise TypeError(
            "model output logits must use "
            "a floating-point dtype. "
            f"Received dtype: "
            f"{logits.dtype}."
        )

    if logits.device != device:
        raise ValueError(
            "model output logits must be on "
            "the requested device. "
            f"Requested: {device}. "
            f"Received: {logits.device}."
        )

    if not torch.isfinite(
        logits
    ).all():
        raise ValueError(
            "model output logits must contain "
            "only finite values."
        )


# =============================================================================
# Loss Validation
# =============================================================================


def _validate_loss(
    loss: Tensor,
) -> None:
    """
    Evaluation Loss를 검증한다.

    정상
    ----
    Tensor

    Scalar:

        ndim == 0

    Floating Point

    유한

    0 이상
    """
    if not isinstance(
        loss,
        Tensor,
    ):
        raise TypeError(
            "loss_function output must be "
            "a torch.Tensor. "
            f"Received type: "
            f"{type(loss).__name__}."
        )

    if loss.ndim != 0:
        raise ValueError(
            "loss_function output must be "
            "a scalar Tensor. "
            f"Received shape: "
            f"{tuple(loss.shape)}."
        )

    if not loss.is_floating_point():
        raise TypeError(
            "loss_function output must use "
            "a floating-point dtype. "
            f"Received dtype: "
            f"{loss.dtype}."
        )

    loss_value = float(
        loss.item()
    )

    if not math.isfinite(
        loss_value
    ):
        raise ValueError(
            "loss_function output must be finite. "
            f"Received value: "
            f"{loss_value}."
        )

    if loss_value < 0.0:
        raise ValueError(
            "loss_function output must be "
            "greater than or equal to 0. "
            f"Received value: "
            f"{loss_value}."
        )