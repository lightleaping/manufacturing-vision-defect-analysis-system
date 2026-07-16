"""
Run the real Day 3 CNN Baseline test evaluation.

이 Script의 역할
----------------
Manufacturing Vision Defect Analysis System의 실제 CNN Baseline
Best Checkpoint를 Test Dataset 전체에서 평가한다.

현재 실제 평가 대상
------------------
Model:

    CNNBaseline

Best Checkpoint:

    models/checkpoints/cnn_baseline_best.pt

현재 실제 Best Epoch:

    Epoch 2

Best Model 선택 기준:

    Lowest Validation Loss

Test Dataset:

    715 Images

NORMAL:

    262 Images

DEFECT:

    453 Images

현재 Binary Class 정의
---------------------
0:

    NORMAL

1:

    DEFECT

Positive Class:

    DEFECT

전체 실행 흐름
-------------
Random Seed 고정

-> Vision DataLoader 생성

-> Test Dataset 검증

-> 새 CNNBaseline 생성

-> CPU Device 이동

-> Best Checkpoint 복원

-> BCEWithLogitsLoss 생성

-> Test Dataset 전체 Evaluation

-> Accuracy 계산

-> Precision 계산

-> Recall 계산

-> F1 Score 계산

-> TN·FP·FN·TP 계산

-> Confusion Matrix 생성

-> Sample별 Prediction Record 생성

-> JSON Artifact 저장

-> 저장 Artifact 재검증

-> PASS

중요
----
Training Pipeline 실행 후 메모리에 남아 있던
마지막 Epoch Model을 사용하지 않는다.

항상 새 CNNBaseline 객체를 생성한 뒤
Best Checkpoint를 복원한다.

현재 Test Dataset은 학습과 Validation에 사용하지 않는다.

이 Script는 Test Dataset에서 다음을 수행하지 않는다.

    Backward

    Optimizer Step

    Threshold 탐색

    Hyperparameter 조정

    Best Epoch 선택

Test Dataset은 현재 고정된 Best Model의
최종 일반화 성능 확인에만 사용한다.

평가 Artifact
-------------
기본 저장 경로:

    reports/artifacts/
    day3_cnn_baseline_test_evaluation.json

저장 내용:

    실행 환경

    Random Seed

    Model Metadata

    Checkpoint Metadata

    Dataset 정보

    Test Loss

    Accuracy

    Precision

    Recall

    F1 Score

    TN

    FP

    FN

    TP

    Confusion Matrix

    715개 Sample별:

        Image Path

        Ground Truth Label

        Raw Logit

        DEFECT Probability

        Prediction

        Correct 여부
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import Tensor

from src.data.data_loader import (
    BATCH_SIZE,
    DROP_LAST,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    VisionDataLoaders,
    create_vision_data_loaders,
)
from src.data.dataset_config import (
    INDEX_TO_CLASS_NAME,
    PROJECT_ROOT,
)
from src.data.dataset_split import (
    ImageSample,
    count_samples_by_label,
)
from src.evaluation.classification_metrics import (
    NEGATIVE_CLASS_LABEL,
    POSITIVE_CLASS_LABEL,
    BinaryClassificationMetrics,
    calculate_binary_classification_metrics,
)
from src.evaluation.evaluation_runner import (
    BinaryEvaluationResult,
    evaluate_binary_classifier,
)
from src.models.cnn_baseline import (
    CNNBaseline,
)
from src.reproducibility import (
    DEFAULT_RANDOM_SEED,
    ReproducibilitySettings,
    set_global_random_seed,
)
from src.training.checkpoint_loader import (
    LoadedCheckpointInfo,
    load_model_checkpoint,
)
from src.training.loss_function import (
    create_binary_classification_loss,
)
from src.training.training_pipeline import (
    DEFAULT_CNN_CHECKPOINT_PATH,
)


# =============================================================================
# Evaluation Artifact Configuration
# =============================================================================

# 현재 JSON Artifact 구조 Version이다.
#
# 향후 JSON Field가 변경되면 Version을 올린다.
EVALUATION_ARTIFACT_VERSION = 1


# 실제 CNN Baseline Test 평가 결과 기본 저장 경로다.
DEFAULT_EVALUATION_OUTPUT_PATH = Path(
    "reports"
) / "artifacts" / (
    "day3_cnn_baseline_test_evaluation.json"
)


# =============================================================================
# Expected Day 3 Test Dataset Configuration
# =============================================================================

EXPECTED_TEST_SAMPLE_COUNT = 715


EXPECTED_TEST_NORMAL_COUNT = 262


EXPECTED_TEST_DEFECT_COUNT = 453


# Batch Size:
#
#     32
#
# Test Samples:
#
#     715
#
# Batch Count:
#
#     ceil(
#         715 / 32
#     )
#
#     =
#
#     23
EXPECTED_TEST_BATCH_COUNT = 23


# 현재 CNNBaseline Parameter 수다.
EXPECTED_CNN_PARAMETER_COUNT = 6_065


# =============================================================================
# Command-line Arguments
# =============================================================================


def create_argument_parser() -> (
    argparse.ArgumentParser
):
    """
    Day 3 CNN Test Evaluation용 Argument Parser를 생성한다.

    지원 인자
    --------
    --checkpoint-path:
        평가할 Best Model Checkpoint

    --output-path:
        평가 JSON Artifact 저장 경로

    --validate-only:
        Dataset·Model·Checkpoint 구성까지만 검증하고
        Test 전체 추론은 실행하지 않는다.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the real Day 3 CNN Baseline "
            "test evaluation."
        )
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=(
            DEFAULT_CNN_CHECKPOINT_PATH
        ),
        help=(
            "CNN Baseline checkpoint path. "
            "A relative path is resolved from "
            "the project root."
        ),
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            DEFAULT_EVALUATION_OUTPUT_PATH
        ),
        help=(
            "Evaluation JSON output path. "
            "A relative path is resolved from "
            "the project root."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate the environment, dataset, "
            "Model, and checkpoint without running "
            "the full Test evaluation."
        ),
    )

    return parser


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """
    Command-line Argument를 파싱한다.
    """
    parser = (
        create_argument_parser()
    )

    return parser.parse_args(
        arguments
    )


# =============================================================================
# Path Resolution
# =============================================================================


def resolve_project_path(
    path: Path,
) -> Path:
    """
    상대 경로를 Project Root 기준 절대 경로로 변환한다.

    입력
    ----
    상대 경로:

        models/checkpoints/
        cnn_baseline_best.pt

    출력
    ----
    PROJECT_ROOT가 결합된 절대 경로

    절대 경로 입력
    --------------
    그대로 사용한다.
    """
    if not isinstance(
        path,
        Path,
    ):
        raise TypeError(
            "path must be a pathlib.Path. "
            f"Received type: "
            f"{type(path).__name__}."
        )

    expanded_path = (
        path.expanduser()
    )

    if expanded_path.is_absolute():
        return (
            expanded_path.resolve()
        )

    return (
        PROJECT_ROOT
        / expanded_path
    ).resolve()


def convert_to_portable_project_path(
    path: Path,
) -> str:
    """
    Project 내부 경로는 상대 POSIX 경로로 변환한다.

    예
    --
    Windows 절대 경로:

        C:\\Users\\...\\project\\data\\image.png

    JSON 저장:

        data/image.png

    Project 외부 경로인 경우:
        절대 경로 문자열을 사용한다.
    """
    resolved_path = (
        path.resolve()
    )

    resolved_project_root = (
        PROJECT_ROOT.resolve()
    )

    try:
        relative_path = (
            resolved_path.relative_to(
                resolved_project_root
            )
        )

    except ValueError:
        return str(
            resolved_path
        )

    return (
        relative_path.as_posix()
    )


# =============================================================================
# Output Helpers
# =============================================================================


def print_major_heading(
    title: str,
) -> None:
    """
    주요 실행 구간 Heading을 출력한다.
    """
    print()

    print(
        "=" * 100
    )

    print(
        title
    )

    print(
        "=" * 100
    )


def print_minor_heading(
    title: str,
) -> None:
    """
    하위 실행 구간 Heading을 출력한다.
    """
    print()

    print(
        f"[{title}]"
    )


def print_name_value(
    name: str,
    value: object,
) -> None:
    """
    이름·값을 정렬하여 출력한다.
    """
    print(
        f"{name:<28}: {value}"
    )


# =============================================================================
# Environment and Reproducibility
# =============================================================================


def print_environment() -> None:
    """
    현재 실제 실행 환경을 출력한다.
    """
    print_minor_heading(
        "ENVIRONMENT"
    )

    print_name_value(
        "Project root",
        PROJECT_ROOT,
    )

    print_name_value(
        "Python",
        platform.python_version(),
    )

    print_name_value(
        "Python executable",
        sys.executable,
    )

    print_name_value(
        "PyTorch",
        torch.__version__,
    )

    print_name_value(
        "Torchvision",
        torchvision.__version__,
    )

    print_name_value(
        "CUDA available",
        torch.cuda.is_available(),
    )


def configure_reproducibility() -> (
    ReproducibilitySettings
):
    """
    Day 3 기준 Random Seed를 적용한다.

    Test Transform은 Random Augmentation을 사용하지 않는다.

    그래도 실행 환경의 Random State를 통일하여
    전체 Pipeline 재현성을 유지한다.
    """
    settings = (
        set_global_random_seed(
            seed=(
                DEFAULT_RANDOM_SEED
            ),
            deterministic_algorithms=False,
        )
    )

    print_minor_heading(
        "REPRODUCIBILITY"
    )

    print_name_value(
        "Random seed",
        settings.seed,
    )

    print_name_value(
        "Deterministic algorithms",
        (
            settings
            .deterministic_algorithms
        ),
    )

    print_name_value(
        "CUDA seed applied",
        (
            settings
            .cuda_seed_applied
        ),
    )

    print_name_value(
        "Device",
        settings.device,
    )

    return settings


# =============================================================================
# Test Dataset Validation
# =============================================================================


def get_validated_test_samples(
    data_loaders: VisionDataLoaders,
) -> tuple[
    ImageSample,
    ...,
]:
    """
    실제 Test Dataset Sample 목록과 구성을 검증한다.

    현재 기대값
    ------------
    Total:

        715

    NORMAL:

        262

    DEFECT:

        453

    Test Batches:

        23
    """
    if not isinstance(
        data_loaders,
        VisionDataLoaders,
    ):
        raise TypeError(
            "data_loaders must be "
            "a VisionDataLoaders instance. "
            f"Received type: "
            f"{type(data_loaders).__name__}."
        )

    test_dataset = (
        data_loaders.test_dataset
    )

    test_loader = (
        data_loaders.test_loader
    )

    if (
        len(
            test_dataset
        )
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "unexpected Test Dataset sample count. "
            f"Expected: "
            f"{EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: "
            f"{len(test_dataset)}."
        )

    if (
        len(
            test_loader
        )
        != EXPECTED_TEST_BATCH_COUNT
    ):
        raise ValueError(
            "unexpected Test DataLoader batch count. "
            f"Expected: "
            f"{EXPECTED_TEST_BATCH_COUNT}. "
            f"Received: "
            f"{len(test_loader)}."
        )

    raw_samples = getattr(
        test_dataset,
        "samples",
        None,
    )

    if raw_samples is None:
        raise AttributeError(
            "test_dataset must expose "
            "a samples attribute."
        )

    try:
        test_samples = tuple(
            raw_samples
        )

    except TypeError as error:
        raise TypeError(
            "test_dataset.samples must be iterable."
        ) from error

    if (
        len(
            test_samples
        )
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "test_dataset.samples length must match "
            "the expected Test Dataset size. "
            f"Expected: "
            f"{EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: "
            f"{len(test_samples)}."
        )

    for (
        sample_index,
        sample,
    ) in enumerate(
        test_samples
    ):
        if not isinstance(
            sample,
            ImageSample,
        ):
            raise TypeError(
                "every Test Dataset sample must be "
                "an ImageSample. "
                f"Index: {sample_index}. "
                f"Received type: "
                f"{type(sample).__name__}."
            )

    label_counts = (
        count_samples_by_label(
            samples=test_samples,
        )
    )

    normal_count = int(
        label_counts.get(
            NEGATIVE_CLASS_LABEL,
            0,
        )
    )

    defect_count = int(
        label_counts.get(
            POSITIVE_CLASS_LABEL,
            0,
        )
    )

    if (
        normal_count
        != EXPECTED_TEST_NORMAL_COUNT
    ):
        raise ValueError(
            "unexpected Test NORMAL sample count. "
            f"Expected: "
            f"{EXPECTED_TEST_NORMAL_COUNT}. "
            f"Received: "
            f"{normal_count}."
        )

    if (
        defect_count
        != EXPECTED_TEST_DEFECT_COUNT
    ):
        raise ValueError(
            "unexpected Test DEFECT sample count. "
            f"Expected: "
            f"{EXPECTED_TEST_DEFECT_COUNT}. "
            f"Received: "
            f"{defect_count}."
        )

    if (
        normal_count
        + defect_count
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "Test Class counts must sum to "
            "the Test sample count."
        )

    print_minor_heading(
        "TEST DATASET"
    )

    print_name_value(
        "Test samples",
        f"{len(test_samples):,}",
    )

    print_name_value(
        "NORMAL samples",
        f"{normal_count:,}",
    )

    print_name_value(
        "DEFECT samples",
        f"{defect_count:,}",
    )

    print_name_value(
        "Test batches",
        f"{len(test_loader):,}",
    )

    print_name_value(
        "Batch size",
        BATCH_SIZE,
    )

    print_name_value(
        "Num workers",
        NUM_WORKERS,
    )

    print_name_value(
        "Pin memory",
        PIN_MEMORY,
    )

    print_name_value(
        "Drop last",
        DROP_LAST,
    )

    print_name_value(
        "Persistent workers",
        PERSISTENT_WORKERS,
    )

    return test_samples


# =============================================================================
# Model Validation
# =============================================================================


def count_model_parameters(
    model: CNNBaseline,
) -> int:
    """
    Model 전체 Parameter 수를 계산한다.
    """
    return sum(
        parameter.numel()
        for parameter in (
            model.parameters()
        )
    )


def validate_cnn_model(
    model: CNNBaseline,
    device: torch.device,
) -> int:
    """
    실제 CNNBaseline 구조·Parameter·Device를 검증한다.
    """
    if not isinstance(
        model,
        CNNBaseline,
    ):
        raise TypeError(
            "model must be a CNNBaseline. "
            f"Received type: "
            f"{type(model).__name__}."
        )

    parameter_count = (
        count_model_parameters(
            model=model,
        )
    )

    if (
        parameter_count
        != EXPECTED_CNN_PARAMETER_COUNT
    ):
        raise ValueError(
            "unexpected CNNBaseline Parameter count. "
            f"Expected: "
            f"{EXPECTED_CNN_PARAMETER_COUNT}. "
            f"Received: "
            f"{parameter_count}."
        )

    for (
        parameter_name,
        parameter,
    ) in model.named_parameters():
        if parameter.device != device:
            raise ValueError(
                "CNNBaseline Parameter is on "
                "an unexpected device. "
                f"Parameter: {parameter_name}. "
                f"Expected: {device}. "
                f"Received: "
                f"{parameter.device}."
            )

        if not torch.isfinite(
            parameter.detach()
        ).all():
            raise ValueError(
                "CNNBaseline Parameter must contain "
                "only finite values. "
                f"Invalid Parameter: "
                f"{parameter_name}."
            )

    return parameter_count


def clone_model_state(
    model: CNNBaseline,
) -> dict[
    str,
    Tensor,
]:
    """
    Model State를 독립적인 CPU Tensor Dictionary로 복사한다.

    Test Evaluation 전후 Model Weight 불변 검증에 사용한다.
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
        ) in (
            model.state_dict().items()
        )
    }


def model_states_are_equal(
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
    두 Model State의 Key·Tensor를 정확하게 비교한다.
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


# =============================================================================
# Evaluation Result Validation
# =============================================================================


def validate_evaluation_results(
    *,
    evaluation_result: (
        BinaryEvaluationResult
    ),
    metrics: (
        BinaryClassificationMetrics
    ),
    test_samples: tuple[
        ImageSample,
        ...,
    ],
    model: CNNBaseline,
    model_state_before: dict[
        str,
        Tensor,
    ],
) -> None:
    """
    Evaluation Runner·Metric·Dataset·Model 결과를 교차 검증한다.

    검증
    ----
    Evaluation Sample:

        715

    Evaluation Batch:

        23

    Dataset Sample 순서:

        Evaluation Label 순서와 일치

    Evaluation Accuracy:

        Classification Metrics Accuracy와 일치

    Confusion Count 합:

        715

    Model Mode:

        Evaluation

    Model Weight:

        평가 전후 동일

    Parameter Gradient:

        모두 None
    """
    if not isinstance(
        evaluation_result,
        BinaryEvaluationResult,
    ):
        raise TypeError(
            "evaluation_result must be "
            "a BinaryEvaluationResult."
        )

    if not isinstance(
        metrics,
        BinaryClassificationMetrics,
    ):
        raise TypeError(
            "metrics must be "
            "a BinaryClassificationMetrics."
        )

    if (
        evaluation_result.sample_count
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "Evaluation sample count is unexpected. "
            f"Expected: "
            f"{EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: "
            f"{evaluation_result.sample_count}."
        )

    if (
        evaluation_result.batch_count
        != EXPECTED_TEST_BATCH_COUNT
    ):
        raise ValueError(
            "Evaluation batch count is unexpected. "
            f"Expected: "
            f"{EXPECTED_TEST_BATCH_COUNT}. "
            f"Received: "
            f"{evaluation_result.batch_count}."
        )

    if (
        metrics.sample_count
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "Metric sample count is unexpected. "
            f"Expected: "
            f"{EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: "
            f"{metrics.sample_count}."
        )

    if not math.isclose(
        evaluation_result.accuracy,
        metrics.accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Evaluation Accuracy and Classification "
            "Metrics Accuracy must match. "
            f"Evaluation: "
            f"{evaluation_result.accuracy}. "
            f"Metrics: "
            f"{metrics.accuracy}."
        )

    confusion_total = (
        metrics.true_negative
        + metrics.false_positive
        + metrics.false_negative
        + metrics.true_positive
    )

    if (
        confusion_total
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "Confusion counts must sum to "
            "the Test sample count. "
            f"Received total: "
            f"{confusion_total}."
        )

    expected_labels = torch.tensor(
        [
            sample.label
            for sample in (
                test_samples
            )
        ],
        dtype=torch.int64,
    )

    if not torch.equal(
        evaluation_result.labels,
        expected_labels,
    ):
        raise ValueError(
            "Evaluation labels must preserve "
            "the Test Dataset sample order."
        )

    if model.training:
        raise ValueError(
            "model must be in Evaluation Mode "
            "after Test evaluation."
        )

    model_state_after = (
        clone_model_state(
            model=model,
        )
    )

    if not model_states_are_equal(
        first_state=(
            model_state_before
        ),
        second_state=(
            model_state_after
        ),
    ):
        raise ValueError(
            "Model State changed during "
            "Test evaluation."
        )

    if not all(
        parameter.grad is None
        for parameter in (
            model.parameters()
        )
    ):
        raise ValueError(
            "Model Parameter Gradient must remain "
            "None during Test evaluation."
        )


# =============================================================================
# Prediction Record
# =============================================================================


def get_class_name(
    label: int,
) -> str:
    """
    Integer Label을 현재 Class 이름으로 변환한다.
    """
    if (
        label
        not in INDEX_TO_CLASS_NAME
    ):
        raise KeyError(
            "label is not defined in "
            "INDEX_TO_CLASS_NAME. "
            f"Received label: "
            f"{label}."
        )

    return str(
        INDEX_TO_CLASS_NAME[
            label
        ]
    )


def build_sample_prediction_records(
    *,
    test_samples: tuple[
        ImageSample,
        ...,
    ],
    evaluation_result: (
        BinaryEvaluationResult
    ),
) -> list[
    dict[str, Any]
]:
    """
    Test Sample별 평가 Record를 생성한다.

    각 Record
    ---------
    sample_index

    image_path

    label

    label_name

    logit

    defect_probability

    prediction

    prediction_name

    is_correct

    왜 저장하는가
    -------------
    다음 단계에서:

        오분류 이미지 찾기

        False Negative 분석

        False Positive 분석

        Confidence 정렬

        Grad-CAM 대상 선택

    에 재사용할 수 있다.
    """
    if (
        len(
            test_samples
        )
        != evaluation_result.sample_count
    ):
        raise ValueError(
            "Test Sample count must match "
            "Evaluation Result sample_count."
        )

    records: list[
        dict[str, Any]
    ] = []

    for (
        sample_index,
        sample,
    ) in enumerate(
        test_samples
    ):
        label = int(
            evaluation_result.labels[
                sample_index
            ].item()
        )

        if (
            sample.label
            != label
        ):
            raise ValueError(
                "Evaluation Label must match "
                "the Dataset Sample Label. "
                f"Sample index: "
                f"{sample_index}. "
                f"Dataset label: "
                f"{sample.label}. "
                f"Evaluation label: "
                f"{label}."
            )

        logit = float(
            evaluation_result.logits[
                sample_index
            ].item()
        )

        defect_probability = float(
            evaluation_result.probabilities[
                sample_index
            ].item()
        )

        prediction = int(
            evaluation_result.predictions[
                sample_index
            ].item()
        )

        is_correct = (
            prediction
            == label
        )

        records.append(
            {
                "sample_index": (
                    sample_index
                ),
                "image_path": (
                    convert_to_portable_project_path(
                        path=(
                            sample.image_path
                        )
                    )
                ),
                "label": label,
                "label_name": (
                    get_class_name(
                        label=label,
                    )
                ),
                "logit": logit,
                "defect_probability": (
                    defect_probability
                ),
                "prediction": (
                    prediction
                ),
                "prediction_name": (
                    get_class_name(
                        label=prediction,
                    )
                ),
                "is_correct": (
                    is_correct
                ),
            }
        )

    return records


# =============================================================================
# JSON Artifact
# =============================================================================


def build_evaluation_artifact(
    *,
    checkpoint_path: Path,
    output_path: Path,
    parameter_count: int,
    reproducibility_settings: (
        ReproducibilitySettings
    ),
    checkpoint_info: (
        LoadedCheckpointInfo
    ),
    evaluation_result: (
        BinaryEvaluationResult
    ),
    metrics: (
        BinaryClassificationMetrics
    ),
    prediction_records: list[
        dict[str, Any]
    ],
    elapsed_seconds: float,
) -> dict[
    str,
    Any,
]:
    """
    실제 CNN Test 평가 JSON Payload를 생성한다.
    """
    predicted_normal_count = int(
        evaluation_result.predictions
        .eq(
            NEGATIVE_CLASS_LABEL
        )
        .sum()
        .item()
    )

    predicted_defect_count = int(
        evaluation_result.predictions
        .eq(
            POSITIVE_CLASS_LABEL
        )
        .sum()
        .item()
    )

    correct_count = (
        metrics.true_negative
        + metrics.true_positive
    )

    incorrect_count = (
        metrics.false_positive
        + metrics.false_negative
    )

    probability_minimum = float(
        evaluation_result.probabilities
        .min()
        .item()
    )

    probability_maximum = float(
        evaluation_result.probabilities
        .max()
        .item()
    )

    probability_mean = float(
        evaluation_result.probabilities
        .mean()
        .item()
    )

    return {
        "artifact_version": (
            EVALUATION_ARTIFACT_VERSION
        ),
        "project_name": (
            "Manufacturing Vision "
            "Defect Analysis System"
        ),
        "project_name_ko": (
            "제조 비전 결함 분석 시스템"
        ),
        "stage": (
            "day3_cnn_baseline_test_evaluation"
        ),
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "environment": {
            "python_version": (
                platform.python_version()
            ),
            "python_executable": (
                sys.executable
            ),
            "torch_version": (
                torch.__version__
            ),
            "torchvision_version": (
                torchvision.__version__
            ),
            "cuda_available": (
                torch.cuda.is_available()
            ),
            "device": (
                reproducibility_settings
                .device
            ),
        },
        "reproducibility": {
            "random_seed": (
                reproducibility_settings
                .seed
            ),
            "deterministic_algorithms": (
                reproducibility_settings
                .deterministic_algorithms
            ),
            "cuda_seed_applied": (
                reproducibility_settings
                .cuda_seed_applied
            ),
        },
        "model": {
            "name": (
                checkpoint_info
                .model_name
            ),
            "module": (
                checkpoint_info
                .model_module
            ),
            "parameter_count": (
                parameter_count
            ),
            "loss_function": (
                checkpoint_info
                .loss_function_name
            ),
            "optimizer": (
                checkpoint_info
                .optimizer_name
            ),
        },
        "checkpoint": {
            "path": (
                convert_to_portable_project_path(
                    path=checkpoint_path,
                )
            ),
            "version": (
                checkpoint_info
                .checkpoint_version
            ),
            "best_epoch": (
                checkpoint_info
                .epoch_number
            ),
            "configured_epoch_count": (
                checkpoint_info
                .configured_epoch_count
            ),
            "best_model_selection_metric": (
                checkpoint_info
                .best_model_selection_metric
            ),
            "classification_threshold": (
                checkpoint_info
                .classification_threshold
            ),
            "best_validation_loss": (
                checkpoint_info
                .validation_loss
            ),
            "best_validation_accuracy": (
                checkpoint_info
                .validation_accuracy
            ),
            "validation_sample_count": (
                checkpoint_info
                .validation_sample_count
            ),
            "validation_batch_count": (
                checkpoint_info
                .validation_batch_count
            ),
        },
        "dataset": {
            "split": "test",
            "sample_count": (
                evaluation_result
                .sample_count
            ),
            "batch_count": (
                evaluation_result
                .batch_count
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "normal_label": (
                NEGATIVE_CLASS_LABEL
            ),
            "defect_label": (
                POSITIVE_CLASS_LABEL
            ),
            "normal_count": (
                EXPECTED_TEST_NORMAL_COUNT
            ),
            "defect_count": (
                EXPECTED_TEST_DEFECT_COUNT
            ),
            "class_names": {
                str(
                    NEGATIVE_CLASS_LABEL
                ): (
                    get_class_name(
                        label=(
                            NEGATIVE_CLASS_LABEL
                        )
                    )
                ),
                str(
                    POSITIVE_CLASS_LABEL
                ): (
                    get_class_name(
                        label=(
                            POSITIVE_CLASS_LABEL
                        )
                    )
                ),
            },
        },
        "evaluation": {
            "average_loss": (
                evaluation_result
                .average_loss
            ),
            "accuracy": (
                metrics.accuracy
            ),
            "precision": (
                metrics.precision
            ),
            "recall": (
                metrics.recall
            ),
            "f1_score": (
                metrics.f1_score
            ),
            "true_negative": (
                metrics.true_negative
            ),
            "false_positive": (
                metrics.false_positive
            ),
            "false_negative": (
                metrics.false_negative
            ),
            "true_positive": (
                metrics.true_positive
            ),
            "confusion_matrix": (
                metrics.confusion_matrix
                .tolist()
            ),
            "correct_count": (
                correct_count
            ),
            "incorrect_count": (
                incorrect_count
            ),
            "predicted_normal_count": (
                predicted_normal_count
            ),
            "predicted_defect_count": (
                predicted_defect_count
            ),
            "defect_probability_min": (
                probability_minimum
            ),
            "defect_probability_max": (
                probability_maximum
            ),
            "defect_probability_mean": (
                probability_mean
            ),
        },
        "execution": {
            "elapsed_seconds": (
                elapsed_seconds
            ),
            "output_path": (
                convert_to_portable_project_path(
                    path=output_path,
                )
            ),
        },
        "sample_predictions": (
            prediction_records
        ),
    }


def validate_evaluation_artifact(
    artifact: dict[
        str,
        Any,
    ],
) -> None:
    """
    저장 전·후 Evaluation Artifact 핵심 구조를 검증한다.
    """
    if not isinstance(
        artifact,
        dict,
    ):
        raise TypeError(
            "artifact must be a dictionary."
        )

    if (
        artifact.get(
            "artifact_version"
        )
        != EVALUATION_ARTIFACT_VERSION
    ):
        raise ValueError(
            "unexpected Evaluation Artifact Version."
        )

    dataset = artifact.get(
        "dataset"
    )

    if not isinstance(
        dataset,
        dict,
    ):
        raise TypeError(
            "artifact.dataset must be "
            "a dictionary."
        )

    if (
        dataset.get(
            "sample_count"
        )
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "artifact Test sample count "
            "is unexpected."
        )

    evaluation = artifact.get(
        "evaluation"
    )

    if not isinstance(
        evaluation,
        dict,
    ):
        raise TypeError(
            "artifact.evaluation must be "
            "a dictionary."
        )

    required_metric_keys = {
        "average_loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "confusion_matrix",
    }

    missing_metric_keys = (
        required_metric_keys
        - set(
            evaluation.keys()
        )
    )

    if missing_metric_keys:
        raise KeyError(
            "artifact.evaluation is missing "
            "required keys. "
            f"Missing: "
            f"{sorted(missing_metric_keys)}."
        )

    prediction_records = (
        artifact.get(
            "sample_predictions"
        )
    )

    if not isinstance(
        prediction_records,
        list,
    ):
        raise TypeError(
            "artifact.sample_predictions "
            "must be a list."
        )

    if (
        len(
            prediction_records
        )
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "artifact Sample Prediction count "
            "is unexpected. "
            f"Expected: "
            f"{EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: "
            f"{len(prediction_records)}."
        )


def write_json_artifact(
    *,
    output_path: Path,
    artifact: dict[
        str,
        Any,
    ],
) -> None:
    """
    Evaluation JSON을 Atomic 방식으로 저장한다.

    저장 흐름
    ---------
    Temporary File 저장

    ->

    JSON 재읽기

    ->

    구조 검증

    ->

    최종 파일 Replace

    장점
    ----
    저장 중 실행이 중단되어
    불완전한 최종 JSON이 남는 위험을 줄인다.
    """
    validate_evaluation_artifact(
        artifact=artifact,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        output_path.with_suffix(
            output_path.suffix
            + ".tmp"
        )
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                artifact,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write(
                "\n"
            )

        with temporary_path.open(
            mode="r",
            encoding="utf-8",
        ) as file:
            loaded_artifact = (
                json.load(
                    file
                )
            )

        validate_evaluation_artifact(
            artifact=(
                loaded_artifact
            ),
        )

        temporary_path.replace(
            output_path
        )

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# =============================================================================
# Result Output
# =============================================================================


def print_checkpoint_information(
    *,
    checkpoint_path: Path,
    checkpoint_info: (
        LoadedCheckpointInfo
    ),
    parameter_count: int,
) -> None:
    """
    복원된 Best Model·Checkpoint 정보를 출력한다.
    """
    print_minor_heading(
        "BEST CHECKPOINT"
    )

    print_name_value(
        "Checkpoint path",
        checkpoint_path,
    )

    print_name_value(
        "Checkpoint version",
        (
            checkpoint_info
            .checkpoint_version
        ),
    )

    print_name_value(
        "Model",
        checkpoint_info.model_name,
    )

    print_name_value(
        "Model module",
        checkpoint_info.model_module,
    )

    print_name_value(
        "Parameters",
        f"{parameter_count:,}",
    )

    print_name_value(
        "Best epoch",
        checkpoint_info.epoch_number,
    )

    print_name_value(
        "Configured epochs",
        (
            checkpoint_info
            .configured_epoch_count
        ),
    )

    print_name_value(
        "Selection metric",
        (
            checkpoint_info
            .best_model_selection_metric
        ),
    )

    print_name_value(
        "Threshold",
        (
            checkpoint_info
            .classification_threshold
        ),
    )

    print_name_value(
        "Best validation loss",
        (
            f"{checkpoint_info.validation_loss:.12f}"
        ),
    )

    print_name_value(
        "Best validation accuracy",
        (
            f"{checkpoint_info.validation_accuracy:.12f}"
        ),
    )

    print_name_value(
        "Best validation percent",
        (
            f"{checkpoint_info.validation_accuracy * 100:.2f}%"
        ),
    )


def print_evaluation_results(
    *,
    evaluation_result: (
        BinaryEvaluationResult
    ),
    metrics: (
        BinaryClassificationMetrics
    ),
    elapsed_seconds: float,
    output_path: Path,
) -> None:
    """
    실제 CNN Test 평가 결과를 출력한다.
    """
    predicted_normal_count = int(
        evaluation_result.predictions
        .eq(
            NEGATIVE_CLASS_LABEL
        )
        .sum()
        .item()
    )

    predicted_defect_count = int(
        evaluation_result.predictions
        .eq(
            POSITIVE_CLASS_LABEL
        )
        .sum()
        .item()
    )

    correct_count = (
        metrics.true_negative
        + metrics.true_positive
    )

    incorrect_count = (
        metrics.false_positive
        + metrics.false_negative
    )

    print_minor_heading(
        "TEST LOSS"
    )

    print_name_value(
        "Average loss",
        (
            f"{evaluation_result.average_loss:.12f}"
        ),
    )

    print_minor_heading(
        "TEST METRICS"
    )

    print_name_value(
        "Accuracy",
        f"{metrics.accuracy:.12f}",
    )

    print_name_value(
        "Accuracy percent",
        f"{metrics.accuracy * 100:.2f}%",
    )

    print_name_value(
        "Precision",
        f"{metrics.precision:.12f}",
    )

    print_name_value(
        "Precision percent",
        f"{metrics.precision * 100:.2f}%",
    )

    print_name_value(
        "Recall",
        f"{metrics.recall:.12f}",
    )

    print_name_value(
        "Recall percent",
        f"{metrics.recall * 100:.2f}%",
    )

    print_name_value(
        "F1 score",
        f"{metrics.f1_score:.12f}",
    )

    print_name_value(
        "F1 percent",
        f"{metrics.f1_score * 100:.2f}%",
    )

    print_minor_heading(
        "CONFUSION COUNTS"
    )

    print_name_value(
        "True negative",
        metrics.true_negative,
    )

    print_name_value(
        "False positive",
        metrics.false_positive,
    )

    print_name_value(
        "False negative",
        metrics.false_negative,
    )

    print_name_value(
        "True positive",
        metrics.true_positive,
    )

    print_name_value(
        "Correct",
        correct_count,
    )

    print_name_value(
        "Incorrect",
        incorrect_count,
    )

    print_minor_heading(
        "CONFUSION MATRIX"
    )

    print(
        metrics.confusion_matrix
    )

    print()

    print(
        "Matrix order:"
    )

    print(
        "["
    )

    print(
        "    [TN, FP],"
    )

    print(
        "    [FN, TP],"
    )

    print(
        "]"
    )

    print_minor_heading(
        "PREDICTION DISTRIBUTION"
    )

    print_name_value(
        "Predicted NORMAL",
        predicted_normal_count,
    )

    print_name_value(
        "Predicted DEFECT",
        predicted_defect_count,
    )

    print_name_value(
        "DEFECT probability min",
        (
            f"{evaluation_result.probabilities.min().item():.12f}"
        ),
    )

    print_name_value(
        "DEFECT probability max",
        (
            f"{evaluation_result.probabilities.max().item():.12f}"
        ),
    )

    print_name_value(
        "DEFECT probability mean",
        (
            f"{evaluation_result.probabilities.mean().item():.12f}"
        ),
    )

    print_minor_heading(
        "EXECUTION"
    )

    print_name_value(
        "Elapsed seconds",
        f"{elapsed_seconds:.2f}",
    )

    print_name_value(
        "Output path",
        output_path,
    )

    print_name_value(
        "Output exists",
        output_path.is_file(),
    )


# =============================================================================
# Main Evaluation
# =============================================================================


def run_evaluation(
    arguments: argparse.Namespace,
) -> int:
    """
    Day 3 CNN Baseline 실제 Test 평가를 실행한다.
    """
    print_major_heading(
        "DAY 3 - CNN BASELINE "
        "REAL TEST EVALUATION"
    )

    print_environment()

    reproducibility_settings = (
        configure_reproducibility()
    )

    device = torch.device(
        reproducibility_settings
        .device
    )

    checkpoint_path = (
        resolve_project_path(
            path=(
                arguments
                .checkpoint_path
            )
        )
    )

    output_path = (
        resolve_project_path(
            path=(
                arguments
                .output_path
            )
        )
    )

    # ---------------------------------------------------------
    # DataLoader
    # ---------------------------------------------------------
    print_major_heading(
        "1. TEST DATA PREPARATION"
    )

    data_loaders = (
        create_vision_data_loaders(
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            drop_last=DROP_LAST,
            persistent_workers=(
                PERSISTENT_WORKERS
            ),
            random_seed=(
                DEFAULT_RANDOM_SEED
            ),
        )
    )

    test_samples = (
        get_validated_test_samples(
            data_loaders=(
                data_loaders
            )
        )
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    print_major_heading(
        "2. BEST MODEL RESTORATION"
    )

    model = CNNBaseline()

    model = model.to(
        device
    )

    parameter_count = (
        validate_cnn_model(
            model=model,
            device=device,
        )
    )

    checkpoint_info = (
        load_model_checkpoint(
            model=model,
            checkpoint_path=(
                checkpoint_path
            ),
            device=device,
        )
    )

    parameter_count = (
        validate_cnn_model(
            model=model,
            device=device,
        )
    )

    print_checkpoint_information(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint_info=(
            checkpoint_info
        ),
        parameter_count=(
            parameter_count
        ),
    )

    # ---------------------------------------------------------
    # Validation-only
    # ---------------------------------------------------------
    if arguments.validate_only:
        print_major_heading(
            "VALIDATION RESULT"
        )

        print(
            "[PASS] Day 3 CNN Baseline "
            "evaluation configuration validation"
        )

        return 0

    # ---------------------------------------------------------
    # Loss
    # ---------------------------------------------------------
    loss_function = (
        create_binary_classification_loss()
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------
    print_major_heading(
        "3. REAL TEST EVALUATION"
    )

    model_state_before = (
        clone_model_state(
            model=model,
        )
    )

    evaluation_started_at = (
        time.perf_counter()
    )

    evaluation_result = (
        evaluate_binary_classifier(
            model=model,
            data_loader=(
                data_loaders
                .test_loader
            ),
            loss_function=(
                loss_function
            ),
            device=device,
            classification_threshold=(
                checkpoint_info
                .classification_threshold
            ),
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result
                .labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - evaluation_started_at
    )

    # ---------------------------------------------------------
    # Cross Validation
    # ---------------------------------------------------------
    validate_evaluation_results(
        evaluation_result=(
            evaluation_result
        ),
        metrics=metrics,
        test_samples=(
            test_samples
        ),
        model=model,
        model_state_before=(
            model_state_before
        ),
    )

    # ---------------------------------------------------------
    # Sample Prediction Records
    # ---------------------------------------------------------
    prediction_records = (
        build_sample_prediction_records(
            test_samples=(
                test_samples
            ),
            evaluation_result=(
                evaluation_result
            ),
        )
    )

    # ---------------------------------------------------------
    # JSON Artifact
    # ---------------------------------------------------------
    print_major_heading(
        "4. EVALUATION ARTIFACT"
    )

    artifact = (
        build_evaluation_artifact(
            checkpoint_path=(
                checkpoint_path
            ),
            output_path=(
                output_path
            ),
            parameter_count=(
                parameter_count
            ),
            reproducibility_settings=(
                reproducibility_settings
            ),
            checkpoint_info=(
                checkpoint_info
            ),
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            prediction_records=(
                prediction_records
            ),
            elapsed_seconds=(
                elapsed_seconds
            ),
        )
    )

    write_json_artifact(
        output_path=(
            output_path
        ),
        artifact=artifact,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    print_major_heading(
        "5. FINAL RESULT"
    )

    print_evaluation_results(
        evaluation_result=(
            evaluation_result
        ),
        metrics=metrics,
        elapsed_seconds=(
            elapsed_seconds
        ),
        output_path=(
            output_path
        ),
    )

    # ---------------------------------------------------------
    # Final Validation
    # ---------------------------------------------------------
    if not output_path.is_file():
        raise FileNotFoundError(
            "Evaluation JSON Artifact was not created. "
            f"Expected path: "
            f"{output_path}."
        )

    if (
        len(
            prediction_records
        )
        != EXPECTED_TEST_SAMPLE_COUNT
    ):
        raise ValueError(
            "unexpected Sample Prediction "
            "Record count."
        )

    print_major_heading(
        "DAY 3 CNN BASELINE "
        "TEST EVALUATION COMPLETE"
    )

    print(
        "[PASS] Day 3 CNN Baseline "
        "real Test evaluation"
    )

    return 0


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """
    Script Entry Point.
    """
    parsed_arguments = (
        parse_arguments(
            arguments=arguments,
        )
    )

    return run_evaluation(
        arguments=(
            parsed_arguments
        )
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )