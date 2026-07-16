"""
Day 3 CNN Baseline real evaluation Script unit tests.

테스트 대상
----------
scripts/run_day3_cnn_baseline_evaluation.py

테스트 목적
----------
실제 CNN Baseline Test 평가 Script의 다음 기능을 검증한다.

    Command-line Argument

    Project Path 처리

    CNNBaseline 검증

    Test Dataset 구성 검증

    Model State 복사·비교

    Evaluation Result 교차 검증

    Sample Prediction Record

    Evaluation JSON Artifact

    Atomic JSON 저장

    --validate-only 실행

    전체 Evaluation Orchestration

중요
----
실제 Test Dataset 715장에 대한 CNN Forward는
이미 실제 실행으로 검증했다.

실제 결과:

    Test Loss:

        0.453337371391

    Accuracy:

        76.92%

    Precision:

        82.88%

    Recall:

        80.13%

    F1 Score:

        81.48%

현재 단위 테스트에서는
실제 이미지 추론을 반복하지 않는다.

대신:

    Synthetic Evaluation Result

    Mock Evaluation Function

    Temporary JSON Path

를 사용한다.

따라서 테스트 실행 시간을 줄이면서
Script 연결 구조와 방어 로직을 검증한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)

import scripts.run_day3_cnn_baseline_evaluation as evaluation_script

from scripts.run_day3_cnn_baseline_evaluation import (
    DEFAULT_EVALUATION_OUTPUT_PATH,
    EVALUATION_ARTIFACT_VERSION,
    EXPECTED_CNN_PARAMETER_COUNT,
    EXPECTED_TEST_BATCH_COUNT,
    EXPECTED_TEST_DEFECT_COUNT,
    EXPECTED_TEST_NORMAL_COUNT,
    EXPECTED_TEST_SAMPLE_COUNT,
    build_evaluation_artifact,
    build_sample_prediction_records,
    clone_model_state,
    convert_to_portable_project_path,
    count_model_parameters,
    get_class_name,
    get_validated_test_samples,
    main,
    model_states_are_equal,
    parse_arguments,
    resolve_project_path,
    run_evaluation,
    validate_cnn_model,
    validate_evaluation_artifact,
    validate_evaluation_results,
    write_json_artifact,
)
from src.data.data_loader import (
    VisionDataLoaders,
)
from src.data.dataset_config import (
    INDEX_TO_CLASS_NAME,
    PROJECT_ROOT,
)
from src.data.dataset_split import (
    ImageSample,
)
from src.evaluation.classification_metrics import (
    BinaryClassificationMetrics,
    calculate_binary_classification_metrics,
)
from src.evaluation.evaluation_runner import (
    BinaryEvaluationResult,
)
from src.models.cnn_baseline import (
    CNNBaseline,
)
from src.reproducibility import (
    ReproducibilitySettings,
)
from src.training.checkpoint_loader import (
    LoadedCheckpointInfo,
)
from src.training.training_pipeline import (
    BEST_MODEL_SELECTION_METRIC,
    CHECKPOINT_VERSION,
    DEFAULT_CNN_CHECKPOINT_PATH,
)


# =============================================================================
# Synthetic Dataset
# =============================================================================


class ImageSampleDataset(
    Dataset[
        tuple[
            Tensor,
            Tensor,
        ]
    ]
):
    """
    ImageSample 목록을 보관하는 Synthetic Dataset.

    실제 이미지 파일은 열지 않는다.

    DataLoader 길이와 Dataset Sample Metadata를
    테스트하기 위한 Dataset이다.
    """

    def __init__(
        self,
        samples: tuple[
            ImageSample,
            ...,
        ],
    ) -> None:
        """
        ImageSample 목록을 저장한다.
        """
        self.samples = samples

    def __len__(self) -> int:
        """
        전체 Sample 수를 반환한다.
        """
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:
        """
        실제 파일 대신 작은 Dummy RGB Image를 반환한다.
        """
        sample = self.samples[
            index
        ]

        image = torch.zeros(
            3,
            8,
            8,
            dtype=torch.float32,
        )

        label = torch.tensor(
            sample.label,
            dtype=torch.int64,
        )

        return (
            image,
            label,
        )


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_samples(
    root_path: Path,
    *,
    normal_count: int = (
        EXPECTED_TEST_NORMAL_COUNT
    ),
    defect_count: int = (
        EXPECTED_TEST_DEFECT_COUNT
    ),
) -> tuple[
    ImageSample,
    ...,
]:
    """
    Synthetic Test ImageSample 목록을 생성한다.

    기본 구성
    --------
    NORMAL:

        262

    DEFECT:

        453

    Total:

        715
    """
    samples: list[
        ImageSample
    ] = []

    for sample_index in range(
        normal_count
    ):
        samples.append(
            ImageSample(
                image_path=(
                    root_path
                    / "normal"
                    / (
                        f"normal_"
                        f"{sample_index:04d}.jpeg"
                    )
                ),
                label=0,
            )
        )

    for sample_index in range(
        defect_count
    ):
        samples.append(
            ImageSample(
                image_path=(
                    root_path
                    / "defect"
                    / (
                        f"defect_"
                        f"{sample_index:04d}.jpeg"
                    )
                ),
                label=1,
            )
        )

    return tuple(
        samples
    )


def create_vision_data_loaders(
    samples: tuple[
        ImageSample,
        ...,
    ],
    *,
    batch_size: int = 32,
) -> VisionDataLoaders:
    """
    Synthetic VisionDataLoaders를 생성한다.

    실제 Image Transform은 실행하지 않는다.

    get_validated_test_samples()의:

        Dataset Size

        Sample Metadata

        Batch Count

    검증에 사용한다.
    """
    dataset = ImageSampleDataset(
        samples=samples,
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return VisionDataLoaders(
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        train_loader=data_loader,
        validation_loader=data_loader,
        test_loader=data_loader,
    )


def create_evaluation_result(
    samples: tuple[
        ImageSample,
        ...,
    ],
    *,
    predictions: Tensor | None = None,
    batch_count: int | None = None,
) -> BinaryEvaluationResult:
    """
    Synthetic BinaryEvaluationResult를 생성한다.

    기본 Prediction
    ---------------
    Ground Truth와 동일

    따라서:

        Accuracy = 1.0

    Logit
    -----
    Prediction 0:

        -2.0

    Prediction 1:

        +2.0
    """
    labels = torch.tensor(
        [
            sample.label
            for sample in samples
        ],
        dtype=torch.int64,
    )

    if predictions is None:
        predictions = labels.clone()

    predictions = (
        predictions
        .detach()
        .to(
            dtype=torch.int64,
            device="cpu",
        )
        .clone()
    )

    logits = torch.where(
        predictions.eq(
            1
        ),
        torch.full(
            (
                len(
                    samples
                ),
            ),
            2.0,
            dtype=torch.float32,
        ),
        torch.full(
            (
                len(
                    samples
                ),
            ),
            -2.0,
            dtype=torch.float32,
        ),
    )

    probabilities = torch.sigmoid(
        logits
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

    if batch_count is None:
        batch_count = max(
            1,
            math.ceil(
                len(
                    samples
                )
                / 32
            ),
        )

    return BinaryEvaluationResult(
        average_loss=0.4,
        accuracy=accuracy,
        sample_count=(
            len(
                samples
            )
        ),
        batch_count=(
            batch_count
        ),
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


def create_checkpoint_info(
    checkpoint_path: Path,
) -> LoadedCheckpointInfo:
    """
    Day 3 Best CNN Checkpoint Metadata를 생성한다.
    """
    return LoadedCheckpointInfo(
        checkpoint_path=(
            checkpoint_path
        ),
        checkpoint_version=(
            CHECKPOINT_VERSION
        ),
        model_name=(
            "CNNBaseline"
        ),
        model_module=(
            "src.models.cnn_baseline"
        ),
        loss_function_name=(
            "BCEWithLogitsLoss"
        ),
        optimizer_name=(
            "Adam"
        ),
        epoch_number=2,
        configured_epoch_count=5,
        classification_threshold=0.5,
        best_model_selection_metric=(
            BEST_MODEL_SELECTION_METRIC
        ),
        validation_loss=(
            0.465498844753
        ),
        validation_accuracy=(
            0.769404672193
        ),
        validation_sample_count=(
            1_327
        ),
        validation_batch_count=(
            42
        ),
    )


def create_reproducibility_settings() -> (
    ReproducibilitySettings
):
    """
    CPU Test용 재현성 설정을 생성한다.
    """
    return ReproducibilitySettings(
        seed=42,
        deterministic_algorithms=False,
        cuda_available=False,
        cuda_seed_applied=False,
        device="cpu",
    )


def create_valid_artifact_bundle(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    tuple[
        ImageSample,
        ...,
    ],
    BinaryEvaluationResult,
    BinaryClassificationMetrics,
]:
    """
    정상 Evaluation Artifact와 관련 객체를 생성한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=(
                EXPECTED_TEST_BATCH_COUNT
            ),
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    prediction_records = (
        build_sample_prediction_records(
            test_samples=samples,
            evaluation_result=(
                evaluation_result
            ),
        )
    )

    checkpoint_path = (
        tmp_path
        / "cnn_baseline_best.pt"
    )

    output_path = (
        tmp_path
        / "evaluation.json"
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
                EXPECTED_CNN_PARAMETER_COUNT
            ),
            reproducibility_settings=(
                create_reproducibility_settings()
            ),
            checkpoint_info=(
                create_checkpoint_info(
                    checkpoint_path=(
                        checkpoint_path
                    )
                )
            ),
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            prediction_records=(
                prediction_records
            ),
            elapsed_seconds=1.25,
        )
    )

    return (
        artifact,
        samples,
        evaluation_result,
        metrics,
    )


# =============================================================================
# Public Constants
# =============================================================================


def test_evaluation_artifact_version() -> None:
    """
    Evaluation Artifact Version을 확인한다.
    """
    assert (
        EVALUATION_ARTIFACT_VERSION
        == 1
    )


def test_default_evaluation_output_path() -> None:
    """
    기본 Evaluation JSON 경로를 확인한다.
    """
    assert (
        DEFAULT_EVALUATION_OUTPUT_PATH
        == (
            Path(
                "reports"
            )
            / "artifacts"
            / (
                "day3_cnn_baseline_"
                "test_evaluation.json"
            )
        )
    )


def test_expected_test_dataset_counts() -> None:
    """
    실제 Test Dataset 고정 구성을 확인한다.
    """
    assert (
        EXPECTED_TEST_SAMPLE_COUNT
        == 715
    )

    assert (
        EXPECTED_TEST_NORMAL_COUNT
        == 262
    )

    assert (
        EXPECTED_TEST_DEFECT_COUNT
        == 453
    )

    assert (
        EXPECTED_TEST_BATCH_COUNT
        == 23
    )

    assert (
        EXPECTED_TEST_NORMAL_COUNT
        + EXPECTED_TEST_DEFECT_COUNT
        == EXPECTED_TEST_SAMPLE_COUNT
    )


def test_expected_cnn_parameter_count() -> None:
    """
    CNNBaseline Parameter 기준값을 확인한다.
    """
    assert (
        EXPECTED_CNN_PARAMETER_COUNT
        == 6_065
    )


# =============================================================================
# Argument Parser
# =============================================================================


def test_parse_arguments_uses_defaults() -> None:
    """
    인자 없이 실행하면 기본 경로를 사용하는지 확인한다.
    """
    arguments = (
        parse_arguments(
            arguments=[]
        )
    )

    assert (
        arguments.checkpoint_path
        == DEFAULT_CNN_CHECKPOINT_PATH
    )

    assert (
        arguments.output_path
        == DEFAULT_EVALUATION_OUTPUT_PATH
    )

    assert (
        arguments.validate_only
        is False
    )


def test_parse_arguments_accepts_custom_paths() -> None:
    """
    사용자 지정 Checkpoint·Output 경로를 파싱하는지 확인한다.
    """
    arguments = (
        parse_arguments(
            arguments=[
                "--checkpoint-path",
                "custom/model.pth",
                "--output-path",
                "custom/result.json",
            ]
        )
    )

    assert (
        arguments.checkpoint_path
        == Path(
            "custom/model.pth"
        )
    )

    assert (
        arguments.output_path
        == Path(
            "custom/result.json"
        )
    )


def test_parse_arguments_accepts_validate_only() -> None:
    """
    --validate-only Flag를 파싱하는지 확인한다.
    """
    arguments = (
        parse_arguments(
            arguments=[
                "--validate-only",
            ]
        )
    )

    assert (
        arguments.validate_only
        is True
    )


# =============================================================================
# Path Resolution
# =============================================================================


def test_resolve_project_path_uses_project_root_for_relative_path() -> None:
    """
    상대 경로를 Project Root 기준으로 변환하는지 확인한다.
    """
    relative_path = (
        Path(
            "reports"
        )
        / "result.json"
    )

    resolved_path = (
        resolve_project_path(
            path=relative_path,
        )
    )

    expected_path = (
        PROJECT_ROOT
        / relative_path
    ).resolve()

    assert (
        resolved_path
        == expected_path
    )


def test_resolve_project_path_preserves_absolute_path(
    tmp_path: Path,
) -> None:
    """
    절대 경로는 Project Root를 다시 결합하지 않는지 확인한다.
    """
    absolute_path = (
        tmp_path
        / "result.json"
    ).resolve()

    resolved_path = (
        resolve_project_path(
            path=absolute_path,
        )
    )

    assert (
        resolved_path
        == absolute_path
    )


@pytest.mark.parametrize(
    "invalid_path",
    [
        None,
        "result.json",
        123,
        object(),
    ],
)
def test_resolve_project_path_rejects_non_path(
    invalid_path: object,
) -> None:
    """
    pathlib.Path가 아닌 입력을 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "path must be "
            "a pathlib.Path"
        ),
    ):
        resolve_project_path(
            path=invalid_path,  # type: ignore[arg-type]
        )


def test_convert_internal_path_to_portable_relative_path() -> None:
    """
    Project 내부 경로를 POSIX 상대 경로로 변환하는지 확인한다.
    """
    internal_path = (
        PROJECT_ROOT
        / "reports"
        / "artifacts"
        / "result.json"
    )

    portable_path = (
        convert_to_portable_project_path(
            path=internal_path,
        )
    )

    assert (
        portable_path
        == (
            "reports/artifacts/"
            "result.json"
        )
    )


def test_convert_external_path_to_absolute_string(
    tmp_path: Path,
) -> None:
    """
    Project 외부 경로는 절대 경로 문자열로 유지하는지 확인한다.
    """
    external_path = (
        tmp_path
        / "result.json"
    )

    portable_path = (
        convert_to_portable_project_path(
            path=external_path,
        )
    )

    assert (
        portable_path
        == str(
            external_path.resolve()
        )
    )


# =============================================================================
# Class Name
# =============================================================================


def test_get_class_name_returns_configured_names() -> None:
    """
    Label 0·1을 현재 Class 이름으로 변환하는지 확인한다.
    """
    assert (
        get_class_name(
            label=0
        )
        == str(
            INDEX_TO_CLASS_NAME[
                0
            ]
        )
    )

    assert (
        get_class_name(
            label=1
        )
        == str(
            INDEX_TO_CLASS_NAME[
                1
            ]
        )
    )


def test_get_class_name_rejects_unknown_label() -> None:
    """
    정의되지 않은 Label을 거부하는지 확인한다.
    """
    with pytest.raises(
        KeyError,
        match=(
            "label is not defined"
        ),
    ):
        get_class_name(
            label=2
        )


# =============================================================================
# CNN Model Validation
# =============================================================================


def test_count_model_parameters_returns_expected_count() -> None:
    """
    CNNBaseline 전체 Parameter 수를 확인한다.
    """
    model = CNNBaseline()

    assert (
        count_model_parameters(
            model=model,
        )
        == EXPECTED_CNN_PARAMETER_COUNT
    )


def test_validate_cnn_model_accepts_valid_model() -> None:
    """
    정상 CPU CNNBaseline을 허용하는지 확인한다.
    """
    model = CNNBaseline()

    parameter_count = (
        validate_cnn_model(
            model=model,
            device=torch.device(
                "cpu"
            ),
        )
    )

    assert (
        parameter_count
        == EXPECTED_CNN_PARAMETER_COUNT
    )


def test_validate_cnn_model_rejects_wrong_model_type() -> None:
    """
    CNNBaseline이 아닌 Model을 거부하는지 확인한다.
    """
    model = nn.Linear(
        2,
        1,
    )

    with pytest.raises(
        TypeError,
        match=(
            "model must be "
            "a CNNBaseline"
        ),
    ):
        validate_cnn_model(
            model=model,  # type: ignore[arg-type]
            device=torch.device(
                "cpu"
            ),
        )


def test_validate_cnn_model_rejects_wrong_parameter_count() -> None:
    """
    CNNBaseline 구조에 예상하지 않은 Parameter가 추가되면 거부한다.
    """

    class ExtraParameterCNN(
        CNNBaseline
    ):
        """
        추가 Parameter를 가진 잘못된 CNNBaseline.
        """

        def __init__(
            self,
        ) -> None:
            super().__init__()

            self.extra_parameter = (
                nn.Parameter(
                    torch.tensor(
                        1.0,
                        dtype=torch.float32,
                    )
                )
            )

    model = ExtraParameterCNN()

    with pytest.raises(
        ValueError,
        match=(
            "unexpected CNNBaseline "
            "Parameter count"
        ),
    ):
        validate_cnn_model(
            model=model,
            device=torch.device(
                "cpu"
            ),
        )


def test_validate_cnn_model_rejects_device_mismatch() -> None:
    """
    Model Parameter와 요청 Device가 다르면 거부하는지 확인한다.
    """
    model = (
        CNNBaseline()
        .to(
            device="meta"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "unexpected device"
        ),
    ):
        validate_cnn_model(
            model=model,
            device=torch.device(
                "cpu"
            ),
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_cnn_model_rejects_non_finite_parameter(
    invalid_value: float,
) -> None:
    """
    NaN·inf Parameter를 거부하는지 확인한다.
    """
    model = CNNBaseline()

    first_parameter = next(
        model.parameters()
    )

    with torch.no_grad():
        first_parameter.view(
            -1
        )[
            0
        ] = invalid_value

    with pytest.raises(
        ValueError,
        match=(
            "must contain only "
            "finite values"
        ),
    ):
        validate_cnn_model(
            model=model,
            device=torch.device(
                "cpu"
            ),
        )


# =============================================================================
# Model State Helpers
# =============================================================================


def test_clone_model_state_creates_independent_copy() -> None:
    """
    복사 후 원본 Model을 변경해도 복사본이 유지되는지 확인한다.
    """
    model = CNNBaseline()

    cloned_state = (
        clone_model_state(
            model=model,
        )
    )

    first_state_name = next(
        iter(
            cloned_state.keys()
        )
    )

    expected_tensor = (
        cloned_state[
            first_state_name
        ]
        .clone()
    )

    with torch.no_grad():
        next(
            model.parameters()
        ).add_(
            1.0
        )

    assert torch.equal(
        cloned_state[
            first_state_name
        ],
        expected_tensor,
    )


def test_model_states_are_equal_returns_true_for_equal_states() -> None:
    """
    동일 State Dictionary를 True로 판단하는지 확인한다.
    """
    model = CNNBaseline()

    first_state = (
        clone_model_state(
            model=model,
        )
    )

    second_state = {
        state_name: (
            state_tensor.clone()
        )
        for (
            state_name,
            state_tensor,
        ) in first_state.items()
    }

    assert (
        model_states_are_equal(
            first_state=(
                first_state
            ),
            second_state=(
                second_state
            ),
        )
        is True
    )


def test_model_states_are_equal_rejects_missing_key() -> None:
    """
    State Key가 다르면 False인지 확인한다.
    """
    model = CNNBaseline()

    first_state = (
        clone_model_state(
            model=model,
        )
    )

    second_state = {
        state_name: (
            state_tensor.clone()
        )
        for (
            state_name,
            state_tensor,
        ) in first_state.items()
    }

    removed_key = next(
        iter(
            second_state.keys()
        )
    )

    del second_state[
        removed_key
    ]

    assert (
        model_states_are_equal(
            first_state=(
                first_state
            ),
            second_state=(
                second_state
            ),
        )
        is False
    )


def test_model_states_are_equal_rejects_changed_tensor() -> None:
    """
    Tensor 값이 다르면 False인지 확인한다.
    """
    model = CNNBaseline()

    first_state = (
        clone_model_state(
            model=model,
        )
    )

    second_state = {
        state_name: (
            state_tensor.clone()
        )
        for (
            state_name,
            state_tensor,
        ) in first_state.items()
    }

    changed_key = next(
        iter(
            second_state.keys()
        )
    )

    second_state[
        changed_key
    ].view(
        -1
    )[
        0
    ] += 1.0

    assert (
        model_states_are_equal(
            first_state=(
                first_state
            ),
            second_state=(
                second_state
            ),
        )
        is False
    )


# =============================================================================
# Test Dataset Validation
# =============================================================================


def test_get_validated_test_samples_accepts_expected_dataset(
    tmp_path: Path,
) -> None:
    """
    715·262·453·23 구성을 허용하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    data_loaders = (
        create_vision_data_loaders(
            samples=samples,
            batch_size=32,
        )
    )

    validated_samples = (
        get_validated_test_samples(
            data_loaders=(
                data_loaders
            )
        )
    )

    assert (
        validated_samples
        == samples
    )

    assert (
        len(
            validated_samples
        )
        == EXPECTED_TEST_SAMPLE_COUNT
    )


def test_get_validated_test_samples_rejects_wrong_total(
    tmp_path: Path,
) -> None:
    """
    Test Sample 수가 715가 아니면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        ),
        normal_count=261,
        defect_count=453,
    )

    data_loaders = (
        create_vision_data_loaders(
            samples=samples,
            batch_size=32,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "unexpected Test Dataset "
            "sample count"
        ),
    ):
        get_validated_test_samples(
            data_loaders=(
                data_loaders
            )
        )


def test_get_validated_test_samples_rejects_wrong_batch_count(
    tmp_path: Path,
) -> None:
    """
    Test Batch 수가 23이 아니면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    data_loaders = (
        create_vision_data_loaders(
            samples=samples,
            batch_size=64,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "unexpected Test DataLoader "
            "batch count"
        ),
    ):
        get_validated_test_samples(
            data_loaders=(
                data_loaders
            )
        )


def test_get_validated_test_samples_rejects_wrong_class_count(
    tmp_path: Path,
) -> None:
    """
    전체 715장이더라도 NORMAL·DEFECT 수가 다르면 거부한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        ),
        normal_count=261,
        defect_count=454,
    )

    data_loaders = (
        create_vision_data_loaders(
            samples=samples,
            batch_size=32,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "unexpected Test NORMAL "
            "sample count"
        ),
    ):
        get_validated_test_samples(
            data_loaders=(
                data_loaders
            )
        )


def test_get_validated_test_samples_rejects_wrong_object_type() -> None:
    """
    VisionDataLoaders가 아닌 객체를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "must be a "
            "VisionDataLoaders"
        ),
    ):
        get_validated_test_samples(
            data_loaders=object(),  # type: ignore[arg-type]
        )


# =============================================================================
# Evaluation Result Cross Validation
# =============================================================================


def test_validate_evaluation_results_accepts_consistent_results(
    tmp_path: Path,
) -> None:
    """
    Dataset·Evaluation·Metric·Model이 모두 일치하면 통과하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=(
                EXPECTED_TEST_BATCH_COUNT
            ),
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    model_state_before = (
        clone_model_state(
            model=model,
        )
    )

    validate_evaluation_results(
        evaluation_result=(
            evaluation_result
        ),
        metrics=metrics,
        test_samples=samples,
        model=model,
        model_state_before=(
            model_state_before
        ),
    )


def test_validate_evaluation_results_rejects_wrong_evaluation_sample_count(
    tmp_path: Path,
) -> None:
    """
    Evaluation Sample 수가 715가 아니면 거부하는지 확인한다.
    """
    samples = (
        create_test_samples(
            root_path=(
                tmp_path
                / "small"
            ),
            normal_count=1,
            defect_count=1,
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=1,
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    with pytest.raises(
        ValueError,
        match=(
            "Evaluation sample count "
            "is unexpected"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            test_samples=samples,
            model=model,
            model_state_before=(
                clone_model_state(
                    model=model,
                )
            ),
        )


def test_validate_evaluation_results_rejects_metric_sample_count(
    tmp_path: Path,
) -> None:
    """
    Metric Sample 수가 715가 아니면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    small_metrics = (
        calculate_binary_classification_metrics(
            labels=torch.tensor(
                [
                    1,
                ],
                dtype=torch.int64,
            ),
            predictions=torch.tensor(
                [
                    1,
                ],
                dtype=torch.int64,
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    with pytest.raises(
        ValueError,
        match=(
            "Metric sample count "
            "is unexpected"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=(
                small_metrics
            ),
            test_samples=samples,
            model=model,
            model_state_before=(
                clone_model_state(
                    model=model,
                )
            ),
        )


def test_validate_evaluation_results_rejects_accuracy_mismatch(
    tmp_path: Path,
) -> None:
    """
    Evaluation Accuracy와 Metric Accuracy가 다르면 거부한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    different_predictions = (
        torch.ones(
            EXPECTED_TEST_SAMPLE_COUNT,
            dtype=torch.int64,
        )
    )

    different_metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                different_predictions
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    with pytest.raises(
        ValueError,
        match=(
            "Evaluation Accuracy and "
            "Classification Metrics Accuracy "
            "must match"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=(
                different_metrics
            ),
            test_samples=samples,
            model=model,
            model_state_before=(
                clone_model_state(
                    model=model,
                )
            ),
        )


def test_validate_evaluation_results_rejects_dataset_order_mismatch(
    tmp_path: Path,
) -> None:
    """
    Evaluation Label 순서와 Dataset 순서가 다르면 거부한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    reversed_samples = tuple(
        reversed(
            samples
        )
    )

    model = CNNBaseline()

    model.eval()

    with pytest.raises(
        ValueError,
        match=(
            "must preserve the "
            "Test Dataset sample order"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            test_samples=(
                reversed_samples
            ),
            model=model,
            model_state_before=(
                clone_model_state(
                    model=model,
                )
            ),
        )


def test_validate_evaluation_results_rejects_training_mode(
    tmp_path: Path,
) -> None:
    """
    평가 후 Model이 Train Mode이면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    model = CNNBaseline()

    model.train()

    with pytest.raises(
        ValueError,
        match=(
            "must be in Evaluation Mode"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            test_samples=samples,
            model=model,
            model_state_before=(
                clone_model_state(
                    model=model,
                )
            ),
        )


def test_validate_evaluation_results_rejects_changed_model_state(
    tmp_path: Path,
) -> None:
    """
    평가 중 Model Weight가 변경되면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    model_state_before = (
        clone_model_state(
            model=model,
        )
    )

    with torch.no_grad():
        next(
            model.parameters()
        ).add_(
            1.0
        )

    with pytest.raises(
        ValueError,
        match=(
            "Model State changed "
            "during Test evaluation"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            test_samples=samples,
            model=model,
            model_state_before=(
                model_state_before
            ),
        )


def test_validate_evaluation_results_rejects_parameter_gradient(
    tmp_path: Path,
) -> None:
    """
    평가 후 Parameter Gradient가 남으면 거부하는지 확인한다.
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    metrics = (
        calculate_binary_classification_metrics(
            labels=(
                evaluation_result.labels
            ),
            predictions=(
                evaluation_result
                .predictions
            ),
        )
    )

    model = CNNBaseline()

    model.eval()

    model_state_before = (
        clone_model_state(
            model=model,
        )
    )

    first_parameter = next(
        model.parameters()
    )

    first_parameter.grad = (
        torch.zeros_like(
            first_parameter
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Gradient must remain None"
        ),
    ):
        validate_evaluation_results(
            evaluation_result=(
                evaluation_result
            ),
            metrics=metrics,
            test_samples=samples,
            model=model,
            model_state_before=(
                model_state_before
            ),
        )


# =============================================================================
# Sample Prediction Records
# =============================================================================


def test_build_sample_prediction_records(
    tmp_path: Path,
) -> None:
    """
    Sample별 Prediction Record를 정확히 생성하는지 확인한다.
    """
    samples = (
        create_test_samples(
            root_path=(
                tmp_path
                / "images"
            ),
            normal_count=2,
            defect_count=2,
        )
    )

    predictions = torch.tensor(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=torch.int64,
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            predictions=predictions,
            batch_count=1,
        )
    )

    records = (
        build_sample_prediction_records(
            test_samples=samples,
            evaluation_result=(
                evaluation_result
            ),
        )
    )

    assert len(
        records
    ) == 4

    assert [
        record[
            "sample_index"
        ]
        for record in records
    ] == [
        0,
        1,
        2,
        3,
    ]

    assert [
        record[
            "label"
        ]
        for record in records
    ] == [
        0,
        0,
        1,
        1,
    ]

    assert [
        record[
            "prediction"
        ]
        for record in records
    ] == [
        0,
        1,
        0,
        1,
    ]

    assert [
        record[
            "is_correct"
        ]
        for record in records
    ] == [
        True,
        False,
        False,
        True,
    ]

    assert (
        records[
            0
        ][
            "label_name"
        ]
        == get_class_name(
            label=0
        )
    )

    assert (
        records[
            2
        ][
            "label_name"
        ]
        == get_class_name(
            label=1
        )
    )

    assert isinstance(
        records[
            0
        ][
            "logit"
        ],
        float,
    )

    assert isinstance(
        records[
            0
        ][
            "defect_probability"
        ],
        float,
    )


def test_build_sample_prediction_records_rejects_length_mismatch(
    tmp_path: Path,
) -> None:
    """
    Sample 수와 Evaluation Result 수가 다르면 거부한다.
    """
    samples = (
        create_test_samples(
            root_path=(
                tmp_path
                / "images"
            ),
            normal_count=1,
            defect_count=1,
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=1,
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Test Sample count must match"
        ),
    ):
        build_sample_prediction_records(
            test_samples=(
                samples[
                    :1
                ]
            ),
            evaluation_result=(
                evaluation_result
            ),
        )


def test_build_sample_prediction_records_rejects_label_mismatch(
    tmp_path: Path,
) -> None:
    """
    Dataset Label과 Evaluation Label이 다르면 거부한다.
    """
    samples = (
        create_test_samples(
            root_path=(
                tmp_path
                / "images"
            ),
            normal_count=1,
            defect_count=1,
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=1,
        )
    )

    changed_samples = (
        ImageSample(
            image_path=(
                samples[
                    0
                ].image_path
            ),
            label=1,
        ),
        samples[
            1
        ],
    )

    with pytest.raises(
        ValueError,
        match=(
            "Evaluation Label must match"
        ),
    ):
        build_sample_prediction_records(
            test_samples=(
                changed_samples
            ),
            evaluation_result=(
                evaluation_result
            ),
        )


# =============================================================================
# Evaluation Artifact
# =============================================================================


def test_build_evaluation_artifact_contains_expected_summary(
    tmp_path: Path,
) -> None:
    """
    Evaluation Artifact 핵심 Metadata·Metric을 확인한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    assert (
        artifact[
            "artifact_version"
        ]
        == 1
    )

    assert (
        artifact[
            "project_name"
        ]
        == (
            "Manufacturing Vision "
            "Defect Analysis System"
        )
    )

    assert (
        artifact[
            "stage"
        ]
        == (
            "day3_cnn_baseline_"
            "test_evaluation"
        )
    )

    assert (
        artifact[
            "model"
        ][
            "parameter_count"
        ]
        == 6_065
    )

    assert (
        artifact[
            "checkpoint"
        ][
            "best_epoch"
        ]
        == 2
    )

    assert (
        artifact[
            "dataset"
        ][
            "sample_count"
        ]
        == 715
    )

    assert (
        artifact[
            "evaluation"
        ][
            "accuracy"
        ]
        == 1.0
    )

    assert (
        artifact[
            "evaluation"
        ][
            "true_negative"
        ]
        == 262
    )

    assert (
        artifact[
            "evaluation"
        ][
            "true_positive"
        ]
        == 453
    )

    assert (
        len(
            artifact[
                "sample_predictions"
            ]
        )
        == 715
    )


def test_validate_evaluation_artifact_accepts_valid_artifact(
    tmp_path: Path,
) -> None:
    """
    정상 Artifact를 허용하는지 확인한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    validate_evaluation_artifact(
        artifact=artifact,
    )


def test_validate_evaluation_artifact_rejects_non_dictionary() -> None:
    """
    Dictionary가 아닌 Artifact를 거부하는지 확인한다.
    """
    with pytest.raises(
        TypeError,
        match=(
            "artifact must be "
            "a dictionary"
        ),
    ):
        validate_evaluation_artifact(
            artifact=[],  # type: ignore[arg-type]
        )


def test_validate_evaluation_artifact_rejects_wrong_version(
    tmp_path: Path,
) -> None:
    """
    지원하지 않는 Artifact Version을 거부한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    invalid_artifact = (
        copy.deepcopy(
            artifact
        )
    )

    invalid_artifact[
        "artifact_version"
    ] = 2

    with pytest.raises(
        ValueError,
        match=(
            "unexpected Evaluation "
            "Artifact Version"
        ),
    ):
        validate_evaluation_artifact(
            artifact=(
                invalid_artifact
            ),
        )


def test_validate_evaluation_artifact_rejects_invalid_dataset(
    tmp_path: Path,
) -> None:
    """
    Dataset Metadata가 Dictionary가 아니면 거부한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    invalid_artifact = (
        copy.deepcopy(
            artifact
        )
    )

    invalid_artifact[
        "dataset"
    ] = []

    with pytest.raises(
        TypeError,
        match=(
            "artifact.dataset must "
            "be a dictionary"
        ),
    ):
        validate_evaluation_artifact(
            artifact=(
                invalid_artifact
            ),
        )


def test_validate_evaluation_artifact_rejects_wrong_sample_count(
    tmp_path: Path,
) -> None:
    """
    Artifact Test Sample 수가 715가 아니면 거부한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    invalid_artifact = (
        copy.deepcopy(
            artifact
        )
    )

    invalid_artifact[
        "dataset"
    ][
        "sample_count"
    ] = 714

    with pytest.raises(
        ValueError,
        match=(
            "sample count is unexpected"
        ),
    ):
        validate_evaluation_artifact(
            artifact=(
                invalid_artifact
            ),
        )


def test_validate_evaluation_artifact_rejects_missing_metric(
    tmp_path: Path,
) -> None:
    """
    필수 Evaluation Metric Key 누락을 거부한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    invalid_artifact = (
        copy.deepcopy(
            artifact
        )
    )

    del invalid_artifact[
        "evaluation"
    ][
        "recall"
    ]

    with pytest.raises(
        KeyError,
        match=(
            "missing required keys"
        ),
    ):
        validate_evaluation_artifact(
            artifact=(
                invalid_artifact
            ),
        )


def test_validate_evaluation_artifact_rejects_prediction_count(
    tmp_path: Path,
) -> None:
    """
    Sample Prediction Record 수가 715가 아니면 거부한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    invalid_artifact = (
        copy.deepcopy(
            artifact
        )
    )

    invalid_artifact[
        "sample_predictions"
    ] = (
        invalid_artifact[
            "sample_predictions"
        ][
            :-1
        ]
    )

    with pytest.raises(
        ValueError,
        match=(
            "Sample Prediction count "
            "is unexpected"
        ),
    ):
        validate_evaluation_artifact(
            artifact=(
                invalid_artifact
            ),
        )


# =============================================================================
# JSON Artifact Writing
# =============================================================================


def test_write_json_artifact_creates_valid_file(
    tmp_path: Path,
) -> None:
    """
    JSON Artifact를 UTF-8 파일로 저장하는지 확인한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    output_path = (
        tmp_path
        / "nested"
        / "evaluation.json"
    )

    write_json_artifact(
        output_path=(
            output_path
        ),
        artifact=artifact,
    )

    assert (
        output_path.is_file()
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded_artifact = (
            json.load(
                file
            )
        )

    assert (
        loaded_artifact
        == artifact
    )

    assert (
        output_path
        .with_suffix(
            ".json.tmp"
        )
        .exists()
        is False
    )


def test_write_json_artifact_replaces_existing_file(
    tmp_path: Path,
) -> None:
    """
    기존 파일을 최종 검증된 JSON으로 교체하는지 확인한다.
    """
    (
        artifact,
        _,
        _,
        _,
    ) = (
        create_valid_artifact_bundle(
            tmp_path=tmp_path
        )
    )

    output_path = (
        tmp_path
        / "evaluation.json"
    )

    output_path.write_text(
        "old-content",
        encoding="utf-8",
    )

    write_json_artifact(
        output_path=(
            output_path
        ),
        artifact=artifact,
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        loaded_artifact = (
            json.load(
                file
            )
        )

    assert (
        loaded_artifact[
            "artifact_version"
        ]
        == 1
    )

    assert (
        loaded_artifact[
            "dataset"
        ][
            "sample_count"
        ]
        == 715
    )


def test_write_json_artifact_rejects_invalid_artifact(
    tmp_path: Path,
) -> None:
    """
    잘못된 Artifact는 파일 저장 전에 거부하는지 확인한다.
    """
    output_path = (
        tmp_path
        / "invalid.json"
    )

    with pytest.raises(
        ValueError,
    ):
        write_json_artifact(
            output_path=(
                output_path
            ),
            artifact={
                "artifact_version": 999,
            },
        )

    assert (
        output_path.exists()
        is False
    )


# =============================================================================
# Validation-only Execution
# =============================================================================


def test_run_evaluation_validate_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    --validate-only가 실제 Test 추론 없이 0을 반환하는지 확인한다.
    """
    checkpoint_path = (
        tmp_path
        / "best.pt"
    )

    output_path = (
        tmp_path
        / "result.json"
    )

    settings = (
        create_reproducibility_settings()
    )

    checkpoint_info = (
        create_checkpoint_info(
            checkpoint_path=(
                checkpoint_path
            )
        )
    )

    fake_data_loaders = (
        object()
    )

    monkeypatch.setattr(
        evaluation_script,
        "print_environment",
        lambda: None,
    )

    monkeypatch.setattr(
        evaluation_script,
        "configure_reproducibility",
        lambda: settings,
    )

    monkeypatch.setattr(
        evaluation_script,
        "create_vision_data_loaders",
        lambda **_: (
            fake_data_loaders
        ),
    )

    monkeypatch.setattr(
        evaluation_script,
        "get_validated_test_samples",
        lambda **_: tuple(),
    )

    monkeypatch.setattr(
        evaluation_script,
        "load_model_checkpoint",
        lambda **_: (
            checkpoint_info
        ),
    )

    monkeypatch.setattr(
        evaluation_script,
        "print_checkpoint_information",
        lambda **_: None,
    )

    arguments = argparse.Namespace(
        checkpoint_path=(
            checkpoint_path
        ),
        output_path=(
            output_path
        ),
        validate_only=True,
    )

    exit_code = run_evaluation(
        arguments=arguments,
    )

    assert exit_code == 0

    assert (
        output_path.exists()
        is False
    )


# =============================================================================
# Full Orchestration without Real Image Inference
# =============================================================================


def test_run_evaluation_full_orchestration_with_mock_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    실제 이미지 Forward 없이 전체 평가 연결 흐름을 검증한다.

    실제로 수행하는 항목
    -------------------
    CNNBaseline 생성

    Model 검증

    Metric 계산

    Result 교차 검증

    Prediction Record 생성

    Artifact 생성

    Atomic JSON 저장

    최종 파일 확인

    Mock 처리
    ---------
    Dataset Loading

    Checkpoint Loading

    CNN Forward
    """
    samples = create_test_samples(
        root_path=(
            tmp_path
            / "images"
        )
    )

    evaluation_result = (
        create_evaluation_result(
            samples=samples,
            batch_count=23,
        )
    )

    checkpoint_path = (
        tmp_path
        / "cnn_baseline_best.pt"
    )

    output_path = (
        tmp_path
        / "evaluation.json"
    )

    settings = (
        create_reproducibility_settings()
    )

    checkpoint_info = (
        create_checkpoint_info(
            checkpoint_path=(
                checkpoint_path
            )
        )
    )

    fake_data_loaders = (
        SimpleNamespace(
            test_loader=object(),
        )
    )

    def fake_evaluate_binary_classifier(
        *,
        model: nn.Module,
        **_: object,
    ) -> BinaryEvaluationResult:
        """
        실제 Forward 대신 고정 Evaluation Result를 반환한다.
        """
        model.eval()

        return evaluation_result

    monkeypatch.setattr(
        evaluation_script,
        "print_environment",
        lambda: None,
    )

    monkeypatch.setattr(
        evaluation_script,
        "configure_reproducibility",
        lambda: settings,
    )

    monkeypatch.setattr(
        evaluation_script,
        "create_vision_data_loaders",
        lambda **_: (
            fake_data_loaders
        ),
    )

    monkeypatch.setattr(
        evaluation_script,
        "get_validated_test_samples",
        lambda **_: samples,
    )

    monkeypatch.setattr(
        evaluation_script,
        "load_model_checkpoint",
        lambda **_: (
            checkpoint_info
        ),
    )

    monkeypatch.setattr(
        evaluation_script,
        "evaluate_binary_classifier",
        (
            fake_evaluate_binary_classifier
        ),
    )

    monkeypatch.setattr(
        evaluation_script,
        "print_checkpoint_information",
        lambda **_: None,
    )

    monkeypatch.setattr(
        evaluation_script,
        "print_evaluation_results",
        lambda **_: None,
    )

    arguments = argparse.Namespace(
        checkpoint_path=(
            checkpoint_path
        ),
        output_path=(
            output_path
        ),
        validate_only=False,
    )

    exit_code = run_evaluation(
        arguments=arguments,
    )

    assert exit_code == 0

    assert (
        output_path.is_file()
    )

    with output_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        artifact = json.load(
            file
        )

    assert (
        artifact[
            "stage"
        ]
        == (
            "day3_cnn_baseline_"
            "test_evaluation"
        )
    )

    assert (
        artifact[
            "dataset"
        ][
            "sample_count"
        ]
        == 715
    )

    assert (
        artifact[
            "evaluation"
        ][
            "accuracy"
        ]
        == 1.0
    )

    assert (
        len(
            artifact[
                "sample_predictions"
            ]
        )
        == 715
    )


# =============================================================================
# Main Entry Point
# =============================================================================


def test_main_parses_arguments_and_calls_run_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    main()이 Argument를 파싱해 run_evaluation()에 전달하는지 확인한다.
    """
    received_arguments: list[
        argparse.Namespace
    ] = []

    def fake_run_evaluation(
        arguments: argparse.Namespace,
    ) -> int:
        received_arguments.append(
            arguments
        )

        return 7

    monkeypatch.setattr(
        evaluation_script,
        "run_evaluation",
        fake_run_evaluation,
    )

    exit_code = main(
        arguments=[
            "--checkpoint-path",
            "custom/model.pt",
            "--output-path",
            "custom/result.json",
            "--validate-only",
        ]
    )

    assert exit_code == 7

    assert (
        len(
            received_arguments
        )
        == 1
    )

    parsed_arguments = (
        received_arguments[
            0
        ]
    )

    assert (
        parsed_arguments
        .checkpoint_path
        == Path(
            "custom/model.pt"
        )
    )

    assert (
        parsed_arguments
        .output_path
        == Path(
            "custom/result.json"
        )
    )

    assert (
        parsed_arguments
        .validate_only
        is True
    )