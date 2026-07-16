"""Day 4 ResNet18 Test 평가 Script 단위 테스트.

실제 715장 Dataset 평가와 44MB Checkpoint 로딩은 수행하지 않는다.
작은 Tensor와 임시 JSON을 사용해 다음을 검증한다.

1. Sample별 결과와 Dataset 순서 정렬
2. Evaluation Result 내부 일관성 검증
3. Day 3 CNN Metric JSON 추출
4. CNN·ResNet18 Delta와 승자 계산
5. Atomic JSON 저장
6. 출력 경로 충돌 방지
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.run_day4_resnet18_evaluation import (
    build_comparison_payload,
    build_sample_results,
    extract_binary_metrics,
    validate_evaluation_result,
    validate_execution_paths,
    write_json_atomically,
)
from src.evaluation.classification_metrics import (
    BinaryClassificationMetrics,
)
from src.evaluation.evaluation_runner import BinaryEvaluationResult


def create_evaluation_result() -> BinaryEvaluationResult:
    """4개 Sample의 일관된 Binary Evaluation Result를 만든다."""
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    logits = torch.tensor([-2.0, 1.0, -1.0, 2.0])
    probabilities = torch.sigmoid(logits)
    predictions = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

    return BinaryEvaluationResult(
        average_loss=0.5,
        accuracy=0.5,
        sample_count=4,
        batch_count=2,
        classification_threshold=0.5,
        labels=labels,
        logits=logits,
        probabilities=probabilities,
        predictions=predictions,
    )


def create_metrics() -> BinaryClassificationMetrics:
    """TN=1, FP=1, FN=1, TP=1인 Metric을 만든다."""
    return BinaryClassificationMetrics(
        accuracy=0.5,
        precision=0.5,
        recall=0.5,
        f1_score=0.5,
        sample_count=4,
        true_negative=1,
        false_positive=1,
        false_negative=1,
        true_positive=1,
        confusion_matrix=torch.tensor(
            [[1, 1], [1, 1]],
            dtype=torch.int64,
        ),
    )


def create_test_dataset(tmp_path: Path) -> SimpleNamespace:
    """실제 ImageSample과 같은 Attribute를 가진 작은 Dataset 대역."""
    samples = [
        SimpleNamespace(
            image_path=tmp_path / f"sample_{index}.jpeg",
            label=label,
        )
        for index, label in enumerate([0, 0, 1, 1])
    ]
    return SimpleNamespace(samples=samples, __len__=lambda: 4)


def test_build_sample_results_preserves_dataset_order(
    tmp_path: Path,
) -> None:
    """Dataset Sample과 Tensor Index가 정확히 결합되는지 확인한다."""
    dataset = create_test_dataset(tmp_path)
    evaluation_result = create_evaluation_result()

    sample_results = build_sample_results(
        test_dataset=dataset,
        evaluation_result=evaluation_result,
    )

    assert len(sample_results) == 4
    assert [item["sample_index"] for item in sample_results] == [0, 1, 2, 3]
    assert [item["ground_truth_label"] for item in sample_results] == [
        0,
        0,
        1,
        1,
    ]
    assert [item["prediction"] for item in sample_results] == [0, 1, 0, 1]
    assert [item["correct"] for item in sample_results] == [
        True,
        False,
        False,
        True,
    ]
    assert sample_results[0]["ground_truth_class_name"] == "NORMAL"
    assert sample_results[2]["ground_truth_class_name"] == "DEFECT"


def test_build_sample_results_rejects_label_misalignment(
    tmp_path: Path,
) -> None:
    """Dataset과 Evaluation Label 순서가 다르면 저장을 차단한다."""
    dataset = create_test_dataset(tmp_path)
    dataset.samples[0].label = 1

    with pytest.raises(ValueError, match="differ at index 0"):
        build_sample_results(
            test_dataset=dataset,
            evaluation_result=create_evaluation_result(),
        )


def test_validate_evaluation_result_accepts_consistent_result(
    tmp_path: Path,
) -> None:
    """Sample·Tensor·Confusion Matrix가 일치하는 정상 Case."""
    dataset = create_test_dataset(tmp_path)

    # validate 함수는 len(dataset)을 호출하므로 작은 실제 Class를 사용한다.
    class DatasetProxy:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

    validate_evaluation_result(
        evaluation_result=create_evaluation_result(),
        metrics=create_metrics(),
        test_dataset=DatasetProxy(dataset.samples),
    )


def test_validate_evaluation_result_rejects_wrong_confusion_total(
    tmp_path: Path,
) -> None:
    """Confusion Matrix 합계가 Sample 수와 다르면 오류를 발생시킨다."""
    dataset = create_test_dataset(tmp_path)

    class DatasetProxy:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

    # BinaryClassificationMetrics는 생성 시점에 Count 합계를 검증한다.
    # 따라서 정상 객체를 만든 뒤 Confusion Matrix만 의도적으로 손상시켜
    # validate_evaluation_result의 추가 방어 로직을 확인한다.
    invalid_metrics = create_metrics()

    object.__setattr__(
        invalid_metrics,
        "confusion_matrix",
        torch.tensor(
            [
                [1, 1],
                [1, 0],
            ],
            dtype=torch.int64,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Confusion Matrix total",
    ):
        validate_evaluation_result(
            evaluation_result=create_evaluation_result(),
            metrics=invalid_metrics,
            test_dataset=DatasetProxy(dataset.samples),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "metrics": {
                "test_loss": 0.45,
                "accuracy": 0.76,
                "precision": 0.82,
                "recall": 0.80,
                "f1_score": 0.81,
            }
        },
        {
            "evaluation": {
                "average_loss": 0.45,
                "classification_metrics": {
                    "accuracy": 0.76,
                    "precision": 0.82,
                    "recall": 0.80,
                    "f1": 0.81,
                },
            }
        },
    ],
)
def test_extract_binary_metrics_supports_expected_json_shapes(
    payload: dict,
) -> None:
    """Day 3 Artifact의 Metric 위치가 달라도 같은 값으로 정규화한다."""
    metrics = extract_binary_metrics(payload)

    assert metrics == {
        "test_loss": 0.45,
        "accuracy": 0.76,
        "precision": 0.82,
        "recall": 0.80,
        "f1_score": 0.81,
    }


def test_extract_binary_metrics_rejects_incomplete_payload() -> None:
    """Accuracy만 있는 불완전 Artifact를 비교에 사용하지 않는다."""
    with pytest.raises(KeyError, match="Could not find"):
        extract_binary_metrics({"metrics": {"accuracy": 0.76}})


def test_build_comparison_payload_calculates_percentage_point_delta(
    tmp_path: Path,
) -> None:
    """ResNet-CNN 차이와 Metric별 승자를 정확히 계산한다."""
    payload = build_comparison_payload(
        cnn_metrics={
            "test_loss": 0.45,
            "accuracy": 0.76,
            "precision": 0.82,
            "recall": 0.80,
            "f1_score": 0.81,
        },
        resnet18_metrics={
            "test_loss": 0.15,
            "accuracy": 0.97,
            "precision": 0.96,
            "recall": 0.98,
            "f1_score": 0.97,
        },
        cnn_evaluation_path=tmp_path / "cnn.json",
        resnet18_evaluation_path=tmp_path / "resnet.json",
        comparison_output_path=tmp_path / "comparison.json",
    )

    comparison = payload["comparison"]

    assert comparison["test_loss"]["better_model"] == "resnet18_transfer"
    assert comparison["accuracy"]["better_model"] == "resnet18_transfer"
    assert comparison["accuracy"]["resnet_minus_cnn_percentage_points"] == pytest.approx(21.0)
    assert comparison["recall"]["resnet_minus_cnn_percentage_points"] == pytest.approx(18.0)
    assert payload["summary"]["primary_winner"] == "resnet18_transfer"


def test_write_json_atomically_creates_valid_utf8_json(
    tmp_path: Path,
) -> None:
    """한글을 포함한 JSON 저장과 임시 파일 정리를 확인한다."""
    output_path = tmp_path / "artifact.json"

    write_json_atomically(
        payload={"project": "제조 비전 결함 분석 시스템"},
        output_path=output_path,
    )

    assert output_path.is_file()
    assert not (tmp_path / "artifact.json.tmp").exists()

    with output_path.open(encoding="utf-8") as input_file:
        assert json.load(input_file) == {
            "project": "제조 비전 결함 분석 시스템"
        }


def test_validate_execution_paths_rejects_output_overwrite(
    tmp_path: Path,
) -> None:
    """평가 출력이 기존 CNN Artifact를 덮어쓰지 못하게 한다."""
    checkpoint_path = tmp_path / "model.pt"
    training_history_path = tmp_path / "training.json"
    cnn_evaluation_path = tmp_path / "cnn.json"

    checkpoint_path.write_bytes(b"checkpoint")
    training_history_path.write_text("{}", encoding="utf-8")
    cnn_evaluation_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must not overwrite"):
        validate_execution_paths(
            checkpoint_path=checkpoint_path,
            training_history_path=training_history_path,
            cnn_evaluation_path=cnn_evaluation_path,
            evaluation_output_path=cnn_evaluation_path,
            comparison_output_path=tmp_path / "comparison.json",
        )
