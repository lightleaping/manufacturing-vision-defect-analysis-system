"""Day 4 ResNet18 실제 Test 평가 및 CNN 비교 Artifact 생성.

실행 예시
---------
기본 경로를 사용한 실제 평가:

    python -m scripts.run_day4_resnet18_evaluation

구조와 경로만 검증하고 평가하지 않음:

    python -m scripts.run_day4_resnet18_evaluation --validate-only

생성 Artifact
-------------
1. reports/artifacts/day4_resnet18_test_evaluation.json
2. reports/artifacts/day4_cnn_resnet18_comparison.json

설계 핵심
---------
1. 학습 때 저장한 Best Checkpoint를 weights=None Model에 복원한다.
2. Day 3과 동일한 Test Dataset·Transform·Threshold·Loss·Metric을 사용한다.
3. Test DataLoader 순서와 Dataset.samples 순서를 검증한 뒤 715개 예측을 저장한다.
4. Day 3 CNN 평가 JSON에서 Metric을 읽어 ResNet18과 공정하게 비교한다.
5. 두 JSON 모두 임시 파일에 먼저 쓴 뒤 원자적으로 교체한다.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from scripts.run_day4_resnet18_training import (
    BATCH_SIZE,
    DEFAULT_RANDOM_SEED,
    DROP_LAST,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    format_project_relative_path,
    restore_best_checkpoint,
)
from src.data.data_loader import create_vision_data_loaders
from src.evaluation.classification_metrics import (
    BinaryClassificationMetrics,
    calculate_binary_classification_metrics,
)
from src.evaluation.evaluation_runner import (
    BinaryEvaluationResult,
    evaluate_binary_classifier,
)
from src.models.resnet18_transfer import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    ResNet18Transfer,
)
from src.reproducibility import set_global_random_seed
from src.training.loss_function import create_binary_classification_loss


PROJECT_NAME = "Manufacturing Vision Defect Analysis System"
PROJECT_NAME_KOREAN = "제조 비전 결함 분석 시스템"
RUN_NAME = "day4_resnet18_transfer_test_evaluation"
COMPARISON_RUN_NAME = "day4_cnn_resnet18_comparison"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESNET18_CHECKPOINT_PATH = Path(
    "models/checkpoints/resnet18_transfer_best.pt"
)
DEFAULT_RESNET18_HISTORY_PATH = Path(
    "reports/artifacts/day4_resnet18_training_history.json"
)
DEFAULT_CNN_EVALUATION_PATH = Path(
    "reports/artifacts/day3_cnn_baseline_test_evaluation.json"
)
DEFAULT_RESNET18_EVALUATION_PATH = Path(
    "reports/artifacts/day4_resnet18_test_evaluation.json"
)
DEFAULT_COMPARISON_PATH = Path(
    "reports/artifacts/day4_cnn_resnet18_comparison.json"
)

EXPECTED_TEST_SAMPLE_COUNT = 715
EXPECTED_NORMAL_COUNT = 262
EXPECTED_DEFECT_COUNT = 453

INDEX_TO_CLASS_NAME = {
    0: "NORMAL",
    1: "DEFECT",
}


# =============================================================================
# Command Line
# =============================================================================


def parse_arguments(
    arguments: Sequence[str] | None = None,
) -> argparse.Namespace:
    """평가 실행 Argument를 해석한다."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Day 4 ResNet18 best checkpoint on the fixed "
            "test split and compare it with the Day 3 CNN baseline."
        )
    )

    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_RESNET18_CHECKPOINT_PATH,
    )
    parser.add_argument(
        "--training-history-path",
        type=Path,
        default=DEFAULT_RESNET18_HISTORY_PATH,
    )
    parser.add_argument(
        "--cnn-evaluation-path",
        type=Path,
        default=DEFAULT_CNN_EVALUATION_PATH,
    )
    parser.add_argument(
        "--evaluation-output-path",
        type=Path,
        default=DEFAULT_RESNET18_EVALUATION_PATH,
    )
    parser.add_argument(
        "--comparison-output-path",
        type=Path,
        default=DEFAULT_COMPARISON_PATH,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate paths, checkpoint restoration, and dataset only.",
    )

    return parser.parse_args(arguments)


# =============================================================================
# Path·JSON Utilities
# =============================================================================


def resolve_project_path(path: Path) -> Path:
    """상대 경로를 Project Root 기준 절대 경로로 변환한다."""
    if not isinstance(path, Path):
        raise TypeError("path must be pathlib.Path.")

    if path.is_absolute():
        return path.resolve()

    return (PROJECT_ROOT / path).resolve()


def read_json_object(path: Path) -> dict[str, Any]:
    """UTF-8 JSON 파일을 Dictionary로 읽고 최상위 형식을 검증한다."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file does not exist: {path}.")

    with path.open(mode="r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    if not isinstance(payload, dict):
        raise TypeError(
            "JSON top-level value must be an object. "
            f"Received: {type(payload).__name__}."
        )

    return payload


def write_json_atomically(
    *,
    payload: Mapping[str, Any],
    output_path: Path,
) -> None:
    """JSON을 임시 파일에 기록한 뒤 최종 경로로 원자적 교체한다."""
    if output_path.suffix.lower() != ".json":
        raise ValueError("output_path must use .json extension.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
            newline="\n",
        ) as output_file:
            json.dump(
                dict(payload),
                output_file,
                ensure_ascii=False,
                indent=2,
            )
            output_file.write("\n")

        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# =============================================================================
# Validation
# =============================================================================


def validate_execution_paths(
    *,
    checkpoint_path: Path,
    training_history_path: Path,
    cnn_evaluation_path: Path,
    evaluation_output_path: Path,
    comparison_output_path: Path,
) -> None:
    """평가 입력 Artifact와 출력 경로 충돌 여부를 검증한다."""
    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError("checkpoint_path must use .pt or .pth extension.")

    for path_name, path in {
        "training_history_path": training_history_path,
        "cnn_evaluation_path": cnn_evaluation_path,
        "evaluation_output_path": evaluation_output_path,
        "comparison_output_path": comparison_output_path,
    }.items():
        if path.suffix.lower() != ".json":
            raise ValueError(f"{path_name} must use .json extension.")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"ResNet18 checkpoint does not exist: {checkpoint_path}."
        )

    if not training_history_path.is_file():
        raise FileNotFoundError(
            "ResNet18 training history does not exist: "
            f"{training_history_path}."
        )

    if not cnn_evaluation_path.is_file():
        raise FileNotFoundError(
            f"CNN evaluation artifact does not exist: {cnn_evaluation_path}."
        )

    if evaluation_output_path == comparison_output_path:
        raise ValueError(
            "evaluation_output_path and comparison_output_path must differ."
        )

    protected_inputs = {
        checkpoint_path,
        training_history_path,
        cnn_evaluation_path,
    }
    for output_path in {evaluation_output_path, comparison_output_path}:
        if output_path in protected_inputs:
            raise ValueError(
                "Output path must not overwrite an input artifact: "
                f"{output_path}."
            )


def validate_test_dataset(data_loaders: Any) -> None:
    """고정 Test Split과 DataLoader 설정이 Day 3 기준과 같은지 검증한다."""
    test_dataset = data_loaders.test_dataset
    samples = test_dataset.samples

    if len(test_dataset) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError(
            "Unexpected Test Dataset size. "
            f"Expected: {EXPECTED_TEST_SAMPLE_COUNT}. "
            f"Received: {len(test_dataset)}."
        )

    if len(samples) != EXPECTED_TEST_SAMPLE_COUNT:
        raise ValueError("Dataset samples length does not match dataset length.")

    label_counts = {0: 0, 1: 0}
    for sample in samples:
        label = int(sample.label)
        if label not in label_counts:
            raise ValueError(f"Unexpected Test label: {label}.")
        label_counts[label] += 1

    if label_counts != {
        0: EXPECTED_NORMAL_COUNT,
        1: EXPECTED_DEFECT_COUNT,
    }:
        raise ValueError(
            "Unexpected Test class counts. "
            f"Received: {label_counts}."
        )

    if len(data_loaders.test_loader) != 23:
        raise ValueError(
            "Unexpected Test batch count. "
            f"Expected: 23. Received: {len(data_loaders.test_loader)}."
        )


def validate_evaluation_result(
    *,
    evaluation_result: BinaryEvaluationResult,
    metrics: BinaryClassificationMetrics,
    test_dataset: Any,
) -> None:
    """Evaluation Runner·Metric·Dataset 간 Sample 단위 일관성을 검증한다."""
    if evaluation_result.sample_count != len(test_dataset):
        raise ValueError("Evaluation sample_count does not match Test Dataset.")

    if metrics.sample_count != len(test_dataset):
        raise ValueError("Metric sample_count does not match Test Dataset.")

    expected_length = len(test_dataset)
    for tensor_name, tensor in {
        "labels": evaluation_result.labels,
        "logits": evaluation_result.logits,
        "probabilities": evaluation_result.probabilities,
        "predictions": evaluation_result.predictions,
    }.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{tensor_name} must be torch.Tensor.")
        if tensor.device.type != "cpu":
            raise ValueError(f"{tensor_name} must be returned on CPU.")
        if tensor.ndim != 1 or tensor.numel() != expected_length:
            raise ValueError(
                f"{tensor_name} must have shape [{expected_length}]."
            )

    dataset_labels = torch.tensor(
        [int(sample.label) for sample in test_dataset.samples],
        dtype=torch.int64,
    )

    if not torch.equal(
        evaluation_result.labels.to(torch.int64),
        dataset_labels,
    ):
        raise ValueError(
            "Evaluation label order does not match test_dataset.samples. "
            "The per-sample Artifact would be misaligned."
        )

    confusion_total = int(metrics.confusion_matrix.sum().item())
    if confusion_total != expected_length:
        raise ValueError("Confusion Matrix total does not match sample count.")

    correct_count = int(
        (evaluation_result.labels == evaluation_result.predictions)
        .sum()
        .item()
    )
    expected_accuracy = correct_count / expected_length

    if not math.isclose(
        metrics.accuracy,
        expected_accuracy,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Metric accuracy does not match collected predictions.")


# =============================================================================
# Per-Sample Result
# =============================================================================


def build_sample_results(
    *,
    test_dataset: Any,
    evaluation_result: BinaryEvaluationResult,
) -> list[dict[str, Any]]:
    """Dataset 순서와 Evaluation Tensor를 결합해 Sample별 결과를 만든다."""
    samples = test_dataset.samples

    if len(samples) != evaluation_result.sample_count:
        raise ValueError("Sample count and evaluation result length differ.")

    sample_results: list[dict[str, Any]] = []

    for sample_index, sample in enumerate(samples):
        ground_truth_label = int(evaluation_result.labels[sample_index].item())
        raw_logit = float(evaluation_result.logits[sample_index].item())
        defect_probability = float(
            evaluation_result.probabilities[sample_index].item()
        )
        prediction = int(evaluation_result.predictions[sample_index].item())

        if int(sample.label) != ground_truth_label:
            raise ValueError(
                "Dataset sample label and evaluation label differ at index "
                f"{sample_index}."
            )

        sample_results.append(
            {
                "sample_index": sample_index,
                "image_path": format_project_relative_path(sample.image_path),
                "ground_truth_label": ground_truth_label,
                "ground_truth_class_name": INDEX_TO_CLASS_NAME[
                    ground_truth_label
                ],
                "raw_logit": raw_logit,
                "defect_probability": defect_probability,
                "prediction": prediction,
                "prediction_class_name": INDEX_TO_CLASS_NAME[prediction],
                "correct": prediction == ground_truth_label,
            }
        )

    return sample_results


# =============================================================================
# Evaluation Artifact
# =============================================================================


def build_evaluation_payload(
    *,
    model: ResNet18Transfer,
    data_loaders: Any,
    evaluation_result: BinaryEvaluationResult,
    metrics: BinaryClassificationMetrics,
    sample_results: Sequence[Mapping[str, Any]],
    device: torch.device,
    evaluation_duration_seconds: float,
    checkpoint_path: Path,
    training_history_path: Path,
    evaluation_output_path: Path,
) -> dict[str, Any]:
    """실제 ResNet18 Test 평가 전체를 JSON 직렬화 가능한 구조로 만든다."""
    counts = model.parameter_counts()
    probabilities = evaluation_result.probabilities

    correct_count = metrics.true_negative + metrics.true_positive
    incorrect_count = metrics.false_positive + metrics.false_negative

    return {
        "project": {
            "name": PROJECT_NAME,
            "name_korean": PROJECT_NAME_KOREAN,
            "run_name": RUN_NAME,
        },
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
        },
        "reproducibility": {
            "random_seed": DEFAULT_RANDOM_SEED,
            "deterministic_algorithms": False,
        },
        "data": {
            "split": "test",
            "image_size": [224, 224],
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "drop_last": DROP_LAST,
            "persistent_workers": PERSISTENT_WORKERS,
            "normal_sample_count": EXPECTED_NORMAL_COUNT,
            "defect_sample_count": EXPECTED_DEFECT_COUNT,
            "sample_count": evaluation_result.sample_count,
            "batch_count": evaluation_result.batch_count,
            "positive_class": "DEFECT",
            "positive_class_index": 1,
        },
        "model": {
            "name": model.__class__.__name__,
            "module": model.__class__.__module__,
            "architecture": "torchvision_resnet18",
            "pretrained_during_training": True,
            "checkpoint_restored_with_weights_none": True,
            "freeze_backbone": model.freeze_backbone,
            "batchnorm_policy": "frozen_backbone_eval",
            "classification_head": "Linear(512, 1)",
            "output": "single_binary_raw_logit",
            "total_parameter_count": counts.total,
            "trainable_parameter_count": counts.trainable,
            "frozen_parameter_count": counts.frozen,
            "gradcam_target_layer": model.gradcam_target_layer_name,
        },
        "evaluation_configuration": {
            "loss_function": "BCEWithLogitsLoss",
            "classification_threshold": (
                evaluation_result.classification_threshold
            ),
            "confusion_matrix_order": [["TN", "FP"], ["FN", "TP"]],
            "zero_division_policy": 0.0,
        },
        "metrics": {
            "test_loss": evaluation_result.average_loss,
            "accuracy": metrics.accuracy,
            "accuracy_percent": metrics.accuracy * 100.0,
            "precision": metrics.precision,
            "precision_percent": metrics.precision * 100.0,
            "recall": metrics.recall,
            "recall_percent": metrics.recall * 100.0,
            "f1_score": metrics.f1_score,
            "f1_score_percent": metrics.f1_score * 100.0,
        },
        "confusion_matrix": {
            "matrix": metrics.confusion_matrix.to(torch.int64).tolist(),
            "true_negative": metrics.true_negative,
            "false_positive": metrics.false_positive,
            "false_negative": metrics.false_negative,
            "true_positive": metrics.true_positive,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
        },
        "probability_statistics": {
            "minimum": float(probabilities.min().item()),
            "maximum": float(probabilities.max().item()),
            "mean": float(probabilities.mean().item()),
        },
        "runtime": {
            "evaluation_duration_seconds": evaluation_duration_seconds,
        },
        "artifacts": {
            "checkpoint_path": format_project_relative_path(checkpoint_path),
            "training_history_path": format_project_relative_path(
                training_history_path
            ),
            "evaluation_output_path": format_project_relative_path(
                evaluation_output_path
            ),
        },
        "sample_results": [dict(item) for item in sample_results],
    }


# =============================================================================
# CNN Metric Extraction·Comparison
# =============================================================================


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "test_loss": ("test_loss", "average_loss", "loss"),
    "accuracy": ("accuracy",),
    "precision": ("precision",),
    "recall": ("recall",),
    "f1_score": ("f1_score", "f1"),
}


def _iter_mapping_nodes(value: Any) -> list[Mapping[str, Any]]:
    """JSON 구조 안의 모든 Mapping Node를 깊이 우선으로 수집한다."""
    nodes: list[Mapping[str, Any]] = []

    def visit(current: Any) -> None:
        if isinstance(current, Mapping):
            nodes.append(current)
            for child in current.values():
                visit(child)
        elif isinstance(current, list):
            for child in current:
                visit(child)

    visit(value)
    return nodes


def _to_finite_float(value: Any, *, name: str) -> float:
    """bool을 제외한 유한 실수로 변환한다."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool.")

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    return numeric_value


def extract_binary_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Day 3 JSON 구조가 조금 달라도 Metric 묶음을 안전하게 찾는다.

    우선순위는 한 Dictionary 안에 Accuracy·Precision·Recall·F1이 모두 있는
    Node이다. Loss는 같은 Node에서 찾고, 없으면 전체 JSON에서 별도로 찾는다.
    """
    nodes = _iter_mapping_nodes(payload)

    best_node: Mapping[str, Any] | None = None
    best_score = -1

    for node in nodes:
        node_keys = set(node.keys())
        score = sum(
            any(alias in node_keys for alias in aliases)
            for metric_name, aliases in _METRIC_ALIASES.items()
            if metric_name != "test_loss"
        )

        if score > best_score:
            best_node = node
            best_score = score

    if best_node is None or best_score < 4:
        raise KeyError(
            "Could not find a binary classification metric object containing "
            "accuracy, precision, recall, and f1_score."
        )

    extracted: dict[str, float] = {}

    for metric_name in ("accuracy", "precision", "recall", "f1_score"):
        for alias in _METRIC_ALIASES[metric_name]:
            if alias in best_node:
                extracted[metric_name] = _to_finite_float(
                    best_node[alias],
                    name=metric_name,
                )
                break
        else:
            raise KeyError(f"Missing metric: {metric_name}.")

    for alias in _METRIC_ALIASES["test_loss"]:
        if alias in best_node:
            extracted["test_loss"] = _to_finite_float(
                best_node[alias],
                name="test_loss",
            )
            break

    if "test_loss" not in extracted:
        for node in nodes:
            for alias in _METRIC_ALIASES["test_loss"]:
                if alias in node:
                    try:
                        extracted["test_loss"] = _to_finite_float(
                            node[alias],
                            name="test_loss",
                        )
                    except (TypeError, ValueError):
                        continue
                    break
            if "test_loss" in extracted:
                break

    if "test_loss" not in extracted:
        raise KeyError("Could not find Test Loss in CNN evaluation artifact.")

    for metric_name in ("accuracy", "precision", "recall", "f1_score"):
        metric_value = extracted[metric_name]
        if not 0.0 <= metric_value <= 1.0:
            raise ValueError(
                f"{metric_name} must be in [0, 1]. Received: {metric_value}."
            )

    return extracted


def build_comparison_payload(
    *,
    cnn_metrics: Mapping[str, float],
    resnet18_metrics: Mapping[str, float],
    cnn_evaluation_path: Path,
    resnet18_evaluation_path: Path,
    comparison_output_path: Path,
) -> dict[str, Any]:
    """CNNBaseline과 ResNet18의 공정 비교 결과를 만든다."""
    metric_names = (
        "test_loss",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    )

    for model_name, model_metrics in {
        "cnn_baseline": cnn_metrics,
        "resnet18_transfer": resnet18_metrics,
    }.items():
        missing = set(metric_names) - set(model_metrics.keys())
        if missing:
            raise KeyError(
                f"{model_name} metrics are missing: {sorted(missing)}."
            )

    comparison: dict[str, Any] = {}

    for metric_name in metric_names:
        cnn_value = float(cnn_metrics[metric_name])
        resnet_value = float(resnet18_metrics[metric_name])
        delta = resnet_value - cnn_value

        if metric_name == "test_loss":
            better_model = (
                "resnet18_transfer"
                if resnet_value < cnn_value
                else "cnn_baseline"
                if cnn_value < resnet_value
                else "tie"
            )
        else:
            better_model = (
                "resnet18_transfer"
                if resnet_value > cnn_value
                else "cnn_baseline"
                if cnn_value > resnet_value
                else "tie"
            )

        comparison[metric_name] = {
            "cnn_baseline": cnn_value,
            "resnet18_transfer": resnet_value,
            "resnet_minus_cnn": delta,
            "resnet_minus_cnn_percentage_points": (
                None if metric_name == "test_loss" else delta * 100.0
            ),
            "better_model": better_model,
        }

    return {
        "project": {
            "name": PROJECT_NAME,
            "name_korean": PROJECT_NAME_KOREAN,
            "run_name": COMPARISON_RUN_NAME,
        },
        "fair_comparison_conditions": {
            "train_sample_count": 5306,
            "validation_sample_count": 1327,
            "test_sample_count": 715,
            "image_size": [224, 224],
            "class_mapping": {"NORMAL": 0, "DEFECT": 1},
            "positive_class": "DEFECT",
            "classification_threshold": 0.5,
            "best_model_selection_metric": "validation_loss",
            "same_train_transform": True,
            "same_validation_transform": True,
            "same_test_transform": True,
            "same_loss_function": "BCEWithLogitsLoss",
            "same_metric_implementation": True,
        },
        "models": {
            "cnn_baseline": {
                "architecture": "custom_cnn_baseline",
                "pretrained": False,
                "parameter_count": 6065,
            },
            "resnet18_transfer": {
                "architecture": "torchvision_resnet18",
                "pretrained": True,
                "freeze_backbone": True,
                "total_parameter_count": 11177025,
                "trainable_parameter_count": 513,
            },
        },
        "comparison": comparison,
        "summary": {
            "accuracy_improvement_percentage_points": (
                comparison["accuracy"][
                    "resnet_minus_cnn_percentage_points"
                ]
            ),
            "recall_improvement_percentage_points": (
                comparison["recall"][
                    "resnet_minus_cnn_percentage_points"
                ]
            ),
            "f1_improvement_percentage_points": (
                comparison["f1_score"][
                    "resnet_minus_cnn_percentage_points"
                ]
            ),
            "primary_winner": comparison["f1_score"]["better_model"],
        },
        "artifacts": {
            "cnn_evaluation_path": format_project_relative_path(
                cnn_evaluation_path
            ),
            "resnet18_evaluation_path": format_project_relative_path(
                resnet18_evaluation_path
            ),
            "comparison_output_path": format_project_relative_path(
                comparison_output_path
            ),
        },
    }


# =============================================================================
# Console Output
# =============================================================================


def print_execution_configuration(
    *,
    checkpoint_path: Path,
    training_history_path: Path,
    cnn_evaluation_path: Path,
    evaluation_output_path: Path,
    comparison_output_path: Path,
    data_loaders: Any,
    model: ResNet18Transfer,
    device: torch.device,
    validate_only: bool,
) -> None:
    """평가 전 고정 조건과 Artifact 경로를 출력한다."""
    counts = model.parameter_counts()

    print("=" * 100)
    print("DAY 4 - RESNET18 TRANSFER LEARNING TEST EVALUATION")
    print("=" * 100)
    print()
    print("[MODEL]")
    print(f"Model                      : {model.__class__.__name__}")
    print(f"Device                     : {device}")
    print(f"Total parameters           : {counts.total}")
    print(f"Trainable parameters       : {counts.trainable}")
    print(f"Frozen parameters          : {counts.frozen}")
    print(f"Model evaluation mode      : {not model.training}")
    print()
    print("[TEST DATASET]")
    print(f"Test samples               : {len(data_loaders.test_dataset)}")
    print(f"NORMAL samples             : {EXPECTED_NORMAL_COUNT}")
    print(f"DEFECT samples             : {EXPECTED_DEFECT_COUNT}")
    print(f"Test batches               : {len(data_loaders.test_loader)}")
    print(f"Batch size                 : {BATCH_SIZE}")
    print("Image size                 : 224 x 224")
    print("Positive class             : DEFECT")
    print(f"Classification threshold  : {DEFAULT_CLASSIFICATION_THRESHOLD}")
    print()
    print("[INPUT ARTIFACTS]")
    print(f"Best checkpoint            : {checkpoint_path}")
    print(f"Training history           : {training_history_path}")
    print(f"CNN evaluation             : {cnn_evaluation_path}")
    print()
    print("[OUTPUT ARTIFACTS]")
    print(f"ResNet18 evaluation        : {evaluation_output_path}")
    print(f"CNN/ResNet18 comparison    : {comparison_output_path}")
    print(f"Validate only              : {validate_only}")


def print_evaluation_completion(
    *,
    evaluation_result: BinaryEvaluationResult,
    metrics: BinaryClassificationMetrics,
    evaluation_duration_seconds: float,
    evaluation_output_path: Path,
    comparison_output_path: Path,
    comparison_payload: Mapping[str, Any],
) -> None:
    """실제 Test 평가와 비교 결과를 읽기 쉬운 형태로 출력한다."""
    print()
    print("=" * 100)
    print("DAY 4 - RESNET18 TEST EVALUATION COMPLETED")
    print("=" * 100)
    print()
    print("[TEST METRICS]")
    print(f"Test loss                  : {evaluation_result.average_loss:.12f}")
    print(f"Accuracy                   : {metrics.accuracy * 100.0:.2f}%")
    print(f"Precision                  : {metrics.precision * 100.0:.2f}%")
    print(f"Recall                     : {metrics.recall * 100.0:.2f}%")
    print(f"F1 score                   : {metrics.f1_score * 100.0:.2f}%")
    print()
    print("[CONFUSION MATRIX]")
    print(metrics.confusion_matrix)
    print(f"True negative              : {metrics.true_negative}")
    print(f"False positive             : {metrics.false_positive}")
    print(f"False negative             : {metrics.false_negative}")
    print(f"True positive              : {metrics.true_positive}")
    print()
    print("[PROBABILITY]")
    print(
        "Minimum                    : "
        f"{evaluation_result.probabilities.min().item():.12f}"
    )
    print(
        "Maximum                    : "
        f"{evaluation_result.probabilities.max().item():.12f}"
    )
    print(
        "Mean                       : "
        f"{evaluation_result.probabilities.mean().item():.12f}"
    )
    print()
    print("[CNN BASELINE COMPARISON]")
    comparison = comparison_payload["comparison"]
    print(
        "Accuracy improvement       : "
        f"{comparison['accuracy']['resnet_minus_cnn_percentage_points']:+.2f}%p"
    )
    print(
        "Recall improvement         : "
        f"{comparison['recall']['resnet_minus_cnn_percentage_points']:+.2f}%p"
    )
    print(
        "F1 improvement             : "
        f"{comparison['f1_score']['resnet_minus_cnn_percentage_points']:+.2f}%p"
    )
    print()
    print("[RUNTIME]")
    print(f"Evaluation seconds         : {evaluation_duration_seconds:.2f}")
    print()
    print("[ARTIFACTS]")
    print(f"ResNet18 evaluation JSON   : {evaluation_output_path}")
    print(f"Comparison JSON            : {comparison_output_path}")
    print(f"Evaluation JSON exists     : {evaluation_output_path.is_file()}")
    print(f"Comparison JSON exists     : {comparison_output_path.is_file()}")
    print()
    print("[PASS] Day 4 ResNet18 transfer learning real test evaluation")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Day 4 ResNet18 실제 Test 평가 Entry Point."""
    arguments = parse_arguments()

    checkpoint_path = resolve_project_path(arguments.checkpoint_path)
    training_history_path = resolve_project_path(
        arguments.training_history_path
    )
    cnn_evaluation_path = resolve_project_path(arguments.cnn_evaluation_path)
    evaluation_output_path = resolve_project_path(
        arguments.evaluation_output_path
    )
    comparison_output_path = resolve_project_path(
        arguments.comparison_output_path
    )

    validate_execution_paths(
        checkpoint_path=checkpoint_path,
        training_history_path=training_history_path,
        cnn_evaluation_path=cnn_evaluation_path,
        evaluation_output_path=evaluation_output_path,
        comparison_output_path=comparison_output_path,
    )

    settings = set_global_random_seed(
        seed=DEFAULT_RANDOM_SEED,
        deterministic_algorithms=False,
    )
    device = torch.device(settings.device)

    data_loaders = create_vision_data_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=DROP_LAST,
        persistent_workers=PERSISTENT_WORKERS,
        random_seed=DEFAULT_RANDOM_SEED,
    )
    validate_test_dataset(data_loaders)

    model = restore_best_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
    loss_function = create_binary_classification_loss()

    print_execution_configuration(
        checkpoint_path=checkpoint_path,
        training_history_path=training_history_path,
        cnn_evaluation_path=cnn_evaluation_path,
        evaluation_output_path=evaluation_output_path,
        comparison_output_path=comparison_output_path,
        data_loaders=data_loaders,
        model=model,
        device=device,
        validate_only=arguments.validate_only,
    )

    if arguments.validate_only:
        _ = read_json_object(training_history_path)
        cnn_payload = read_json_object(cnn_evaluation_path)
        cnn_metrics = extract_binary_metrics(cnn_payload)

        print()
        print("[CNN METRIC EXTRACTION]")
        for metric_name, metric_value in cnn_metrics.items():
            print(f"{metric_name:<26}: {metric_value}")

        print()
        print("=" * 100)
        print("[PASS] Day 4 ResNet18 evaluation structure validation")
        print("=" * 100)
        return

    evaluation_started_at = time.perf_counter()

    evaluation_result = evaluate_binary_classifier(
        model=model,
        data_loader=data_loaders.test_loader,
        loss_function=loss_function,
        device=device,
        classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    )

    evaluation_duration_seconds = time.perf_counter() - evaluation_started_at

    metrics = calculate_binary_classification_metrics(
        labels=evaluation_result.labels,
        predictions=evaluation_result.predictions,
    )

    validate_evaluation_result(
        evaluation_result=evaluation_result,
        metrics=metrics,
        test_dataset=data_loaders.test_dataset,
    )

    sample_results = build_sample_results(
        test_dataset=data_loaders.test_dataset,
        evaluation_result=evaluation_result,
    )

    evaluation_payload = build_evaluation_payload(
        model=model,
        data_loaders=data_loaders,
        evaluation_result=evaluation_result,
        metrics=metrics,
        sample_results=sample_results,
        device=device,
        evaluation_duration_seconds=evaluation_duration_seconds,
        checkpoint_path=checkpoint_path,
        training_history_path=training_history_path,
        evaluation_output_path=evaluation_output_path,
    )

    write_json_atomically(
        payload=evaluation_payload,
        output_path=evaluation_output_path,
    )

    cnn_payload = read_json_object(cnn_evaluation_path)
    cnn_metrics = extract_binary_metrics(cnn_payload)
    resnet18_metrics = extract_binary_metrics(evaluation_payload)

    comparison_payload = build_comparison_payload(
        cnn_metrics=cnn_metrics,
        resnet18_metrics=resnet18_metrics,
        cnn_evaluation_path=cnn_evaluation_path,
        resnet18_evaluation_path=evaluation_output_path,
        comparison_output_path=comparison_output_path,
    )

    write_json_atomically(
        payload=comparison_payload,
        output_path=comparison_output_path,
    )

    if not evaluation_output_path.is_file():
        raise FileNotFoundError(
            "ResNet18 evaluation JSON was not created: "
            f"{evaluation_output_path}."
        )

    if not comparison_output_path.is_file():
        raise FileNotFoundError(
            "CNN/ResNet18 comparison JSON was not created: "
            f"{comparison_output_path}."
        )

    print_evaluation_completion(
        evaluation_result=evaluation_result,
        metrics=metrics,
        evaluation_duration_seconds=evaluation_duration_seconds,
        evaluation_output_path=evaluation_output_path,
        comparison_output_path=comparison_output_path,
        comparison_payload=comparison_payload,
    )


if __name__ == "__main__":
    main()
