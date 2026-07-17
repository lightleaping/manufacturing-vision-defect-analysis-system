"""Day 12 Best Checkpoint의 Validation·Test 최종 평가와 Failure Analysis.

실행 예:
    python -m scripts.run_day12_detection_evaluation `
        --project-root . `
        --torch-num-threads 2 `
        --log-interval 50

이 Script에서 Day 12 최초로 Test Split을 사용한다. Test 결과를 본 뒤 학습
설정이나 Best Checkpoint를 변경하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import time
from typing import Any
from urllib.parse import urlparse

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)

from src.detection.checkpoint import load_detection_checkpoint_payload
from src.detection.data_loader import detection_collate_fn
from src.detection.detection_dataset import NeuDetDetectionDataset
from src.detection.epoch_runner import (
    build_detection_checkpoint_class_mapping,
    run_detection_evaluation_epoch,
)
from src.detection.evaluation import calculate_detection_iou_sweep
from src.detection.failure_analysis import analyze_detection_failures
from src.detection.model_config import DetectionModelConfig
from src.detection.model_factory import create_detection_model
from src.detection.training_visualization import (
    create_detection_failure_montage,
    create_detection_prediction_montage,
    plot_detection_class_metrics,
    plot_detection_training_history,
)
from src.detection.transforms import create_detection_transform


DEFAULT_MANIFEST = Path("data/processed/neu_det/splits.json")
DEFAULT_BEST_CHECKPOINT = Path("models/detection/day12_detection_best.pt")
DEFAULT_EVALUATION_ARTIFACT = Path(
    "reports/artifacts/day12_detection_evaluation.json"
)
DEFAULT_FAILURE_ARTIFACT = Path(
    "reports/artifacts/day12_detection_failure_analysis.json"
)
DEFAULT_TRAINING_HISTORY_FIGURE = Path(
    "reports/figures/day12_detection_training_history.png"
)
DEFAULT_CLASS_METRICS_FIGURE = Path(
    "reports/figures/day12_detection_class_metrics.png"
)
DEFAULT_PREDICTIONS_FIGURE = Path(
    "reports/figures/day12_detection_predictions.png"
)
DEFAULT_FAILURE_FIGURE = Path(
    "reports/figures/day12_detection_failure_analysis.png"
)
MINIMUM_FREE_GIB = 0.5


def _torch_cache_checkpoint_directory() -> Path:
    torch_home = os.getenv("TORCH_HOME")
    root = (
        Path(torch_home).expanduser().resolve()
        if torch_home
        else (Path.home() / ".cache" / "torch").resolve()
    )
    return root / "hub" / "checkpoints"


def _expected_weight_path() -> Path:
    weight = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    filename = Path(urlparse(weight.url).path).name
    return _torch_cache_checkpoint_directory() / filename


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _create_loader(dataset: NeuDetDetectionDataset) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        persistent_workers=False,
        collate_fn=detection_collate_fn,
    )


def _progress_printer(payload: dict[str, Any]) -> None:
    if payload.get("event") != "evaluation_progress":
        return
    split = str(payload["split"]).upper()
    print(
        f"[{split}] "
        f"batch={payload['batch_count']} "
        f"samples={payload['sample_count']} "
        f"boxes={payload['prediction_box_count']} "
        f"elapsed={payload['elapsed_seconds']:.2f}s",
        flush=True,
    )


def run_day12_detection_evaluation(
    *,
    project_root: Path,
    torch_num_threads: int = 2,
    log_interval: int = 50,
    score_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Validation과 Test를 평가하고 JSON·Figure Artifact를 생성한다."""
    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path.")
    for name, value in (
        ("torch_num_threads", torch_num_threads),
        ("log_interval", log_interval),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive int.")
    for name, value in (
        ("score_threshold", score_threshold),
        ("iou_threshold", iou_threshold),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0.0 < float(value) <= 1.0
        ):
            raise ValueError(f"{name} must be in (0, 1].")

    project_root = project_root.resolve()
    manifest_path = project_root / DEFAULT_MANIFEST
    best_path = project_root / DEFAULT_BEST_CHECKPOINT
    evaluation_path = project_root / DEFAULT_EVALUATION_ARTIFACT
    failure_path = project_root / DEFAULT_FAILURE_ARTIFACT
    figure_paths = {
        "training_history": project_root / DEFAULT_TRAINING_HISTORY_FIGURE,
        "class_metrics": project_root / DEFAULT_CLASS_METRICS_FIGURE,
        "predictions": project_root / DEFAULT_PREDICTIONS_FIGURE,
        "failure_analysis": project_root / DEFAULT_FAILURE_FIGURE,
    }
    for path, description in (
        (manifest_path, "Split manifest"),
        (best_path, "Best checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{description} does not exist: {path}.")
    free_before = shutil.disk_usage(project_root).free / (1024**3)
    if free_before < MINIMUM_FREE_GIB:
        raise OSError(
            f"At least {MINIMUM_FREE_GIB:.1f} GiB free space is required. "
            f"Current: {free_before:.3f} GiB."
        )
    expected_weight = _expected_weight_path()
    if not expected_weight.is_file():
        raise FileNotFoundError("COCO pretrained Detection weight is not cached.")

    model_config = DetectionModelConfig(
        min_size=320,
        max_size=320,
        use_pretrained_weights=True,
        use_pretrained_backbone=False,
        progress=False,
    )
    checkpoint_mapping = build_detection_checkpoint_class_mapping(
        model_config.index_to_class
    )
    checkpoint_payload = load_detection_checkpoint_payload(
        best_path,
        map_location="cpu",
    )
    if checkpoint_payload["class_mapping"] != checkpoint_mapping:
        raise ValueError("Best checkpoint class mapping does not match NEU-DET.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(torch_num_threads)
    started = time.perf_counter()
    try:
        validation_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="validation",
            transform=create_detection_transform(training=False),
            duplicate_box_policy="preserve",
        )
        test_dataset = NeuDetDetectionDataset.from_manifest(
            manifest_path=manifest_path,
            project_root=project_root,
            split="test",
            transform=create_detection_transform(training=False),
            duplicate_box_policy="preserve",
        )
        validation_loader = _create_loader(validation_dataset)
        test_loader = _create_loader(test_dataset)

        model_started = time.perf_counter()
        model_result = create_detection_model(
            config=model_config,
            device="cpu",
            training=False,
            proposal_limits=None,
        )
        model = model_result.model
        model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
        model.eval()
        model_prepare_seconds = time.perf_counter() - model_started

        print("=" * 100)
        print("DAY 12 - DETECTION FINAL VALIDATION, TEST AND FAILURE ANALYSIS")
        print("=" * 100)
        print(f"Project root             : {project_root}")
        print(f"Best checkpoint epoch    : {checkpoint_payload['epoch']}")
        print(f"Best validation mAP@0.50 : {checkpoint_payload['best_metric']:.6f}")
        print(f"Validation images        : {len(validation_dataset)}")
        print(f"Test images              : {len(test_dataset)}")
        print(f"Score threshold          : {float(score_threshold):.2f}")
        print(f"IoU threshold            : {float(iou_threshold):.2f}")
        print("[BEST CHECKPOINT FROZEN] [TEST SPLIT USED FOR FINAL EVALUATION]")

        validation_result = run_detection_evaluation_epoch(
            model=model,
            data_loader=validation_loader,
            split="validation",
            num_classes=model_config.num_classes,
            index_to_class=model_config.index_to_class,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
            device="cpu",
            log_interval=log_interval,
            progress_callback=_progress_printer,
        )
        test_result = run_detection_evaluation_epoch(
            model=model,
            data_loader=test_loader,
            split="test",
            num_classes=model_config.num_classes,
            index_to_class=model_config.index_to_class,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
            device="cpu",
            log_interval=log_interval,
            progress_callback=_progress_printer,
        )
        validation_sweep = calculate_detection_iou_sweep(
            predictions=validation_result.predictions,
            targets=validation_result.targets,
            index_to_class=model_config.index_to_class,
            score_threshold=float(score_threshold),
        )
        test_sweep = calculate_detection_iou_sweep(
            predictions=test_result.predictions,
            targets=test_result.targets,
            index_to_class=model_config.index_to_class,
            score_threshold=float(score_threshold),
        )
        sample_ids = [sample.record_id for sample in test_dataset.samples]
        failure_analysis = analyze_detection_failures(
            predictions=test_result.predictions,
            targets=test_result.targets,
            index_to_class=model_config.index_to_class,
            sample_ids=sample_ids,
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
        )

        plot_detection_training_history(
            history=checkpoint_payload["history"],
            output_path=figure_paths["training_history"],
        )
        plot_detection_class_metrics(
            class_metrics=test_result.metrics["class_metrics"],
            output_path=figure_paths["class_metrics"],
        )
        create_detection_prediction_montage(
            dataset=test_dataset,
            predictions=test_result.predictions,
            targets=test_result.targets,
            index_to_class=model_config.index_to_class,
            output_path=figure_paths["predictions"],
            score_threshold=float(score_threshold),
            iou_threshold=float(iou_threshold),
        )
        create_detection_failure_montage(
            dataset=test_dataset,
            predictions=test_result.predictions,
            targets=test_result.targets,
            failure_analysis=failure_analysis,
            index_to_class=model_config.index_to_class,
            output_path=figure_paths["failure_analysis"],
        )

        evaluation_payload: dict[str, Any] = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "day12_detection_final_evaluation",
            "environment": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "torchvision": str(torchvision.__version__),
                "device": "cpu",
                "torch_num_threads": torch_num_threads,
            },
            "evaluation_policy": {
                "best_checkpoint_selected_on_validation": True,
                "best_checkpoint_frozen_before_test": True,
                "test_split_used": True,
                "test_result_used_for_model_selection": False,
                "score_threshold": float(score_threshold),
                "iou_threshold": float(iou_threshold),
                "duplicate_box_policy": "preserve",
            },
            "checkpoint": {
                "path": _relative_or_absolute(best_path, project_root),
                "epoch_index": int(checkpoint_payload["epoch"]),
                "completed_epoch_number": int(checkpoint_payload["epoch"]) + 1,
                "best_validation_metric_name": "map_50",
                "best_validation_metric": float(checkpoint_payload["best_metric"]),
                "class_mapping": checkpoint_payload["class_mapping"],
            },
            "data": {
                "validation_images": len(validation_dataset),
                "test_images": len(test_dataset),
            },
            "model": model_result.metadata,
            "validation": validation_result.summary(),
            "validation_iou_sweep": validation_sweep,
            "test": test_result.summary(),
            "test_iou_sweep": test_sweep,
            "timing": {
                "model_prepare_seconds": model_prepare_seconds,
                "total_execution_seconds": time.perf_counter() - started,
            },
            "storage": {
                "disk_free_gib_before": round(free_before, 3),
                "disk_free_gib_after": round(
                    shutil.disk_usage(project_root).free / (1024**3),
                    3,
                ),
            },
            "artifacts": {
                "evaluation": _relative_or_absolute(evaluation_path, project_root),
                "failure_analysis": _relative_or_absolute(failure_path, project_root),
                "figures": {
                    name: _relative_or_absolute(path, project_root)
                    for name, path in figure_paths.items()
                },
            },
        }
        failure_payload = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "day12_detection_test_failure_analysis",
            "checkpoint": evaluation_payload["checkpoint"],
            "split": "test",
            "analysis": failure_analysis,
            "figure": _relative_or_absolute(
                figure_paths["failure_analysis"],
                project_root,
            ),
        }
        _write_json(evaluation_path, evaluation_payload)
        _write_json(failure_path, failure_payload)

        validation_overall = validation_result.metrics["overall"]
        test_overall = test_result.metrics["overall"]
        print("-" * 100)
        print(f"Validation Precision     : {validation_overall['precision']:.6f}")
        print(f"Validation Recall        : {validation_overall['recall']:.6f}")
        print(f"Validation F1            : {validation_overall['f1']:.6f}")
        print(f"Validation mAP@0.50      : {validation_overall['map_50']}")
        print(f"Test Precision           : {test_overall['precision']:.6f}")
        print(f"Test Recall              : {test_overall['recall']:.6f}")
        print(f"Test F1                  : {test_overall['f1']:.6f}")
        print(f"Test mAP@0.50            : {test_overall['map_50']}")
        print(f"Test project mAP@.50:.95 : {test_sweep['summary']['map_50_95']}")
        print(f"Failure events           : {failure_analysis['summary']['event_count']}")
        print(f"Evaluation artifact      : {evaluation_path}")
        print(f"Failure artifact         : {failure_path}")
        print("[RESULT]                 : PASS")
        print("[TEST RESULT NOT USED FOR CHECKPOINT SELECTION]")
        return evaluation_payload
    finally:
        torch.set_num_threads(previous_threads)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Day 12 final Detection validation/test evaluation."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--torch-num-threads", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    return parser


def main() -> None:
    arguments = _build_parser().parse_args()
    run_day12_detection_evaluation(
        project_root=arguments.project_root,
        torch_num_threads=arguments.torch_num_threads,
        log_interval=arguments.log_interval,
        score_threshold=arguments.score_threshold,
        iou_threshold=arguments.iou_threshold,
    )


if __name__ == "__main__":
    main()
